# Transcription Service

Audio in, timestamped JSON transcript out.

Uploads are validated by content rather than filename, normalized to a single
audio format, split at natural pauses in speech, transcribed in parallel by
Gemini, and stitched back onto one continuous timeline. Accepts WAV and MP3.
One endpoint, one request, one response — however long the file.

## Architecture

```
  POST /transcribe (multipart)
        |
        v  spool to temp file ......... streamed, over MAX_UPLOAD_BYTES -> 413
        v  ffprobe validate ........... real codec, not the extension -> 415
        v  ffmpeg normalize ........... 16 kHz mono s16le WAV
        |
        +-- duration <= CHUNK_SIZE + WINDOW ? --> yes: one chunk, VAD skipped
        |                                          |
        +-- no: Silero VAD (ONNX) -> speech spans  |
                chunk planner -> cut at nearest    |
                silence within +/-WINDOW, no overlap
                                     |             |
                                     +------+------+
                                            |
                    +-----------+-----------+-----------+
                    |  chunk 0  |  chunk 1  |  chunk N  |  <= MAX_PARALLEL_CHUNKS
                    |  Gemini   |  Gemini   |  Gemini   |  retry 3x, backoff+jitter
                    +-----------+-----------+-----------+  (a failure is recorded,
                                            |              not fatal)
                                            v
                       timestamp repair (clamp/drop/reorder, counted)
                       + offset stitching (chunk-local time + chunk.start)
                                            |
                                            v
                                  200 TranscriptionResult
```

Chunks are cut inside detected silence, so no word spans a boundary and the
chunks tile the timeline exactly — `chunk[i].start == chunk[i-1].end`, no
overlap and no gap. Gemini returns times relative to each chunk; those are
validated and clamped to the chunk before the chunk's offset is added, so a
bad timestamp can never leak into its neighbour. Every correction is counted
and reported in `metadata.timestamps_repaired`.

| Path | Responsibility |
|---|---|
| `app/api.py` | HTTP surface, request ids, error -> status |
| `app/cli.py` | same pipeline, file in / JSON out |
| `app/config.py` | every tunable, one `Settings` object |
| `app/asr.py` | Gemini call, structured output, error classification |
| `app/audio/` | `probe` (validate), `normalize`, `vad`, `chunking` (pure planning) |
| `app/pipeline/` | `runner` (orchestration), `repair` (timestamps), `retry` |

## Setup

Requires a `GEMINI_API_KEY` — the service refuses to start without one.
Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
cp .env.example .env        # then put your key in it
docker build -t transcription-service .
docker run --rm -p 8000:8000 --env-file .env transcription-service
```

Or locally (Python 3.11, `ffmpeg` on PATH):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...
uvicorn app.api:app --port 8000
```

The image bakes in the Silero VAD model; running locally it is downloaded once
(~2 MB) to `VAD_MODEL_CACHE_DIR`.

## Usage

```bash
curl localhost:8000/health

curl -X POST localhost:8000/transcribe -F "file=@audio.mp3"

curl -s -X POST localhost:8000/transcribe -F "file=@audio.mp3" | jq -r .text
```

The transcript comes back in the same response:

```json
{
  "segments": [
    { "id": 0, "start": 0.0, "end": 4.25, "text": "First phrase." },
    { "id": 1, "start": 4.5, "end": 9.1,  "text": "Second phrase." }
  ],
  "text": "First phrase. Second phrase.",
  "language": "en",
  "duration_sec": 9.1,
  "chunk_count": 1,
  "failed_chunks": [],
  "metadata": {
    "model": "gemini-3.1-flash-lite", "backend": "gemini",
    "original_format": "mp3", "sample_rate": 22050, "channels": 1,
    "processing_time_sec": 2.14, "timestamps_repaired": 0,
    "repair_detail": { "clamped": 0, "reordered": 0, "dropped": 0 }
  }
}
```

`sample_rate` and `channels` describe the **original** upload; the model always
sees 16 kHz mono. A chunk that exhausts its retries appears in `failed_chunks`
with its time range and error, and the rest of the transcript still returns.

**Errors:** `413` too large · `415` not WAV/MP3, undecodable, or empty ·
`500` ffmpeg failed. Every response carries `x-request-id`, which appears on
every log line for that request.

### CLI

```bash
python -m app.cli audio.mp3 --out result.json
python -m app.cli audio.mp3 | jq -r .text     # JSON on stdout, logs on stderr
```

Exit codes: `0` ok · `1` bad audio · `2` no such file.

## Configuration

See `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | **required** |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | model id |
| `CHUNK_SIZE_SEC` | `120` | target chunk length; shorter audio skips the VAD |
| `CHUNK_SEARCH_WINDOW_SEC` | `30` | how far either side to hunt for silence |
| `MAX_PARALLEL_CHUNKS` | `4` | chunks in flight to Gemini at once |
| `MAX_RETRIES` | `3` | attempts per chunk, on 429/503/timeout |
| `MAX_UPLOAD_BYTES` | `104857600` | 100 MB |
| `GEMINI_USE_VERTEX` | `false` | route via Vertex AI instead of the Developer API; needs `GOOGLE_CLOUD_PROJECT` |
