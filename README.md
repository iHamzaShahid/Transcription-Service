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

## Design decisions

**Normalize with ffmpeg at the front door.** Input can be WAV or MP3, any
sample rate, mono or stereo. Instead of teaching the VAD, the chunker and the
Gemini request to each handle all of that, everything past the entry point
sees one format: 16 kHz mono PCM. Silero needs 16 kHz and Gemini downsamples
to it anyway, so sending anything richer is wasted upload. It also means one
decoder in the whole system, and the tool that reads the bytes is the same one
that decides whether we can read them at all.

**Cut at silence, not at the clock.** 120 seconds is what the model handles
well in one request, but cutting exactly at 120.000s lands mid-word about half
the time, and that word gets mangled in both chunks. So the target is a
suggestion: the planner looks for the nearest silence within ±30s and cuts in
the middle of it, which leaves the most room for error on both sides. Measured
with a 60s target, cuts landed at 60.53s, 121.14s and 181.58s, always inside a
detected silence. If there's no silence in the window at all (music, noise,
unbroken speech) it hard-cuts at the target and marks the chunk.

**No overlap.** Overlap is the usual fix for split words, but cutting at
silence already solved that. What it costs is real: duplicate audio, duplicate
tokens, and a de-duplication step that has to work out which copy of "and then
he said" is the real one. That's fuzzy text matching over approximate
timestamps, which fails quietly and corrupts transcripts. Since no word spans
a boundary, chunks tile the timeline exactly instead
(`chunk[i].start == chunk[i-1].end`).

**Bound the parallelism.** Chunks are independent, so a long file is easy to
parallelise. But sending all of them at once means ~60 simultaneous requests
for a two hour file, a 429 on the whole batch, and 60 chunks of decoded audio
in memory together. A semaphore caps what's in flight regardless of file
length, so load depends on configuration and not on what someone uploaded.
Retries use jitter for the same reason: a batch rate-limited together would
otherwise all retry at the same instant.

**Validate timestamps before stitching.** These times come from a model
emitting numbers as tokens, not from a forced aligner, so they drift: ends
before starts, times past the end of the clip, segments going backwards. And
because each chunk's times get shifted by that chunk's offset, an unchecked
error doesn't stay local. A segment that overruns its chunk lands on top of the
next chunk's segments and the merged transcript reads out of order. So each
chunk is clamped to `[0, duration]` and forced monotonic *before* the offset is
added, which keeps every error inside its own chunk. The fixes are counted and
returned in `metadata.timestamps_repaired` rather than applied silently, since
200 repairs across 300 segments means something very different from zero.
