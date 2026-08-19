"""Format detection with ffprobe.

The file extension and the multipart content-type are both attacker//user
controlled, so neither is consulted. Only what ffprobe reads out of the actual
bytes decides whether we accept the upload.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.audio import proc
from app.config import Settings
from app.errors import UnsupportedMediaError
from app.schemas import AudioInfo

log = logging.getLogger(__name__)

# Codecs we accept == "WAV and MP3". WAV is a container, so accept the PCM
# family it usually carries; MP3 shows up as `mp3` or `mp3float` depending on
# which decoder ffmpeg picked.
ACCEPTED_CODECS = frozenset(
    {
        "mp3",
        "mp3float",
        "pcm_s16le",
        "pcm_s24le",
        "pcm_s32le",
        "pcm_u8",
        "pcm_f32le",
        "pcm_f64le",
    }
)
# ffprobe reports demuxer names; wav files may probe as `wav`, mp3 as `mp3`.
ACCEPTED_CONTAINERS = frozenset({"wav", "mp3"})


async def probe(path: Path, settings: Settings) -> AudioInfo:
    """Return facts about `path`, or raise UnsupportedMediaError."""
    result = await proc.run(
        [
            settings.ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=settings.ffmpeg_timeout_sec,
    )

    if not result.ok:
        raise UnsupportedMediaError(
            "Could not decode the uploaded file as audio. "
            f"ffprobe said: {result.stderr_tail(200) or 'unknown error'}"
        )

    try:
        payload = json.loads(result.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise UnsupportedMediaError("ffprobe returned unreadable output") from exc

    audio_streams = [
        s for s in payload.get("streams", []) if s.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise UnsupportedMediaError(
            "The uploaded file contains no audio stream. Send WAV or MP3."
        )

    stream = audio_streams[0]
    codec = str(stream.get("codec_name") or "unknown").lower()
    fmt = payload.get("format", {})
    # format_name is a comma-separated list of candidate demuxers, e.g. "mov,mp4".
    containers = {c.strip().lower() for c in str(fmt.get("format_name", "")).split(",")}

    if codec not in ACCEPTED_CODECS or not (containers & ACCEPTED_CONTAINERS):
        raise UnsupportedMediaError(
            f"Unsupported audio format: codec={codec!r} "
            f"container={','.join(sorted(containers)) or 'unknown'!r}. "
            "This service accepts WAV (PCM) and MP3 only."
        )

    duration = _first_float(stream.get("duration"), fmt.get("duration"))
    if duration is None or duration <= 0:
        raise UnsupportedMediaError(
            "Could not determine the duration of the uploaded audio; "
            "the file is likely truncated or corrupt."
        )

    info = AudioInfo(
        codec=codec,
        container=_pick_container(containers),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        duration_sec=duration,
        bit_rate=_int_or_none(fmt.get("bit_rate") or stream.get("bit_rate")),
    )
    log.info(
        "probed upload",
        extra={
            "codec": info.codec,
            "container": info.container,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "duration_sec": round(info.duration_sec, 3),
        },
    )
    return info


def _pick_container(containers: set[str]) -> str:
    for candidate in sorted(ACCEPTED_CONTAINERS):
        if candidate in containers:
            return candidate
    return "unknown"


def _first_float(*values: object) -> float | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _int_or_none(value: object) -> int | None:
    result = _first_float(value)
    return int(result) if result is not None else None
