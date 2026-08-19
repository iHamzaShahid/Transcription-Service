"""Normalize any accepted input to 16 kHz mono 16-bit PCM WAV.

Everything downstream (VAD, chunk slicing, the bytes handed to the ASR
backend) assumes exactly this format, so there is one decoder in the system
and it is ffmpeg's.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

import numpy as np

from app.audio import proc
from app.config import Settings
from app.errors import AudioProcessingError, UnsupportedMediaError

log = logging.getLogger(__name__)


async def normalize(src: Path, dst: Path, settings: Settings) -> Path:
    """Transcode `src` into `dst` as 16 kHz mono s16le WAV."""
    result = await proc.run(
        [
            settings.ffmpeg_bin,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(settings.target_sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(dst),
        ],
        timeout=settings.ffmpeg_timeout_sec,
    )

    if not result.ok:
        # ffprobe accepted the header but the stream would not decode: that is
        # still a bad upload, not a server fault.
        raise UnsupportedMediaError(
            "The audio stream could not be decoded. "
            f"ffmpeg said: {result.stderr_tail(200) or 'unknown error'}"
        )
    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioProcessingError("ffmpeg produced an empty output file")

    log.info("normalized audio", extra={"bytes": dst.stat().st_size})
    return dst


def read_wav_mono16(path: Path, expected_rate: int) -> np.ndarray:
    """Read a normalized WAV into a float32 array in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise AudioProcessingError("normalized file is not mono 16-bit PCM")
        if handle.getframerate() != expected_rate:
            raise AudioProcessingError(
                f"normalized file is {handle.getframerate()} Hz, "
                f"expected {expected_rate} Hz"
            )
        frames = handle.readframes(handle.getnframes())

    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    return pcm / 32768.0


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Serialize a float32 [-1, 1] array as an in-memory 16-bit PCM WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
