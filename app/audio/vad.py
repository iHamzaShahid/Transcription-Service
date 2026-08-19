"""Voice activity detection: Silero VAD v5 on onnxruntime (no torch).

Answers one question — *where is there speech?* — and chunking
(app/audio/chunking.py) turns that into cut points.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)

# Silero v5 consumes exactly 512 samples per step at 16 kHz.
SILERO_WINDOW = 512
SILERO_STATE_SHAPE = (2, 1, 128)


@dataclass(frozen=True)
class TimeSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# Shared post-processing
# --------------------------------------------------------------------------


def _flags_to_segments(
    flags: np.ndarray,
    frame_sec: float,
    total_sec: float,
    *,
    min_speech_sec: float,
    min_silence_sec: float,
    pad_sec: float,
) -> list[TimeSpan]:
    """Turn a per-frame speech mask into padded, merged, filtered spans."""
    if flags.size == 0:
        return []

    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    # float() matters: numpy scalars would otherwise ride all the way into the
    # JSON response, where they are not serializable.
    spans = [
        TimeSpan(float(start) * frame_sec, float(end) * frame_sec)
        for start, end in zip(edges[0::2], edges[1::2])
    ]

    spans = [s for s in spans if s.duration >= min_speech_sec]
    spans = _merge(spans, gap_sec=min_silence_sec)
    if pad_sec > 0:
        spans = [
            TimeSpan(max(0.0, s.start - pad_sec), min(total_sec, s.end + pad_sec))
            for s in spans
        ]
        spans = _merge(spans, gap_sec=0.0)
    return spans


def _merge(spans: list[TimeSpan], *, gap_sec: float) -> list[TimeSpan]:
    """Merge spans separated by a gap shorter than `gap_sec`."""
    merged: list[TimeSpan] = []
    for span in spans:
        if merged and span.start - merged[-1].end < gap_sec:
            merged[-1] = TimeSpan(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


# --------------------------------------------------------------------------
# Silero (ONNX)
# --------------------------------------------------------------------------

_session_lock = threading.Lock()
_session_cache: dict[str, object] = {}


def resolve_model_path(settings: Settings) -> Path:
    """Find silero_vad.onnx, downloading it into the cache dir if needed."""
    if settings.vad_model_path:
        path = Path(settings.vad_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"VAD_MODEL_PATH does not exist: {path}")
        return path

    cached = Path(settings.vad_model_cache_dir) / "silero_vad.onnx"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    cached.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading silero vad model", extra={"url": settings.vad_model_url})
    with urllib.request.urlopen(settings.vad_model_url, timeout=30) as response:
        payload = response.read()
    # Write atomically so a half-downloaded file is never cached.
    handle, tmp_name = tempfile.mkstemp(dir=str(cached.parent))
    with os.fdopen(handle, "wb") as tmp:
        tmp.write(payload)
    os.replace(tmp_name, cached)
    return cached


def _load_session(model_path: Path):  # type: ignore[no-untyped-def]
    key = str(model_path)
    with _session_lock:
        if key not in _session_cache:
            import onnxruntime as ort

            options = ort.SessionOptions()
            # One job already saturates the box via chunk-level concurrency;
            # intra-op threads on top of that just cause contention.
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            _session_cache[key] = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        return _session_cache[key]


class SileroVAD:
    """Silero VAD v5 inference loop, implemented directly against onnxruntime."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = _load_session(resolve_model_path(settings))

    def speech_probabilities(
        self, samples: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        n_windows = len(samples) // SILERO_WINDOW
        if n_windows == 0:
            return np.zeros(0, dtype=np.float32)

        window_view = samples[: n_windows * SILERO_WINDOW].reshape(
            n_windows, SILERO_WINDOW
        )
        state = np.zeros(SILERO_STATE_SHAPE, dtype=np.float32)
        sr = np.array(sample_rate, dtype=np.int64)
        probs = np.empty(n_windows, dtype=np.float32)

        # The model is stateful across windows, so this loop is sequential by
        # construction; batching would break the recurrence.
        for index in range(n_windows):
            frame = window_view[index : index + 1].astype(np.float32, copy=False)
            output, state = self._session.run(
                ["output", "stateN"],
                {"input": frame, "state": state, "sr": sr},
            )
            probs[index] = float(output[0][0])
        return probs

    def speech_segments(self, samples: np.ndarray, sample_rate: int) -> list[TimeSpan]:
        probs = self.speech_probabilities(samples, sample_rate)
        if probs.size == 0:
            return []

        threshold = self._settings.vad_speech_threshold
        # Hysteresis: it takes a clear signal to open a speech span and a
        # clearly quiet one to close it, which stops flicker around the
        # threshold from shredding continuous speech.
        release = max(threshold - 0.15, 0.01)
        flags = np.zeros(probs.size, dtype=bool)
        triggered = False
        for index, prob in enumerate(probs):
            if triggered:
                triggered = prob >= release
            else:
                triggered = prob >= threshold
            flags[index] = triggered

        return _flags_to_segments(
            flags,
            frame_sec=SILERO_WINDOW / sample_rate,
            total_sec=len(samples) / sample_rate,
            min_speech_sec=0.2,
            min_silence_sec=self._settings.min_silence_sec,
            pad_sec=self._settings.vad_speech_pad_sec,
        )
