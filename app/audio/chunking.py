"""Turn VAD output into chunk boundaries.

Aim for `target` second chunks, but move the cut to the silence gap nearest
the target within a +/- `window` search band. Cuts therefore land where nobody
is speaking, which is why the chunks need no overlap: no word is ever split,
so there is no duplicated audio to reconcile afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.audio.vad import TimeSpan


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float  # global offset, seconds
    end: float
    cut_at_silence: bool  # False => the hard-cut fallback fired

    @property
    def duration(self) -> float:
        return self.end - self.start


def silence_gaps(
    speech: list[TimeSpan], duration: float, min_silence_sec: float
) -> list[TimeSpan]:
    """Complement of the speech spans, keeping only gaps worth cutting in."""
    gaps: list[TimeSpan] = []
    cursor = 0.0
    for span in sorted(speech, key=lambda s: s.start):
        if span.start - cursor >= min_silence_sec:
            gaps.append(TimeSpan(cursor, span.start))
        cursor = max(cursor, span.end)
    if duration - cursor >= min_silence_sec:
        gaps.append(TimeSpan(cursor, duration))
    return gaps


def plan_chunks(
    duration: float,
    speech: list[TimeSpan],
    *,
    target_sec: float,
    window_sec: float,
    min_silence_sec: float,
) -> list[Chunk]:
    """Plan cut points for `duration` seconds of audio. Pure and testable."""
    if duration <= 0:
        return []

    gaps = silence_gaps(speech, duration, min_silence_sec)
    chunks: list[Chunk] = []
    cursor = 0.0

    # Stop cutting once the remainder would fit inside the search band: any
    # further cut could only leave a stub shorter than the window we were
    # willing to search anyway.
    while duration - cursor > target_sec + window_sec:
        target = cursor + target_sec
        low = max(target - window_sec, cursor + min_silence_sec)
        high = min(target + window_sec, duration)
        cut = _nearest_silence_cut(gaps, target=target, low=low, high=high)
        if cut is None:
            # Fallback: no silence anywhere in the window (continuous speech,
            # music, noise). Take the hard cut and accept the split word.
            chunks.append(Chunk(len(chunks), cursor, target, cut_at_silence=False))
            cursor = target
        else:
            chunks.append(Chunk(len(chunks), cursor, cut, cut_at_silence=True))
            cursor = cut

    chunks.append(Chunk(len(chunks), cursor, duration, cut_at_silence=True))
    return chunks


def _nearest_silence_cut(
    gaps: list[TimeSpan], *, target: float, low: float, high: float
) -> float | None:
    """Best cut point inside [low, high], preferring the middle of a gap."""
    if high <= low:
        return None

    best: float | None = None
    best_distance = float("inf")
    for gap in gaps:
        overlap_start = max(gap.start, low)
        overlap_end = min(gap.end, high)
        if overlap_end <= overlap_start:
            continue
        # Cut in the middle of the usable silence: maximum margin on both
        # sides against VAD boundary error.
        candidate = (overlap_start + overlap_end) / 2.0
        distance = abs(candidate - target)
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best


def slice_samples(samples: np.ndarray, chunk: Chunk, sample_rate: int) -> np.ndarray:
    start = max(0, int(round(chunk.start * sample_rate)))
    end = min(len(samples), int(round(chunk.end * sample_rate)))
    return samples[start:end]
