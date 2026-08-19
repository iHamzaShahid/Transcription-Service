"""Timestamp validation, repair, and offset stitching.

An LLM emits timestamps as *tokens*. They are approximations, not the output
of a forced aligner, and they do drift: ends before starts, times past the end
of the clip, segments that walk backwards. This module makes the output
internally consistent and counts every correction it had to make, so the
caller can see how much to trust it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas import ChunkTranscript, RawSegment, RepairCounts


@dataclass
class RepairedChunk:
    segments: list[RawSegment]  # still chunk-relative
    counts: RepairCounts


# Segments shorter than this after clamping are treated as collapsed noise.
MIN_SEGMENT_SEC = 0.01


def repair_chunk(transcript: ChunkTranscript, chunk_duration: float) -> RepairedChunk:
    """Force one chunk's segments to be sane and monotonic.

    Rules, applied per segment in order:
      * non-finite times or empty text  -> drop
      * time outside [0, chunk_duration] -> clamp
      * start before the previous segment's end -> push forward
      * end still not after start -> drop

    Each segment contributes at most one count per category, and a dropped
    segment is only ever counted as dropped.
    """
    counts = RepairCounts()
    kept: list[RawSegment] = []
    previous_end = 0.0

    for segment in transcript.segments:
        if not segment.text.strip():
            counts.dropped += 1
            continue
        if not (math.isfinite(segment.start) and math.isfinite(segment.end)):
            counts.dropped += 1
            continue

        start = _clamp(segment.start, chunk_duration)
        end = _clamp(segment.end, chunk_duration)
        was_clamped = start != segment.start or end != segment.end

        was_reordered = start < previous_end
        if was_reordered:
            # Monotonic starts: never let a segment begin before the last one
            # ended, or the stitched transcript reads out of order.
            start = min(previous_end, chunk_duration)

        if end - start < MIN_SEGMENT_SEC:
            # Nothing survives, so the repairs above are moot: a discarded
            # segment is counted once, as dropped, and never twice.
            counts.dropped += 1
            continue

        if was_clamped:
            counts.clamped += 1
        if was_reordered:
            counts.reordered += 1
        kept.append(RawSegment(start=start, end=end, text=segment.text.strip()))
        previous_end = end

    return RepairedChunk(segments=kept, counts=counts)


def _clamp(value: float, upper: float) -> float:
    return min(max(float(value), 0.0), upper)


def apply_offset(segments: list[RawSegment], offset_sec: float) -> list[RawSegment]:
    """Shift chunk-relative times into the global timeline."""
    return [
        RawSegment(
            start=round(s.start + offset_sec, 3),
            end=round(s.end + offset_sec, 3),
            text=s.text,
        )
        for s in segments
    ]
