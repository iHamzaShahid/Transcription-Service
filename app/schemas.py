"""Pydantic models: the public API contract plus a few internal value objects."""

from __future__ import annotations

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# What a backend returns for one chunk (also used as the Gemini response_schema)
# --------------------------------------------------------------------------


class RawSegment(BaseModel):
    """A segment as produced by the ASR backend, relative to the chunk start."""

    start: float = Field(description="Segment start in seconds from the chunk start.")
    end: float = Field(description="Segment end in seconds from the chunk start.")
    text: str = Field(description="Verbatim transcript of the segment.")


class ChunkTranscript(BaseModel):
    """One chunk's transcript, chunk-relative. No free-form prose, ever."""

    segments: list[RawSegment] = Field(default_factory=list)
    language: str = Field(
        default="unknown", description="BCP-47-ish language code, e.g. 'en'."
    )


# --------------------------------------------------------------------------
# Probe / audio description
# --------------------------------------------------------------------------


class AudioInfo(BaseModel):
    """Facts about the *original* upload, as reported by ffprobe."""

    codec: str
    container: str
    sample_rate: int
    channels: int
    duration_sec: float
    bit_rate: int | None = None


# --------------------------------------------------------------------------
# Public response
# --------------------------------------------------------------------------


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class FailedChunk(BaseModel):
    """A chunk that exhausted its retries. The job still returns everything else."""

    index: int
    start: float
    end: float
    error: str


class RepairCounts(BaseModel):
    """Breakdown behind `metadata.timestamps_repaired`."""

    clamped: int = 0
    reordered: int = 0
    dropped: int = 0

    @property
    def total(self) -> int:
        return self.clamped + self.reordered + self.dropped


class ResultMetadata(BaseModel):
    model: str
    backend: str
    original_format: str
    sample_rate: int
    channels: int
    processing_time_sec: float
    timestamps_repaired: int
    repair_detail: RepairCounts


class TranscriptionResult(BaseModel):
    segments: list[Segment]
    text: str
    language: str
    duration_sec: float
    chunk_count: int
    failed_chunks: list[FailedChunk]
    metadata: ResultMetadata


class ErrorResponse(BaseModel):
    detail: str
