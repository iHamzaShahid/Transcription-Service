"""The pipeline. One synchronous path, shared by the API and the CLI.

    validate -> normalize -> (VAD -> chunk) -> transcribe -> repair -> stitch

Short audio skips the VAD and chunking steps entirely and goes to Gemini in
one piece.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.asr import GeminiBackend
from app.audio import normalize as norm
from app.audio.chunking import Chunk, plan_chunks, slice_samples
from app.audio.probe import probe
from app.audio.vad import SileroVAD
from app.config import Settings
from app.errors import ASRError
from app.pipeline.repair import apply_offset, repair_chunk
from app.pipeline.retry import with_retries
from app.schemas import (
    AudioInfo,
    ChunkTranscript,
    FailedChunk,
    RawSegment,
    RepairCounts,
    ResultMetadata,
    Segment,
    TranscriptionResult,
)

log = logging.getLogger(__name__)


@dataclass
class _ChunkOutcome:
    chunk: Chunk
    segments: list[RawSegment]  # global timestamps, already repaired
    counts: RepairCounts
    language: str | None
    failure: FailedChunk | None


class TranscriptionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Both are expensive to build and safe to share across requests.
        self.backend = GeminiBackend(settings)
        self.vad = SileroVAD(settings)

    async def run(self, source: Path) -> TranscriptionResult:
        started = time.monotonic()
        settings = self.settings
        # Everything derived from the upload lives here and dies in `finally`.
        workdir = Path(tempfile.mkdtemp(prefix="transcribe-", dir=settings.temp_dir))
        try:
            # 1. Validate. Raises UnsupportedMediaError -> 415.
            info = await probe(source, settings)

            # 2. Normalize to the one format everything downstream assumes.
            normalized = await norm.normalize(
                source, workdir / "normalized.wav", settings
            )
            samples = norm.read_wav_mono16(normalized, settings.target_sample_rate)
            duration = len(samples) / settings.target_sample_rate

            # 3. Split, or don't.
            chunks = await self._plan(samples, duration)

            # 4. Transcribe, bounded. 5. Repair and stitch.
            outcomes = await self._transcribe_chunks(samples, chunks)
            return self._assemble(
                outcomes,
                info=info,
                duration=duration,
                elapsed=time.monotonic() - started,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # -- steps -------------------------------------------------------------

    async def _plan(self, samples: np.ndarray, duration: float) -> list[Chunk]:
        """Chunk the audio, skipping the VAD pass when it cannot matter."""
        settings = self.settings

        if duration <= settings.skip_vad_below_sec:
            # Too short to split: even the furthest cut the planner would
            # consider leaves one chunk, so the VAD pass would be wasted work.
            log.info(
                "short audio, transcribing whole",
                extra={"duration_sec": round(duration, 3)},
            )
            return [Chunk(index=0, start=0.0, end=duration, cut_at_silence=True)]

        # VAD is CPU-bound; keep it off the event loop.
        speech = await asyncio.to_thread(
            self.vad.speech_segments, samples, settings.target_sample_rate
        )
        chunks = plan_chunks(
            duration,
            speech,
            target_sec=settings.chunk_size_sec,
            window_sec=settings.chunk_search_window_sec,
            min_silence_sec=settings.min_silence_sec,
        )
        log.info(
            "planned chunks",
            extra={
                "chunk_count": len(chunks),
                "duration_sec": round(duration, 3),
                "speech_spans": len(speech),
                "hard_cuts": sum(1 for c in chunks if not c.cut_at_silence),
            },
        )
        return chunks

    async def _transcribe_chunks(
        self, samples: np.ndarray, chunks: list[Chunk]
    ) -> list[_ChunkOutcome]:
        semaphore = asyncio.Semaphore(self.settings.max_parallel_chunks)

        async def worker(chunk: Chunk) -> _ChunkOutcome:
            async with semaphore:
                return await self._transcribe_one(samples, chunk)

        # Bounded by the semaphore above: at most max_parallel_chunks in flight.
        return await asyncio.gather(*(worker(chunk) for chunk in chunks))

    async def _transcribe_one(
        self, samples: np.ndarray, chunk: Chunk
    ) -> _ChunkOutcome:
        settings = self.settings
        sliced = slice_samples(samples, chunk, settings.target_sample_rate)
        audio = norm.to_wav_bytes(sliced, settings.target_sample_rate)
        # The audio actually sent, which can be a hair shorter than the planned
        # window at the tail of the file. Repairs clamp against this, not the plan.
        audio_duration = len(sliced) / settings.target_sample_rate

        try:
            transcript: ChunkTranscript = await with_retries(
                lambda: self.backend.transcribe(audio, duration_sec=audio_duration),
                max_attempts=settings.max_retries,
                base_delay=settings.retry_base_delay_sec,
                max_delay=settings.retry_max_delay_sec,
                label=f"chunk-{chunk.index}",
            )
        except Exception as exc:
            # One chunk's failure must not take down the request: record it and
            # let the rest of the transcript through.
            reason = str(exc) if isinstance(exc, ASRError) else repr(exc)
            log.error(
                "chunk failed after retries",
                extra={"chunk_index": chunk.index, "error": reason},
            )
            return _ChunkOutcome(
                chunk=chunk,
                segments=[],
                counts=RepairCounts(),
                language=None,
                failure=FailedChunk(
                    index=chunk.index,
                    start=round(chunk.start, 3),
                    end=round(chunk.end, 3),
                    error=reason,
                ),
            )

        repaired = repair_chunk(transcript, audio_duration)
        if repaired.counts.total:
            log.info(
                "repaired chunk timestamps",
                extra={
                    "chunk_index": chunk.index,
                    "clamped": repaired.counts.clamped,
                    "reordered": repaired.counts.reordered,
                    "dropped": repaired.counts.dropped,
                },
            )
        return _ChunkOutcome(
            chunk=chunk,
            # Shift chunk-local times onto the global timeline.
            segments=apply_offset(repaired.segments, chunk.start),
            counts=repaired.counts,
            language=transcript.language,
            failure=None,
        )

    def _assemble(
        self,
        outcomes: list[_ChunkOutcome],
        *,
        info: AudioInfo,
        duration: float,
        elapsed: float,
    ) -> TranscriptionResult:
        # Chunks finish out of order; the transcript must not.
        outcomes = sorted(outcomes, key=lambda o: o.chunk.index)

        totals = RepairCounts()
        segments: list[Segment] = []
        failures: list[FailedChunk] = []
        languages: Counter[str] = Counter()

        for outcome in outcomes:
            totals.clamped += outcome.counts.clamped
            totals.reordered += outcome.counts.reordered
            totals.dropped += outcome.counts.dropped
            if outcome.failure:
                failures.append(outcome.failure)
            if outcome.language and outcome.language != "unknown":
                languages[outcome.language] += 1
            for raw in outcome.segments:
                segments.append(
                    Segment(
                        id=len(segments),
                        start=raw.start,
                        end=raw.end,
                        text=raw.text,
                    )
                )

        language = languages.most_common(1)[0][0] if languages else "unknown"
        return TranscriptionResult(
            segments=segments,
            text=" ".join(s.text for s in segments).strip(),
            language=language,
            duration_sec=round(duration, 3),
            chunk_count=len(outcomes),
            failed_chunks=failures,
            metadata=ResultMetadata(
                model=self.backend.model,
                backend=self.backend.name,
                original_format=info.codec,
                sample_rate=info.sample_rate,
                channels=info.channels,
                processing_time_sec=round(elapsed, 3),
                timestamps_repaired=totals.total,
                repair_detail=totals,
            ),
        )
