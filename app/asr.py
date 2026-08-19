"""Gemini transcription: the only ASR backend.

Timestamps come back relative to the chunk; the pipeline adds the global
offset. Transient failures are raised as ASRError(retryable=True) so the
retry policy can act on them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.errors import ASRError
from app.schemas import ChunkTranscript

log = logging.getLogger(__name__)

TRANSCRIPTION_PROMPT = (
    "Transcribe this audio verbatim.\n"
    "Rules:\n"
    "- Return segments covering only the speech that is actually audible.\n"
    "- start and end are seconds from the beginning of THIS audio clip.\n"
    "- Keep segments to single sentences or natural phrases.\n"
    "- Do not translate, summarize, censor, or add commentary.\n"
    "- If there is no intelligible speech, return an empty segment list.\n"
    "- language is the BCP-47 code of the dominant spoken language."
)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# A response we cannot use, but re-rolling would not help either.
TERMINAL_FINISH_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII"}
)


# The wire schema is deliberately strict and default-free: the Gemini API
# rejects `default` in a response schema, and we want every field back on
# every call. The lenient ChunkTranscript is what the rest of the app sees.
class _WireSegment(BaseModel):
    start: float = Field(description="Start time in seconds from the clip start.")
    end: float = Field(description="End time in seconds from the clip start.")
    text: str = Field(description="Verbatim transcript of this segment.")


class _WireTranscript(BaseModel):
    segments: list[_WireSegment]
    language: str = Field(description="BCP-47 code of the dominant language, e.g. en.")


class GeminiBackend:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        from google import genai

        self._settings = settings
        self.model = settings.gemini_model
        # audio_timestamp only exists on the Vertex transport, so which client
        # we build decides whether we can ask for it at all.
        self.use_vertex = settings.gemini_use_vertex

        if self.use_vertex:
            if not settings.google_cloud_project:
                raise ASRError(
                    "GEMINI_USE_VERTEX is on but GOOGLE_CLOUD_PROJECT is not set."
                )
            self._client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
            )
        else:
            if not settings.gemini_api_key:
                raise ASRError(
                    "GEMINI_API_KEY is not set. The service cannot transcribe "
                    "without it; set it in the environment or in .env."
                )
            self._client = genai.Client(api_key=settings.gemini_api_key)

    async def transcribe(
        self, audio: bytes, *, duration_sec: float, language_hint: str | None = None
    ) -> ChunkTranscript:
        from google.genai import errors as genai_errors
        from google.genai import types

        prompt = self.build_prompt(language_hint)
        config = self.build_config()

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=audio, mime_type="audio/wav"),
                        types.Part.from_text(text=prompt),
                    ],
                    config=config,
                ),
                timeout=self._settings.request_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ASRError(
                f"gemini request exceeded {self._settings.request_timeout_sec:.0f}s",
                retryable=True,
            ) from exc
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            raise ASRError(
                f"gemini API error {code}: {exc}",
                retryable=code in RETRYABLE_STATUS,
            ) from exc
        except (ConnectionError, OSError) as exc:
            raise ASRError(f"gemini transport error: {exc}", retryable=True) from exc
        except ValueError as exc:
            # The SDK validates the request before sending it. Retrying an
            # invalid request just wastes three round trips.
            raise ASRError(f"gemini rejected the request: {exc}") from exc

        return self._parse(response)

    def build_prompt(self, language_hint: str | None = None) -> str:
        if not language_hint:
            return TRANSCRIPTION_PROMPT
        return f"{TRANSCRIPTION_PROMPT}\n- The audio is expected to be in {language_hint}."

    def build_config(self) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            # Improves timestamps on audio-only input, but it exists only on
            # the Vertex transport. Sending it on the Developer API raises:
            #   ValueError: audio_timestamp parameter is only supported in
            #   Gemini Enterprise Agent Platform mode, not in Gemini Developer
            #   API mode.
            # So with a plain API key it is omitted and timestamps come from
            # the model's own sense of time, which is why repair.py matters.
            audio_timestamp=True if self.use_vertex else None,
            thinking_config=self._thinking_config(types),
            # Structured output: the transcript is parsed, never read as prose.
            response_mime_type="application/json",
            response_schema=_WireTranscript,
            temperature=0.0,
        )

    def _thinking_config(self, types: Any) -> Any:
        """Minimal thinking: this is ASR, there is nothing to reason about."""
        try:
            return types.ThinkingConfig(thinking_level="minimal")
        except (ValidationError, TypeError, AttributeError):
            # Older SDKs only expose the numeric budget.
            return types.ThinkingConfig(thinking_budget=0)

    def _parse(self, response: Any) -> ChunkTranscript:
        finish_reason = _finish_reason(response)
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, _WireTranscript):
            return _to_chunk_transcript(parsed)

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise ASRError(
                f"gemini returned no content (finish_reason={finish_reason})",
                retryable=finish_reason not in TERMINAL_FINISH_REASONS,
            )
        try:
            return _to_chunk_transcript(_WireTranscript(**json.loads(text)))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ASRError(
                f"gemini returned output that does not match the schema: {exc}",
                retryable=finish_reason not in TERMINAL_FINISH_REASONS,
            ) from exc


def _to_chunk_transcript(wire: _WireTranscript) -> ChunkTranscript:
    return ChunkTranscript(
        segments=[
            {"start": s.start, "end": s.end, "text": s.text} for s in wire.segments
        ],
        language=wire.language or "unknown",
    )


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "NONE"
    reason = getattr(candidates[0], "finish_reason", None)
    return str(getattr(reason, "name", reason) or "NONE")
