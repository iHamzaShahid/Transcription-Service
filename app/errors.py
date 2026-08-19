"""Domain errors. The API layer maps these onto HTTP status codes."""

from __future__ import annotations


class TranscriptionError(Exception):
    """Base class for everything this service raises deliberately."""


class UnsupportedMediaError(TranscriptionError):
    """Input is not decodable audio, or not an accepted codec/container -> 415."""


class PayloadTooLargeError(TranscriptionError):
    """Upload exceeded MAX_UPLOAD_BYTES -> 413."""


class AudioProcessingError(TranscriptionError):
    """ffmpeg/ffprobe failed for a reason that is not the caller's fault -> 500."""


class ASRError(TranscriptionError):
    """A transcription backend failed for a single chunk."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
