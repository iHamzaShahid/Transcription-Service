"""FastAPI surface. All real work lives in app/pipeline/runner.py."""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.errors import (
    AudioProcessingError,
    PayloadTooLargeError,
    TranscriptionError,
    UnsupportedMediaError,
)
from app.logging_conf import configure_logging, new_id, request_id_var
from app.pipeline.runner import TranscriptionPipeline
from app.schemas import ErrorResponse, TranscriptionResult

log = logging.getLogger(__name__)

UPLOAD_BUFFER_BYTES = 1024 * 1024
# Slack for multipart boundaries and part headers, so a file that is exactly
# at the limit is never rejected by the Content-Length pre-check.
MULTIPART_OVERHEAD_BYTES = 8192


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    # Built once: the ONNX session and the genai client are both expensive to
    # create and safe to share across requests. A missing API key or an
    # unloadable VAD model fails here, at boot, not on the first upload.
    app.state.settings = settings
    app.state.pipeline = TranscriptionPipeline(settings)
    log.info(
        "service ready",
        extra={
            "model": app.state.pipeline.backend.model,
            "chunk_size_sec": settings.chunk_size_sec,
            "max_parallel_chunks": settings.max_parallel_chunks,
        },
    )
    yield


app = FastAPI(
    title="Transcription Service",
    version="2.0.0",
    summary="Chunked, VAD-aware audio transcription with repaired timestamps.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("x-request-id") or new_id()
    token = request_id_var.set(request_id)
    started = time.monotonic()
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_sec": round(time.monotonic() - started, 3),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.middleware("http")
async def upload_limit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Reject oversized uploads before the body is parsed into a temp file.

    The check inside the endpoint is authoritative, but it only runs after
    Starlette has already spooled the whole multipart body. Refusing on
    Content-Length first means a well-behaved client is turned away before
    the bytes are buffered at all.
    """
    settings: Settings = request.app.state.settings
    declared = request.headers.get("content-length")
    if declared and declared.isdigit():
        if int(declared) > settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES:
            log.warning("rejected oversized upload", extra={"declared_bytes": declared})
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Upload exceeds the {settings.max_upload_bytes} byte limit."
                    )
                },
            )
    return await call_next(request)


@app.exception_handler(TranscriptionError)
async def domain_error_handler(request: Request, exc: TranscriptionError):  # type: ignore[no-untyped-def]
    status = {
        UnsupportedMediaError: 415,
        PayloadTooLargeError: 413,
        AudioProcessingError: 500,
    }.get(type(exc), 500)
    log.warning("request rejected", extra={"status": status, "error": str(exc)})
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    pipeline: TranscriptionPipeline = request.app.state.pipeline
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "model": pipeline.backend.model,
        "chunk_size_sec": settings.chunk_size_sec,
        "max_parallel_chunks": settings.max_parallel_chunks,
        "max_upload_bytes": settings.max_upload_bytes,
    }


@app.post(
    "/transcribe",
    response_model=TranscriptionResult,
    responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}},
)
async def transcribe(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Transcribe an audio file and return the transcript in this response."""
    settings: Settings = request.app.state.settings
    pipeline: TranscriptionPipeline = request.app.state.pipeline

    upload_path = await _spool_upload(file, settings)
    try:
        result = await pipeline.run(upload_path)
        log.info(
            "transcription complete",
            extra={
                "chunk_count": result.chunk_count,
                "failed_chunks": len(result.failed_chunks),
                "timestamps_repaired": result.metadata.timestamps_repaired,
                "processing_time_sec": result.metadata.processing_time_sec,
            },
        )
        return JSONResponse(status_code=200, content=result.model_dump(mode="json"))
    finally:
        upload_path.unlink(missing_ok=True)


async def _spool_upload(file: UploadFile, settings: Settings) -> Path:
    """Stream the upload to disk, aborting as soon as it exceeds the cap.

    ffprobe and ffmpeg take a path, so the bytes have to land on disk before
    anything can look at them.
    """
    handle = tempfile.NamedTemporaryFile(
        prefix="upload-", delete=False, dir=settings.temp_dir
    )
    path = Path(handle.name)
    size = 0
    try:
        with handle:
            while data := await file.read(UPLOAD_BUFFER_BYTES):
                size += len(data)
                if size > settings.max_upload_bytes:
                    raise PayloadTooLargeError(
                        f"Upload exceeds the {settings.max_upload_bytes} byte limit."
                    )
                handle.write(data)
        if size == 0:
            raise UnsupportedMediaError("The uploaded file is empty.")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    # NB: not "filename" — that is a reserved LogRecord attribute.
    log.info("received upload", extra={"upload_name": file.filename, "bytes": size})
    return path
