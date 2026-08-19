"""Structured (JSON) logging with a request/job id carried via contextvars.

Anything passed as `extra={...}` is merged into the JSON line. Keys must not
collide with LogRecord's own attributes (`filename`, `module`, `name`, `msg`,
`args`, `levelname`, ...) — the stdlib raises KeyError at the call site if they
do. See `_RESERVED` below.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import sys
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, TextIO

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "job_id", default=None
)

# Attributes present on every LogRecord; anything else was passed via `extra`
# and therefore belongs in the JSON payload.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if (rid := request_id_var.get()) is not None:
            payload["request_id"] = rid
        if (jid := job_id_var.get()) is not None:
            payload["job_id"] = jid
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Send JSON logs to `stream` (default stdout, as containers expect).

    The CLI passes stderr, so that piping its stdout into `jq` gets the
    result and nothing else.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn installs its own handlers; make them use ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def new_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def bind(request_id: str | None = None, job_id: str | None = None) -> Iterator[None]:
    """Bind ids for the duration of a block (works across await points)."""
    tokens = []
    if request_id is not None:
        tokens.append((request_id_var, request_id_var.set(request_id)))
    if job_id is not None:
        tokens.append((job_id_var, job_id_var.set(job_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
