"""Exponential backoff with full jitter, for the ASR calls."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

from app.errors import ASRError

log = logging.getLogger(__name__)

T = TypeVar("T")


def is_retryable(exc: BaseException) -> bool:
    """Backends classify their own failures; anything else is fatal."""
    return isinstance(exc, ASRError) and exc.retryable


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    label: str = "operation",
) -> T:
    """Call `operation`, retrying transient failures up to `max_attempts` times."""
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            if not is_retryable(exc) or attempt == max_attempts:
                raise
            # Full jitter: sleep anywhere in [0, capped backoff]. Spreads a
            # herd of chunks that all got rate-limited at the same instant.
            backoff = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay = random.uniform(0, backoff)
            log.warning(
                "retrying after transient failure",
                extra={
                    "label": label,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "delay_sec": round(delay, 3),
                    "error": str(exc),
                },
            )
            await asyncio.sleep(delay)

    assert last_error is not None  # unreachable: the loop either returns or raises
    raise last_error
