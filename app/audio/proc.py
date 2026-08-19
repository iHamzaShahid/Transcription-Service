"""Thin async wrapper around subprocess, used by the ffmpeg/ffprobe callers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.errors import AudioProcessingError


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def stderr_tail(self, limit: int = 400) -> str:
        text = self.stderr.decode("utf-8", errors="replace").strip()
        return text[-limit:]


async def run(argv: list[str], *, timeout: float) -> CommandResult:
    """Run a command to completion. Raises AudioProcessingError on timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:  # binary missing from the image
        raise AudioProcessingError(f"{argv[0]} is not installed") from exc

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        await _terminate(proc)
        raise AudioProcessingError(
            f"{argv[0]} timed out after {timeout:.0f}s"
        ) from exc
    except asyncio.CancelledError:
        # The job was cancelled (shutdown, or the caller gave up). Reap the
        # child before unwinding, or ffmpeg outlives the task that started it.
        await _terminate(proc)
        raise

    return CommandResult(proc.returncode or 0, stdout, stderr)


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:  # already gone
        return
    await proc.wait()
