"""Command line entry point: python -m app.cli <file> --out result.json

Runs the exact same TranscriptionPipeline the API runs. The only difference
is where the bytes come from and where the JSON goes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.errors import TranscriptionError
from app.logging_conf import bind, configure_logging, new_id
from app.pipeline.runner import TranscriptionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Transcribe a WAV or MP3 file to JSON.",
    )
    parser.add_argument("file", type=Path, help="Path to a WAV or MP3 file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON here instead of stdout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON log stream on stderr.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()

    if not args.file.is_file():
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 2

    try:
        pipeline = TranscriptionPipeline(settings)
        result = await pipeline.run(args.file)
    except TranscriptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(payload)

    if result.failed_chunks:
        print(
            f"warning: {len(result.failed_chunks)} chunk(s) failed",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    # Logs to stderr: stdout carries the JSON result and nothing else.
    configure_logging(
        "ERROR" if args.quiet else settings.log_level, stream=sys.stderr
    )
    with bind(request_id=new_id()):
        return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
