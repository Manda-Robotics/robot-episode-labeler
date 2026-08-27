"""Command-line entry point.

    rel annotate episode.mp4 "A robot arm folds a cardboard box."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .annotation.llm import DEFAULT_MODEL, LLMError
from .formats import render
from .pipeline import annotate
from .schemas import AnnotateRequest, Quality
from .video.decode import VideoError


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rel", description="Annotate a robot manipulation video with timestamped subtasks."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ann = sub.add_parser("annotate", help="annotate one episode")
    ann.add_argument("video", help="path to the episode video")
    ann.add_argument("prompt", help="what the robot is doing")
    ann.add_argument("--subtasks", help="comma-separated closed vocabulary")
    ann.add_argument("--attributes", help="comma-separated failure/attribute rubric")
    ann.add_argument("--quality", default="balanced", choices=[q.value for q in Quality])
    ann.add_argument("--model", default=DEFAULT_MODEL)
    ann.add_argument("--format", default="table", choices=["table", "json", "jsonl", "csv"])
    ann.add_argument("-o", "--out", help="write to a file instead of stdout")

    args = parser.parse_args(argv)

    try:
        response = annotate(
            AnnotateRequest(
                video=args.video,
                prompt=args.prompt,
                subtasks=_split(args.subtasks),
                attributes=_split(args.attributes),
                quality=Quality(args.quality),
            ),
            model=args.model,
        )
    except (VideoError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = render(response, args.format)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)

    for warning in response.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
