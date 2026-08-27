"""Output rendering. JSON is canonical; the rest are conveniences."""

from __future__ import annotations

import csv
import io
import json

from ..schemas import AnnotateResponse

FORMATS = ("json", "jsonl", "csv", "table")


def render(response: AnnotateResponse, fmt: str = "json") -> str:
    if fmt == "json":
        return json.dumps(response.model_dump(mode="json"), indent=2)
    if fmt == "jsonl":
        return "\n".join(
            json.dumps(s.model_dump(mode="json")) for s in response.segments
        )
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["start_seconds", "end_seconds", "label", "result",
                    "attributes", "confidence", "flags", "description"])
        for s in response.segments:
            w.writerow([s.start_seconds, s.end_seconds, s.label, s.result.value,
                        ";".join(s.attributes), s.confidence.value,
                        ";".join(s.flags), s.description])
        return buf.getvalue().rstrip("\n")
    if fmt == "table":
        return _table(response)
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


def _table(response: AnnotateResponse) -> str:
    lines = [
        f"{response.task}",
        f"{response.duration_seconds:.2f}s, {len(response.segments)} subtasks"
        f"  [{response.metadata.get('model', '?')}, {response.metadata.get('quality', '?')}]",
        "",
        f"{'start':>7s} {'end':>7s}  {'result':7s} {'conf':6s} label",
    ]
    for s in response.segments:
        flags = f"  ({', '.join(s.flags)})" if s.flags else ""
        attrs = f" [{', '.join(s.attributes)}]" if s.attributes else ""
        lines.append(
            f"{s.start_seconds:7.2f} {s.end_seconds:7.2f}  {s.result.value:7s} "
            f"{s.confidence.value:6s} {s.label}{attrs}{flags}"
        )
    return "\n".join(lines)
