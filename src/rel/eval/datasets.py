"""Benchmark loading for any dataset materialised in the WGO sidecar layout.

`data/<dataset>/<id>.mp4` + `data/<dataset>/<id>.json`, where the JSON carries
`id`, `instruction`, `segments: [{start_sec, end_sec, subtask}]`, `family` and
free-form `metadata`. WGO-Bench is `data/wgo/`; `scripts/fetch_*.py` write the
others. One loader means one scoring path for every corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from .wgo import DATA, Episode, GoldSegment, _family_from_id


def available() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if p.is_dir() and any(p.glob("*.json")))


def load(dataset: str = "wgo", limit: int | None = None, family: str | None = None) -> list[Episode]:
    root = DATA / dataset
    if not root.is_dir():
        raise FileNotFoundError(f"no such dataset directory: {root} (have {available()})")
    eps: list[Episode] = []
    for side in sorted(root.glob("*.json")):
        d = json.loads(side.read_text())
        if "segments" not in d:
            continue
        fam = d.get("family") or _family_from_id(d["id"])
        if family and fam != family:
            continue
        video = side.with_suffix(".mp4")
        if not video.exists():
            continue
        segs = sorted(
            (GoldSegment(float(s["start_sec"]), float(s["end_sec"]), s["subtask"]) for s in d["segments"]),
            key=lambda g: g.start,
        )
        eps.append(Episode(
            id=d["id"], video=video, instruction=d.get("instruction", ""),
            segments=segs, family=fam, meta=d.get("metadata", {}),
        ))
    eps.sort(key=lambda e: e.id)
    return eps[:limit] if limit else eps
