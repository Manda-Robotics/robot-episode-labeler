"""WGO-Bench loader.

WGO-Bench (macrodata/WGO-Bench) is 100 manually annotated robot and egocentric
episodes with 743 gold subtask segments. We use it as our development benchmark
only: it is CC-BY-NC-SA-4.0, so it must not be redistributed or shipped inside a
commercial artifact. Scores computed against it are fine to publish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data"
PARQUET = DATA / "wgo_annotations.parquet"
EPISODES = DATA / "wgo"


@dataclass(frozen=True)
class GoldSegment:
    start: float
    end: float
    subtask: str


@dataclass(frozen=True)
class Episode:
    id: str
    video: Path
    instruction: str
    segments: list[GoldSegment]
    family: str
    meta: dict

    @property
    def duration_hint(self) -> float:
        return self.segments[-1].end if self.segments else 0.0


def materialise(limit: int | None = None, force: bool = False) -> int:
    """Unpack embedded MP4 bytes and gold labels into data/wgo/ as plain files."""
    import pyarrow.parquet as pq

    EPISODES.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(PARQUET)
    written = 0
    for rg in range(pf.num_row_groups):
        if limit is not None and written >= limit:
            break
        table = pf.read_row_group(rg)
        for row in table.to_pylist():
            ep_id = row["id"]
            mp4 = EPISODES / f"{ep_id}.mp4"
            side = EPISODES / f"{ep_id}.json"
            if force or not mp4.exists():
                mp4.write_bytes(row["video"])
            if force or not side.exists():
                meta = json.loads(row["metadata"]) if row.get("metadata") else {}
                ps = row.get("perception_state")
                side.write_text(json.dumps({
                    "id": ep_id,
                    "instruction": row["instruction"],
                    "segments": row["segments"],
                    "metadata": meta,
                    "family": (ps or {}).get("robot_family") or _family_from_id(ep_id),
                    "perception_state": ps,
                }))
            written += 1
            if limit is not None and written >= limit:
                break
    return written


def _family_from_id(ep_id: str) -> str:
    return ep_id.split("_")[0]


def load(limit: int | None = None, family: str | None = None) -> list[Episode]:
    eps: list[Episode] = []
    for side in sorted(EPISODES.glob("*.json")):
        d = json.loads(side.read_text())
        fam = d.get("family") or _family_from_id(d["id"])
        if family and fam != family:
            continue
        eps.append(Episode(
            id=d["id"],
            video=side.with_suffix(".mp4"),
            instruction=d["instruction"],
            segments=[GoldSegment(s["start_sec"], s["end_sec"], s["subtask"])
                      for s in d["segments"]],
            family=fam,
            meta=d.get("metadata", {}),
        ))
    eps.sort(key=lambda e: e.id)
    return eps[:limit] if limit else eps


def perception(ep_id: str) -> dict | None:
    side = EPISODES / f"{ep_id}.json"
    if not side.exists():
        return None
    return json.loads(side.read_text()).get("perception_state")
