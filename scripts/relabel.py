"""Run only the labeling stage over an existing run's predicted segments.

Labeling is scored on temporally matched segments, so its prompt can be measured
without re-running segmentation: take a run's segments, label them with a given
prompt, write a new run file, then `score_labels.py` it. Same gold, same
boundaries, one variable.
"""

from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rel.config import config_for, parse_overrides
from rel.eval import datasets
from rel.pipeline import _label_all, client_for
from rel.annotation.validate import clean
from rel.schemas import AnnotateRequest, Quality, Segment
from rel.video.decode import probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="run tag whose segments are relabeled")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default=None, help="e.g. label_prompt=label_v2.md")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    src = json.loads((Path("results") / f"{args.source}.json").read_text())
    cfg = config_for(Quality.balanced, **parse_overrides(args.config))
    eps = {e.id: e for e in datasets.load(src.get("dataset", "wgo"))}

    def work(row: dict) -> dict:
        ep = eps[row["id"]]
        client = client_for(cfg)
        req = AnnotateRequest(video=str(ep.video), prompt=ep.instruction, quality=Quality.balanced)
        info = probe(ep.video)
        segs = [Segment(start_seconds=a, end_seconds=b, label=l) for a, b, l in row["pred"]]
        t0 = time.time()
        try:
            segs = _label_all(client, req, segs, info, cfg)
            segs, warnings = clean(segs, info.duration, req)
            return {**row, "pred": [[s.start_seconds, s.end_seconds, s.label] for s in segs],
                    "labels": [s.model_dump(mode="json") for s in segs],
                    "warnings": warnings, "elapsed": round(time.time() - t0, 2),
                    "usage": client.usage.as_dict()}
        except Exception as exc:  # noqa: BLE001
            return {**row, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    rows = [r for r in src["rows"] if r.get("ok") and r["id"] in eps]
    out_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); out_rows.append(r)
            print(f"  [{i:3d}/{len(rows)}] {'ok ' if r.get('ok') else 'ERR'} {r['id']}", flush=True)
    out_rows.sort(key=lambda r: r["id"])
    out = {**src, "tag": args.tag, "source": args.source, "label_config": cfg.to_dict(),
           "rows": out_rows, "failed": sum(1 for r in out_rows if not r.get("ok"))}
    out["scores"].pop("label_accuracy", None)
    (Path("results") / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"-> results/{args.tag}.json ({len(out_rows)} episodes)")


if __name__ == "__main__":
    main()
