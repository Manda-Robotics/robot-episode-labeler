"""Failure taxonomy for a run: why was each gold segment missed?

Aggregate metrics say how far off we are; this says in which direction, which is
what decides the next change. Each unmatched gold segment is classed as:

  merged   one predicted segment covers most of it AND most of a neighbour
  split    two or more predicted segments lie mostly inside it
  shifted  a predicted segment overlaps it substantially but under IoU 0.5
  missing  nothing overlaps it to speak of
"""

from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rel.eval.metrics import iou, match_segments


def _cov(p, g):
    lo, hi = max(p[0], g[0]), min(p[1], g[1])
    return max(0.0, hi - lo) / (g[1] - g[0]) if g[1] > g[0] else 0.0


def classify(pred, gold):
    matched = {m.gold_index for m in match_segments(pred, gold)}
    out = []
    for gi, g in enumerate(gold):
        if gi in matched:
            out.append("ok"); continue
        merged = False
        for p in pred:
            if _cov(p, g) >= 0.7:
                for nj in (gi - 1, gi + 1):
                    if 0 <= nj < len(gold) and _cov(p, gold[nj]) >= 0.7:
                        merged = True
        if merged:
            out.append("merged"); continue
        inside = [p for p in pred if (p[1] - p[0]) > 0 and _cov(g, p) >= 0.7]
        if len(inside) >= 2:
            out.append("split"); continue
        best = max((iou(p, g) for p in pred), default=0.0)
        out.append("shifted" if best >= 0.2 else "missing")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    args = ap.parse_args()
    for tag in args.tags:
        run = json.loads((Path("results") / f"{tag}.json").read_text())
        by_fam: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        by_dur: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for r in run["rows"]:
            if not r.get("ok"):
                continue
            pred = [(a, b) for a, b, _ in r["pred"]]
            gold = [(a, b) for a, b, _ in r["gold"]]
            for g, cls in zip(gold, classify(pred, gold)):
                by_fam[r["family"]][cls] += 1
                d = g[1] - g[0]
                bucket = "<2s" if d < 2 else "2-4s" if d < 4 else "4-8s" if d < 8 else "8s+"
                by_dur[bucket][cls] += 1
        print(f"=== {tag}")
        cols = ["ok", "merged", "split", "shifted", "missing"]
        print(f"  {'family':9s} " + " ".join(f"{c:>8s}" for c in cols) + "   n")
        for fam, c in sorted(by_fam.items()):
            n = sum(c.values())
            print(f"  {fam:9s} " + " ".join(f"{c[k]/n:8.2f}" for k in cols) + f"   {n}")
        for b in ("<2s", "2-4s", "4-8s", "8s+"):
            c = by_dur[b]; n = sum(c.values()) or 1
            print(f"  {b:9s} " + " ".join(f"{c[k]/n:8.2f}" for k in cols) + f"   {n}")


if __name__ == "__main__":
    main()
