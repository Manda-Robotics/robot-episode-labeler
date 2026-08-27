"""Compare two runs on the episodes that succeeded in BOTH.

Runs differ in which episodes fail, and failed episodes are excluded from scoring,
so aggregate numbers from two runs can be computed over different gold sets and
are not comparable. This restricts both to the same episodes.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rel.eval.metrics import TOLERANCES, Aggregate


def rows_ok(tag: str) -> dict[str, dict]:
    run = json.loads((Path("results") / f"{tag}.json").read_text())
    return {r["id"]: r for r in run["rows"] if r.get("ok")}


def score(rows: list[dict]) -> dict:
    agg, per = Aggregate(), {}
    for r in rows:
        pred = [(p[0], p[1]) for p in r["pred"]]
        gold = [(g[0], g[1]) for g in r["gold"]]
        agg.add(pred, gold)
        per.setdefault(r["family"], Aggregate()).add(pred, gold)
    return {"overall": agg.report(), "by_family": {k: v.report() for k, v in sorted(per.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    args = ap.parse_args()

    A, B = rows_ok(args.a), rows_ok(args.b)
    common = sorted(set(A) & set(B))
    print(f"{args.a}: {len(A)} ok | {args.b}: {len(B)} ok | compared on {len(common)} common\n")

    sa, sb = score([A[i] for i in common]), score([B[i] for i in common])

    def line(name, ga, gb, fmt="{:.3f}"):
        va, vb = ga, gb
        delta = vb - va
        arrow = "+" if delta > 0 else ""
        print(f"  {name:26s} {fmt.format(va):>8s} -> {fmt.format(vb):>8s}   {arrow}{delta:+.3f}"
              .replace("++", "+"))

    print(f"{'metric':28s} {args.a[:8]:>8s}    {args.b[:8]:>8s}   delta")
    line("segmentation F1", sa["overall"]["segmentation_f1"], sb["overall"]["segmentation_f1"])
    line("  precision", sa["overall"]["segmentation_precision"], sb["overall"]["segmentation_precision"])
    line("  recall", sa["overall"]["segmentation_recall"], sb["overall"]["segmentation_recall"])
    for t in TOLERANCES:
        line(f"boundary recall +/-{t}s",
             sa["overall"]["boundary_recall_at"][str(t)], sb["overall"]["boundary_recall_at"][str(t)])
    line("median boundary err (s)",
         sa["overall"]["median_boundary_error_sec"], sb["overall"]["median_boundary_error_sec"], "{:.3f}")
    print(f"  {'segments predicted':26s} {sa['overall']['segments_pred']:>8d} -> "
          f"{sb['overall']['segments_pred']:>8d}   (gold {sa['overall']['segments_gold']})")

    print("\nby family (segmentation F1 / boundary recall +/-1.0s):")
    for fam in sorted(sa["by_family"]):
        fa, fb = sa["by_family"][fam], sb["by_family"][fam]
        print(f"  {fam:9s} F1 {fa['segmentation_f1']:.3f} -> {fb['segmentation_f1']:.3f}"
              f"   +/-1s {fa['boundary_recall_at']['1.0']:.3f} -> {fb['boundary_recall_at']['1.0']:.3f}"
              f"   pred {fa['segments_pred']:4d} -> {fb['segments_pred']:4d} (gold {fa['segments_gold']})")


if __name__ == "__main__":
    main()
