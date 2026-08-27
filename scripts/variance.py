"""Quantify how much of a difference between two runs is real.

Two sources of uncertainty are separated:

  * sampling -- 100 episodes is a small corpus, so a corpus-level score has a wide
    confidence interval, obtained by resampling episodes with replacement;
  * nondeterminism -- the same configuration run twice does not give the same
    answer, even at temperature 0.

Per-episode counts are computed once and the bootstrap only re-sums them, because
pooled precision, recall and boundary recall are all ratios of summed counts.
"""

from __future__ import annotations

import argparse, json, random, statistics
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rel.eval.metrics import TOLERANCES, internal_boundaries, match_segments

BOOTSTRAP = 4000
SEED = 20260827


@dataclass
class EpisodeCounts:
    n_pred: int
    n_gold: int
    n_matched: int
    b_total: int
    b_hits: dict[float, int]

    @property
    def f1(self) -> float:
        p = self.n_matched / self.n_pred if self.n_pred else 0.0
        r = self.n_matched / self.n_gold if self.n_gold else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0


def counts(row: dict) -> EpisodeCounts:
    pred = [(s[0], s[1]) for s in row["pred"]]
    gold = [(s[0], s[1]) for s in row["gold"]]
    matched = len(match_segments(pred, gold))
    gb, pb = internal_boundaries(gold), internal_boundaries(pred)
    hits = {t: 0 for t in TOLERANCES}
    if gb and pb:
        errs = [min(abs(g - p) for p in pb) for g in gb]
        for t in TOLERANCES:
            hits[t] = sum(e <= t for e in errs)
    return EpisodeCounts(len(pred), len(gold), matched, len(gb), hits)


def pooled_f1(cs: list[EpisodeCounts]) -> float:
    m = sum(c.n_matched for c in cs)
    p = m / sum(c.n_pred for c in cs) if sum(c.n_pred for c in cs) else 0.0
    r = m / sum(c.n_gold for c in cs) if sum(c.n_gold for c in cs) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def pooled_boundary(cs: list[EpisodeCounts], tol: float) -> float:
    tot = sum(c.b_total for c in cs)
    return sum(c.b_hits[tol] for c in cs) / tot if tot else 0.0


def ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    v = sorted(values)
    return v[int(len(v) * alpha / 2)], v[min(len(v) - 1, int(len(v) * (1 - alpha / 2)))]


def rows_ok(tag: str) -> dict[str, dict]:
    run = json.loads((Path("results") / f"{tag}.json").read_text())
    return {r["id"]: r for r in run["rows"] if r.get("ok")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--boot", type=int, default=BOOTSTRAP)
    args = ap.parse_args()

    A, B = rows_ok(args.a), rows_ok(args.b)
    common = sorted(set(A) & set(B))
    ca = [counts(A[i]) for i in common]
    cb = [counts(B[i]) for i in common]
    n = len(common)

    print(f"paired on {n} episodes (of {len(A)} and {len(B)} successful)\n")
    fa, fb = pooled_f1(ca), pooled_f1(cb)
    print(f"  {'segmentation F1':26s} {fa:.4f} -> {fb:.4f}   delta {fb - fa:+.4f}")

    rng = random.Random(SEED)
    idxs = [[rng.randrange(n) for _ in range(n)] for _ in range(args.boot)]

    bd = [pooled_f1([cb[i] for i in ix]) - pooled_f1([ca[i] for i in ix]) for ix in idxs]
    lo, hi = ci(bd)
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant (CI spans 0)"
    print(f"  {'':26s} 95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
    print(f"  {'':26s} P(delta>0) = {sum(d > 0 for d in bd) / len(bd):.3f}\n")

    print("  boundary recall:")
    for t in TOLERANCES:
        va, vb = pooled_boundary(ca, t), pooled_boundary(cb, t)
        d = [pooled_boundary([cb[i] for i in ix], t) - pooled_boundary([ca[i] for i in ix], t)
             for ix in idxs]
        lo, hi = ci(d)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"    +/-{t:<4} {va:.4f} -> {vb:.4f}  delta {vb - va:+.4f}"
              f"  CI [{lo:+.4f}, {hi:+.4f}]  {sig}")

    diffs = [b.f1 - a.f1 for a, b in zip(ca, cb)]
    changed = [d for d in diffs if abs(d) > 1e-9]
    print(f"\n  episodes whose F1 changed: {len(changed)}/{n}")
    if changed:
        mags = [abs(d) for d in changed]
        print(f"  per-episode |delta|: median {statistics.median(mags):.3f}"
              f"  mean {statistics.mean(mags):.3f}  max {max(mags):.3f}")
    print(f"  segments predicted: {sum(c.n_pred for c in ca)} -> {sum(c.n_pred for c in cb)}"
          f"  (gold {sum(c.n_gold for c in ca)})")


if __name__ == "__main__":
    main()
