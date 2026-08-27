"""Eyeball one episode: predicted vs gold as aligned timelines.

Aggregate metrics say how well we do; this says what we got wrong, which is what
actually drives prompt changes.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rel.eval.metrics import internal_boundaries, match_segments

WIDTH = 72


def bar(spans: list[tuple[float, float]], duration: float, labels: list[str]) -> str:
    """One character per time slice, cycling glyphs so adjacent segments differ."""
    glyphs = "#=+*o~"
    row = [" "] * WIDTH
    for i, (a, b) in enumerate(spans):
        lo = int(a / duration * WIDTH)
        hi = max(lo + 1, int(b / duration * WIDTH))
        for x in range(lo, min(hi, WIDTH)):
            row[x] = glyphs[i % len(glyphs)]
    return "".join(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episode", help="episode id; default = the worst-scoring one")
    ap.add_argument("--worst", type=int, default=1, help="show N worst episodes")
    args = ap.parse_args()

    run = json.loads((Path("results") / f"{args.tag}.json").read_text())
    rows = [r for r in run["rows"] if r.get("ok")]

    if args.episode:
        chosen = [r for r in rows if r["id"] == args.episode]
    else:
        def f1(r):
            pred = [(p[0], p[1]) for p in r["pred"]]
            gold = [(g[0], g[1]) for g in r["gold"]]
            m = len(match_segments(pred, gold))
            p = m / len(pred) if pred else 0.0
            rc = m / len(gold) if gold else 0.0
            return 2 * p * rc / (p + rc) if (p + rc) else 0.0
        chosen = sorted(rows, key=f1)[: args.worst]

    for r in chosen:
        pred = [(p[0], p[1]) for p in r["pred"]]
        gold = [(g[0], g[1]) for g in r["gold"]]
        duration = max(r.get("duration", 0), gold[-1][1] if gold else 0) or 1.0
        matched = {m.gold_index for m in match_segments(pred, gold)}

        print(f"\n=== {r['id']} ({r['family']}) — {duration:.1f}s ===")
        print(f"  gold |{bar(gold, duration, [])}|  {len(gold)} segments")
        print(f"  pred |{bar(pred, duration, [])}|  {len(pred)} segments")
        gb, pb = internal_boundaries(gold), internal_boundaries(pred)
        if gb:
            errs = [min(abs(g - p) for p in pb) if pb else float("inf") for g in gb]
            print(f"  boundary errors: " +
                  ", ".join(f"{g:.1f}s->{e:.2f}" if e != float('inf') else f"{g:.1f}s->miss"
                            for g, e in zip(gb, errs)))
        print("\n  GOLD")
        for i, g in enumerate(r["gold"]):
            print(f"    {'MATCH' if i in matched else '  -  '} {g[0]:6.2f}-{g[1]:6.2f}  {g[2][:64]}")
        print("  PREDICTED")
        for p in r["pred"]:
            print(f"          {p[0]:6.2f}-{p[1]:6.2f}  {p[2][:64]}")
        if r.get("warnings"):
            print("  WARNINGS")
            for w in r["warnings"][:5]:
                print(f"    - {w}")


if __name__ == "__main__":
    main()
