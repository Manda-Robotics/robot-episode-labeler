"""Compare eval runs side by side. Prints the table we actually make decisions on."""

from __future__ import annotations

import argparse, json
from pathlib import Path

TOL = ["0.25", "0.5", "1.0", "2.0"]

# Published list price per 1M tokens (input / output incl. thinking), by model prefix.
# gemini-3.7-flash and 3.5-flash are $0.75 / $3.75 through 2026 (ai.google.dev/pricing).
PRICES = {
    "gemini-3.7-flash": (0.75, 3.75), "gemini-3.6-flash": (0.75, 3.75), "gemini-3.5-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-robotics-er-2-preview": (2.00, 10.00), "gemini-2.5-flash": (0.30, 2.50),
}


def load(tag: str) -> dict:
    return json.loads((Path("results") / f"{tag}.json").read_text())


def row(run: dict) -> dict:
    s = run["scores"]["overall"]
    vid_h = run.get("video_seconds", 0) / 3600 or None
    tok = run.get("tokens", {})
    price_in, price_out = PRICES.get(run["model"], (0.75, 3.75))
    cost = (tok.get("prompt", 0) / 1e6 * price_in) + (tok.get("output", 0) / 1e6 * price_out)
    lab = run.get("scores", {}).get("label_accuracy", {})
    return {
        "tag": run["tag"],
        "model": run["model"].replace("gemini-", ""),
        "quality": run["quality"],
        "f1": s["segmentation_f1"],
        "wgo": s.get("f1_wgo"),
        "short": (s.get("recall_by_duration") or {}).get("1-2", {}).get("recall"),
        "p": s["segmentation_precision"],
        "r": s["segmentation_recall"],
        "bnd": s["boundary_recall_at"],
        "med": s["median_boundary_error_sec"],
        "pred": s["segments_pred"],
        "gold": s["segments_gold"],
        "label_acc": lab.get("accuracy"),
        "cost_per_video_hour": round(cost / vid_h, 2) if vid_h else None,
        "failed": run.get("failed", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--family", action="store_true", help="also break down per family")
    args = ap.parse_args()
    runs = [load(t) for t in args.tags]
    rows = [row(r) for r in runs]

    head = (f"{'run':26s} {'model':11s} {'qual':9s} {'segF1':>6s} {'wgo':>6s} {'P':>5s} {'R':>5s} "
            + " ".join(f"{'±'+t:>6s}" for t in TOL) + f" {'med':>6s} {'R1-2s':>6s} {'lbl':>5s} {'$/vid-h':>8s}")
    print(head); print("-" * len(head))
    for r in rows:
        print(f"{r['tag'][:26]:26s} {r['model'][:11]:11s} {r['quality']:9s} "
              f"{r['f1']:6.3f} {(r['wgo'] if r['wgo'] is not None else float('nan')):6.3f} {r['p']:5.3f} {r['r']:5.3f} "
              + " ".join(f"{r['bnd'].get(t, 0):6.3f}" for t in TOL)
              + f" {(r['med'] if r['med'] is not None else 0):6.2f}"
              + f" {(r['short'] if r['short'] is not None else float('nan')):6.3f}"
              + f" {r['label_acc'] if r['label_acc'] is not None else float('nan'):5.3f}"
              + f" {r['cost_per_video_hour'] if r['cost_per_video_hour'] else 0:8.2f}")
    print()
    for r in rows:
        note = f"  ({r['failed']} episodes failed)" if r["failed"] else ""
        print(f"{r['tag']}: {r['pred']} predicted vs {r['gold']} gold segments{note}")

    if args.family:
        fams = sorted({f for run in runs for f in run["scores"]["by_family"]})
        print()
        h = f"{'run':26s} " + " ".join(f"{f:>22s}" for f in fams)
        print(h); print("-" * len(h))
        for run in runs:
            cells = []
            for f in fams:
                fs = run["scores"]["by_family"].get(f)
                cells.append(f"{'F1 '+format(fs['segmentation_f1'],'.3f')+' ±1s '+format(fs['boundary_recall_at'].get('1.0',0),'.2f'):>22s}"
                             if fs else f"{'-':>22s}")
            print(f"{run['tag'][:26]:26s} " + " ".join(cells))


if __name__ == "__main__":
    main()
