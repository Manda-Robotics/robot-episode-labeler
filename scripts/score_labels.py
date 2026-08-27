"""Score label accuracy on temporally matched segments of an existing eval run."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rel.annotation.llm import GeminiClient
from rel.eval.judge import score_labels
from rel.eval.metrics import match_segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    # Judged by a stronger model from a different tier than the system under test,
    # so the judge does not share the annotator's blind spots. Recorded in the
    # result either way.
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    args = ap.parse_args()

    path = Path("results") / f"{args.tag}.json"
    run = json.loads(path.read_text())

    pairs, provenance = [], []
    for r in run["rows"]:
        if not r.get("ok"):
            continue
        pred, gold = r["pred"], r["gold"]
        for m in match_segments([(p[0], p[1]) for p in pred],
                                [(g[0], g[1]) for g in gold], args.iou):
            pairs.append((pred[m.pred_index][2], gold[m.gold_index][2]))
            provenance.append({"episode": r["id"], "family": r["family"], "iou": round(m.iou, 3)})

    if not pairs:
        print("no temporally matched segments to score"); return

    client = GeminiClient(model=args.model)
    acc, verdicts = score_labels(client, pairs)
    by_index = {v.index: v for v in verdicts}

    by_family: dict[str, list[int]] = {}
    for i, prov in enumerate(provenance):
        v = by_index.get(i)
        by_family.setdefault(prov["family"], []).append(1 if (v and v.same_event) else 0)

    run.setdefault("scores", {})["label_accuracy"] = {
        "matched_pairs": len(pairs),
        "iou_threshold": args.iou,
        "accuracy": round(acc, 4),
        "by_family": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_family.items())},
        "judge_model": args.model,
    }
    run["label_verdicts"] = [
        {**provenance[i], "predicted": pairs[i][0], "gold": pairs[i][1],
         "same_event": bool(by_index.get(i) and by_index[i].same_event),
         "note": (by_index[i].note if by_index.get(i) else "no verdict returned")}
        for i in range(len(pairs))
    ]
    path.write_text(json.dumps(run, indent=2))

    print(f"label accuracy on {len(pairs)} matched segments (IoU>={args.iou}): {acc:.3f}")
    for fam, vals in sorted(by_family.items()):
        print(f"  {fam:9s} {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
