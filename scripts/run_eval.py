"""Run the pipeline over WGO-Bench and score it. Writes results/<tag>.json."""

from __future__ import annotations

import argparse, json, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rel.annotation.llm import DEFAULT_MODEL, GeminiClient
from rel.eval import wgo
from rel.eval.metrics import Aggregate, TOLERANCES
from rel.pipeline import annotate
from rel.schemas import AnnotateRequest, Quality


def run_one(ep: wgo.Episode, model: str, quality: Quality, subdivide: bool | None) -> dict:
    client = GeminiClient(model=model)
    t0 = time.time()
    try:
        resp = annotate(
            AnnotateRequest(video=str(ep.video), prompt=ep.instruction, quality=quality),
            client=client, subdivide=subdivide,
        )
        return {
            "id": ep.id, "family": ep.family, "ok": True,
            "elapsed": round(time.time() - t0, 2),
            "duration": resp.duration_seconds,
            "pred": [[s.start_seconds, s.end_seconds, s.label] for s in resp.segments],
            "gold": [[g.start, g.end, g.subtask] for g in ep.segments],
            "warnings": resp.warnings,
            "usage": client.usage.as_dict(),
        }
    except Exception as exc:
        return {"id": ep.id, "family": ep.family, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "trace": traceback.format_exc()[-400:],
                "gold": [[g.start, g.end, g.subtask] for g in ep.segments],
                "usage": client.usage.as_dict()}


def score(rows: list[dict]) -> dict:
    overall = Aggregate()
    per_family: dict[str, Aggregate] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        pred = [(p[0], p[1]) for p in r["pred"]]
        gold = [(g[0], g[1]) for g in r["gold"]]
        overall.add(pred, gold)
        per_family.setdefault(r["family"], Aggregate()).add(pred, gold)
    return {
        "overall": overall.report(),
        "by_family": {k: v.report() for k, v in sorted(per_family.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quality", default="fast", choices=[q.value for q in Quality])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--family", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true",
                    help="continue from results/<tag>.partial.jsonl")
    ap.add_argument("--subdivide", dest="subdivide", action="store_true", default=None,
                    help="force the subdivision pass on")
    ap.add_argument("--no-subdivide", dest="subdivide", action="store_false",
                    help="force the subdivision pass off")
    args = ap.parse_args()

    eps = wgo.load(limit=args.limit, family=args.family)
    print(f"{args.tag}: {len(eps)} episodes | model={args.model} quality={args.quality}", flush=True)

    rows: list[dict] = []
    # Episodes are appended as they complete: a run killed part way through still
    # leaves usable results, and --resume picks up from them.
    checkpoint = Path("results") / f"{args.tag}.partial.jsonl"
    done: dict[str, dict] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        rows.extend(done.values())
        eps = [e for e in eps if e.id not in done]
        print(f"  resuming: {len(done)} already done, {len(eps)} to go", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, e, args.model, Quality(args.quality), args.subdivide): e for e in eps}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            with checkpoint.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            mark = "ok " if r.get("ok") else "ERR"
            extra = f"{len(r.get('pred', []))}seg" if r.get("ok") else r.get("error", "")[:60]
            print(f"  [{i:3d}/{len(eps)}] {mark} {r['id']:14s} {extra}", flush=True)

    rows.sort(key=lambda r: r["id"])
    tokens_in = sum(r["usage"]["prompt_tokens"] for r in rows)
    tokens_out = sum(r["usage"]["output_tokens"] for r in rows)
    video_sec = sum(r.get("duration", 0) for r in rows if r.get("ok"))
    out = {
        "tag": args.tag, "model": args.model, "quality": args.quality,
        "subdivide": args.subdivide,
        "episodes": len(rows), "failed": sum(1 for r in rows if not r.get("ok")),
        "wall_seconds": round(time.time() - t0, 1),
        "video_seconds": round(video_sec, 1),
        "tokens": {"prompt": tokens_in, "output": tokens_out},
        "scores": score(rows), "rows": rows,
    }
    path = Path("results") / f"{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    checkpoint.unlink(missing_ok=True)

    s = out["scores"]["overall"]
    print(f"\n=== {args.tag} ===")
    print(f"  seg F1 {s['segmentation_f1']:.3f}  (P {s['segmentation_precision']:.3f} / R {s['segmentation_recall']:.3f})")
    print(f"  boundary recall @ {{{', '.join(f'{t}s: ' + format(s['boundary_recall_at'][str(t)], '.3f') for t in TOLERANCES)}}}")
    print(f"  median boundary error: {s['median_boundary_error_sec']}s")
    for fam, fs in out["scores"]["by_family"].items():
        print(f"  {fam:8s} F1 {fs['segmentation_f1']:.3f}  pred/gold {fs['segments_pred']}/{fs['segments_gold']}")
    print(f"  failed: {out['failed']}  wall {out['wall_seconds']}s  tokens in/out {tokens_in}/{tokens_out}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
