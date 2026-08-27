"""Smoke-test the deployed Replicate model with one real episode."""

from __future__ import annotations

import json, os, sys, time, urllib.request

MODEL = "mandarobotics/robot-episode-labeler"
API = "https://api.replicate.com/v1"
VIDEO = "data/wgo/galaxea_065.mp4"
PROMPT = "use a gripper to pick the target object and place on the gray plate."


def call(path: str, token: str, data: dict | None = None) -> dict:
    req = urllib.request.Request(
        path if path.startswith("http") else f"{API}{path}",
        data=json.dumps(data).encode() if data else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def main() -> int:
    token = os.environ.get("REPLICATE_API_TOKEN")
    gemini = os.environ.get("GEMINI_API_KEY")
    if not token or not gemini:
        print("REPLICATE_API_TOKEN and GEMINI_API_KEY must both be set", file=sys.stderr)
        return 2

    version = (call(f"/models/{MODEL}", token).get("latest_version") or {}).get("id")
    if not version:
        print(f"{MODEL} has no published version", file=sys.stderr)
        return 1
    print(f"version {version[:16]}...")

    # Upload the episode; multipart is hand-rolled to keep this dependency-free.
    boundary = "----smoke"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="content"; filename="episode.mp4"\r\n',
        b"Content-Type: video/mp4\r\n\r\n",
        open(VIDEO, "rb").read(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{API}/files", data=body,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        video_url = json.loads(r.read())["urls"]["get"]
    print("uploaded episode")

    pred = call("/predictions", token, {
        "version": version,
        "input": {"video": video_url, "prompt": PROMPT,
                  "quality": "balanced", "gemini_api_key": gemini},
    })
    print(f"prediction {pred['id']} -> {pred['status']}")

    while pred["status"] in ("starting", "processing"):
        time.sleep(4)
        pred = call(pred["urls"]["get"], token)

    if pred["status"] != "succeeded":
        print(f"FAILED: {pred['status']}: {pred.get('error')}", file=sys.stderr)
        return 1

    out = json.loads(pred["output"])
    print(f"\n{out['task']}  ({out['duration_seconds']}s, {len(out['segments'])} subtasks)")
    for s in out["segments"]:
        print(f"  {s['start_seconds']:6.2f}-{s['end_seconds']:6.2f}  {s['result']:7s} {s['label']}")
    print(f"\nmetrics: {pred.get('metrics')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
