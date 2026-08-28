"""Smoke-test the deployed Replicate model with one real episode."""

from __future__ import annotations

import json, os, ssl, sys, time, urllib.request

# Some framework Python builds ship without a usable root store; certifi's is used
# when available so this runs the same way everywhere.
try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL = ssl.create_default_context()

MODEL = "mandarobotics/robot-episode-labeler"
API = "https://api.replicate.com/v1"
VIDEO = "data/wgo/galaxea_065.mp4"
PROMPT = "use a gripper to pick the target object and place on the gray plate."


def call(path: str, token: str, data: dict | None = None) -> dict:
    req = urllib.request.Request(
        path if path.startswith("http") else f"{API}{path}",
        data=json.dumps(data).encode() if data else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 # urllib's default User-Agent is rejected by the edge with 403.
                 "User-Agent": "robot-episode-labeler-smoke/1.0"},
    )
    with urllib.request.urlopen(req, timeout=90, context=_SSL) as r:
        return json.loads(r.read())


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    prompt = sys.argv[2] if len(sys.argv) > 2 else PROMPT
    subtasks = sys.argv[3] if len(sys.argv) > 3 else ""
    attributes = sys.argv[4] if len(sys.argv) > 4 else ""
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
        open(video, "rb").read(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{API}/files", data=body,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "robot-episode-labeler-smoke/1.0"},
    )
    with urllib.request.urlopen(req, timeout=180, context=_SSL) as r:
        video_url = json.loads(r.read())["urls"]["get"]
    print("uploaded episode")

    pred = call("/predictions", token, {
        "version": version,
        "input": {"video": video_url, "prompt": prompt,
                  "subtasks": subtasks, "attributes": attributes,
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
        attrs = f"  [{', '.join(s['attributes'])}]" if s.get("attributes") else ""
        print(f"  {s['start_seconds']:6.2f}-{s['end_seconds']:6.2f}  {s['result']:7s} "
              f"{s['label']}{attrs}")
    print(f"\nmetrics: {pred.get('metrics')}")
    print(f"prediction id: {pred['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
