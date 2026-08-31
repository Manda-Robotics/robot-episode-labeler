"""Fetch RoboCerebra (qiukingballball/RoboCerebra, MIT) into data/robocerebra/.

Each trainset case ships a 512x512 60 fps H.264 mp4 plus task_description.json
whose steps carry sim timesteps. Timesteps are frame indices of that mp4
(verified: ffprobe frame count == last step end + 1 on nearly every case), so
seconds = timestep / video_fps. Cases are spread across the three scenes.

Usage: uv run python scripts/fetch_robocerebra.py [--limit N] [--force]
"""

from __future__ import annotations

import argparse, json, re, shutil, subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "qiukingballball/RoboCerebra"
SCENES = ["coffee_table", "kitchen_table", "study_table"]
OUT = Path(__file__).resolve().parents[1] / "data" / "robocerebra"
RAW = OUT / "_raw"
FAMILY = "robocerebra"
LICENSE = "MIT"


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"fps": float(num) / float(den), "frames": int(s["nb_read_frames"]),
            "duration": float(s["duration"]), "codec": s["codec_name"],
            "width": int(s["width"]), "pix_fmt": s.get("pix_fmt")}


def to_h264(src: Path, dst: Path, max_w: int = 640) -> None:
    info = ffprobe(src)
    if info["codec"] == "h264" and info["pix_fmt"] == "yuv420p" and info["width"] <= max_w:
        shutil.copyfile(src, dst)
        return
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-vf", f"scale='min({max_w},iw)':-2", "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def list_cases(api: HfApi, scene: str) -> list[str]:
    names = [e.path.split("/")[-1] for e in
             api.list_repo_tree(REPO, path_in_repo=f"RoboCerebra_trainset/{scene}", repo_type="dataset")]
    cases = [n for n in names if re.fullmatch(r"case\d+", n)]
    return sorted(cases, key=lambda c: int(c[4:]))


def spread(items: list[str], n: int) -> list[str]:
    if n >= len(items):
        return items
    return [items[round(i * (len(items) - 1) / max(n - 1, 1))] for i in range(n)]


def fetch_case(scene: str, case: str, force: bool) -> str | None:
    ep_id = f"robocerebra_{scene}_{case}"
    mp4, side = OUT / f"{ep_id}.mp4", OUT / f"{ep_id}.json"
    if mp4.exists() and side.exists() and not force:
        return "exists"
    sub = f"RoboCerebra_trainset/{scene}/{case}"
    try:
        jpath = hf_hub_download(REPO, f"{sub}/task_description.json", repo_type="dataset", local_dir=RAW)
        vpath = hf_hub_download(REPO, f"{sub}/{case}.mp4", repo_type="dataset", local_dir=RAW)
    except Exception as e:  # missing file in this case dir
        print(f"  skip {scene}/{case}: download failed ({type(e).__name__})")
        return None
    try:
        d = json.loads(Path(jpath).read_text())
        instruction = d["high_level_instruction"].strip()
        steps = d["steps"]
        raw = [(int(s["timestep"]["start"]), int(s["timestep"]["end"]), s["subtask_description"].strip())
               for s in steps]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        print(f"  skip {scene}/{case}: malformed task_description.json ({e})")
        return None
    if not raw or any(e <= s for s, e, _ in raw) or any(raw[i + 1][0] < raw[i][1] for i in range(len(raw) - 1)):
        print(f"  skip {scene}/{case}: non-monotonic / empty steps")
        return None

    info = ffprobe(Path(vpath))
    fps, nframes, dur = info["fps"], info["frames"], info["duration"]
    last_end = raw[-1][1]
    notes = ("timestep unit = video frame index at %.0f fps; seconds = timestep / fps. " % fps
             + f"annotation last end={last_end}, mp4 frames={nframes}")
    if abs(nframes - last_end) > 2:
        notes += f" (MISMATCH {nframes - last_end:+d} frames; end times clamped to video duration)"
    segs = []
    for s, e, text in raw:
        ss, ee = min(s / fps, dur), min(e / fps, dur)
        if ee <= ss:
            continue
        segs.append({"start_sec": round(ss, 3), "end_sec": round(ee, 3), "subtask": text})
    if not segs:
        print(f"  skip {scene}/{case}: no segments inside video")
        return None

    to_h264(Path(vpath), mp4)
    side.write_text(json.dumps({
        "id": ep_id, "instruction": instruction, "segments": segs, "family": FAMILY,
        "metadata": {"source_dataset": REPO, "source_episode": sub, "fps": fps,
                     "duration_sec": round(dur, 3), "license": LICENSE,
                     "annotation_provenance": "human", "camera": "agentview (third-person, sim)",
                     "notes": notes}}, indent=1))
    return "written"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    per_scene = [a.limit // 3 + (1 if i < a.limit % 3 else 0) for i in range(3)]
    written = skipped = existing = 0
    for scene, n in zip(SCENES, per_scene):
        cases = spread(list_cases(api, scene), n)
        print(f"{scene}: {n} of {len(list_cases(api, scene))} cases -> {cases}")
        for case in cases:
            r = fetch_case(scene, case, a.force)
            if r == "written":
                written += 1; print(f"  wrote {OUT / f'robocerebra_{scene}_{case}'}.mp4/.json")
            elif r == "exists":
                existing += 1
            else:
                skipped += 1
    shutil.rmtree(RAW, ignore_errors=True)
    print(f"done: wrote {written}, already present {existing}, skipped {skipped} -> {OUT}")


if __name__ == "__main__":
    main()
