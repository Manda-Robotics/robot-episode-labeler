"""Fetch AgiBot World 2026 (agibot-world/AgiBotWorld2026, CC BY-NC-SA 4.0, real
G2 robot, human step annotations) into data/agibot/ using the head camera.

The repo ships LeRobot v2.1 sub-datasets as multi-GB tar.gz chunks that are
~93% depth video. We stream each chunk over HTTP and keep only meta/ plus the
observation.images.top_head mp4s, so nothing large ever touches disk.
Step segments come from meta/info.json["instruction_segments"][ep] (track
"default"; frame indices at 30 fps -> seconds = frame / fps). The coarse
"Subtask" track gives the episode-level instruction. top_head is AV1, decoded
with the static imageio-ffmpeg binary when the system ffmpeg cannot.

Usage: uv run python scripts/fetch_agibot.py [--limit N] [--archives a,b] [--force]
"""

from __future__ import annotations

import argparse, json, re, shutil, ssl, subprocess, tarfile, urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_url

REPO = "agibot-world/AgiBotWorld2026"
VIDEO_KEY = "observation.images.top_head"
# smallest chunks with real step annotations (0.33 GB task_4439 is a single
# "random interaction" clip with one segment and is deliberately excluded)
DEFAULT_ARCHIVES = [
    "ImitationLearning/Home/task_4713/509995_510027.tar.gz",            # 3.09 GB, 7 eps
    "ImitationLearning/CommercialSpaces/task_4542/553049_553079.tar.gz",  # 1.44 GB, 4 eps
]
OUT = Path(__file__).resolve().parents[1] / "data" / "agibot"
RAW = OUT / "_raw"
FAMILY = "agibot"
LICENSE = "CC-BY-NC-SA-4.0"


def static_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,r_frame_rate,nb_read_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"fps": float(num) / float(den), "frames": int(s["nb_read_frames"]),
            "duration": float(s["duration"]), "codec": s["codec_name"], "width": int(s["width"])}


def to_h264(src: Path, dst: Path, max_w: int = 640) -> None:
    cmd = ["-y", "-v", "error", "-i", str(src), "-vf", f"scale='min({max_w},iw)':-2",
           "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(dst)]
    r = subprocess.run(["ffmpeg", *cmd], capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        subprocess.run([static_ffmpeg(), *cmd], check=True)


def stream_extract(archive: str, dest: Path) -> None:
    """Stream the tar.gz from the Hub and keep only meta/ and top_head videos."""
    marker = dest / ".complete"
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(REPO, archive, repo_type="dataset")
    print(f"streaming {archive} (keeping meta + {VIDEO_KEY} only) ...")
    kept = 0
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if "/meta/" in m.name or f"/{VIDEO_KEY}/" in m.name:
                rel = Path(m.name)
                target = dest / rel.relative_to(rel.parts[0]) if rel.parts[0] == "data" else dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as f, open(target, "wb") as o:
                    shutil.copyfileobj(f, o)
                kept += 1
    marker.write_text(str(kept))
    print(f"  kept {kept} files")


def episode_id(info: dict, ek: str, archive: str) -> tuple[str, str]:
    h5 = (info.get("h5_path") or {}).get(ek, "")
    m = re.search(r"/(\d+)/(\d+)/proprio_stats\.h5$", h5)
    if m:
        return m.group(1), m.group(2)
    task = re.search(r"task_(\d+)", archive).group(1)
    return task, f"{Path(archive).name.split('.')[0]}_{int(ek):03d}"


def convert(dest: Path, archive: str, force: bool, remaining: int) -> tuple[int, int, int]:
    info = json.loads((dest / "meta" / "info.json").read_text())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in (dest / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()}
    eps = {e["episode_index"]: e for e in
           (json.loads(l) for l in (dest / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip())}
    fps_meta = info.get("fps", 30)
    written = skipped = existing = 0
    for ek, segs_all in sorted(info["instruction_segments"].items(), key=lambda kv: int(kv[0])):
        if remaining - written <= 0:
            break
        task, epid = episode_id(info, ek, archive)
        ep_id = f"agibot_task{task}_{epid}"
        mp4, side = OUT / f"{ep_id}.mp4", OUT / f"{ep_id}.json"
        if mp4.exists() and side.exists() and not force:
            existing += 1; continue
        src = dest / "videos" / f"chunk-{int(ek) // info.get('chunks_size', 1000):03d}" / VIDEO_KEY / f"episode_{int(ek):06d}.mp4"
        if not src.exists():
            print(f"  skip {ep_id}: no {VIDEO_KEY} video in archive"); skipped += 1; continue
        steps = sorted((s for s in segs_all if s.get("track") == "default" and (s.get("instruction") or "").strip()),
                       key=lambda s: (s["start_frame_index"], s["end_frame_index"]))
        raw = [(int(s["start_frame_index"]), int(s["end_frame_index"]), s["instruction"].strip(), s.get("skill") or "")
               for s in steps]
        if not raw or any(e <= s for s, e, *_ in raw) or any(raw[i + 1][0] < raw[i][1] for i in range(len(raw) - 1)):
            print(f"  skip {ep_id}: empty/overlapping default-track segments"); skipped += 1; continue
        coarse = []
        for s in sorted((s for s in segs_all if s.get("track", "").lower() == "subtask"), key=lambda s: s["start_frame_index"]):
            t = (s.get("instruction") or "").strip()
            if t and t not in coarse:
                coarse.append(t)
        task_text = tasks.get(int(eps.get(int(ek), {}).get("task_index", 0)), next(iter(tasks.values()), ""))
        instruction = " ".join(coarse) if coarse else task_text

        to_h264(src, mp4)
        pinfo = ffprobe(mp4)
        fps, nframes, dur = pinfo["fps"], pinfo["frames"], pinfo["duration"]
        length = eps.get(int(ek), {}).get("length")
        last_end = raw[-1][1]
        notes = (f"frame indices at {fps:.0f} fps (meta fps={fps_meta}); seconds = frame / fps. "
                 f"last end={last_end}, mp4 frames={nframes}, episodes.jsonl length={length}")
        if last_end > nframes + 2:
            notes += f" (MISMATCH: annotation exceeds video by {last_end - nframes} frames; clamped)"
        elif nframes - last_end > 2:
            notes += f"; trailing {nframes - last_end} frames unannotated"
        segs = [{"start_sec": round(min(s / fps, dur), 3), "end_sec": round(min(e / fps, dur), 3), "subtask": t}
                for s, e, t, _ in raw]
        segs = [s for s in segs if s["end_sec"] > s["start_sec"]]
        side.write_text(json.dumps({
            "id": ep_id, "instruction": instruction, "segments": segs, "family": FAMILY,
            "metadata": {"source_dataset": REPO, "source_episode": f"{archive}#episode_{int(ek):06d} (task {task}, episode {epid})",
                         "task_name": task_text, "coarse_subtasks": coarse, "video_key": VIDEO_KEY,
                         "fps": fps, "duration_sec": round(dur, 3), "license": LICENSE,
                         "annotation_provenance": "human",
                         "notes": notes + ". Segments = instruction_segments track 'default'; instruction = 'Subtask' track text."}}, indent=1))
        written += 1
        print(f"  wrote {mp4} ({len(segs)} segments, {dur:.1f}s)")
    return written, existing, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--archives", type=str, default=",".join(DEFAULT_ARCHIVES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tot_w = tot_e = tot_s = 0
    for archive in [x.strip() for x in a.archives.split(",") if x.strip()]:
        if tot_w + tot_e >= a.limit:
            break
        dest = RAW / Path(archive).name.split(".")[0]
        stream_extract(archive, dest)
        w, e, s = convert(dest, archive, a.force, a.limit - tot_w - tot_e)
        tot_w += w; tot_e += e; tot_s += s
    if not a.keep_raw:
        shutil.rmtree(RAW, ignore_errors=True)
    print(f"done: wrote {tot_w}, already present {tot_e}, skipped {tot_s} -> {OUT}")


if __name__ == "__main__":
    main()
