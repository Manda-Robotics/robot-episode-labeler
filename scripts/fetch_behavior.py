"""Fetch BEHAVIOR-1K 2025 challenge demos (behavior-1k/2025-challenge-demos, MIT)
into data/behavior/ using only the head RGB camera.

annotations/task-XXXX/episode_XXXXXXXX.json -> skill_annotation, whose
frame_duration [start, end] are frame indices of the 30 fps episode videos
(verified: last end <= ffprobe frame count on every checked episode; a short
unannotated tail is common). Subtask text is built from skill_description +
the referenced object ids.

Usage: uv run python scripts/fetch_behavior.py [--limit N] [--tasks 0,34,...] [--force]
"""

from __future__ import annotations

import argparse, json, math, re, shutil, subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "behavior-1k/2025-challenge-demos"
VIDEO_KEY = "observation.images.rgb.head"
OUT = Path(__file__).resolve().parents[1] / "data" / "behavior"
RAW = OUT / "_raw"
FAMILY = "behavior"
LICENSE = "MIT"
# short-to-medium tasks (median 1-4 min) so the eval stays tractable
DEFAULT_TASKS = [0, 34, 35, 40, 10, 42, 6, 1]
PREPS = ["next to", "in front of", "from", "onto", "into", "on", "in", "to", "with", "under", "at", "over", "inside"]
NO_PREP_DEFAULT = {"hang": ["on"], "attach": ["to"], "hand over": ["to"], "hold": ["with"], "release": ["from"],
                   "pour": ["from", "into"], "chop": ["on", "with"], "cut": ["on", "with"], "dice": ["on", "with"]}
TOOL_VERBS = {"chop", "cut", "dice", "slice", "scrub", "sweep", "spray", "wipe", "brush", "stir"}
TRAILING_NOUN = {"door", "switch", "lid", "drawer", "button"}
SPATIAL = {"right": "to the right of", "left": "to the left of", "in_front_of": "in front of", "behind": "behind",
           "center": "at the center of", "top": "on top of", "under": "under", "next_to": "next to", "near": "near"}


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
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-vf", f"scale='min({max_w},iw)':-2", "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


VOCAB: set[str] = set()   # per-episode word list (from meta inst_to_name synsets) to tell words from model ids


def obj_name(oid: str) -> str:
    # radio_89 -> radio ; coffee_table_koagbh_0 -> coffee table ; electric_kettle_207 -> electric kettle
    if isinstance(oid, (list, tuple)):  # substances arrive as a nested list, e.g. [["diced__vidalia_onion"], "bowl_73"]
        return " and ".join(obj_name(o) for o in oid)
    base = re.sub(r"(_\d+)+$", "", oid)   # half_vidalia_onion_75_0 -> half_vidalia_onion
    toks = [t for t in base.split("_") if t]
    if len(toks) > 1 and re.fullmatch(r"[a-z]{6}", toks[-1]) and toks[-1] not in VOCAB:
        toks = toks[:-1]  # 6-letter OmniGibson model id
    return " ".join(toks)


def load_vocab(meta_path: Path) -> None:
    VOCAB.clear()
    try:
        cfg = json.loads(json.loads(meta_path.read_text())["config"])
        inst = cfg["scene"]["scene_file"]["metadata"]["task"]["inst_to_name"]
    except Exception:
        return
    for synset in inst:  # category words only; instance names carry the random model ids we want to drop
        VOCAB.update(re.split(r"[._]", synset.split(".n.")[0]))


def _flat(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    out: list[str] = []
    for y in x or []:
        out += _flat(y)
    return out


def _split_desc(desc: str) -> tuple[str, list[str]]:
    """'place on next to' -> ('place', ['on', 'next to']); 'pick up from' -> ('pick up', ['from'])."""
    rest, preps = desc.strip(), []
    while True:
        hit = None
        for p in PREPS:
            if rest.endswith(" " + p) or rest == p:
                hit = p; break
        if not hit or rest == hit:
            break
        rest = rest[: -len(hit)].strip(); preps.insert(0, hit)
    return rest, preps


def phrase(desc: str, objs: list[str], memory, spatial, manip: list[str] | None = None) -> str:
    names = [obj_name(o) for o in objs]
    verb, preps = _split_desc(desc)
    if not names:
        return desc.strip()
    if verb in TOOL_VERBS and len(names) >= 2 and manip and objs[0] == manip[0]:
        return f"{verb} {' and '.join(names[1:])} with {names[0]}"       # objs = [tool, target]
    if desc.strip() == "hand over" and len(names) == 3 and {names[1], names[2]} <= {"left", "right"}:
        return f"hand over {names[0]} from {names[1]} hand to {names[2]} hand"
    # spatial modifiers are aligned per object ([["", "", "right"]]); memory ones are free-form
    spa = _flat(spatial)
    spa = spa + [""] * (len(names) - len(spa))
    mem = [m for m in _flat(memory) if m]
    first = names[0]
    if mem and "back" in mem:
        if len(names) == 1 and preps:
            verb += " back"          # move to X + back -> move back to X
        else:
            first += " back"         # place X back on Y
    pre = " ".join(m for m in mem if m != "back")
    if pre:
        first = f"{pre} {first}"
    text = f"{verb} {first}"
    if len(names) == 1:
        words = verb.split()
        if not preps and len(words) > 1 and words[-1] in TRAILING_NOUN:
            text = f"{' '.join(words[:-1])} {first} {words[-1]}"   # open door + microwave -> open microwave door
        else:
            text = f"{desc.strip()} {first}" if not preps else f"{verb} {' '.join(preps)} {first}"
    for i, name in enumerate(names[1:], start=1):
        fallback = NO_PREP_DEFAULT.get(verb, ["and"])
        prep = preps[i - 1] if i - 1 < len(preps) else (preps[-1] if preps else fallback[min(i - 1, len(fallback) - 1)])
        if spa[i]:
            prep = SPATIAL.get(spa[i], spa[i].replace("_", " "))
        text += f" {prep} {name}"
    return re.sub(r"\s+", " ", text).strip()


def build_subtask(skill: dict) -> str:
    descs = skill["skill_description"]
    objs = skill.get("object_id") or [[] for _ in descs]
    parts = []
    for i, d in enumerate(descs):
        o = objs[i] if i < len(objs) else []
        parts.append(phrase(d, list(o), skill.get("memory_prefix") or [], skill.get("spatial_prefix") or [],
                            skill.get("manipulating_object_id") or []))
    return " and ".join(parts)


def load_meta() -> tuple[dict, list[dict]]:
    tasks = {}
    for line in Path(hf_hub_download(REPO, "meta/tasks.jsonl", repo_type="dataset", local_dir=RAW)).read_text().splitlines():
        if line.strip():
            t = json.loads(line); tasks[t["task"]] = t
    eps = [json.loads(l) for l in Path(hf_hub_download(REPO, "meta/episodes.jsonl", repo_type="dataset", local_dir=RAW)).read_text().splitlines() if l.strip()]
    for e in eps:
        e["task"] = tasks[e["tasks"][0]]
    return tasks, eps


def fetch_episode(ep: dict, force: bool, relabel: bool = False) -> str | None:
    ti, ei = ep["task"]["task_index"], ep["episode_index"]
    ep_id = f"behavior_{ep['task']['task_name']}_{ei}"
    mp4, side = OUT / f"{ep_id}.mp4", OUT / f"{ep_id}.json"
    relabel = relabel and mp4.exists()
    if mp4.exists() and side.exists() and not force and not relabel:
        return "exists"
    apath = f"annotations/task-{ti:04d}/episode_{ei:08d}.json"
    vpath = f"videos/task-{ti:04d}/{VIDEO_KEY}/episode_{ei:08d}.mp4"
    mpath = f"meta/episodes/task-{ti:04d}/episode_{ei:08d}.json"
    try:
        load_vocab(Path(hf_hub_download(REPO, mpath, repo_type="dataset", local_dir=RAW)))
    except Exception as e:
        print(f"  warn {ep_id}: no episode meta for object vocabulary ({type(e).__name__})"); VOCAB.clear()
    try:
        a = json.loads(Path(hf_hub_download(REPO, apath, repo_type="dataset", local_dir=RAW)).read_text())
        raw = [(int(s["frame_duration"][0]), int(s["frame_duration"][1]), build_subtask(s))
               for s in a["skill_annotation"]]
    except Exception as e:
        print(f"  skip {ep_id}: annotation missing/malformed ({type(e).__name__}: {e})")
        return None
    if not raw or any(e <= s for s, e, _ in raw) or any(raw[i + 1][0] < raw[i][1] for i in range(len(raw) - 1)):
        print(f"  skip {ep_id}: empty/non-monotonic skill_annotation")
        return None
    if relabel:
        v = None
    else:
        try:
            v = Path(hf_hub_download(REPO, vpath, repo_type="dataset", local_dir=RAW))
        except Exception as e:
            print(f"  skip {ep_id}: video download failed ({type(e).__name__})")
            return None
    info = ffprobe(v or mp4)
    fps, nframes, dur = info["fps"], info["frames"], info["duration"]
    last_end = raw[-1][1]
    notes = (f"frame_duration are frame indices of the {fps:.0f} fps video; seconds = frame / fps. "
             f"annotation valid_duration={a.get('meta_data', {}).get('valid_duration')}, last end={last_end}, "
             f"mp4 frames={nframes} (episodes.jsonl length={ep['length']})")
    if last_end > nframes + 2:
        notes += f" (MISMATCH: annotation exceeds video by {last_end - nframes} frames; clamped)"
    elif nframes - last_end > 2:
        notes += f"; trailing {nframes - last_end} frames unannotated"
    segs = [{"start_sec": round(min(s / fps, dur), 3), "end_sec": round(min(e / fps, dur), 3), "subtask": t}
            for s, e, t in raw]
    segs = [s for s in segs if s["end_sec"] > s["start_sec"]]
    if v is not None:
        to_h264(v, mp4)
        v.unlink(missing_ok=True)
    side.write_text(json.dumps({
        "id": ep_id, "instruction": ep["task"]["task"], "segments": segs, "family": FAMILY,
        "metadata": {"source_dataset": REPO, "source_episode": f"task-{ti:04d}/episode_{ei:08d}",
                     "task_name": ep["task"]["task_name"], "video_key": VIDEO_KEY,
                     "fps": fps, "duration_sec": round(dur, 3), "license": LICENSE,
                     "annotation_provenance": "human-teleop",
                     "notes": notes + ". Subtask text auto-composed from skill_description + object ids."}}, indent=1))
    return "written"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--tasks", type=str, default=",".join(map(str, DEFAULT_TASKS)))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--relabel", action="store_true", help="rewrite sidecar json for episodes whose mp4 exists")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    task_ids = [int(x) for x in a.tasks.split(",") if x.strip()]
    tasks, eps = load_meta()
    per_task = math.ceil(a.limit / len(task_ids))
    chosen: list[dict] = []
    for ti in task_ids:
        chosen += [e for e in eps if e["task"]["task_index"] == ti][:per_task]
    chosen = chosen[:a.limit]
    written = skipped = existing = 0
    for ep in chosen:
        r = fetch_episode(ep, a.force, a.relabel)
        if r == "written":
            written += 1; print(f"  wrote {OUT}/behavior_{ep['task']['task_name']}_{ep['episode_index']}.mp4/.json")
        elif r == "exists":
            existing += 1
        else:
            skipped += 1
    shutil.rmtree(RAW, ignore_errors=True)
    print(f"done: wrote {written}, already present {existing}, skipped {skipped} -> {OUT}")


if __name__ == "__main__":
    main()
