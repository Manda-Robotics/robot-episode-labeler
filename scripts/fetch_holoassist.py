"""Fetch HoloAssist (holoassist.github.io, CDLA-Permissive-2.0, egocentric human)
into data/holoassist/.

HoloAssist ships only one 155 GB uncompressed tar of the 256px videos, so we
walk its tar headers with HTTP Range requests starting at a known offset in the
(alphabetical, contiguous) mp4 region and download just the mp4s of the
sessions we want. The 111 MB annotation JSON gives per-session events in
seconds: "Coarse grained action" (Action sentence) and "Fine grained action"
(Verb/Adjective/Noun/adverbial). Long sessions are cut into <= --max-sec
episodes at coarse-action boundaries; fine-grained actions become segments and
the coarse sentences of the chunk become the instruction.

Usage: uv run python scripts/fetch_holoassist.py [--limit N] [--sessions N] [--force]
"""

from __future__ import annotations

import argparse, json, re, shutil, ssl, subprocess, urllib.request
from pathlib import Path

BASE = "https://hl2data.z5.web.core.windows.net/holoassist-data-release/"
ANNOT_URL = BASE + "data-annotation-trainval-v1_1.json"
TAR_URL = BASE + "video_compress.tar"
# byte offset of the tar header of R039-13July-DSLR/Export_py/Video_compress.mp4
# (found by probing; the pose/timing txt region occupies the first ~5 GB)
TAR_ANCHOR = 6010863616
OUT = Path(__file__).resolve().parents[1] / "data" / "holoassist"
RAW = OUT / "_raw"
FAMILY = "holoassist"
LICENSE = "CDLA-Permissive-2.0"


def _ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_range(start: int, length: int, dest: Path | None = None) -> bytes:
    req = urllib.request.Request(TAR_URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, context=_ctx()) as r:
        if dest is None:
            return r.read()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        return b""


def _pad(n: int) -> int:
    return (n + 511) // 512 * 512


def tar_entries(offset: int):
    """Yield (name, size, data_offset, next_offset) walking headers from offset."""
    while True:
        blk = fetch_range(offset, 2048)
        pos = 0
        name = None
        while True:
            h = blk[pos:pos + 512]
            if len(h) < 512 or h[257:262] != b"ustar":
                return
            n = h[:100].rstrip(b"\0").decode(errors="replace")
            size = int((h[124:136].rstrip(b"\0 ") or b"0"), 8)
            typ = h[156:157]
            if typ == b"x":  # pax header: may carry a long path
                pax = blk[pos + 512:pos + 512 + size].decode(errors="replace")
                m = re.search(r"\d+ path=(.*)\n", pax)
                if m:
                    name = m.group(1)
                pos += 512 + _pad(size)
                continue
            name = name or n
            data_off = offset + pos + 512
            yield name, size, data_off, data_off + _pad(size)
            offset = data_off + _pad(size)
            break


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,r_frame_rate,nb_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"fps": float(num) / float(den), "frames": int(s.get("nb_frames") or 0),
            "duration": float(s["duration"]), "codec": s["codec_name"], "width": int(s["width"])}


def cut_h264(src: Path, dst: Path, start: float, end: float, max_w: int = 640) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
                    "-vf", f"scale='min({max_w},iw)':-2", "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def fine_text(attrs: dict) -> str:
    parts = [attrs.get("Verb", ""), attrs.get("Adjective", ""), attrs.get("Noun", ""), attrs.get("adverbial", "")]
    parts = [p.strip() for p in parts if p and p.strip().lower() != "none"]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def chunks_for(session: dict, max_sec: float) -> list[tuple[float, float, list[str]]]:
    coarse = sorted((e for e in session["events"] if e["label"] == "Coarse grained action"), key=lambda e: e["start"])
    out: list[tuple[float, float, list[str]]] = []
    cur: list[dict] = []
    for e in coarse:
        if cur and (e["end"] - cur[0]["start"] > max_sec):
            out.append((cur[0]["start"], cur[-1]["end"], [c["attributes"].get("Action sentence", "").strip() for c in cur]))
            cur = []
        cur.append(e)
    if cur:
        out.append((cur[0]["start"], cur[-1]["end"], [c["attributes"].get("Action sentence", "").strip() for c in cur]))
    return out


def write_session(name: str, session: dict, video: Path, max_sec: float, force: bool, remaining: int) -> tuple[int, int]:
    info = ffprobe(video)
    fine = sorted((e for e in session["events"] if e["label"] == "Fine grained action"), key=lambda e: (e["start"], e["end"]))
    ann_end = max((e["end"] for e in session["events"]), default=0.0)
    written = skipped = 0
    for i, (cs, ce, sentences) in enumerate(chunks_for(session, max_sec)):
        if written >= remaining:
            break
        ep_id = f"holoassist_{name}_{i:02d}"
        mp4, side = OUT / f"{ep_id}.mp4", OUT / f"{ep_id}.json"
        if mp4.exists() and side.exists() and not force:
            continue
        ce = min(ce, info["duration"])
        if ce - cs < 2:
            skipped += 1; continue
        segs, trimmed, dropped, prev_end = [], 0, 0, 0.0
        for e in fine:
            s, t = max(e["start"], cs), min(e["end"], ce)
            if t <= s:
                continue
            mid = (e["start"] + e["end"]) / 2
            if not (cs <= mid <= ce):
                continue
            txt = fine_text(e["attributes"])
            if not txt:
                dropped += 1; continue
            if s < prev_end:      # tiny overlaps between consecutive fine actions
                s = prev_end; trimmed += 1
                if t - s <= 0.05:
                    dropped += 1; continue
            segs.append({"start_sec": round(s - cs, 3), "end_sec": round(t - cs, 3), "subtask": txt})
            prev_end = t
        if not segs:
            print(f"  skip {ep_id}: no fine-grained actions in chunk"); skipped += 1; continue
        cut_h264(video, mp4, cs, ce)
        out_info = ffprobe(mp4)
        notes = (f"annotation times are seconds on the session video ({info['fps']:.3g} fps, {info['duration']:.1f}s; "
                 f"last annotated event ends {ann_end:.1f}s); episode = session[{cs:.3f}s, {ce:.3f}s] cut at coarse-action "
                 f"boundaries, segment times shifted by {cs:.3f}s. {trimmed} overlapping fine-action starts trimmed, "
                 f"{dropped} empty/degenerate dropped.")
        if abs(ann_end - info["duration"]) > 5:
            notes += f" NOTE: annotation span ({ann_end:.1f}s) vs video duration ({info['duration']:.1f}s) differ by {ann_end - info['duration']:+.1f}s."
        side.write_text(json.dumps({
            "id": ep_id, "instruction": " ".join(s for s in sentences if s) or session.get("taskType", ""),
            "segments": segs, "family": FAMILY,
            "metadata": {"source_dataset": "HoloAssist (holoassist.github.io) data-annotation-trainval-v1_1 + video_compress.tar",
                         "source_episode": f"{name}#chunk{i}", "task_type": session.get("taskType"),
                         "session_video": f"{name}/Export_py/Video_compress.mp4",
                         "fps": out_info["fps"], "duration_sec": round(out_info["duration"], 3), "license": LICENSE,
                         "annotation_provenance": "human",
                         "notes": notes + " Subtask text = fine-grained Verb+Adjective+Noun+adverbial; instruction = coarse 'Action sentence's of the chunk."}}, indent=1))
        written += 1
        print(f"  wrote {mp4} ({len(segs)} segments, {out_info['duration']:.1f}s)")
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30, help="max episodes to write")
    ap.add_argument("--sessions", type=int, default=10, help="max sessions to download")
    ap.add_argument("--max-sec", type=float, default=180.0)
    ap.add_argument("--tar-offset", type=int, default=TAR_ANCHOR)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    ann = RAW / "data-annotation-trainval-v1_1.json"
    if not ann.exists():
        print(f"downloading {ANNOT_URL} (111 MB)")
        with urllib.request.urlopen(ANNOT_URL, context=_ctx()) as r, open(ann, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    sessions = {s["video_name"]: s for s in json.loads(ann.read_text())}
    written = skipped = used = 0
    for name, size, data_off, _ in tar_entries(a.tar_offset):
        if written >= a.limit or used >= a.sessions:
            break
        m = re.fullmatch(r"([^/]+)/Export_py/Video_compress\.mp4", name)
        if not m:
            continue
        sid = m.group(1)
        if sid not in sessions:
            print(f"  skip {sid}: not in annotation file"); continue
        n_fine = sum(1 for e in sessions[sid]["events"] if e["label"] == "Fine grained action")
        n_coarse = sum(1 for e in sessions[sid]["events"] if e["label"] == "Coarse grained action")
        if n_fine < 3 or n_coarse < 1:
            print(f"  skip {sid}: {n_fine} fine / {n_coarse} coarse actions"); skipped += 1; continue
        video = RAW / f"{sid}.mp4"
        if not video.exists() or video.stat().st_size != size:
            print(f"downloading {sid} ({size / 1e6:.0f} MB) from tar offset {data_off}")
            fetch_range(data_off, size, video)
        used += 1
        w, s = write_session(sid, sessions[sid], video, a.max_sec, a.force, a.limit - written)
        written += w; skipped += s
        video.unlink(missing_ok=True)
    for v in RAW.glob("*.mp4"):
        v.unlink()
    print(f"done: wrote {written} episodes from {used} sessions, skipped {skipped} -> {OUT}")


if __name__ == "__main__":
    main()
