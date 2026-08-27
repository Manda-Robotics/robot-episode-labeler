"""Frame sampling. We own the sampling grid on purpose.

Handing a whole MP4 to a video model means accepting that model's default frame
rate, which is typically 1 fps. Manipulation boundaries -- a gripper closing, an
object leaving a surface -- routinely happen inside one such frame. Sampling
ourselves is what makes sub-second boundaries reachable at all.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class Frame:
    """A sampled frame and the episode time it represents, in seconds."""

    t: float
    image: Image.Image


@lru_cache(maxsize=1)
def _ffmpeg_binaries() -> tuple[str, ...]:
    """Decoders to try, in order.

    A host ffmpeg can be missing a software decoder for a codec callers really
    send: Homebrew's macOS build ships an AV1 decoder that is videotoolbox-only,
    so every AV1 episode fails on hardware that cannot do AV1. imageio-ffmpeg
    provides a static build with libaom as a fallback, which keeps decoding a
    property of this package rather than of the machine it runs on.
    """
    found: list[str] = []
    system = shutil.which("ffmpeg")
    if system:
        found.append(system)
    try:
        import imageio_ffmpeg

        static = imageio_ffmpeg.get_ffmpeg_exe()
        if static and static not in found:
            found.append(static)
    except Exception:  # noqa: BLE001 - the fallback is optional
        pass
    if not found:
        raise VideoError("no ffmpeg binary available; install ffmpeg or imageio-ffmpeg")
    return tuple(found)


def _require_ffmpeg() -> None:
    if shutil.which("ffprobe") is None:
        raise VideoError("ffprobe not found on PATH; it is required to read video")
    _ffmpeg_binaries()


def _run_ffmpeg(args: list[str], path: Path) -> None:
    """Run ffmpeg, falling back to another binary if the first cannot decode."""
    errors: list[str] = []
    for binary in _ffmpeg_binaries():
        proc = subprocess.run([binary, *args], capture_output=True, text=True)
        if proc.returncode == 0:
            return
        errors.append(f"{Path(binary).name}: {proc.stderr.strip()[:200]}")
    raise VideoError(f"ffmpeg failed on {path}: " + " | ".join(errors))


def probe(path: str | Path) -> VideoInfo:
    _require_ffmpeg()
    path = Path(path)
    if not path.exists():
        raise VideoError(f"no such video: {path}")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise VideoError(f"ffprobe failed on {path}: {out.stderr.strip()[:300]}")
    meta = json.loads(out.stdout)
    if not meta.get("streams"):
        raise VideoError(f"no video stream in {path}")
    stream = meta["streams"][0]
    num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    duration = float(meta.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise VideoError(f"could not determine duration of {path}")
    return VideoInfo(duration=duration, width=int(stream["width"]),
                     height=int(stream["height"]), fps=fps)


def sample_frames(
    path: str | Path,
    interval: float = 0.5,
    width: int = 224,
    start: float | None = None,
    end: float | None = None,
) -> list[Frame]:
    """Sample one frame every `interval` seconds, scaled to `width` px wide.

    `start`/`end` trim to a window, which is how boundary refinement zooms in
    without re-decoding the whole episode at a high frame rate.
    """
    _require_ffmpeg()
    if interval <= 0:
        raise ValueError("interval must be positive")
    info = probe(path)
    t0 = max(0.0, start if start is not None else 0.0)
    t1 = min(info.duration, end if end is not None else info.duration)
    if t1 <= t0:
        return []

    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["-v", "error", "-nostdin"]
        # -ss before -i seeks by keyframe and is far faster on long episodes.
        if t0 > 0:
            cmd += ["-ss", f"{t0:.3f}"]
        cmd += ["-i", str(path)]
        if t1 < info.duration:
            cmd += ["-t", f"{max(t1 - t0, interval):.3f}"]
        cmd += [
            "-vf", f"fps=1/{interval},scale={width}:-2:flags=bicubic",
            "-fps_mode", "passthrough",
            f"{tmp}/f_%05d.png",
        ]
        _run_ffmpeg(cmd, path)

        frames: list[Frame] = []
        for i, png in enumerate(sorted(Path(tmp).glob("f_*.png"))):
            # The fps filter emits frame i at input time t0 + i*interval.
            t = t0 + i * interval
            if t > t1 + 1e-6:
                break
            with Image.open(png) as im:
                frames.append(Frame(t=round(t, 3), image=im.convert("RGB").copy()))
    return frames
