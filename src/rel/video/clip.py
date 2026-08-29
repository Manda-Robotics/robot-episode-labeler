"""Cut a window of an episode into a small, self-contained MP4.

Native video input to the model is only useful if we control what it sees: a
window we chose, small enough to send inline, with timestamps that start at
zero so the offset is added back in code rather than trusted to the model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .decode import _run_ffmpeg, probe


def clip_bytes(
    path: str | Path,
    start: float,
    end: float,
    width: int = 480,
    crf: int = 26,
) -> bytes:
    """Return MP4 bytes for [start, end) of `path`, scaled to `width` px wide.

    The clip is re-encoded, so the cut is frame-accurate and any input codec
    (including AV1 that the host cannot decode) becomes H.264.
    """
    info = probe(path)
    start = max(0.0, start)
    end = min(info.duration, end)
    if end - start <= 0:
        raise ValueError(f"empty clip window {start}-{end}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        cmd = ["-v", "error", "-nostdin", "-y"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(path), "-t", f"{end - start:.3f}",
                "-vf", f"scale={width}:-2:flags=bicubic",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        _run_ffmpeg(cmd, Path(path))
        return out.read_bytes()
