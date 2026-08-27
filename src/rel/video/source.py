"""Resolve whatever the caller passed into a local file.

Callers send a path, an http(s) URL, or a data URI. Downloads are capped and
streamed, because an endpoint that will happily pull an unbounded file is a
denial-of-service surface, not a feature.
"""

from __future__ import annotations

import base64
import shutil
import tempfile
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
CHUNK = 1 << 20
TIMEOUT_S = 120


class SourceError(RuntimeError):
    pass


def is_remote(video: str) -> bool:
    return urllib.parse.urlparse(video).scheme in ("http", "https", "data")


@contextmanager
def resolve(video: str, max_bytes: int = MAX_BYTES):
    """Yield a local Path for `video`, cleaning up anything we downloaded."""
    if not is_remote(video):
        path = Path(video)
        if not path.exists():
            raise SourceError(f"no such video: {video}")
        yield path
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="rel-src-"))
    try:
        target = tmpdir / "episode"
        if video.startswith("data:"):
            _, _, payload = video.partition(",")
            raw = base64.b64decode(payload)
            if len(raw) > max_bytes:
                raise SourceError(f"video exceeds {max_bytes} bytes")
            target.write_bytes(raw)
        else:
            req = urllib.request.Request(video, headers={"User-Agent": "robot-episode-labeler"})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                    declared = int(resp.headers.get("Content-Length") or 0)
                    if declared > max_bytes:
                        raise SourceError(f"video is {declared} bytes, over the {max_bytes} limit")
                    written = 0
                    with target.open("wb") as fh:
                        while chunk := resp.read(CHUNK):
                            written += len(chunk)
                            if written > max_bytes:
                                raise SourceError(f"video exceeds {max_bytes} bytes")
                            fh.write(chunk)
            except SourceError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise SourceError(f"could not fetch {video}: {exc}") from exc
        yield target
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
