"""Download WGO-Bench and unpack it into data/wgo/ as plain mp4 + json."""

from __future__ import annotations

import sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rel.eval import wgo

URL = "https://huggingface.co/datasets/macrodata/WGO-Bench/resolve/main/data/annotations.parquet"


def main() -> None:
    wgo.DATA.mkdir(parents=True, exist_ok=True)
    if not wgo.PARQUET.exists():
        print(f"downloading {URL}\n  -> {wgo.PARQUET} (~1.3 GB)")
        urllib.request.urlretrieve(URL, wgo.PARQUET)
    n = wgo.materialise()
    eps = wgo.load()
    print(f"unpacked {n} episodes, {sum(len(e.segments) for e in eps)} gold segments "
          f"into {wgo.EPISODES}")


if __name__ == "__main__":
    main()
