"""Stage and publish the Hugging Face Space in deploy/hf_space/.

Dry-run by default: assembles the Space into a staging directory (app files,
a copy of `src/rel`, the two public example clips) and lists what would be
uploaded. `--push` uploads it to the Space repo with the token in HF_KEY,
creating the Space if it does not exist. The token is passed explicitly so a
cached login for another account can never be used by accident.
"""

from __future__ import annotations

import argparse, os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "deploy" / "hf_space"
EXAMPLES = ["droid_blue_block_in_green_bowl.mp4", "droid_duvet_tip_left.mp4"]
DEFAULT_REPO = "mandarobotics/robot-episode-labeler"


def stage(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    for f in SPACE_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            shutil.copy2(f, dest / f.name)
    shutil.copytree(ROOT / "src" / "rel", dest / "rel",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dest / "examples").mkdir(exist_ok=True)
    for name in EXAMPLES:
        shutil.copy2(ROOT / "examples" / name, dest / "examples" / name)
    shutil.copy2(ROOT / "examples" / "ATTRIBUTION.md", dest / "examples" / "ATTRIBUTION.md")
    return sorted(p for p in dest.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Space id, e.g. mandarobotics/robot-episode-labeler")
    ap.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--stage-dir", default=None, help="keep the staged files here instead of a temp dir")
    args = ap.parse_args()

    if not args.repo.startswith("mandarobotics/"):
        print(f"refusing: Space must live under mandarobotics/, got {args.repo}", file=sys.stderr)
        return 2

    tmp = Path(args.stage_dir) if args.stage_dir else Path(tempfile.mkdtemp(prefix="hf_space_"))
    files = stage(tmp)
    total = sum(f.stat().st_size for f in files)
    print(f"staged {len(files)} files, {total / 2**20:.1f} MB, in {tmp}")
    for f in files:
        print(f"  {f.relative_to(tmp)}  ({f.stat().st_size / 1024:.0f} KB)")

    if not args.push:
        print("\ndry run. Re-run with --push to upload to", args.repo)
        return 0

    token = os.environ.get("HF_KEY")
    if not token:
        print("HF_KEY is not set (see .env)", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    me = api.whoami()
    orgs = [o.get("name") for o in me.get("orgs", [])]
    print(f"authenticated as {me.get('name')} (orgs: {orgs})")
    if "mandarobotics" not in orgs:
        print("refusing: this token is not a member of mandarobotics", file=sys.stderr)
        return 2
    url = api.create_repo(args.repo, repo_type="space", space_sdk="gradio",
                          private=args.private, exist_ok=True)
    print("space:", url)
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=str(tmp),
                      commit_message="Publish robot-episode-labeler Space")
    print(f"pushed. https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
