"""Replicate (Cog) entry point.

The build context is the repository root (see cog.yaml), so `src/rel` ships in the
image and is placed on the path here rather than being pip-installed.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

from cog import BasePredictor, Input, Path, Secret

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Fail at startup, not on a caller's first request.
        from rel.video.decode import _ffmpeg_binaries

        _ffmpeg_binaries()

    def run(
        self,
        video: Path = Input(description="Robot manipulation episode (mp4/mov/webm)."),
        prompt: str = Input(
            description="What the robot is doing. Optional: leave blank to annotate "
                        "without a task hint.",
            default="",
        ),
        subtasks: str = Input(
            description="Optional comma-separated subtask vocabulary. Supplied, labels "
                        "are constrained to it and snapped to it in code.",
            default="",
        ),
        attributes: str = Input(
            description="Optional comma-separated failure/attribute rubric, "
                        "e.g. retry,missed_grasp,dropped_object.",
            default="",
        ),
        quality: str = Input(
            description="fast = segmentation only. balanced = + subdivision of long "
                        "segments + labeling (recommended). strict = + boundary "
                        "refinement and disagreement flags.",
            choices=["fast", "balanced", "strict"],
            default="balanced",
        ),
        gemini_api_key: Secret = Input(
            description="Your Gemini API key, from https://aistudio.google.com/apikey. "
                        "Write-only: it is never stored or returned.",
            default=None,
        ),
    ) -> str:
        # Replicate has no model-level secret store, so the key is a write-only
        # input. It is scoped to this call and never written to the response.
        key = gemini_api_key.get_secret_value() if gemini_api_key else os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key. Pass gemini_api_key, or set GEMINI_API_KEY in "
                "the environment. Get one at https://aistudio.google.com/apikey"
            )
        os.environ["GEMINI_API_KEY"] = key

        from rel.pipeline import annotate
        from rel.schemas import AnnotateRequest, Quality

        def split(value: str) -> list[str]:
            return [v.strip() for v in value.split(",") if v.strip()]

        result = annotate(AnnotateRequest(
            video=str(video),
            prompt=prompt,
            subtasks=split(subtasks),
            attributes=split(attributes),
            quality=Quality(quality),
        ))
        return json.dumps(result.model_dump(mode="json"), indent=2)
