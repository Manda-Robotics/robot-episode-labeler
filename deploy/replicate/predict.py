"""Replicate (Cog) entry point."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from cog import BasePredictor, Input, Path as CogPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY must be set as a Replicate secret")

    def predict(
        self,
        video: CogPath = Input(description="Robot manipulation episode (mp4/mov/webm)."),
        prompt: str = Input(description="What the robot is doing.",
                            default="A robot arm manipulates objects on a table."),
        subtasks: str = Input(
            description="Optional comma-separated subtask vocabulary. Supplied, labels "
                        "are constrained to it.",
            default="",
        ),
        attributes: str = Input(
            description="Optional comma-separated failure/attribute rubric, "
                        "e.g. retry,missed_grasp,dropped_object.",
            default="",
        ),
        quality: str = Input(description="How much inference to spend.",
                             choices=["fast", "balanced", "strict"], default="balanced"),
    ) -> str:
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
