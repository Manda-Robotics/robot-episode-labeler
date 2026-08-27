"""fal serverless app.

Billing is per second of video annotated, which is the unit the customer already
thinks in. The pipeline underneath can change completely without changing what a
40-second episode costs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import fal
from fastapi import Response
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class AnnotateInput(BaseModel):
    video_url: str = Field(
        description="URL of the robot episode video (mp4/mov/webm).",
        examples=["https://storage.example.com/episode_0001.mp4"],
    )
    prompt: str = Field(
        description="What the robot is doing.",
        examples=["A robot arm folds a cardboard box."],
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description="Optional closed vocabulary; labels are constrained to it.",
    )
    attributes: list[str] = Field(
        default_factory=list,
        description="Optional failure/attribute rubric, e.g. retry, missed_grasp.",
    )
    quality: str = Field(default="balanced", description="fast | balanced | strict")


class SegmentOut(BaseModel):
    start_seconds: float
    end_seconds: float
    label: str
    result: str
    attributes: list[str]
    description: str
    confidence: str
    flags: list[str]


class AnnotateOutput(BaseModel):
    task: str
    duration_seconds: float
    segments: list[SegmentOut]
    warnings: list[str]
    metadata: dict


class RobotEpisodeLabeler(fal.App, name="robot-episode-labeler", keep_alive=300):
    # CPU-only: the heavy lifting is ffmpeg plus a hosted multimodal model.
    machine_type = "S"
    requirements = [
        "google-genai>=1.0.0",
        "pillow>=10.4",
        "pydantic>=2.9",
        "imageio-ffmpeg>=0.5",
    ]

    def setup(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY must be set as a fal secret")

    @fal.endpoint("/")
    def annotate_episode(self, payload: AnnotateInput, response: Response) -> AnnotateOutput:
        from rel.pipeline import annotate
        from rel.schemas import AnnotateRequest, Quality

        result = annotate(AnnotateRequest(
            video=payload.video_url,
            prompt=payload.prompt,
            subtasks=payload.subtasks,
            attributes=payload.attributes,
            quality=Quality(payload.quality),
        ))

        # Charge per whole second of video annotated, minimum one.
        response.headers["x-fal-billable-units"] = str(max(1, round(result.duration_seconds)))

        return AnnotateOutput(
            task=result.task,
            duration_seconds=result.duration_seconds,
            segments=[SegmentOut(**s.model_dump(mode="json")) for s in result.segments],
            warnings=result.warnings,
            metadata=result.metadata,
        )
