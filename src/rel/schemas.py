"""Public API contract for the robot episode labeler.

Two things matter here: the input stays small enough that a first-time caller can
guess it, and the output stays stable while the pipeline underneath changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

PIPELINE_NAME = "robot-episode-labeler"

# Reserved label for a stretch in which no subtask completes: the robot fumbles,
# retries without success, or is idle. Without a way to say this, a closed
# vocabulary forces every stretch into a task-shaped label, which fabricates
# events on exactly the failure-heavy episodes an eval customer cares about.
NO_EVENT_LABEL = "no_completed_subtask"
PIPELINE_VERSION = "0.1.0"


class Quality(str, Enum):
    """How much inference to spend. Maps to a pipeline shape, not a knob count."""

    fast = "fast"          # coarse segmentation only
    balanced = "balanced"  # + subdivision of long segments + context labeling
    strict = "strict"      # + boundary refinement, repeat pass, disagreement flags


class Result(str, Enum):
    passed = "pass"
    failed = "fail"
    unknown = "unknown"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class AnnotateRequest(BaseModel):
    """Only `video` and `prompt` are required; everything else sharpens the output."""

    video: str = Field(description="Local path or URL to the episode video.")
    prompt: str = Field(
        default="",
        description="What the robot is doing. Optional: left empty, the episode is "
                    "annotated without a task hint.",
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description="Optional closed vocabulary. Supplied -> labels are constrained to it.",
    )
    attributes: list[str] = Field(
        default_factory=list,
        description="Optional failure/attribute rubric, e.g. retry, missed_grasp, dropped_object.",
    )
    quality: Quality = Quality.balanced

    @field_validator("prompt")
    @classmethod
    def _tidy_prompt(cls, v: str) -> str:
        return v.strip()

    @property
    def described(self) -> bool:
        """False when the caller gave no task hint at all."""
        return bool(self.prompt)

    @field_validator("subtasks", "attributes")
    @classmethod
    def _clean_vocab(cls, v: list[str]) -> list[str]:
        # Dedupe while preserving caller order; their order is a hint about task order.
        seen, out = set(), []
        for item in v:
            s = item.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    @property
    def schema_mode(self) -> bool:
        """True when the caller supplied a vocabulary and labels must be constrained."""
        return bool(self.subtasks)


class Segment(BaseModel):
    start_seconds: float
    end_seconds: float
    label: str
    result: Result = Result.unknown
    attributes: list[str] = Field(default_factory=list)
    description: str = ""
    confidence: Confidence = Confidence.medium
    flags: list[str] = Field(
        default_factory=list,
        description="Why confidence dropped, e.g. boundary_moved_1.25s, label_disagreement.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> "Segment":
        if self.end_seconds < self.start_seconds:
            raise ValueError(f"segment ends before it starts: {self}")
        return self

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


class AnnotateResponse(BaseModel):
    task: str
    duration_seconds: float
    segments: list[Segment]
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def base_metadata(model: str, quality: Quality, **extra: Any) -> dict[str, Any]:
    """Provenance stamped on every response so results stay reproducible across releases."""
    return {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "model": model,
        "quality": quality.value,
        **extra,
    }
