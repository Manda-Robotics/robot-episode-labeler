"""Stage 1: whole-episode coarse temporal segmentation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import AnnotateRequest
from ..video.contact_sheet import build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, load_prompt


class CoarseSegment(BaseModel):
    start_seconds: float = Field(description="Segment start, from the frame stamps.")
    end_seconds: float = Field(description="Segment end, from the frame stamps.")
    label: str = Field(description="Short label for the completed manipulation event.")


class CoarseSegments(BaseModel):
    segments: list[CoarseSegment]


def build_segment_prompt(request: AnnotateRequest, duration: float, interval: float) -> str:
    if request.schema_mode:
        vocab = load_prompt("vocabulary_closed.md").format(
            labels="\n".join(f"    - {s}" for s in request.subtasks)
        )
    else:
        vocab = load_prompt("vocabulary_open.md")
    return load_prompt("segment.md").format(
        interval=interval,
        instruction=request.prompt,
        duration=duration,
        vocabulary=vocab,
    )


def segment_episode(
    client: GeminiClient,
    video: str,
    request: AnnotateRequest,
    duration: float,
    interval: float = 0.5,
    width: int = 224,
    per_sheet: int = 20,
    columns: int = 5,
) -> list[CoarseSegment]:
    """Sample the whole episode into stamped contact sheets and segment it in one call."""
    frames = sample_frames(video, interval=interval, width=width)
    if not frames:
        return []
    sheets = build_sheets(frames, per_sheet=per_sheet, columns=columns)
    prompt = build_segment_prompt(request, duration, interval)
    result = client.json(
        "segment", prompt, CoarseSegments, images=[s.image for s in sheets]
    )
    return list(result.segments)
