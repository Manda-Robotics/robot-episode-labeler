"""Stage 3: per-segment labeling, success judgement and attributes.

Labeling a segment in isolation loses the thread of the episode, so each call
sees the segment plus its neighbours by name and a little video either side.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import AnnotateRequest, Confidence, Result, Segment
from ..video.contact_sheet import build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, LLMError, load_prompt

LABEL_CONTEXT = 1.0
LABEL_MAX_FRAMES = 20
LABEL_WIDTH = 256


class SegmentLabel(BaseModel):
    label: str
    result: str = Field(default="unknown", description="pass | fail | unknown")
    attributes: list[str] = Field(default_factory=list)
    description: str = ""


def label_segment(
    client: GeminiClient,
    video: str,
    segment: Segment,
    request: AnnotateRequest,
    duration: float,
    previous: str | None = None,
    following: str | None = None,
) -> Segment:
    """Return a copy of `segment` with label, result, attributes and description."""
    lo = max(0.0, segment.start_seconds - LABEL_CONTEXT)
    hi = min(duration, segment.end_seconds + LABEL_CONTEXT)
    span = max(hi - lo, 0.5)
    interval = max(span / LABEL_MAX_FRAMES, 0.1)

    frames = sample_frames(video, interval=interval, width=LABEL_WIDTH, start=lo, end=hi)
    if not frames:
        return segment
    sheets = build_sheets(frames, per_sheet=LABEL_MAX_FRAMES, columns=5)

    neighbours = ""
    if previous or following:
        parts = []
        if previous:
            parts.append(f"The preceding subtask was: {previous}")
        if following:
            parts.append(f"The following subtask is: {following}")
        neighbours = "\n".join(parts) + "\n"

    if request.schema_mode:
        vocab = (" You MUST use exactly one of these labels verbatim:\n"
                 + "\n".join(f"    - {s}" for s in request.subtasks))
    else:
        vocab = " Use a short verb-object phrase in lower snake_case."
    attrs = (" Allowed attributes: " + ", ".join(request.attributes)
             if request.attributes else " No rubric was supplied; return an empty list.")

    prompt = load_prompt("label.md").format(
        instruction=request.prompt, start=segment.start_seconds, end=segment.end_seconds,
        neighbours=neighbours, vocabulary=vocab, attributes=attrs,
    )
    try:
        out = client.json("label", prompt, SegmentLabel, images=[s.image for s in sheets])
    except LLMError:
        return segment

    updated = segment.model_copy(deep=True)

    updated.label = out.label or segment.label
    try:
        updated.result = Result(out.result.strip().lower())
    except ValueError:
        updated.result = Result.unknown
    updated.attributes = list(out.attributes or [])
    updated.description = (out.description or "").strip()
    # Disagreement between the segmentation pass and this one is a real signal --
    # but only when the two passes named a genuinely different event. In discovery
    # mode the wording varies freely ("pick_up_banana" vs "pick_banana"), so
    # compare meaning-bearing tokens rather than strings, otherwise almost every
    # segment gets flagged and the signal becomes noise.
    if segment.label and out.label and _disagrees(segment.label, out.label, request.schema_mode):
        updated.flags = [*updated.flags, "label_disagreement"]
        updated.confidence = Confidence.low
    return updated


_STOPWORDS = {"the", "a", "an", "to", "on", "in", "of", "up", "down", "and", "with", "at", "into"}


def _tokens(label: str) -> set[str]:
    raw = label.lower().replace("_", " ").replace("-", " ").split()
    return {t for t in raw if t not in _STOPWORDS} or set(raw)


def _disagrees(first: str, second: str, schema_mode: bool) -> bool:
    if schema_mode:
        # A closed vocabulary makes exact comparison meaningful.
        return first.strip().lower() != second.strip().lower()
    a, b = _tokens(first), _tokens(second)
    if not a or not b:
        return first.strip().lower() != second.strip().lower()
    overlap = len(a & b) / len(a | b)
    return overlap < 0.34
