"""Stage 2: localized boundary refinement.

Coarse segmentation is the dominant error source, and it is cheapest to attack
where the uncertainty actually is: a short window either side of a proposed
boundary, sampled densely enough to see the transition, instead of paying a high
frame rate across the whole episode.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..video.contact_sheet import build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, LLMError, load_prompt

REFINE_INTERVAL = 0.25
REFINE_HALF_WINDOW = 1.5
REFINE_WIDTH = 320  # wider tiles than the coarse pass: the transition is subtle


class BoundaryChoice(BaseModel):
    boundary_seconds: float = Field(description="Chosen boundary, from the frame stamps.")
    reason: str = Field(default="", description="One short sentence.")


def refine_boundary(
    client: GeminiClient,
    video: str,
    candidate: float,
    before: str,
    after: str,
    duration: float,
    interval: float = REFINE_INTERVAL,
    half_window: float = REFINE_HALF_WINDOW,
) -> tuple[float, str]:
    """Re-place one boundary. Returns (time, reason); falls back to the candidate."""
    low = max(0.0, candidate - half_window)
    high = min(duration, candidate + half_window)
    if high - low < interval * 2:
        return candidate, "window too small to refine"

    frames = sample_frames(video, interval=interval, width=REFINE_WIDTH, start=low, end=high)
    if len(frames) < 3:
        return candidate, "too few frames to refine"
    sheets = build_sheets(frames, per_sheet=len(frames), columns=5)

    prompt = load_prompt("refine_boundary.md").format(
        interval=interval, before=before, after=after,
        candidate=candidate, low=frames[0].t, high=frames[-1].t,
    )
    try:
        choice = client.json("refine", prompt, BoundaryChoice, images=[s.image for s in sheets])
    except LLMError:
        return candidate, "refinement call failed"

    t = float(choice.boundary_seconds)
    if not (frames[0].t - 1e-6 <= t <= frames[-1].t + 1e-6):
        # Out-of-window answers are the common failure mode; keep the candidate.
        return candidate, "refined value fell outside the inspected window"
    # Snap to the sampling grid we actually showed the model.
    nearest = min(frames, key=lambda f: abs(f.t - t))
    return nearest.t, choice.reason
