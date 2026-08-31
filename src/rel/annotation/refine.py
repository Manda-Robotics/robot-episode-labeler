"""Stage 2: localized boundary refinement.

Coarse segmentation is the dominant error source, and it is cheapest to attack
where the uncertainty actually is: a short window either side of a proposed
boundary, sampled densely enough to see the transition, instead of paying a high
frame rate across the whole episode.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..video.clip import clip_bytes
from ..video.contact_sheet import build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, LLMError, VideoClip, load_prompt

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
    input_mode: str = "sheets",
    fps: float = 8.0,
    width: int = 480,
) -> tuple[float, str]:
    """Re-place one boundary. Returns (time, reason); falls back to the candidate."""
    low = max(0.0, candidate - half_window)
    high = min(duration, candidate + half_window)
    if input_mode == "video":
        return _refine_video(client, video, candidate, before, after, low, high, fps, width)
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


def _refine_video(
    client: GeminiClient, video: str, candidate: float, before: str, after: str,
    low: float, high: float, fps: float, width: int,
) -> tuple[float, str]:
    """Native-video variant: a short clip at high frame rate around the candidate.

    The sheet variant quantises the answer to a 0.25 s grid and was measured as
    not moving any boundary metric. A clip at 8 fps gives the model per-frame
    timestamp tokens at 0.125 s, which is the resolution the ±0.25 s metric needs.
    """
    if high - low < 0.5:
        return candidate, "window too small to refine"
    try:
        clip = VideoClip(data=clip_bytes(video, low, high, width=width), fps=fps)
    except Exception as exc:  # noqa: BLE001 - refinement is best-effort
        return candidate, f"clip failed: {exc}"[:80]
    span = high - low
    prompt = load_prompt("refine_boundary_video.md").format(
        fps=fps, span=span, before=before, after=after, candidate=candidate - low,
    )
    try:
        choice = client.json("refine", prompt, BoundaryChoice, videos=[clip])
    except LLMError:
        return candidate, "refinement call failed"
    t = float(choice.boundary_seconds)
    if not (-1e-6 <= t <= span + 1e-6):
        return candidate, "refined value fell outside the inspected window"
    return round(low + t, 2), choice.reason
