"""Stage 1b: split coarse segments that contain more than one event.

Measured on WGO-Bench, recall is almost entirely a function of how long an event
lasts: 0.70 for events over 8s, 0.44 between 2 and 4s, and 0.00 under a second.
At 0.5s sampling a one-second event is two frames inside a sheet spanning ten,
so it is not that the model judges it wrongly -- it can barely see it.

This pass re-reads long segments at a finer interval, where those events are
actually visible, and spends inference only where an event can hide.
"""

from __future__ import annotations

from ..schemas import AnnotateRequest
from ..video.contact_sheet import build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, LLMError, load_prompt
from .segment import CoarseSegment, CoarseSegments

# Only segments longer than this can plausibly hide a second event.
SUBDIVIDE_MIN_DURATION = 3.0
SUBDIVIDE_INTERVAL = 0.25
SUBDIVIDE_WIDTH = 256
SUBDIVIDE_MAX_FRAMES = 24
# A split producing slivers is over-segmentation, not discovery.
MIN_PIECE = 0.4


def subdivide_segment(
    client: GeminiClient,
    video: str,
    segment: CoarseSegment,
    request: AnnotateRequest,
) -> list[CoarseSegment]:
    """Return the pieces of `segment`; the segment itself if it is one event."""
    span = segment.end_seconds - segment.start_seconds
    if span < SUBDIVIDE_MIN_DURATION:
        return [segment]

    interval = max(SUBDIVIDE_INTERVAL, span / SUBDIVIDE_MAX_FRAMES)
    frames = sample_frames(
        video, interval=interval, width=SUBDIVIDE_WIDTH,
        start=segment.start_seconds, end=segment.end_seconds,
    )
    if len(frames) < 4:
        return [segment]
    sheets = build_sheets(frames, per_sheet=SUBDIVIDE_MAX_FRAMES, columns=6)

    if request.schema_mode:
        vocab = load_prompt("vocabulary_closed.md").format(
            labels="\n".join(f"    - {s}" for s in request.subtasks)
        )
    else:
        vocab = load_prompt("vocabulary_open.md")
    instruction_block = (
        f"TASK BEING PERFORMED: {request.prompt}"
        if request.described
        else "No task description was supplied. Work out what is being done from the frames."
    )

    prompt = load_prompt("subdivide.md").format(
        instruction_block=instruction_block,
        start=segment.start_seconds, end=segment.end_seconds,
        label=segment.label, interval=interval, vocabulary=vocab,
    )
    try:
        result = client.json("subdivide", prompt, CoarseSegments,
                             images=[s.image for s in sheets])
    except LLMError:
        return [segment]

    pieces = [
        p for p in sorted(result.segments, key=lambda s: s.start_seconds)
        if p.end_seconds - p.start_seconds >= MIN_PIECE
    ]
    if len(pieces) < 2:
        return [segment]

    # Clamp to the original span and re-anchor the ends, so subdivision can never
    # move the boundaries the coarse pass established -- only add interior ones.
    lo, hi = segment.start_seconds, segment.end_seconds
    kept: list[CoarseSegment] = []
    for p in pieces:
        p.start_seconds = min(max(p.start_seconds, lo), hi)
        p.end_seconds = min(max(p.end_seconds, lo), hi)
        if p.end_seconds - p.start_seconds >= MIN_PIECE:
            kept.append(p)
    if len(kept) < 2:
        return [segment]
    kept[0].start_seconds = lo
    kept[-1].end_seconds = hi
    for a, b in zip(kept, kept[1:]):
        if b.start_seconds < a.end_seconds:
            b.start_seconds = a.end_seconds
    return [k for k in kept if k.end_seconds - k.start_seconds >= MIN_PIECE]
