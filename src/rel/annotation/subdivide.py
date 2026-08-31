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
from ..video.clip import clip_bytes
from ..video.contact_sheet import build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, LLMError, VideoClip, load_prompt
from .segment import CoarseSegment, CoarseSegments

# Only segments longer than this can plausibly hide a second event.
SUBDIVIDE_MIN_DURATION = 3.0
SUBDIVIDE_INTERVAL = 0.25
SUBDIVIDE_WIDTH = 256
FRAMES_PER_SHEET = 24
SHEET_COLUMNS = 6
# Total frames the pass may look at. Capping frames *per sheet* instead made the
# sampling interval grow with segment length, so a 36 s segment was re-read at
# 1.5 s -- three times coarser than the 0.5 s pass that produced it. A pass meant
# to find events that coarse sampling missed must never sample more coarsely than
# the pass that missed them.
MAX_TOTAL_FRAMES = 168
# A split producing slivers is over-segmentation, not discovery.
MIN_PIECE = 0.4


def subdivide_segment(
    client: GeminiClient,
    video: str,
    segment: CoarseSegment,
    request: AnnotateRequest,
    min_duration: float = SUBDIVIDE_MIN_DURATION,
    interval: float = SUBDIVIDE_INTERVAL,
    width: int = SUBDIVIDE_WIDTH,
    max_frames: int = MAX_TOTAL_FRAMES,
    prompt_name: str = "subdivide.md",
    input_mode: str = "sheets",
    fps: float = 4.0,
    video_width: int = 480,
) -> list[CoarseSegment]:
    """Return the pieces of `segment`; the segment itself if it is one event."""
    span = segment.end_seconds - segment.start_seconds
    if span < min_duration:
        return [segment]

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

    lo, hi = segment.start_seconds, segment.end_seconds
    try:
        if input_mode == "video":
            clip = VideoClip(data=clip_bytes(video, lo, hi, width=video_width), fps=fps)
            prompt = load_prompt("subdivide_video.md").format(
                instruction_block=instruction_block, fps=fps, span=span,
                start=lo, end=hi, label=segment.label, vocabulary=vocab,
            )
            result = client.json("subdivide", prompt, CoarseSegments, videos=[clip])
            # Clip-relative -> episode time.
            for p in result.segments:
                p.start_seconds += lo
                p.end_seconds += lo
        else:
            interval = max(interval, span / max_frames)
            frames = sample_frames(video, interval=interval, width=width, start=lo, end=hi)
            if len(frames) < 4:
                return [segment]
            sheets = build_sheets(frames, per_sheet=FRAMES_PER_SHEET, columns=SHEET_COLUMNS)
            prompt = load_prompt(prompt_name).format(
                instruction_block=instruction_block, start=lo, end=hi,
                label=segment.label, interval=interval, vocabulary=vocab,
            )
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
