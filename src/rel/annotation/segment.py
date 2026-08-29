"""Stage 1: whole-episode coarse temporal segmentation.

Long episodes are segmented in overlapping windows rather than one enormous call.
A 160-second episode is 17 contact sheets; asking a model to hold all of them at
once produces a coarse summary of the episode instead of its individual events.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import AnnotateRequest
from ..video.contact_sheet import Sheet, build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, load_prompt

# One sheet is `per_sheet * interval` seconds (20 * 0.5s = 10s by default), so
# six sheets is a minute of video per call.
MAX_SHEETS_PER_CALL = 6
SHEET_OVERLAP = 1


class CoarseSegment(BaseModel):
    start_seconds: float = Field(description="Segment start, from the frame stamps.")
    end_seconds: float = Field(description="Segment end, from the frame stamps.")
    label: str = Field(description="Short label for the completed manipulation event.")


class CoarseSegments(BaseModel):
    segments: list[CoarseSegment]


def build_segment_prompt(
    request: AnnotateRequest,
    duration: float,
    interval: float,
    window: tuple[float, float] | None = None,
    previous_label: str | None = None,
    prompt_name: str = "segment_v2.md",
) -> str:
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

    if window is None:
        window_block = ""
    else:
        lo, hi = window
        lines = [
            f"You are shown only the window from {lo:.2f}s to {hi:.2f}s of this episode.",
            f"Annotate ONLY what happens inside it: every timestamp you return must lie "
            f"between {lo:.2f} and {hi:.2f}.",
            "Earlier and later parts of the episode are handled separately, so do not "
            "try to cover them.",
        ]
        if previous_label:
            lines.append(
                f"The subtask still in progress at the start of this window is "
                f"'{previous_label}'. If it finishes here, its segment ends at that moment; "
                f"if this window begins mid-subtask, start your first segment at {lo:.2f}."
            )
        window_block = "\n".join(lines) + "\n"

    return load_prompt(prompt_name).format(
        interval=interval,
        instruction_block=instruction_block,
        duration=duration,
        window_block=window_block,
        vocabulary=vocab,
    )


def _chunks(sheets: list[Sheet], max_sheets: int, overlap: int) -> list[list[Sheet]]:
    """Consecutive windows of sheets that overlap by `overlap` sheets."""
    if len(sheets) <= max_sheets:
        return [sheets]
    step = max(1, max_sheets - overlap)
    out: list[list[Sheet]] = []
    start = 0
    while start < len(sheets):
        out.append(sheets[start : start + max_sheets])
        if start + max_sheets >= len(sheets):
            break
        start += step
    return out


def _stitch(segments: list[CoarseSegment]) -> list[CoarseSegment]:
    """Merge duplicates produced by overlapping windows.

    Two windows both see the seconds they share, so the same event can be
    reported twice. Adjacent or overlapping segments carrying the same label are
    one event and are merged; the validator handles any remaining overlap.
    """
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s.start_seconds, s.end_seconds))
    merged = [ordered[0]]
    for seg in ordered[1:]:
        last = merged[-1]
        same_label = seg.label.strip().lower() == last.label.strip().lower()
        touching = seg.start_seconds <= last.end_seconds + 1e-6
        if same_label and touching:
            last.end_seconds = max(last.end_seconds, seg.end_seconds)
            continue
        merged.append(seg)
    return merged


def segment_episode(
    client: GeminiClient,
    video: str,
    request: AnnotateRequest,
    duration: float,
    interval: float = 0.5,
    width: int = 224,
    per_sheet: int = 20,
    columns: int = 5,
    max_sheets: int | None = MAX_SHEETS_PER_CALL,
    overlap: int = SHEET_OVERLAP,
    prompt_name: str = "segment_v2.md",
) -> list[CoarseSegment]:
    """Sample the episode into stamped contact sheets and segment it.

    Short episodes are one call. Longer ones are segmented window by window and
    stitched, which keeps the number of frames competing for the model's
    attention roughly constant regardless of episode length.
    """
    frames = sample_frames(video, interval=interval, width=width)
    if not frames:
        return []
    sheets = build_sheets(frames, per_sheet=per_sheet, columns=columns)
    windows = _chunks(sheets, max_sheets, overlap) if max_sheets else [sheets]

    collected: list[CoarseSegment] = []
    previous_label: str | None = None
    for window in windows:
        span = (window[0].start, window[-1].end) if len(windows) > 1 else None
        prompt = build_segment_prompt(request, duration, interval, span, previous_label,
                                      prompt_name=prompt_name)
        result = client.json("segment", prompt, CoarseSegments,
                             images=[s.image for s in window])
        got = list(result.segments)
        if span is not None:
            lo, hi = span
            # Windows sometimes answer outside their remit; keep only what is theirs.
            got = [g for g in got if g.end_seconds > lo - 1e-6 and g.start_seconds < hi + 1e-6]
        collected.extend(got)
        if got:
            previous_label = got[-1].label

    return _stitch(collected) if len(windows) > 1 else collected
