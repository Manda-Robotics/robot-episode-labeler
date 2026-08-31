"""Stage 1, dense-state variant: classify every frame, derive boundaries in code.

Asking a model for timestamps makes it interpolate between what it saw. Asking
it what is happening *in each stamped frame* is a classification the model is
much better at (REZE reports 51.5 vs 6.5 mIoU for the same model on Charades
when grounding is reformulated this way). Boundaries then fall out of the label
sequence deterministically, at the sampling grid's resolution.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import AnnotateRequest
from ..video.contact_sheet import Sheet, build_sheets
from ..video.decode import sample_frames
from .llm import GeminiClient, load_prompt
from .segment import CoarseSegment, _chunks

NONE_LABELS = {"", "none", "null", "idle", "no subtask", "n/a", "-"}
# A run of fewer frames than this is noise in the label sequence, not an event.
MIN_RUN = 2


class FrameState(BaseModel):
    t: float = Field(description="Timestamp stamped on the frame, copied exactly.")
    held: str = Field(default="none", description="What the gripper/hand holds, or 'none'.")
    subtask: str = Field(description="Subtask in progress in this frame, or 'none'.")


class FrameStates(BaseModel):
    rows: list[FrameState]


class Run(BaseModel):
    t: float = Field(description="Stamp of the first frame of this run, copied exactly.")
    subtask: str = Field(description="Subtask in progress from this frame on, or 'none'.")


class Runs(BaseModel):
    runs: list[Run]


def runs_to_rows(runs: list[Run], times: list[float], interval: float) -> list[FrameState]:
    """Expand run starts into one row per stamped frame."""
    starts = sorted(((r.t, r.subtask) for r in runs), key=lambda x: x[0])
    rows: list[FrameState] = []
    label = "none"
    k = 0
    for t in times:
        while k < len(starts) and starts[k][0] <= t + interval * 0.49:
            label = starts[k][1]
            k += 1
        rows.append(FrameState(t=t, subtask=label))
    return rows


def _is_none(label: str) -> bool:
    return label.strip().lower() in NONE_LABELS


def _snap_rows(rows: list[FrameState], times: list[float], interval: float) -> dict[float, FrameState]:
    """Assign each returned row to the nearest stamped time; last write wins."""
    out: dict[float, FrameState] = {}
    for r in rows:
        nearest = min(times, key=lambda t: abs(t - r.t))
        if abs(nearest - r.t) <= interval * 0.51:
            out[nearest] = r
    return out


def _smooth(labels: list[str]) -> list[str]:
    """Absorb runs shorter than MIN_RUN into the neighbouring run."""
    if len(labels) < 3:
        return labels
    out = list(labels)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            j = i
            while j < len(out) and out[j] == out[i]:
                j += 1
            if j - i < MIN_RUN and (i > 0 or j < len(out)):
                fill = out[i - 1] if i > 0 else out[j]
                for k in range(i, j):
                    out[k] = fill
                changed = True
            i = j
    return out


def rows_to_segments(rows: dict[float, FrameState], times: list[float], interval: float) -> list[CoarseSegment]:
    """Turn a per-frame label sequence into contiguous segments.

    A boundary sits halfway between the last frame of one run and the first frame
    of the next, which is the unbiased estimate given that the change happened
    somewhere in that gap. Runs of "none" become gaps, not segments.
    """
    labels = [rows[t].subtask.strip() if t in rows else "none" for t in times]
    labels = ["none" if _is_none(l) else l for l in labels]
    labels = _smooth(labels)
    segments: list[CoarseSegment] = []
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j].lower() == labels[i].lower():
            j += 1
        if not _is_none(labels[i]):
            start = times[i] if i == 0 else (times[i - 1] + times[i]) / 2.0
            end = times[j - 1] + interval / 2.0 if j < len(times) else times[j - 1] + interval / 2.0
            if segments and start < segments[-1].end_seconds:
                start = segments[-1].end_seconds
            segments.append(CoarseSegment(start_seconds=round(start, 2), end_seconds=round(end, 2),
                                          label=labels[i]))
        i = j
    if segments:
        # Gold convention (every family in WGO-Bench): the first subtask starts at
        # 0.00 -- the initial reach belongs to the first pick.
        segments[0].start_seconds = 0.0
    return segments


def _prompt(request: AnnotateRequest, duration: float, interval: float,
            window: tuple[float, float] | None, prompt_name: str) -> str:
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
    window_block = ""
    if window is not None:
        lo, hi = window
        window_block = (f"You are shown only the frames from {lo:.2f}s to {hi:.2f}s of this "
                        f"episode; earlier and later parts are handled separately.\n")
    return load_prompt(prompt_name).format(
        interval=interval, instruction_block=instruction_block, duration=duration,
        window_block=window_block, vocabulary=vocab,
    )


def segment_episode_state(
    client: GeminiClient,
    video: str,
    request: AnnotateRequest,
    duration: float,
    interval: float = 0.5,
    width: int = 224,
    per_sheet: int = 20,
    columns: int = 5,
    max_sheets: int | None = 6,
    overlap: int = 1,
    prompt_name: str = "segment_state.md",
    output: str = "rows",
) -> list[CoarseSegment]:
    """`output` is "rows" (one row per frame) or "runs" (one row per change; ~5x
    fewer output tokens, at the risk of the model inventing off-grid stamps)."""
    frames = sample_frames(video, interval=interval, width=width)
    if not frames:
        return []
    sheets = build_sheets(frames, per_sheet=per_sheet, columns=columns)
    windows = _chunks(sheets, max_sheets, overlap) if max_sheets else [sheets]

    rows: dict[float, FrameState] = {}
    for window in windows:
        times = [t for s in window for t in s.times]
        span = (window[0].start, window[-1].end) if len(windows) > 1 else None
        prompt = _prompt(request, duration, interval, span, prompt_name)
        if output == "runs":
            result = client.json("segment", prompt, Runs, images=[s.image for s in window])
            returned = runs_to_rows(list(result.runs), times, interval)
        else:
            result = client.json("segment", prompt, FrameStates, images=[s.image for s in window])
            returned = list(result.rows)
        got = _snap_rows(returned, times, interval)
        # Later windows overwrite the overlap: they saw more of what follows.
        rows.update(got)
    all_times = [f.t for f in frames]
    return rows_to_segments(rows, all_times, interval)
