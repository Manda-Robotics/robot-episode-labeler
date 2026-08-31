"""Stage 1, native-video variant: the model watches a clip instead of a sheet.

Contact sheets quantise boundaries to the sampling grid (84% of predicted
boundaries in the sheet pipeline land exactly on the 0.5 s grid). A video part
with a controlled frame rate gives the model per-frame timestamp tokens instead,
so a boundary can be reported where it happened. Windows are cut with ffmpeg so
the model only ever sees a clip whose timeline starts at zero; the offset is
added back here, not trusted to the model.
"""

from __future__ import annotations

from ..schemas import AnnotateRequest
from ..video.clip import clip_bytes
from .llm import GeminiClient, VideoClip, load_prompt
from .segment import CoarseSegment, CoarseSegments, _stitch


def _windows(duration: float, window: float, overlap: float) -> list[tuple[float, float]]:
    if window <= 0 or duration <= window:
        return [(0.0, duration)]
    step = max(window - overlap, 1.0)
    out: list[tuple[float, float]] = []
    lo = 0.0
    while True:
        hi = min(duration, lo + window)
        out.append((lo, hi))
        if hi >= duration - 1e-6:
            break
        lo += step
    return out


def build_video_prompt(
    request: AnnotateRequest,
    fps: float,
    lo: float,
    hi: float,
    windowed: bool,
    previous_label: str | None,
    prompt_name: str = "segment_video_v2.md",
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
    window_block = ""
    if windowed:
        lines = [
            "This clip is one window of a longer episode; earlier and later parts are "
            "handled separately, so annotate only what happens inside it.",
        ]
        if previous_label:
            lines.append(
                f"The subtask still in progress when this clip starts is '{previous_label}'. "
                "If it finishes here, its segment starts at 0.00 and ends at that moment."
            )
        window_block = "\n".join(lines) + "\n"
    return load_prompt(prompt_name).format(
        fps=fps, lo=lo, hi=hi, span=hi - lo, instruction_block=instruction_block,
        window_block=window_block, vocabulary=vocab,
    )


def segment_episode_video(
    client: GeminiClient,
    video: str,
    request: AnnotateRequest,
    duration: float,
    fps: float = 2.0,
    window: float = 60.0,
    overlap: float = 5.0,
    width: int = 480,
    prompt_name: str = "segment_video_v2.md",
) -> list[CoarseSegment]:
    windows = _windows(duration, window, overlap)
    collected: list[CoarseSegment] = []
    previous_label: str | None = None
    for lo, hi in windows:
        clip = VideoClip(data=clip_bytes(video, lo, hi, width=width), fps=fps)
        prompt = build_video_prompt(request, fps, lo, hi, len(windows) > 1, previous_label,
                                    prompt_name)
        result = client.json("segment", prompt, CoarseSegments, videos=[clip])
        got: list[CoarseSegment] = []
        for g in result.segments:
            # Clip-relative -> episode time. Clamp to the window: the model has
            # no way to know anything outside it.
            s = max(0.0, min(g.start_seconds, hi - lo)) + lo
            e = max(0.0, min(g.end_seconds, hi - lo)) + lo
            if e > s:
                got.append(CoarseSegment(start_seconds=round(s, 2), end_seconds=round(e, 2),
                                         label=g.label))
        collected.extend(got)
        if got:
            previous_label = got[-1].label
    return _stitch(collected) if len(windows) > 1 else collected
