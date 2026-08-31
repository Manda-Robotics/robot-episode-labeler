"""Stage orchestration: one video in, validated annotations out.

`quality` selects a preset `PipelineConfig`; callers who are measuring rather
than shipping pass an explicit config. The response contract does not change
with either.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .annotation.label import label_segment
from .annotation.llm import GeminiClient, prompts_fingerprint
from .annotation.refine import refine_boundary
from .annotation.segment import CoarseSegment, segment_episode
from .annotation.segment_state import segment_episode_state
from .annotation.segment_video import segment_episode_video
from .annotation.subdivide import subdivide_segment
from .annotation.validate import clean
from .config import DEFAULT_MODEL, PipelineConfig, config_for
from .schemas import (
    AnnotateRequest, AnnotateResponse, Confidence, Quality, Segment, base_metadata,
)
from .video.decode import VideoInfo, probe
from .video.source import resolve

# A refinement that moves a boundary further than this is reported, not trusted.
BOUNDARY_MOVE_FLAG = 0.75


def _to_segments(coarse: list[CoarseSegment]) -> list[Segment]:
    return [
        Segment(start_seconds=c.start_seconds, end_seconds=c.end_seconds, label=c.label)
        for c in coarse
    ]


def _refine_all(
    client: GeminiClient, request: AnnotateRequest, segments: list[Segment],
    info: VideoInfo, cfg: PipelineConfig,
) -> list[str]:
    """Re-place every internal boundary. Mutates `segments` in place."""
    if len(segments) < 2:
        return []
    jobs = [
        (i, (segments[i].end_seconds + segments[i + 1].start_seconds) / 2.0)
        for i in range(len(segments) - 1)
    ]

    def work(job: tuple[int, float]) -> tuple[int, float, str]:
        i, candidate = job
        t, reason = refine_boundary(
            client, request.video, candidate,
            before=segments[i].label, after=segments[i + 1].label,
            duration=info.duration, half_window=cfg.refine_half_window,
            input_mode=cfg.refine_input, fps=cfg.refine_fps, width=cfg.refine_width,
        )
        return i, t, reason

    with ThreadPoolExecutor(max_workers=cfg.max_parallel) as pool:
        results = sorted(pool.map(work, jobs), key=lambda r: r[0])

    notes: list[str] = []
    for i, t, _reason in results:
        prev, nxt = segments[i], segments[i + 1]
        # Never let a refinement invert a segment.
        t = min(max(t, prev.start_seconds + 0.05), nxt.end_seconds - 0.05)
        moved = abs(t - (prev.end_seconds + nxt.start_seconds) / 2.0)
        prev.end_seconds, nxt.start_seconds = round(t, 2), round(t, 2)
        if moved >= BOUNDARY_MOVE_FLAG:
            flag = f"boundary_moved_{moved:.2f}s"
            prev.flags = [*prev.flags, flag]
            nxt.flags = [*nxt.flags, flag]
            prev.confidence = nxt.confidence = Confidence.low
            notes.append(
                f"boundary between '{prev.label}' and '{nxt.label}' moved {moved:.2f}s in refinement"
            )
    return notes


def _label_all(
    client: GeminiClient, request: AnnotateRequest, segments: list[Segment],
    info: VideoInfo, cfg: PipelineConfig,
) -> list[Segment]:
    def work(i: int) -> Segment:
        return label_segment(
            client, request.video, segments[i], request, info.duration,
            previous=segments[i - 1].label if i > 0 else None,
            following=segments[i + 1].label if i + 1 < len(segments) else None,
            context=cfg.label_context, width=cfg.label_width,
            max_frames=cfg.label_max_frames, prompt_name=cfg.label_prompt,
        )

    with ThreadPoolExecutor(max_workers=cfg.max_parallel) as pool:
        return list(pool.map(work, range(len(segments))))


def _subdivide_all(
    client: GeminiClient, request: AnnotateRequest, coarse: list[CoarseSegment],
    cfg: PipelineConfig,
) -> list[CoarseSegment]:
    if not coarse:
        return coarse

    def work(seg: CoarseSegment) -> list[CoarseSegment]:
        return subdivide_segment(
            client, request.video, seg, request,
            min_duration=cfg.subdivide_min_duration, interval=cfg.subdivide_interval,
            width=cfg.subdivide_width, max_frames=cfg.subdivide_max_frames,
            prompt_name=cfg.subdivide_prompt, input_mode=cfg.subdivide_input,
            fps=cfg.subdivide_fps, video_width=cfg.video_width,
        )

    with ThreadPoolExecutor(max_workers=cfg.max_parallel) as pool:
        pieces = list(pool.map(work, coarse))
    return [p for group in pieces for p in group]


def _segment(
    client: GeminiClient, request: AnnotateRequest, info: VideoInfo, cfg: PipelineConfig,
) -> list[CoarseSegment]:
    if cfg.segment_input == "video":
        return segment_episode_video(
            client, request.video, request, duration=info.duration,
            fps=cfg.video_fps, window=cfg.video_window, overlap=cfg.video_overlap,
            width=cfg.video_width, prompt_name=cfg.segment_video_prompt,
        )
    if cfg.segment_input == "state":
        return segment_episode_state(
            client, request.video, request, duration=info.duration,
            interval=cfg.coarse_interval, width=cfg.tile_width,
            per_sheet=cfg.frames_per_sheet, columns=cfg.sheet_columns,
            max_sheets=cfg.max_sheets_per_call, overlap=cfg.sheet_overlap,
            prompt_name=cfg.segment_state_prompt, output=cfg.state_output,
        )
    return segment_episode(
        client, request.video, request, duration=info.duration,
        interval=cfg.coarse_interval, width=cfg.tile_width,
        per_sheet=cfg.frames_per_sheet, columns=cfg.sheet_columns,
        max_sheets=cfg.max_sheets_per_call, overlap=cfg.sheet_overlap,
        prompt_name=cfg.segment_prompt,
    )


def client_for(cfg: PipelineConfig, api_key: str | None = None) -> GeminiClient:
    """`api_key` overrides GEMINI_API_KEY; hosted front ends pass the caller's own
    key here rather than through the process environment, which is shared
    between concurrent requests."""
    return GeminiClient(
        model=cfg.model, api_key=api_key, temperature=cfg.temperature,
        thinking_level=cfg.thinking_level, thinking_budget=cfg.thinking_budget,
        media_resolution=cfg.media_resolution, media_processing=cfg.media_processing,
    )


def resolve_config(
    request: AnnotateRequest, config: PipelineConfig | None, model: str | None,
    subdivide: bool | None,
) -> PipelineConfig:
    cfg = config or config_for(request.quality)
    overrides = {}
    if model is not None and config is None:
        overrides["model"] = model
    if subdivide is not None:
        overrides["subdivide"] = subdivide
    return cfg.with_overrides(**overrides) if overrides else cfg


def annotate(
    request: AnnotateRequest,
    client: GeminiClient | None = None,
    model: str | None = None,
    subdivide: bool | None = None,
    config: PipelineConfig | None = None,
) -> AnnotateResponse:
    """Accepts a local path, an http(s) URL or a data URI as `request.video`.

    `config` overrides the quality preset entirely; `model` and `subdivide` are
    convenience overrides on top of the preset so a stage can be measured in
    isolation.
    """
    cfg = resolve_config(request, config, model, subdivide)
    if client is None:
        client = client_for(cfg)
    with resolve(request.video) as local:
        return _annotate_local(request.model_copy(update={"video": str(local)}), client, cfg)


def _annotate_local(
    request: AnnotateRequest, client: GeminiClient, cfg: PipelineConfig,
) -> AnnotateResponse:
    started = time.time()
    info = probe(request.video)
    quality = request.quality

    coarse = _segment(client, request, info, cfg)
    if cfg.subdivide:
        coarse = _subdivide_all(client, request, coarse, cfg)

    segments = _to_segments(coarse)
    notes: list[str] = []

    if cfg.repeat_pass and segments:
        # A second independent segmentation pass; where the two disagree about
        # how many events happened, say so rather than presenting one as fact.
        second = _to_segments(_segment(client, request, info, cfg))
        if len(second) != len(segments):
            notes.append(
                f"repeat segmentation disagreed on segment count "
                f"({len(segments)} vs {len(second)}); treat boundaries as unstable"
            )
            for s in segments:
                s.flags = [*s.flags, "segment_count_unstable"]
                s.confidence = Confidence.low

    segments, warnings = clean(segments, info.duration, request)
    if cfg.refine:
        # Measured on WGO-Bench: 31% of all calls for no significant movement in
        # any boundary metric, so only `strict` pays for it. See docs/results.md.
        notes += _refine_all(client, request, segments, info, cfg)
    if cfg.label:
        segments = _label_all(client, request, segments, info, cfg)
        segments, more = clean(segments, info.duration, request)
        warnings += more

    return AnnotateResponse(
        task=request.prompt,
        duration_seconds=round(info.duration, 2),
        segments=segments,
        warnings=warnings + notes,
        metadata=base_metadata(
            client.model, quality,
            sampling_interval=(1.0 / cfg.video_fps if cfg.segment_input == "video"
                               else cfg.coarse_interval),
            subdivided=cfg.subdivide,
            config=cfg.to_dict(),
            prompts=prompts_fingerprint(),
            elapsed_seconds=round(time.time() - started, 2),
            usage=client.usage.as_dict(),
        ),
    )
