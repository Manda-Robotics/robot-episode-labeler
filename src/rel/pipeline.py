"""Stage orchestration: one video in, validated annotations out.

`quality` selects a pipeline shape rather than exposing individual knobs. The
internals are free to change; the response contract is not.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .annotation.label import label_segment
from .annotation.llm import DEFAULT_MODEL, GeminiClient, prompts_fingerprint
from .annotation.refine import refine_boundary
from .annotation.segment import CoarseSegment, segment_episode
from .annotation.subdivide import subdivide_segment
from .annotation.validate import clean
from .schemas import (
    AnnotateRequest, AnnotateResponse, Confidence, Quality, Segment, base_metadata,
)
from .video.decode import VideoInfo, probe
from .video.source import resolve

COARSE_INTERVAL = 0.5
TILE_WIDTH = 224
FRAMES_PER_SHEET = 20
SHEET_COLUMNS = 5
MAX_PARALLEL = 4
# A refinement that moves a boundary further than this is reported, not trusted.
BOUNDARY_MOVE_FLAG = 0.75


def _to_segments(coarse: list[CoarseSegment]) -> list[Segment]:
    return [
        Segment(start_seconds=c.start_seconds, end_seconds=c.end_seconds, label=c.label)
        for c in coarse
    ]


def _refine_all(
    client: GeminiClient, request: AnnotateRequest, segments: list[Segment], info: VideoInfo
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
            duration=info.duration,
        )
        return i, t, reason

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
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
    client: GeminiClient, request: AnnotateRequest, segments: list[Segment], info: VideoInfo
) -> list[Segment]:
    def work(i: int) -> Segment:
        return label_segment(
            client, request.video, segments[i], request, info.duration,
            previous=segments[i - 1].label if i > 0 else None,
            following=segments[i + 1].label if i + 1 < len(segments) else None,
        )

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        return list(pool.map(work, range(len(segments))))


def annotate(
    request: AnnotateRequest,
    client: GeminiClient | None = None,
    model: str = DEFAULT_MODEL,
    subdivide: bool | None = None,
) -> AnnotateResponse:
    """Accepts a local path, an http(s) URL or a data URI as `request.video`.

    `subdivide` overrides the quality mode's default second-pass behaviour; it
    exists so the pass can be measured in isolation.
    """
    with resolve(request.video) as local:
        return _annotate_local(request.model_copy(update={"video": str(local)}),
                               client or GeminiClient(model=model), subdivide)


def _subdivide_all(
    client: GeminiClient, request: AnnotateRequest, coarse: list[CoarseSegment]
) -> list[CoarseSegment]:
    if not coarse:
        return coarse
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        pieces = list(pool.map(
            lambda seg: subdivide_segment(client, request.video, seg, request), coarse
        ))
    return [p for group in pieces for p in group]


def _annotate_local(
    request: AnnotateRequest, client: GeminiClient, subdivide: bool | None = None
) -> AnnotateResponse:
    started = time.time()
    info = probe(request.video)
    quality = request.quality
    if subdivide is None:
        subdivide = quality in (Quality.balanced, Quality.strict)

    coarse = segment_episode(
        client, request.video, request, duration=info.duration,
        interval=COARSE_INTERVAL, width=TILE_WIDTH,
        per_sheet=FRAMES_PER_SHEET, columns=SHEET_COLUMNS,
    )
    if subdivide:
        coarse = _subdivide_all(client, request, coarse)

    segments = _to_segments(coarse)
    notes: list[str] = []

    if quality is Quality.strict and segments:
        # A second independent segmentation pass; where the two disagree about
        # how many events happened, say so rather than presenting one as fact.
        second = _to_segments(segment_episode(
            client, request.video, request, duration=info.duration,
            interval=COARSE_INTERVAL, width=TILE_WIDTH,
            per_sheet=FRAMES_PER_SHEET, columns=SHEET_COLUMNS,
        ))
        if len(second) != len(segments):
            notes.append(
                f"repeat segmentation disagreed on segment count "
                f"({len(segments)} vs {len(second)}); treat boundaries as unstable"
            )
            for s in segments:
                s.flags = [*s.flags, "segment_count_unstable"]
                s.confidence = Confidence.low

    if quality in (Quality.balanced, Quality.strict):
        segments, warnings = clean(segments, info.duration, request)
        notes += _refine_all(client, request, segments, info)
        segments = _label_all(client, request, segments, info)
        segments, more = clean(segments, info.duration, request)
        warnings += more
    else:
        segments, warnings = clean(segments, info.duration, request)

    return AnnotateResponse(
        task=request.prompt,
        duration_seconds=round(info.duration, 2),
        segments=segments,
        warnings=warnings + notes,
        metadata=base_metadata(
            client.model, quality,
            sampling_interval=COARSE_INTERVAL,
            subdivided=subdivide,
            prompts=prompts_fingerprint(),
            elapsed_seconds=round(time.time() - started, 2),
            usage=client.usage.as_dict(),
        ),
    )
