"""Every knob of the pipeline in one place, so an experiment is a named, recorded
configuration rather than an edit to a module constant.

The three public quality modes are presets over this. Evaluation runs stamp the
full config into their results, so any number can be tied back to exactly what
produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from .schemas import Quality

DEFAULT_MODEL = "gemini-3.7-flash"


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "balanced"
    model: str = DEFAULT_MODEL

    # --- coarse segmentation -------------------------------------------------
    coarse_interval: float = 0.5        # seconds between sampled frames
    tile_width: int = 224               # px; the width each frame is scaled to
    frames_per_sheet: int = 20
    sheet_columns: int = 5
    max_sheets_per_call: int = 6        # 6 sheets * 20 frames * 0.5 s = one minute per call
    sheet_overlap: int = 1              # sheets shared between consecutive windows
    # segment_v2.md adds the pick/place decomposition rule: +0.097 F1 on
    # WGO-Bench, every boundary tolerance significant (docs/research-log.md).
    # segment.md is kept to replicate the recorded first-pass runs.
    segment_prompt: str = "segment_v2.md"
    # Alternative input for the coarse pass: native video instead of contact sheets.
    segment_input: str = "sheets"       # "sheets" | "video" | "state"
    video_fps: float = 2.0              # frames per second the model samples the clip at
    video_window: float = 60.0          # seconds of video per call; 0 = whole episode
    video_overlap: float = 5.0          # seconds shared between consecutive windows
    video_width: int = 480              # px; clip is re-encoded at this width
    segment_video_prompt: str = "segment_video_v2.md"
    segment_state_prompt: str = "segment_state.md"
    state_output: str = "rows"          # "rows" (one per frame) | "runs" (one per change)

    # --- subdivision of long segments ---------------------------------------
    subdivide: bool = True
    subdivide_min_duration: float = 3.0
    subdivide_interval: float = 0.25
    subdivide_width: int = 256
    subdivide_max_frames: int = 168
    subdivide_prompt: str = "subdivide.md"
    subdivide_input: str = "sheets"     # "sheets" (0.25 s grid) | "video" (clip at subdivide_fps)
    subdivide_fps: float = 4.0

    # --- per-segment labeling -----------------------------------------------
    label: bool = True
    # label_v2.md names the event that completes at the segment's END instead of
    # the episode goal: 0.750 -> 0.796 on 480 identical matched segments,
    # +0.046 [+0.020, +0.077]; DROID 0.500 -> 0.594. See docs/research-log.md.
    label_prompt: str = "label_v2.md"
    label_context: float = 1.0          # seconds of video shown either side of the segment
    label_width: int = 256
    label_max_frames: int = 20

    # --- strict-mode extras -------------------------------------------------
    refine: bool = False
    refine_input: str = "sheets"        # "sheets" (0.25 s grid) | "video" (clip at refine_fps)
    refine_fps: float = 8.0
    refine_half_window: float = 1.5     # seconds either side of the candidate boundary
    refine_width: int = 480
    repeat_pass: bool = False

    # --- model call options -------------------------------------------------
    temperature: float | None = 0.0     # None = API default (recommended for 3.x)
    thinking_level: str | None = None   # gemini-3.x: "low" | "medium" | "high"; None = API default
    thinking_budget: int | None = None  # older models; None = API default
    media_resolution: str | None = None # "low" | "medium" | "high"; None = API default
    media_processing: str | None = None # video: "static" | "agentic"; None = API default
    max_parallel: int = 4

    def with_overrides(self, **overrides: Any) -> "PipelineConfig":
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[Quality, PipelineConfig] = {
    Quality.fast: PipelineConfig(name="fast", subdivide=False, label=False),
    Quality.balanced: PipelineConfig(name="balanced"),
    Quality.strict: PipelineConfig(name="strict", refine=True, repeat_pass=True),
}


def config_for(quality: Quality, **overrides: Any) -> PipelineConfig:
    return PRESETS[quality].with_overrides(**overrides)


def parse_overrides(spec: str | None) -> dict[str, Any]:
    """Parse `key=value,key=value` from a command line into typed overrides."""
    if not spec:
        return {}
    types = {f.name: f.type for f in fields(PipelineConfig)}
    out: dict[str, Any] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in types:
            raise ValueError(f"unknown config field '{key}'")
        t = str(types[key])
        if raw.lower() in ("none", "null"):
            out[key] = None
        elif t.startswith("bool"):
            out[key] = raw.lower() in ("1", "true", "yes", "on")
        elif t.startswith("int"):
            out[key] = int(raw)
        elif t.startswith("float"):
            out[key] = float(raw)
        else:
            out[key] = raw
    return out
