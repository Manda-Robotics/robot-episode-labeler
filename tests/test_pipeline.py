"""Pipeline orchestration, exercised against a fake model.

These assert the parts we control -- stage wiring, invariant enforcement,
flagging -- without spending inference or depending on model behaviour.
"""

import subprocess, sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rel.annotation.llm import Usage
from rel.annotation.refine import BoundaryChoice
from rel.annotation.segment import CoarseSegment, CoarseSegments
from rel.annotation.label import SegmentLabel
from rel.pipeline import annotate
from rel.schemas import AnnotateRequest, Confidence, Quality, Result


class FakeClient:
    """Stands in for GeminiClient; records calls and returns scripted answers."""

    def __init__(self, segments, boundary=None, label=None, second_pass=None):
        self.model = "fake-model"
        self.usage = Usage()
        self._segments = segments
        self._second = second_pass
        self._boundary = boundary
        self._label = label
        self.calls = []
        self._segment_calls = 0

    def json(self, stage, text, schema, images=None):
        self.calls.append(stage)
        self.usage.add(stage, 10, 5)
        if stage == "segment":
            self._segment_calls += 1
            if self._segment_calls > 1 and self._second is not None:
                return CoarseSegments(segments=self._second)
            return CoarseSegments(segments=self._segments)
        if stage == "refine":
            return BoundaryChoice(boundary_seconds=self._boundary, reason="fake")
        if stage == "label":
            return self._label or SegmentLabel(label="labelled", result="pass")
        raise AssertionError(f"unexpected stage {stage}")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    out = d / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=s=160x120:r=10:d=10", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    return out


def req(clip, **kw):
    return AnnotateRequest(video=str(clip), prompt="a robot does a thing", **kw)


def test_fast_mode_runs_one_call_and_skips_refinement(clip):
    c = FakeClient([CoarseSegment(start_seconds=0, end_seconds=5, label="a"),
                    CoarseSegment(start_seconds=5, end_seconds=10, label="b")])
    r = annotate(req(clip, quality=Quality.fast), client=c)
    assert c.calls == ["segment"]
    assert len(r.segments) == 2
    assert r.metadata["quality"] == "fast"


def test_balanced_mode_refines_then_labels(clip):
    c = FakeClient([CoarseSegment(start_seconds=0, end_seconds=5, label="a"),
                    CoarseSegment(start_seconds=5, end_seconds=10, label="b")],
                   boundary=5.25, label=SegmentLabel(label="a", result="pass"))
    r = annotate(req(clip, quality=Quality.balanced), client=c)
    assert "refine" in c.calls and "label" in c.calls
    assert c.calls.count("refine") == 1        # one internal boundary
    assert c.calls.count("label") == 2        # one per segment
    assert r.segments[0].end_seconds == r.segments[1].start_seconds


def test_large_boundary_move_is_flagged_and_lowers_confidence(clip):
    c = FakeClient([CoarseSegment(start_seconds=0, end_seconds=5, label="a"),
                    CoarseSegment(start_seconds=5, end_seconds=10, label="b")],
                   boundary=3.5, label=SegmentLabel(label="a", result="pass"))
    r = annotate(req(clip, quality=Quality.balanced), client=c)
    assert any(f.startswith("boundary_moved_") for f in r.segments[0].flags)
    assert r.segments[0].confidence is Confidence.low
    assert any("moved" in w for w in r.warnings)


def test_refinement_never_inverts_a_segment(clip):
    # A refinement that would push the boundary before the segment's own start.
    c = FakeClient([CoarseSegment(start_seconds=4, end_seconds=5, label="a"),
                    CoarseSegment(start_seconds=5, end_seconds=10, label="b")],
                   boundary=0.0, label=SegmentLabel(label="a", result="pass"))
    r = annotate(req(clip, quality=Quality.balanced), client=c)
    for s in r.segments:
        assert s.end_seconds > s.start_seconds


def test_strict_mode_flags_unstable_segment_counts(clip):
    c = FakeClient([CoarseSegment(start_seconds=0, end_seconds=5, label="a"),
                    CoarseSegment(start_seconds=5, end_seconds=10, label="b")],
                   second_pass=[CoarseSegment(start_seconds=0, end_seconds=10, label="a")],
                   boundary=5.0, label=SegmentLabel(label="a", result="pass"))
    r = annotate(req(clip, quality=Quality.strict), client=c)
    assert c.calls.count("segment") == 2
    assert any("segment_count_unstable" in s.flags for s in r.segments)


def test_label_disagreement_is_flagged(clip):
    c = FakeClient([CoarseSegment(start_seconds=0, end_seconds=10, label="pick_cup")],
                   label=SegmentLabel(label="place_cup", result="fail", attributes=[]))
    r = annotate(req(clip, quality=Quality.balanced), client=c)
    assert "label_disagreement" in r.segments[0].flags
    assert r.segments[0].result is Result.failed


def test_out_of_bounds_model_output_is_clamped(clip):
    c = FakeClient([CoarseSegment(start_seconds=-5, end_seconds=999, label="a")])
    r = annotate(req(clip, quality=Quality.fast), client=c)
    assert r.segments[0].start_seconds == 0.0
    assert r.segments[0].end_seconds <= r.duration_seconds


def test_empty_segmentation_yields_warning_not_crash(clip):
    c = FakeClient([])
    r = annotate(req(clip, quality=Quality.fast), client=c)
    assert r.segments == [] and r.warnings


def test_metadata_records_provenance(clip):
    c = FakeClient([CoarseSegment(start_seconds=0, end_seconds=10, label="a")])
    r = annotate(req(clip, quality=Quality.fast), client=c)
    m = r.metadata
    assert m["model"] == "fake-model" and m["pipeline_version"]
    assert m["usage"]["calls"] == 1
