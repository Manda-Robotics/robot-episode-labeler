import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rel.annotation.validate import clean
from rel.schemas import AnnotateRequest, Confidence, Segment


def req(**kw):
    return AnnotateRequest(video="v.mp4", prompt="do a thing", **kw)


def seg(a, b, label="x", **kw):
    return Segment(start_seconds=a, end_seconds=b, label=label, **kw)


def test_clamps_to_episode_bounds():
    out, warns = clean([seg(-3, 5), seg(5, 99)], duration=10.0, request=req())
    assert out[0].start_seconds == 0.0 and out[-1].end_seconds == 10.0
    assert any("clamped" in w for w in warns)


def test_drops_sub_threshold_segments():
    out, _ = clean([seg(0, 5), seg(5, 5.01), seg(5.01, 9)], duration=9.0, request=req())
    assert len(out) == 2


def test_trims_overlaps_and_keeps_order():
    out, warns = clean([seg(0, 6, "a"), seg(4, 9, "b")], duration=9.0, request=req())
    assert out[0].end_seconds == out[1].start_seconds == 4.0
    assert any("overlap" in w for w in warns)
    assert all(a.end_seconds <= b.start_seconds for a, b in zip(out, out[1:]))


def test_closed_vocabulary_is_actually_closed():
    r = req(subtasks=["Pick Box", "Fold Left"])
    out, warns = clean([seg(0, 5, "grabbing the box thing")], duration=5.0, request=r)
    assert out[0].label in r.subtasks
    assert out[0].confidence is Confidence.low
    assert "label_outside_vocabulary" in out[0].flags


def test_attributes_restricted_to_rubric():
    r = req(attributes=["retry"])
    out, warns = clean([seg(0, 5, "x", attributes=["retry", "invented"])], duration=5.0, request=r)
    assert out[0].attributes == ["retry"]
    assert any("dropped attributes" in w for w in warns)


def test_gaps_are_snapped_so_output_is_contiguous():
    out, _ = clean([seg(0, 3, "a"), seg(6, 9, "b")], duration=9.0, request=req())
    assert out[0].end_seconds == out[1].start_seconds


def test_empty_input_warns_rather_than_raises():
    out, warns = clean([], duration=9.0, request=req())
    assert out == [] and warns


def test_no_event_label_is_kept_not_snapped_in_schema_mode():
    from rel.schemas import NO_EVENT_LABEL, Result
    req = AnnotateRequest(video="x.mp4", prompt="p", subtasks=["Pick Up Block", "Place Block"])
    segs = [Segment(start_seconds=0, end_seconds=4, label="Pick Up Block"),
            Segment(start_seconds=4, end_seconds=9, label=NO_EVENT_LABEL)]
    out, warnings = clean(segs, 10.0, req)
    assert out[1].label == NO_EVENT_LABEL
    assert out[1].result is Result.failed
    assert "no_completed_subtask" in out[1].flags
    assert not any("snapped" in w for w in warnings)
