import json, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rel.formats import render
from rel.formats.mcap_out import TOPIC_INSTRUCTION, TOPIC_SUBTASK, write_mcap
from rel.schemas import (
    AnnotateResponse, Confidence, Quality, Result, Segment, base_metadata,
)


def sample() -> AnnotateResponse:
    return AnnotateResponse(
        task="fold a cardboard box",
        duration_seconds=12.5,
        segments=[
            Segment(start_seconds=0.0, end_seconds=4.25, label="Pick Box",
                    result=Result.passed, description="Lifts the box."),
            Segment(start_seconds=4.25, end_seconds=12.5, label="Fold Left",
                    result=Result.failed, attributes=["retry"],
                    confidence=Confidence.low, flags=["boundary_moved_0.90s"]),
        ],
        metadata=base_metadata("gemini-3.5-flash", Quality.balanced),
    )


def test_json_round_trips():
    d = json.loads(render(sample(), "json"))
    assert d["segments"][1]["result"] == "fail"
    assert d["metadata"]["pipeline"] == "robot-episode-labeler"


def test_jsonl_is_one_segment_per_line():
    lines = render(sample(), "jsonl").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["label"] == "Pick Box"


def test_csv_has_header_and_flattens_lists():
    rows = render(sample(), "csv").splitlines()
    assert rows[0].startswith("start_seconds,end_seconds,label")
    assert "retry" in rows[2] and "boundary_moved_0.90s" in rows[2]


def test_unknown_format_rejected():
    try:
        render(sample(), "yaml")
    except ValueError as exc:
        assert "yaml" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_mcap_is_readable_and_time_based(tmp_path):
    from mcap.reader import make_reader

    origin = 1_700_000_000_000_000_000
    out = write_mcap(sample(), tmp_path / "annotation.mcap", time_origin_ns=origin)
    with out.open("rb") as fh:
        msgs = [(ch.topic, json.loads(msg.data), msg.log_time)
                for _, ch, msg in make_reader(fh).iter_messages()]

    topics = {t for t, _, _ in msgs}
    assert topics == {TOPIC_INSTRUCTION, TOPIC_SUBTASK}

    subtasks = [(m, t) for topic, m, t in msgs if topic == TOPIC_SUBTASK]
    assert len(subtasks) == 2
    # Segment times stay episode-relative; log times are absolute.
    assert subtasks[0][0]["start_time"] == 0.0 and subtasks[0][0]["end_time"] == 4.25
    assert subtasks[0][1] == origin + int(4.25 * 1e9)
    assert subtasks[1][0]["attributes"] == ["retry"]
