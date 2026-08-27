"""Windowed segmentation: chunking, stitching, and window discipline."""

import sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rel.annotation.segment import (
    CoarseSegment, _chunks, _stitch, build_segment_prompt,
)
from rel.schemas import AnnotateRequest
from rel.video.contact_sheet import Sheet


def sheet(index: int, per_sheet: int = 20, interval: float = 0.5) -> Sheet:
    base = index * per_sheet * interval
    return Sheet(image=None, times=[base + j * interval for j in range(per_sheet)])


def test_short_episode_is_a_single_window():
    sheets = [sheet(i) for i in range(6)]
    assert len(_chunks(sheets, 6, 1)) == 1


def test_long_episode_windows_overlap_and_cover_everything():
    sheets = [sheet(i) for i in range(17)]
    windows = _chunks(sheets, 6, 1)
    assert len(windows) > 1
    # every sheet appears at least once
    seen = {id(s) for w in windows for s in w}
    assert len(seen) == 17
    # consecutive windows share a sheet, so no event falls in a seam
    for a, b in zip(windows, windows[1:]):
        assert a[-1].start >= b[0].start


def test_windows_never_exceed_the_cap():
    for n in (1, 5, 6, 7, 13, 40):
        assert all(len(w) <= 6 for w in _chunks([sheet(i) for i in range(n)], 6, 1))


def test_stitch_merges_the_same_event_seen_by_two_windows():
    segs = [CoarseSegment(start_seconds=0, end_seconds=5, label="pick"),
            CoarseSegment(start_seconds=4.5, end_seconds=9, label="pick"),
            CoarseSegment(start_seconds=9, end_seconds=12, label="place")]
    out = _stitch(segs)
    assert [(s.start_seconds, s.end_seconds, s.label) for s in out] == [
        (0, 9, "pick"), (9, 12, "place")]


def test_stitch_keeps_a_genuine_repeat_of_the_same_label():
    # Same label twice with a real gap is two events, not one.
    segs = [CoarseSegment(start_seconds=0, end_seconds=2, label="pick"),
            CoarseSegment(start_seconds=5, end_seconds=7, label="pick")]
    assert len(_stitch(segs)) == 2


def test_stitch_orders_output():
    segs = [CoarseSegment(start_seconds=9, end_seconds=12, label="b"),
            CoarseSegment(start_seconds=0, end_seconds=5, label="a")]
    out = _stitch(segs)
    assert out[0].start_seconds < out[1].start_seconds


def test_prompt_mentions_the_window_bounds():
    req = AnnotateRequest(video="v.mp4", prompt="fold a box")
    text = build_segment_prompt(req, 120.0, 0.5, window=(50.0, 109.5), previous_label="Pick Box")
    assert "50.00s to 109.50s" in text
    assert "Pick Box" in text


def test_prompt_without_instruction_still_reads_sensibly():
    req = AnnotateRequest(video="v.mp4")
    text = build_segment_prompt(req, 30.0, 0.5)
    assert "No task description was supplied" in text
    assert "{" not in text.replace("{{", "").replace("}}", "")


def test_closed_vocabulary_is_listed_in_the_prompt():
    req = AnnotateRequest(video="v.mp4", prompt="x", subtasks=["Pick Box", "Fold Left"])
    text = build_segment_prompt(req, 30.0, 0.5)
    assert "Pick Box" in text and "Fold Left" in text
    assert "MUST" in text
