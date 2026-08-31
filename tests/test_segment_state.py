import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rel.annotation.segment_state import FrameState, rows_to_segments, _snap_rows


def rows(seq, interval=0.5):
    times = [round(i * interval, 3) for i in range(len(seq))]
    return {t: FrameState(t=t, subtask=l) for t, l in zip(times, seq)}, times


def test_label_runs_become_segments_with_midpoint_boundaries():
    r, times = rows(["pick", "pick", "pick", "place", "place", "none", "none"])
    segs = rows_to_segments(r, times, 0.5)
    assert [s.label for s in segs] == ["pick", "place"]
    assert segs[0].start_seconds == 0.0
    assert segs[0].end_seconds == 1.25 and segs[1].start_seconds == 1.25   # between 1.0 and 1.5
    assert segs[1].end_seconds == 2.25                                       # trailing "none" is a gap


def test_single_frame_blips_are_absorbed():
    r, times = rows(["pick", "pick", "place", "pick", "pick", "pick"])
    segs = rows_to_segments(r, times, 0.5)
    assert [s.label for s in segs] == ["pick"]


def test_rows_snap_to_nearest_stamp_and_ignore_far_ones():
    times = [0.0, 0.5, 1.0]
    got = _snap_rows([FrameState(t=0.49, subtask="a"), FrameState(t=3.0, subtask="b")], times, 0.5)
    assert set(got) == {0.5} and got[0.5].subtask == "a"


def test_missing_rows_count_as_none():
    times = [0.0, 0.5, 1.0, 1.5]
    r = {0.0: FrameState(t=0.0, subtask="pick"), 0.5: FrameState(t=0.5, subtask="pick")}
    segs = rows_to_segments(r, times, 0.5)
    assert len(segs) == 1 and segs[0].end_seconds == 0.75


def test_first_segment_starts_at_zero_even_after_idle_frames():
    r, times = rows(["none", "none", "pick", "pick", "pick"])
    segs = rows_to_segments(r, times, 0.5)
    assert segs[0].start_seconds == 0.0 and segs[0].label == "pick"


def test_runs_expand_to_per_frame_rows():
    from rel.annotation.segment_state import Run, runs_to_rows
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    rows = runs_to_rows([Run(t=0.0, subtask="pick"), Run(t=1.0, subtask="place"), Run(t=2.0, subtask="none")],
                        times, 0.5)
    assert [r.subtask for r in rows] == ["pick", "pick", "place", "place", "none"]
