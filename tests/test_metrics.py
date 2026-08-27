import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rel.eval.metrics import (
    Aggregate, boundary_score, internal_boundaries, iou, match_segments, segmentation_score,
)

GOLD = [(0.0, 2.0), (2.0, 5.0), (5.0, 9.0)]


def test_iou_basics():
    assert iou((0, 2), (0, 2)) == 1.0
    assert iou((0, 2), (2, 4)) == 0.0
    assert iou((0, 4), (2, 6)) == 2 / 6


def test_perfect_prediction_scores_one():
    s = segmentation_score(GOLD, GOLD)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_matching_is_one_to_one():
    # Three overlapping guesses at one gold segment must not count three times.
    pred = [(0.0, 2.0), (0.1, 2.0), (0.0, 1.9)]
    assert len(match_segments(pred, [(0.0, 2.0)])) == 1


def test_over_segmentation_hurts_precision_not_recall():
    pred = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 9)]
    s = segmentation_score(pred, GOLD)
    assert s.recall == 1.0 and s.precision < 0.6


def test_single_blob_scores_zero():
    assert segmentation_score([(0.0, 9.0)], GOLD).f1 == 0.0


def test_internal_boundaries_exclude_episode_ends():
    assert internal_boundaries(GOLD) == [2.0, 5.0]
    assert internal_boundaries([(0.0, 9.0)]) == []


def test_boundary_tolerance_is_graded():
    shifted = [(0, 2.4), (2.4, 5.4), (5.4, 9)]
    b = boundary_score(shifted, GOLD)
    assert b.recall_at[0.25] == 0.0
    assert b.recall_at[0.5] == 1.0
    assert abs(b.median_error - 0.4) < 1e-6


def test_empty_prediction_is_zero_not_error():
    s = segmentation_score([], GOLD)
    assert s.f1 == 0.0
    assert boundary_score([], GOLD).recall_at[2.0] == 0.0


def test_aggregate_pools_segments_across_episodes():
    a = Aggregate()
    a.add(GOLD, GOLD)
    a.add([(0.0, 9.0)], GOLD)
    r = a.report()
    assert r["episodes"] == 2 and r["segments_gold"] == 6 and r["segments_matched"] == 3


def test_aggregate_penalises_an_episode_with_no_predicted_boundaries():
    # One blob predicted against three gold segments: the two gold boundaries
    # must count against recall, not be silently skipped.
    a = Aggregate()
    a.add([(0.0, 9.0)], GOLD)
    r = a.report()
    assert r["boundary_recall_at"]["2.0"] == 0.0
    assert a.boundary_total == 2


def test_aggregate_pools_boundaries_across_episodes():
    a = Aggregate()
    a.add(GOLD, GOLD)          # 2 boundaries, both hit
    a.add([(0.0, 9.0)], GOLD)  # 2 boundaries, both missed
    assert a.boundary_total == 4
    assert a.report()["boundary_recall_at"]["0.25"] == 0.5


def test_gold_with_gaps_still_yields_a_boundary():
    gapped = [(0.0, 2.0), (3.0, 5.0)]
    assert internal_boundaries(gapped) == [2.5]


def test_duplicate_boundaries_are_not_double_counted():
    # Zero-length segments must not manufacture extra boundaries.
    spans = [(0.0, 2.0), (2.0, 2.0), (2.0, 5.0)]
    assert internal_boundaries(spans) == [2.0]
