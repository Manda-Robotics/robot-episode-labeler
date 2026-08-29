"""Temporal and semantic scoring.

Segmentation and labeling are scored separately on purpose. A system can find
the right moments and name them badly, or name things well having cut the
episode in the wrong places, and those two failures need different fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Span = tuple[float, float]

TOLERANCES = (0.25, 0.5, 1.0, 2.0)
# Segment-level F1 is reported at several IoU thresholds, as in the temporal
# action segmentation literature (F1@10/25/50). 0.5 is the headline.
IOU_THRESHOLDS = (0.1, 0.25, 0.5, 0.75)
# Recall by how long the gold event lasts: the dominant failure mode is short events.
DURATION_BUCKETS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf")))


def iou(a: Span, b: Span) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Match:
    pred_index: int
    gold_index: int
    iou: float


def match_segments(pred: list[Span], gold: list[Span], threshold: float = 0.5) -> list[Match]:
    """Greedy one-to-one matching by descending IoU, above `threshold`.

    Greedy is standard for this kind of temporal scoring and avoids rewarding a
    system that emits many overlapping guesses for one gold segment.
    """
    candidates = [
        Match(pi, gi, iou(p, g))
        for pi, p in enumerate(pred)
        for gi, g in enumerate(gold)
        if iou(p, g) >= threshold
    ]
    candidates.sort(key=lambda m: m.iou, reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[Match] = []
    for m in candidates:
        if m.pred_index in used_p or m.gold_index in used_g:
            continue
        used_p.add(m.pred_index)
        used_g.add(m.gold_index)
        out.append(m)
    return out


@dataclass
class SegmentationScore:
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    n_matched: int


def segmentation_score(pred: list[Span], gold: list[Span], threshold: float = 0.5) -> SegmentationScore:
    matches = match_segments(pred, gold, threshold)
    n = len(matches)
    p = n / len(pred) if pred else 0.0
    r = n / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return SegmentationScore(p, r, f1, len(pred), len(gold), n)


def internal_boundaries(spans: list[Span], eps: float = 1e-6) -> list[float]:
    """Boundaries strictly inside the episode: the moments a system must find.

    The first start and last end are dictated by the episode, not discovered, so
    counting them would flatter every system equally.
    """
    if len(spans) < 2:
        return []
    bounds: list[float] = []
    for a, b in zip(spans, spans[1:]):
        t = (a[1] + b[0]) / 2.0
        if not bounds or abs(t - bounds[-1]) > eps:
            bounds.append(t)
    return bounds


@dataclass
class BoundaryScore:
    """Fraction of gold boundaries with a prediction within each tolerance."""

    recall_at: dict[float, float] = field(default_factory=dict)
    precision_at: dict[float, float] = field(default_factory=dict)
    median_error: float | None = None
    n_gold: int = 0
    n_pred: int = 0


def boundary_score(pred: list[Span], gold: list[Span],
                   tolerances: tuple[float, ...] = TOLERANCES) -> BoundaryScore:
    gb = internal_boundaries(gold)
    pb = internal_boundaries(pred)
    score = BoundaryScore(n_gold=len(gb), n_pred=len(pb))
    if not gb:
        return score
    if not pb:
        score.recall_at = {t: 0.0 for t in tolerances}
        score.precision_at = {t: 0.0 for t in tolerances}
        return score

    g_err = [min(abs(g - p) for p in pb) for g in gb]
    p_err = [min(abs(p - g) for g in gb) for p in pb]
    score.recall_at = {t: sum(e <= t for e in g_err) / len(g_err) for t in tolerances}
    score.precision_at = {t: sum(e <= t for e in p_err) / len(p_err) for t in tolerances}
    ordered = sorted(g_err)
    mid = len(ordered) // 2
    score.median_error = (
        ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    )
    return score


def _bucket(duration: float) -> tuple[float, float]:
    for lo, hi in DURATION_BUCKETS:
        if lo <= duration < hi:
            return (lo, hi)
    return DURATION_BUCKETS[-1]


@dataclass
class EpisodeCounts:
    """Everything a corpus metric needs from one episode, as raw counts.

    Pooled precision, recall and boundary recall are ratios of summed counts, so
    a bootstrap over episodes only has to re-sum these.
    """

    n_pred: int
    n_gold: int
    matched_at: dict           # IoU threshold (or "wgo") -> matched segments
    b_total: int                           # internal gold boundaries
    b_hits: dict[float, int]               # tolerance -> gold boundaries hit
    pb_total: int                          # internal predicted boundaries
    pb_hits: dict[float, int]              # tolerance -> predicted boundaries near a gold one
    errors: list[float]                    # per gold boundary, distance to nearest prediction
    recall_by_duration: dict[tuple[float, float], tuple[int, int]]  # bucket -> (hit, total)

    @property
    def n_matched(self) -> int:
        return self.matched_at[0.5]

    @property
    def f1(self) -> float:
        p = self.n_matched / self.n_pred if self.n_pred else 0.0
        r = self.n_matched / self.n_gold if self.n_gold else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0


def snap_ends(pred: list[Span], gold: list[Span]) -> list[Span]:
    """WGO-Bench's convention: the first start and last end are dictated by the
    episode, so they are snapped to gold before matching and only the discovered
    boundaries are scored. Used for the `f1_wgo` number, which is the one
    comparable to Macrodata's published 0.306 (IoU 0.75)."""
    if not pred or not gold:
        return list(pred)
    out = list(pred)
    a, b = out[0]
    out[0] = (min(gold[0][0], b), b)
    a, b = out[-1]
    out[-1] = (a, max(gold[-1][1], a))
    return out


def episode_counts(pred: list[Span], gold: list[Span]) -> EpisodeCounts:
    matched_at = {t: len(match_segments(pred, gold, t)) for t in IOU_THRESHOLDS}
    if 0.5 not in matched_at:
        matched_at[0.5] = len(match_segments(pred, gold, 0.5))
    # WGO-comparable: IoU 0.75 with the outer boundaries snapped to gold.
    matched_at["wgo"] = len(match_segments(snap_ends(pred, gold), gold, 0.75))
    gb, pb = internal_boundaries(gold), internal_boundaries(pred)
    b_hits = {t: 0 for t in TOLERANCES}
    pb_hits = {t: 0 for t in TOLERANCES}
    errors: list[float] = []
    if gb and pb:
        errors = [min(abs(g - p) for p in pb) for g in gb]
        p_err = [min(abs(p - g) for g in gb) for p in pb]
        for t in TOLERANCES:
            b_hits[t] = sum(e <= t for e in errors)
            pb_hits[t] = sum(e <= t for e in p_err)
    hit_gold = {m.gold_index for m in match_segments(pred, gold, 0.5)}
    by_dur: dict[tuple[float, float], tuple[int, int]] = {}
    for gi, (a, b) in enumerate(gold):
        k = _bucket(b - a)
        h, n = by_dur.get(k, (0, 0))
        by_dur[k] = (h + (gi in hit_gold), n + 1)
    return EpisodeCounts(len(pred), len(gold), matched_at, len(gb), b_hits, len(pb), pb_hits,
                         errors, by_dur)


def _f1(matched: int, n_pred: int, n_gold: int) -> float:
    p = matched / n_pred if n_pred else 0.0
    r = matched / n_gold if n_gold else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class Aggregate:
    """Corpus-level roll-up. Segment counts are pooled, not averaged per episode,
    so long episodes are not down-weighted against short ones."""

    n_episodes: int = 0
    n_pred: int = 0
    n_gold: int = 0
    n_matched: int = 0
    matched_at: dict[float, int] = field(default_factory=dict)
    boundary_hits: dict[float, int] = field(default_factory=dict)
    boundary_total: int = 0
    pred_boundary_hits: dict[float, int] = field(default_factory=dict)
    pred_boundary_total: int = 0
    errors: list[float] = field(default_factory=list)
    by_duration: dict[tuple[float, float], list[int]] = field(default_factory=dict)

    def add(self, pred: list[Span], gold: list[Span], threshold: float = 0.5) -> None:
        self.add_counts(episode_counts(pred, gold))

    def add_counts(self, c: EpisodeCounts) -> None:
        self.n_episodes += 1
        self.n_pred += c.n_pred
        self.n_gold += c.n_gold
        self.n_matched += c.n_matched
        for t, m in c.matched_at.items():
            self.matched_at[t] = self.matched_at.get(t, 0) + m
        self.boundary_total += c.b_total
        self.pred_boundary_total += c.pb_total
        self.errors.extend(c.errors)
        for t in TOLERANCES:
            self.boundary_hits[t] = self.boundary_hits.get(t, 0) + c.b_hits[t]
            self.pred_boundary_hits[t] = self.pred_boundary_hits.get(t, 0) + c.pb_hits[t]
        for k, (h, n) in c.recall_by_duration.items():
            cur = self.by_duration.setdefault(k, [0, 0])
            cur[0] += h
            cur[1] += n

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def report(self) -> dict:
        med = None
        if self.errors:
            o = sorted(self.errors)
            m = len(o) // 2
            med = o[m] if len(o) % 2 else (o[m - 1] + o[m]) / 2
        return {
            "episodes": self.n_episodes,
            "segments_pred": self.n_pred,
            "segments_gold": self.n_gold,
            "segments_matched": self.n_matched,
            "segmentation_precision": round(self.precision, 4),
            "segmentation_recall": round(self.recall, 4),
            "segmentation_f1": round(self.f1, 4),
            "f1_at_iou": {
                str(t): round(_f1(self.matched_at.get(t, 0), self.n_pred, self.n_gold), 4)
                for t in IOU_THRESHOLDS
            },
            "f1_wgo": round(_f1(self.matched_at.get("wgo", 0), self.n_pred, self.n_gold), 4),
            "recall_by_duration": {
                f"{lo:g}-{hi:g}": {"recall": round(h / n, 4), "n": n}
                for (lo, hi), (h, n) in sorted(self.by_duration.items()) if n
            },
            "boundary_recall_at": {
                str(t): round(self.boundary_hits.get(t, 0) / self.boundary_total, 4)
                for t in TOLERANCES
            } if self.boundary_total else {},
            "boundary_precision_at": {
                str(t): round(self.pred_boundary_hits.get(t, 0) / self.pred_boundary_total, 4)
                for t in TOLERANCES
            } if self.pred_boundary_total else {},
            "median_boundary_error_sec": round(med, 3) if med is not None else None,
        }
