"""Deterministic cleanup. The model proposes; this module decides.

Invariants a caller is entitled to assume -- ordering, bounds, contiguity, a
closed vocabulary actually being closed -- are enforced in code. Asking a
language model to respect them and hoping is not a contract.
"""

from __future__ import annotations

from ..schemas import AnnotateRequest, Confidence, Result, Segment

# Segments shorter than this are treated as noise rather than events.
MIN_DURATION = 0.15


def _closest(label: str, vocab: list[str]) -> str:
    """Snap a label to the caller's vocabulary; exact, then case-insensitive,
    then token overlap. Never invents a label outside the list."""
    if not vocab:
        return label
    for v in vocab:
        if v == label:
            return v
    low = {v.lower(): v for v in vocab}
    if label.lower() in low:
        return low[label.lower()]
    words = set(label.lower().replace("_", " ").split())
    best, best_score = vocab[0], -1.0
    for v in vocab:
        vw = set(v.lower().replace("_", " ").split())
        score = len(words & vw) / len(words | vw) if (words | vw) else 0.0
        if score > best_score:
            best, best_score = v, score
    return best


def clean(
    segments: list[Segment],
    duration: float,
    request: AnnotateRequest,
    snap_gaps: bool = True,
) -> tuple[list[Segment], list[str]]:
    """Return validated segments plus human-readable warnings about what changed."""
    warnings: list[str] = []
    if not segments:
        return [], ["no segments were produced for this episode"]

    kept: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start_seconds, s.end_seconds)):
        s = seg.model_copy(deep=True)
        start = min(max(s.start_seconds, 0.0), duration)
        end = min(max(s.end_seconds, 0.0), duration)
        if (s.start_seconds, s.end_seconds) != (start, end):
            warnings.append(
                f"segment '{s.label}' clamped to the episode "
                f"({s.start_seconds:.2f}-{s.end_seconds:.2f} -> {start:.2f}-{end:.2f})"
            )
        if end - start < MIN_DURATION:
            warnings.append(f"dropped segment '{s.label}' shorter than {MIN_DURATION}s")
            continue
        s.start_seconds, s.end_seconds = round(start, 2), round(end, 2)

        if request.schema_mode:
            snapped = _closest(s.label, request.subtasks)
            if snapped != s.label:
                warnings.append(f"label '{s.label}' snapped to '{snapped}'")
                s.flags = [*s.flags, "label_outside_vocabulary"]
                s.confidence = Confidence.low
                s.label = snapped
        if request.attributes:
            allowed = {a.lower(): a for a in request.attributes}
            valid, dropped = [], []
            for a in s.attributes:
                (valid.append(allowed[a.lower()]) if a.lower() in allowed else dropped.append(a))
            if dropped:
                warnings.append(f"dropped attributes not in the rubric: {sorted(set(dropped))}")
            s.attributes = valid
        if s.result not in tuple(Result):
            s.result = Result.unknown
        kept.append(s)

    if not kept:
        return [], warnings + ["every proposed segment was invalid"]

    # Overlaps: trim the earlier segment back to where the next one starts.
    ordered: list[Segment] = [kept[0]]
    for s in kept[1:]:
        prev = ordered[-1]
        if s.start_seconds < prev.end_seconds:
            overlap = prev.end_seconds - s.start_seconds
            warnings.append(
                f"trimmed {overlap:.2f}s overlap between '{prev.label}' and '{s.label}'"
            )
            prev.end_seconds = s.start_seconds
            if prev.duration < MIN_DURATION:
                ordered.pop()
                if not ordered:
                    ordered.append(s)
                    continue
        ordered.append(s)

    if snap_gaps:
        for a, b in zip(ordered, ordered[1:]):
            if b.start_seconds > a.end_seconds:
                midpoint = round((a.end_seconds + b.start_seconds) / 2, 2)
                a.end_seconds, b.start_seconds = midpoint, midpoint

    return ordered, warnings
