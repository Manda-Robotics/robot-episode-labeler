"""Semantic label scoring.

Gold labels in WGO-Bench are free-text descriptions ("pick up the left-most pink
plastic craft stick"), so string equality would score a correct system at zero.
A model judges whether two descriptions name the same manipulation event.

The judge only ever sees temporally matched pairs, so this measures naming in
isolation from segmentation, which is the decomposition we want.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..annotation.llm import GeminiClient

BATCH = 20

_PROMPT = """You are scoring an automatic robot-video annotator against human labels.

For each numbered pair below, decide whether the predicted label describes the
SAME completed manipulation event as the human label.

Judge the event, not the wording. "pick_cup" and "picks up the white mug from
the table" are the same event. "place_cup" and "pick up the cup" are not.
Ignore differences in verbosity, tense, object detail and phrasing. If the
predicted label names a different action, or a different object where the object
is what distinguishes the events, it is not a match.

{pairs}

Return a verdict for every pair, in order, using the same indices."""


class Verdict(BaseModel):
    index: int
    same_event: bool
    note: str = Field(default="")


class Verdicts(BaseModel):
    verdicts: list[Verdict]


def score_labels(
    client: GeminiClient, pairs: list[tuple[str, str]], batch: int = BATCH
) -> tuple[float, list[Verdict]]:
    """pairs is [(predicted, gold), ...]. Returns (accuracy, verdicts)."""
    if not pairs:
        return 0.0, []
    all_v: list[Verdict] = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        rendered = "\n".join(
            f"{start + i}. predicted: {p!r}\n   human:     {g!r}"
            for i, (p, g) in enumerate(chunk)
        )
        out = client.json("judge", _PROMPT.format(pairs=rendered), Verdicts)
        all_v.extend(out.verdicts)

    by_index = {v.index: v for v in all_v}
    hits = sum(1 for i in range(len(pairs)) if by_index.get(i) and by_index[i].same_event)
    return hits / len(pairs), all_v
