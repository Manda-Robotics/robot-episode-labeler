# robot-episode-labeler

Turn a robot manipulation video into timestamped subtask annotations, through one
inference endpoint.

```
video + task description  ->  [{start, end, label, result, attributes, description}, ...]
```

No onboarding, no ontology workshop, no annotation contract. You send an episode
and a sentence describing it; you get back structured, queryable labels.

## Why this exists

Robot teams sit on thousands of teleoperation episodes that are effectively
opaque: to find "every failed grasp of the yellow block" you have to watch them.
Making episodes temporally and semantically searchable is the useful part of
what dedicated annotation vendors sell, and most of it can now be produced
automatically. This packages that as a primitive rather than a service.

Deliberately **not** in scope: object masks, 6D pose, 3D reconstruction, dense
captioning, or a human-in-the-loop review operation. The bottleneck in public
benchmarks is temporal boundaries, so that is what the engineering effort goes to.

## Status

Alpha. Automatically generated labels, no human review.

Timestamps have sub-second *resolution*. Their *accuracy* is a different and much
weaker claim: on our development benchmark, **29% of true boundaries are found
within half a second** and the median boundary error is **1.1 s**. Recall depends
strongly on how long an event lasts — 0.70 for events over 8 s, **0.00 under one
second**. Measured numbers, scoring protocol, and what is not yet measured are all
in [`docs/results.md`](docs/results.md). Do not describe this as "sub-second
accurate".

## Interface

```python
from rel.schemas import AnnotateRequest
from rel.pipeline import annotate

resp = annotate(AnnotateRequest(
    video="episode.mp4",
    prompt="A robot arm folds a cardboard box.",
    subtasks=["Pick Box", "Fold Left", "Fold Right", "Stack"],  # optional
    attributes=["retry", "missed_grasp", "dropped_object"],     # optional
    quality="balanced",                                          # fast | balanced | strict
))
```

Supplying `subtasks` switches on **schema mode**: labels are constrained to your
vocabulary and snapped to it in code, not merely requested in a prompt. Without
it the model discovers its own labels. Schema mode is the mode robotics teams
actually want, because they know their SOP; the hard part is applying it
consistently across thousands of episodes.

### Quality modes

| Mode | Pipeline | For |
|---|---|---|
| `fast` | coarse segmentation | dataset browsing, prototyping |
| `balanced` | + boundary refinement + context labeling | default |
| `strict` | + repeat segmentation and disagreement flags | curation / QA |

## How it works

1. **Sample.** ffmpeg pulls frames on a grid we control (0.5 s), scaled to 224 px.
   We do not hand the raw MP4 to a video model, because the default there is
   ~1 fps and manipulation boundaries routinely happen inside one such frame.
2. **Contact sheets.** Frames are tiled 20 per sheet, 5 columns, each stamped with
   its episode time. Burned-in stamps are the only channel a model reliably reads
   a timestamp from.
3. **Segment.** One call over the whole episode, with a prompt that defines
   boundaries as *completed world-state changes* and explicitly rules out
   approach, retreat, hesitation and regrasping.
4. **Refine.** Each boundary is re-placed from a dense (0.25 s) window around it.
   Paying for high frame rate only where the uncertainty is.
5. **Label.** Each segment is named and judged pass/fail with its neighbours as
   context.
6. **Validate.** Ordering, bounds, contiguity, closed vocabularies and rubrics are
   enforced in code. The model proposes; `annotation/validate.py` decides.

Confidence is derived from observable disagreement between stages, not from
asking the model how sure it is.

## Development

```bash
uv sync --extra dev
cp .env.example .env        # add a Gemini key from https://aistudio.google.com/apikey
uv run pytest -q
```

### Evaluation

Development benchmark is [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench):
100 episodes, 743 gold segments, across DROID, Galaxea and HomER egocentric video.

```bash
uv run python scripts/fetch_eval.py                       # downloads + unpacks (1.3 GB)
uv run python scripts/run_eval.py --tag mine --quality balanced
uv run python scripts/score_labels.py --tag mine
```

Segmentation and labeling are scored separately: a system can find the right
moments and name them badly, or name well having cut in the wrong places, and
those need different fixes. The headline number for a robotics user is the
**boundary tolerance curve** — what fraction of true boundaries we land within
±0.25 / 0.5 / 1 / 2 s.

> WGO-Bench is CC-BY-NC-SA-4.0. It is a development benchmark only: not
> redistributed, not shipped in any commercial artifact. Reporting scores is fine.

## Layout

```
src/rel/
  schemas.py        public request/response contract
  pipeline.py       stage orchestration, quality modes
  video/            frame sampling, timestamped contact sheets
  annotation/       segment, refine, label, validate, llm client
  eval/             WGO-Bench loader, temporal metrics, label judge
  prompts/          the annotation philosophy, as prompt text
scripts/            eval runners
results/            measured runs, committed as evidence
```
