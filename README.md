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

Timestamps have sub-second *resolution*. Their *accuracy* is a different and
weaker claim. On the development benchmark the segmentation pass currently finds
**51% of true boundaries within half a second** (median boundary error 0.49 s),
segmentation F1 0.70 at IoU 0.5, and 65% of events shorter than two seconds
(previously 18%). Those are segmentation-only numbers; the end-to-end `balanced`
path with the new segmentation has not been re-measured yet. Measured numbers,
the scoring protocol, and what is not yet measured are in
[`docs/results.md`](docs/results.md); the working notes, including what did not
work, are in [`docs/research-log.md`](docs/research-log.md). Do not describe this
as "sub-second accurate".

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
| `fast` | windowed coarse segmentation | dataset browsing, prototyping |
| `balanced` | + subdivision of long segments + context labeling | default |
| `strict` | + boundary refinement, repeat pass, disagreement flags | curation / QA |

Each mode is a preset over `rel.config.PipelineConfig`; every knob (sampling
interval, tile size, prompts, model, thinking level, input modality) is a field,
and the full config is stamped into each response's metadata. Pass
`config=PipelineConfig(...)` to `annotate` to override a preset.

Boundary refinement sits in `strict` rather than `balanced` because it was measured
and did not earn its cost: 31% of all model calls for no significant movement in
any boundary metric. See [`docs/results.md`](docs/results.md).

## How it works

1. **Sample.** ffmpeg pulls frames on a grid we control (0.5 s), scaled to 224 px.
   We do not hand the raw MP4 to a video model, because the default there is
   ~1 fps and manipulation boundaries routinely happen inside one such frame.
2. **Contact sheets.** Frames are tiled 20 per sheet, 5 columns, each stamped with
   its episode time. Burned-in stamps are the only channel a model reliably reads
   a timestamp from.
3. **Segment.** One call per 60 s window, with a prompt that defines
   boundaries as *completed world-state changes*, rules out approach, retreat,
   hesitation and regrasping, and — the single largest measured improvement —
   states that a pick and the following place are two subtasks, never one.
   Before that rule the model merged them into one segment named after the
   goal, which was most of what the benchmark counted as missed events.
   Two alternative segmentation inputs are implemented and measured:
   native video at a controlled frame rate (`segment_input="video"`, equal
   accuracy at ~35% lower cost) and per-frame state classification with
   boundaries derived in code (`segment_input="state"`, finds more short events,
   places their edges less precisely).
4. **Subdivide.** Any segment longer than 3 s is re-read at 0.25 s sampling and
   split where a second completed event is visible. Recall is otherwise a function
   of event duration — measured at 0.00 below one second — because at coarse
   sampling a short event is only a frame or two. This is the single largest
   measured improvement in the pipeline.
5. **Label.** Each segment is named and judged pass/fail with its neighbours as
   context.
6. **Refine** (`strict` only). Each boundary is re-placed from a dense window
   around it. Measured as not earning its cost, so it is off by default.
7. **Validate.** Ordering, bounds, contiguity, closed vocabularies and rubrics are
   enforced in code. The model proposes; `annotation/validate.py` decides.

Confidence is derived from observable disagreement between stages, not from
asking the model how sure it is.

## Development

```bash
uv sync --extra dev
cp .env.example .env        # add a Gemini key from https://aistudio.google.com/apikey
uv run pytest -q
```

Run the CLI as a module rather than through the console script:

```bash
PYTHONPATH=src uv run python -m rel.cli annotate episode.mp4 "A robot folds a box."
```

The installed `rel` entry point works, but this venv intermittently stops
processing `.pth` files, which leaves the editable install invisible and the
console script unable to import `rel`. Recreating the venv (`rm -rf .venv &&
uv sync --extra dev`) fixes it; invoking the module directly avoids it entirely.

### Evaluation

Development benchmark is [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench):
100 episodes, 743 gold segments, across DROID, Galaxea and HomER egocentric video.

```bash
uv run python scripts/fetch_eval.py                       # downloads + unpacks (1.3 GB)
uv run python scripts/run_eval.py --tag mine --quality balanced
uv run python scripts/score_labels.py --tag mine
uv run python scripts/variance.py g37_base mine           # paired bootstrap: is the change real?
uv run python scripts/errors.py mine                      # why each gold segment was missed
uv run python scripts/run_eval.py --tag exp --quality fast --config segment_input=video,video_fps=2
```

A single 100-episode run cannot resolve a change smaller than about 0.05 F1
(identical configurations differ by that much run to run), so no change is
adopted without `variance.py` and a replication. Four further benchmarks —
RoboCerebra and BEHAVIOR-1K (sim, human-teleop boundaries), AgiBotWorld (real
bimanual robot), HoloAssist (egocentric human) — are fetched by
`scripts/fetch_<name>.py` into the same layout and selected with `--dataset`.

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
  config.py         every pipeline knob; quality modes are presets over it
  pipeline.py       stage orchestration
  video/            frame sampling, timestamped contact sheets, window clips
  annotation/       segment (sheets / video / state), subdivide, refine, label, validate, llm client
  eval/             dataset loaders, temporal metrics, label judge
  prompts/          the annotation philosophy, as prompt text (v2 = current)
scripts/            eval runners, paired bootstrap, failure taxonomy, dataset fetchers
results/            measured runs, committed as evidence
docs/               results (claims), research-log (what was tried), decisions (ADRs)
```
