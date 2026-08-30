# robot-episode-labeler

Turns a robot manipulation video into timestamped subtask annotations through one
inference endpoint.

```
video + task description  ->  [{start, end, label, result, attributes, description}, ...]
```

You send an episode and a sentence describing it. You get back structured,
queryable labels. Object masks, 6D pose, 3D reconstruction, dense captioning and
human review are out of scope. Temporal boundaries are the bottleneck on public
benchmarks, so that is where the engineering effort goes.

## Status

Alpha. Labels are generated automatically, with no human review.

Timestamps have sub-second resolution. Their accuracy is a weaker claim. On the
development benchmark the current segmentation pass (`fast` mode, default prompt)
measures:

| metric | value |
|---|---:|
| true boundaries within 0.5 s | 51% |
| median boundary error | 0.49 s |
| segmentation F1 at IoU 0.5 | 0.70 |
| events lasting 1–2 s found | 65% (previously 18%) |
| events under 1 s found | 19% (previously 0%) |

These are segmentation-only numbers. The end-to-end `balanced` path (segmentation,
subdivision, labeling) was last measured on the previous prompt: F1 0.629, 40.5%
of boundaries within 0.5 s, label accuracy 0.72 on matched segments. It has not
been re-measured with the new segmentation. The scoring protocol and what is not
yet measured are in [`docs/results.md`](docs/results.md); working notes, including
what did not work, are in [`docs/research-log.md`](docs/research-log.md). Do not
describe this as "sub-second accurate".

## Install

```bash
uv sync --extra dev
cp .env.example .env        # add a Gemini key from https://aistudio.google.com/apikey
uv run pytest -q
```

## Usage

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

From the command line, run the module directly:

```bash
PYTHONPATH=src uv run python -m rel.cli annotate episode.mp4 "A robot folds a box." \
    --subtasks "Pick Box,Fold Left,Fold Right,Stack" --quality balanced
```

The installed `rel` entry point also works, but this venv intermittently stops
processing `.pth` files, which leaves the console script unable to import `rel`.
`rm -rf .venv && uv sync --extra dev` fixes it; invoking the module avoids it.

### Schema mode

Supplying `subtasks` switches on schema mode: labels are constrained to your
vocabulary and snapped to it in code, not only requested in the prompt. Without
`subtasks` the model discovers its own labels. Most robotics teams know their
SOP; the hard part is applying it consistently across thousands of episodes.

### Quality modes

| Mode | Pipeline | For |
|---|---|---|
| `fast` | windowed coarse segmentation | dataset browsing, prototyping |
| `balanced` | + subdivision of long segments + context labeling | default |
| `strict` | + boundary refinement, repeat pass, disagreement flags | curation, QA |

Each mode is a preset over `rel.config.PipelineConfig`. Every knob (sampling
interval, tile size, prompts, model, thinking level, input modality) is a field,
the full config is stamped into each response's metadata, and
`config=PipelineConfig(...)` overrides a preset.

Boundary refinement is in `strict` rather than `balanced` because it was measured
at 31% of all model calls for no significant movement in any boundary metric
([`docs/results.md`](docs/results.md)).

## How it works

1. **Sample.** ffmpeg pulls frames on a 0.5 s grid, scaled to 224 px. Video
   models default to about 1 fps, and manipulation boundaries routinely fall
   inside one such frame, so the raw MP4 is not handed over.
2. **Contact sheets.** Frames are tiled 20 per sheet, 5 columns, each stamped
   with its episode time. Burned-in stamps are the only channel a model reliably
   reads a timestamp from.
3. **Segment.** One call per 60 s window. The prompt defines a boundary as a
   completed world-state change, excludes approach, retreat, hesitation and
   regrasping, and states that a pick and the following place are two subtasks.
   That last rule is the largest measured gain in the project: before it, the
   model merged pick and place into one segment named after the goal, which was
   most of the missed events. Two other inputs are implemented and measured:
   native video at a controlled frame rate (`segment_input="video"`, equal
   accuracy at about 35% lower cost) and per-frame state classification with
   boundaries derived in code (`segment_input="state"`, more short events found,
   edges placed less precisely).
4. **Subdivide.** Any segment longer than 3 s is re-read at 0.25 s sampling and
   split where a second completed event is visible. Without this, recall is a
   function of event duration (0.00 below one second at coarse sampling). This
   was the first pass's largest gain.
5. **Label.** Each segment is named and judged pass/fail with its neighbours as
   context.
6. **Refine** (`strict` only). Each boundary is re-placed from a dense window
   around it. Measured as not earning its cost.
7. **Validate.** Ordering, bounds, contiguity, closed vocabularies and rubrics
   are enforced in `annotation/validate.py`. The model proposes; the code decides.

Confidence is derived from observable disagreement between stages, not from the
model's self-report.

## Evaluation

The development benchmark is [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench):
100 episodes and 743 gold segments across DROID, Galaxea and HomER egocentric
video. The scoring protocol is in [`docs/results.md`](docs/results.md).

```bash
uv run python scripts/fetch_eval.py                       # downloads + unpacks (1.3 GB)
uv run python scripts/run_eval.py --tag mine --quality balanced
uv run python scripts/score_labels.py --tag mine
uv run python scripts/variance.py g37_base mine           # paired bootstrap on the delta
uv run python scripts/errors.py mine                      # why each gold segment was missed
uv run python scripts/run_eval.py --tag exp --quality fast --config segment_input=video,video_fps=2
```

Identical configurations differ by up to about 0.05 F1 run to run, so a single
100-episode run cannot resolve a smaller change. No change is adopted without
`variance.py` and a replication.

Segmentation and labeling are scored separately: finding the right moments and
naming them are different failures with different fixes. The headline number for
a robotics user is the boundary tolerance curve, the fraction of true boundaries
with a prediction within ±0.25 / 0.5 / 1 / 2 s.

Four further benchmarks, fetched by `scripts/fetch_<name>.py` into the same
layout and selected with `--dataset`: RoboCerebra and BEHAVIOR-1K (sim,
human-teleop boundaries), AgiBotWorld (real bimanual robot), HoloAssist
(egocentric human).

WGO-Bench is CC-BY-NC-SA-4.0. It is a development benchmark only, not
redistributed and not shipped in any commercial artifact. Reporting scores is fine.

## Deployment

All targets wrap the same `rel.pipeline.annotate`. Details, including the
bring-your-own-key model and open blockers, are in [`deploy/README.md`](deploy/README.md).

- **Replicate.** Live (private) at [replicate.com/mandarobotics/robot-episode-labeler](https://replicate.com/mandarobotics/robot-episode-labeler).
- **Hugging Face Space.** Gradio front end in `deploy/hf_space/`, prepared, not yet pushed.
- **fal.** `deploy/fal/app.py`, validated as far as possible without deploying; blocked on serverless access for the account.

## Layout

```
src/rel/
  schemas.py        public request/response contract
  config.py         every pipeline knob; quality modes are presets over it
  pipeline.py       stage orchestration
  video/            frame sampling, timestamped contact sheets, window clips
  annotation/       segment (sheets / video / state), subdivide, refine, label, validate, llm client
  eval/             dataset loaders, temporal metrics, label judge
  prompts/          annotation rules as prompt text (v2 = current)
scripts/            eval runners, paired bootstrap, failure taxonomy, dataset fetchers
results/            measured runs, committed as evidence
docs/               results (claims), research-log (what was tried), decisions (ADRs)
deploy/             Replicate, Hugging Face Space and fal wrappers
examples/           licensed demo clips and their inputs
```

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.

Built by [Manda Robotics](https://github.com/Manda-Robotics).
