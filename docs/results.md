# Measured results

Every number here comes from a committed run in `results/`. Where something is
not measured, it says so.

## Scoring protocol

Read this before comparing our numbers to anyone else's.

- **Segmentation F1.** Predicted and gold segments are matched greedily by
  descending temporal IoU, one-to-one, keeping matches with **IoU ≥ 0.5**.
  Precision is matches / predicted, recall is matches / gold. Counts are pooled
  across the corpus rather than averaged per episode, so a long episode is not
  down-weighted against a short one.
- **Boundary recall @ t.** Of the *internal* boundaries in the gold annotation
  (the first start and last end are dictated by the episode, not discovered), the
  fraction with a predicted boundary within ±t seconds.
- **Label accuracy.** Judged by a model, on temporally matched segments only, so
  naming is measured separately from segmentation. Gold labels are free-text
  descriptions, so string equality would score a correct system at zero.

**This is our protocol, not WGO-Bench's.** Macrodata report 0.306 segmentation F1
for their best configuration. We do not know their matching rule, IoU threshold or
pooling, and their published repository contains no scoring code, so **our numbers
are not a head-to-head comparison** and must not be presented as beating theirs.
What is comparable is the pipeline: their sampling parameters (0.5 s, 224 px,
20 frames per sheet, 5 columns) are the ones we use.

## The noise floor comes first

Everything below is interpreted against this measurement, because without it the
numbers mean very little.

The **same configuration run twice** (`fast_v3_chunked` vs `fast_v3_repeat`,
`gemini-3.5-flash`, temperature 0, 99 episodes both completed):

| metric | run 1 | run 2 | delta |
|---|---:|---:|---:|
| segmentation F1 | 0.5327 | 0.4992 | −0.0335 |
| boundary recall ±0.25s | 0.1490 | 0.1538 | +0.0048 |
| boundary recall ±0.5s | 0.2917 | 0.2692 | −0.0224 |
| boundary recall ±1.0s | 0.4551 | 0.4231 | −0.0321 |

33 of 99 episodes changed score, with a **median per-episode |ΔF1| of 0.145**.

So temperature 0 is not determinism, and on a 100-episode corpus that produces
swings of roughly **±0.03–0.08 in corpus F1**. The practical consequence:

> **A single run cannot resolve a change smaller than about 0.05 F1.** Any
> comparison of two single runs at that scale is reading noise. Comparisons must
> either use a paired bootstrap whose interval excludes zero, replicate across
> independent runs, or average several runs per configuration.

`scripts/variance.py` does the paired bootstrap (4000 resamples over episodes) and
prints a confidence interval for the delta.

## Runs

`fast` = coarse segmentation only. 100 episodes, 743 gold segments,
`gemini-3.5-flash`, temperature 0.

| run | change | seg F1 | ±0.25s | ±0.5s | ±1.0s | ±2.0s | median err | failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fast_v2` | one call per episode | 0.544 | 0.113 | 0.234 | 0.425 | 0.637 | 1.275s | 11 |
| `fast_v3_chunked` | 60 s windows | 0.533 | 0.149 | 0.292 | 0.455 | 0.659 | 1.183s | 1 |
| `fast_v3_repeat` | *identical to above* | 0.499 | 0.154 | 0.269 | 0.423 | 0.617 | — | 1 |

Raw rows are not comparable across runs when failures differ, because failed
episodes leave the gold denominator. `scripts/paired.py` and `scripts/variance.py`
restrict to episodes both runs completed.

### Does windowing help? Partly, and less than it first looked

Windowed segmentation compared against the single-call run, twice — once against
each independent windowed run, so a claim has to replicate to count:

| metric | vs run 1 | vs run 2 | verdict |
|---|---|---|---|
| segmentation F1 | +0.024, CI [−0.039, +0.089] | −0.014, CI [−0.079, +0.053] | **no effect shown** |
| boundary recall ±0.25s | +0.037, CI [**+0.005**, +0.072] | +0.039, CI [**+0.002**, +0.076] | **significant, replicated** |
| boundary recall ±0.5s | +0.062, CI [+0.002, +0.117] | +0.039, CI [−0.015, +0.097] | mixed, unproven |
| boundary recall ±1.0s | +0.047, CI [−0.032, +0.125] | +0.014, CI [−0.052, +0.084] | no effect shown |

**What we can claim:** windowing improves the tightest boundary localisation
(±0.25 s) by about 3–4 points, and that replicates across two independent runs. It
also reduced failures from 11 to 1, which is deterministic rather than statistical.

**What we cannot claim:** that it improves segmentation F1. The earlier "+0.024 F1"
read was inside the noise floor, and the second run put the same comparison at
−0.014. The per-family numbers from a single pair of runs (e.g. "HomER F1
0.451 → 0.501", "DROID regressed") are likewise unproven — per-family subsets are
smaller and therefore noisier than the corpus.

## The dominant failure mode: short events

Recall on `fast_v3_chunked`, by how long the gold segment lasts:

| gold segment duration | recall | n |
|---|---:|---:|
| 0–1 s | **0.000** | 52 |
| 1–2 s | 0.150 | 120 |
| 2–4 s | 0.436 | 227 |
| 4–8 s | 0.602 | 181 |
| 8 s + | 0.699 | 143 |

Found segments have a median duration of 5.49 s; missed ones 2.44 s.

This is not primarily a reasoning failure. At 0.5 s sampling with 20 frames per
sheet, a one-second event is two frames inside a sheet spanning ten seconds — the
model can barely see it. Macrodata report the same shape (7.4% recall for subtasks
under two seconds), which suggests it is a property of the contact-sheet approach
rather than of a particular model.

The `subdivide` pass in `annotation/subdivide.py` targets exactly this: it re-reads
segments longer than 3 s at 0.25 s sampling and splits them where a second event is
visible, spending inference only where an event can hide. **It is implemented and
tested but not yet measured** — the credit outage hit first.

## Cost

Measured over two full 100-episode passes (135 minutes of video, 1.57 M tokens):

- **~$0.62 per video-hour** at list price for `gemini-3.5-flash`
- ~$1.39 total for both complete benchmark runs

For scale, Macrodata report $0.86/video-hour for segmentation only and
$5.27/video-hour with per-segment relabeling. Nothing here needs cost engineering
yet.

One observation worth acting on: a typical segmentation call spends **5,838
thinking tokens against 95 output tokens**. Thinking dominates both cost and
latency by roughly 60:1, and `thinking_budget` is an untested lever.

## What is not measured yet

- The `subdivide` recall pass.
- `balanced` mode end to end: what boundary refinement and context labeling buy.
- Label accuracy: the judge is implemented, the scoring run died on the credit
  outage.
- `gemini-3.6-flash` / `gemini-3.7-flash` against 3.5.
- Any measurement of `strict` mode.
