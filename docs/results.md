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

### Subdivision fixes it, and the fix is paid for in precision

`annotation/subdivide.py` re-reads any predicted segment longer than 3 s at 0.25 s
sampling and splits it where a second completed event is visible, spending
inference only where an event can hide.

Recall by gold-segment duration, paired on 93 episodes:

| gold segment duration | baseline | subdivided | n |
|---|---:|---:|---:|
| 0–1 s | 0.000 | **0.109** | 46 |
| 1–2 s | 0.200 | **0.529** | 70 |
| 2–4 s | 0.451 | 0.646 | 164 |
| 4–8 s | 0.573 | 0.685 | 143 |
| 8 s + | 0.706 | 0.675 | 126 |

Corpus effect, paired against **two** independent baseline runs. Every tolerance is
significant in both, at 3–6× the noise floor:

| metric | baseline | subdivided | delta (95% CI) | replication |
|---|---:|---:|---|---|
| boundary recall ±0.25 s | 0.149 | **0.263** | +0.114 [+0.070, +0.158] | +0.099 [+0.062, +0.137] |
| boundary recall ±0.5 s | 0.290 | **0.447** | +0.158 [+0.091, +0.223] | +0.165 [+0.109, +0.216] |
| boundary recall ±1.0 s | 0.452 | **0.654** | +0.202 [+0.128, +0.275] | +0.239 [+0.170, +0.303] |
| boundary recall ±2.0 s | 0.640 | **0.816** | +0.175 [+0.111, +0.240] | +0.224 [+0.169, +0.276] |
| segment recall | 0.472 | **0.603** | | |
| segment precision | 0.635 | 0.480 | | |
| segmentation F1 | 0.541 | 0.534 | −0.007, not significant | +0.014, not significant |

The trade is explicit: we discover many more real events and also over-split some
genuinely single ones, so F1 is unchanged while boundary recall improves sharply.
For making episodes searchable, a missed event is worse than a spurious boundary a
user can see and skip — and low-confidence splits carry flags. Cost roughly triples,
$0.62 → $1.95 per video-hour on `gemini-3.5-flash`.

The obvious next lever on precision is to reject splits whose pieces all carry the
same label, which is a cut through one event rather than the discovery of two.
Untested.

## Current best configuration

`gemini-3.7-flash`, windowed segmentation, subdivision on. Paired on 97 episodes
against the same model without subdivision:

| metric | 3.5 base | 3.7 base | **3.7 + subdivide** |
|---|---:|---:|---:|
| segmentation F1 | 0.531 | 0.598 | **0.643** |
| precision | 0.635 | 0.743 | 0.716 |
| recall | 0.472 | 0.498 | **0.583** |
| boundary recall ±0.25 s | 0.148 | 0.174 | **0.217** |
| boundary recall ±0.5 s | 0.291 | 0.330 | **0.405** |
| boundary recall ±1.0 s | 0.454 | 0.490 | **0.569** |
| cost per video-hour | $0.71 | **$0.61** | $1.40 |
| failed episodes | 1 | 2 | 2 |

**Subdivision replicates on a second model.** On 3.7 it is significant at ±0.25 s
(+0.043, CI [+0.004, +0.086]), ±0.5 s (+0.076, CI [+0.026, +0.125]) and ±1.0 s
(+0.079, CI [+0.014, +0.143]). Together with the two 3.5 comparisons that is three
independent confirmations across two models, so this is the most solid result we
have.

Notably, 3.7 subdivides more judiciously than 3.5: precision falls only
0.743 → 0.716 rather than 0.635 → 0.480, so on 3.7 subdivision raises F1 as well
(+0.047, P(delta>0)=0.933, just short of significance) instead of leaving it flat.

Recall by gold-segment duration on 3.7:

| gold duration | base | + subdivide | n |
|---|---:|---:|---:|
| 0–1 s | 0.000 | 0.038 | 52 |
| 1–2 s | 0.183 | **0.367** | 120 |
| 2–4 s | 0.509 | 0.619 | 226 |
| 4–8 s | 0.637 | 0.709 | 179 |
| 8 s + | 0.754 | 0.746 | 142 |

**3.5 vs 3.7 as models:** +0.066 F1, CI [−0.003, +0.146], P(delta>0)=0.969 — just
short of significance, but 3.7 is also *cheaper* ($0.61 vs $0.71 per video-hour) and
faster, so there is no trade to weigh. 3.7 is the default; 3.5 is kept only to
replicate recorded runs.

Sub-second events remain largely invisible: 0.038 recall under one second. Nothing
here should be described as sub-second accurate.

## Shipping configuration

`balanced` on `gemini-3.7-flash`: windowed segmentation + subdivision + context
labeling. This is what the endpoint runs by default.

| metric | value |
|---|---:|
| segmentation F1 | 0.629 (P 0.677 / R 0.588) |
| boundary recall ±0.25 s | 0.215 |
| boundary recall ±0.5 s | 0.405 |
| boundary recall ±1.0 s | 0.609 |
| boundary recall ±2.0 s | 0.788 |
| median boundary error | **0.703 s** |
| label accuracy (matched segments) | 0.720 |
| cost | $2.39 / video-hour |
| failed episodes | 1 / 100 |

Dropping refinement from this path cost nothing measurable and saved 26%
($3.25 → $2.39 per video-hour, identical ±0.5 s boundary recall).

Label accuracy is judged by `gemini-3.1-pro-preview` — a different tier from the
annotator, so the judge does not share its blind spots — on temporally matched
segments only. By family: **droid 0.461**, galaxea 0.806, homer 0.800. DROID is the
clear weak spot and is the obvious next thing to look at; its gold labels read like
task instructions ("Move the silver cup to the right") rather than event
descriptions, so some of the gap may be a scoring-convention mismatch rather than a
model failure. Not yet investigated.

### Progress over the session

| metric | start (3.5, single call) | now (3.7, balanced) |
|---|---:|---:|
| segmentation F1 | 0.544 | 0.629 |
| boundary recall ±0.5 s | 0.234 | 0.405 |
| boundary recall ±1.0 s | 0.425 | 0.609 |
| median boundary error | 1.275 s | 0.703 s |
| failed episodes | 11 | 1 |

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

- Why DROID label accuracy (0.461) trails Galaxea and HomER (~0.80).
- Whether rejecting same-label splits recovers subdivision's precision loss.
- Any measurement of `strict` mode.
