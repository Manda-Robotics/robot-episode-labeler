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

## Runs

`fast` = coarse segmentation only. 100 episodes, 743 gold segments,
`gemini-3.5-flash`, temperature 0.

| run | change | seg F1 | ±0.25s | ±0.5s | ±1.0s | ±2.0s | median err | failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fast_v2` | one call per episode | 0.544 | 0.113 | 0.234 | 0.425 | 0.637 | 1.275s | 11 |
| `fast_v3_chunked` | 60 s windows | 0.533 | 0.149 | 0.292 | 0.455 | 0.659 | 1.183s | 1 |

Those two rows are **not directly comparable**: failed episodes are excluded from
scoring, and the runs failed on different episodes, so their gold denominators
differ (HomER gold 342 vs 450). Paired on the 89 episodes both completed
(`scripts/paired.py`):

| metric | one call | windowed | delta |
|---|---:|---:|---:|
| segmentation F1 | 0.544 | **0.568** | +0.024 |
| precision | 0.679 | 0.667 | −0.012 |
| recall | 0.454 | **0.495** | +0.041 |
| boundary recall ±0.25s | 0.113 | **0.150** | +0.037 |
| boundary recall ±0.5s | 0.234 | **0.296** | +0.062 |
| boundary recall ±1.0s | 0.425 | **0.472** | +0.047 |
| median boundary error | 1.275s | **1.115s** | −0.160s |
| segments predicted | 402 | 447 | (gold 602) |

By family:

| family | F1 (one call → windowed) | ±1.0s | predicted → | gold |
|---|---|---|---|---:|
| droid | 0.602 → 0.551 | 0.378 → 0.344 | 142 → 146 | 137 |
| galaxea | 0.679 → 0.749 | 0.653 → 0.633 | 101 → 104 | 123 |
| homer | 0.451 → 0.501 | 0.369 → 0.459 | 159 → 197 | 342 |

Windowing was aimed at HomER, whose episodes run to 171 s — 17 contact sheets in a
single call — and that is where it paid off. The reference implementation sends
every sheet in one call, so this is a real difference rather than a reinvention.

> **Caveat: the noise floor is not yet measured.** DROID moved 0.602 → 0.551 on
> identical episodes through an unchanged code path (short episodes never window),
> which can only be model nondeterminism at temperature 0. A repeat run under an
> identical configuration was started to quantify this and was cut short by the API
> account running out of credit. **Until that is measured, treat any delta smaller
> than roughly ±0.05 F1 as unproven**, including the per-family numbers above. The
> corpus-level boundary-recall gains are larger and more consistent, but they are
> not yet proven either.

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

- Run-to-run variance (the noise floor). **Blocks confident interpretation of every
  delta above.**
- The `subdivide` recall pass.
- `balanced` mode end to end: what boundary refinement and context labeling buy.
- Label accuracy: the judge is implemented, the scoring run died on the credit
  outage.
- `gemini-3.6-flash` / `gemini-3.7-flash` against 3.5.
- Any measurement of `strict` mode.
