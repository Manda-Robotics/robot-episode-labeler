# Research log — second pass (2026-08-28)

Working notes for the rework. `docs/results.md` stays the record of *measured*
claims; this file records what was tried, in order, including what did not work.
Numbers here are on WGO-Bench unless stated. Every corpus number is subject to
the ±0.03–0.08 F1 noise floor in `docs/results.md`; nothing is claimed without
`scripts/variance.py`.

## Where the first pass left it

`g37_balanced_v2` (gemini-3.7-flash, 0.5 s contact sheets, 60 s windows,
subdivision, context labeling): F1@0.5 0.629, ±0.5 s boundary recall 0.405,
median boundary error 0.70 s, label accuracy 0.72 (DROID 0.46).

### Diagnosis of the existing predictions (no inference spent)

1. **Grid quantisation.** 84% of predicted boundaries sit exactly on the 0.5 s
   sampling grid. ±0.25 s recall (0.22) is capped by the grid, not by judgement.
2. **Pick and place are merged into one event named after the place.** Of gold
   `pick X` → `place X` pairs, one predicted segment covers both in 42% (HomER),
   19% (Galaxea), 8% (DROID) of cases. The labeling pass then names the pick
   segment after the task goal (`place_lid_on_table` on a gold `pick up the lid`),
   which is most of DROID's 0.46 label accuracy: 47 of 102 DROID verdicts are
   this exact pattern. Diagnosis: the prompt describes boundaries as world-state
   changes but the model still thinks in task-level units; the label prompt asks
   "what event is this" with the task instruction in view.
3. **HomER under-segments** (341 predicted vs 450 gold). Egocentric, many 1–2 s
   events.
4. **The WGO-comparable metric.** Macrodata's 0.306 is F1 at IoU 0.75 with the
   outer boundaries snapped to gold. On that metric (`f1_wgo`) our runs score:
   fast 0.385, fast+subdivide 0.369, balanced 0.352. Subdivision *lowers* it —
   over-splitting is punished at tight IoU — so it is a recall-vs-precision trade
   and not a free win. Per family (fast): Galaxea 0.61, DROID 0.58, HomER 0.24.

### Negative result: pixel motion priors (no inference spent)

Consecutive-frame motion-energy minima are *worse* than a uniform grid as
boundary candidates (Galaxea 0.41 recall@0.5 s vs 0.66 for a uniform grid of
the same count). 1-s change maxima beat the grid on DROID (0.45 vs 0.29) and
Galaxea (0.81 vs 0.65) but not HomER (camera motion dominates), and snapping the
existing predicted boundaries to them within ±0.75 s does not improve any
tolerance (±0.5 s: 0.407 → 0.398). The model's boundaries already carry more
timing information than pixel motion can add. Dropped. (Script:
scratchpad `motion_prior.py`; results in this file only.)

## Levers identified by research (see agent reports summarised here)

- **Native video input with `fps`.** Gemini 3.x accepts video parts with
  `VideoMetadata(fps≤24, start_offset, end_offset)`; 70 tokens/frame at default
  resolution, 280 at HIGH; timestamp tokens are interleaved per frame. Moment-Video
  reports Gemini 3.1 Pro going 27% → 38% on momentary events from 1 → 5 fps.
  Probe on one DROID episode: 2 fps placed the boundary at 5.88 s (gold 5.89), off
  the grid, in 2 s and 1.5k tokens. Whole-corpus test required.
- **Temperature.** Google now recommends 1.0 for 3.x and says lower "may lead to
  looping or degraded performance"; the parameter is deprecated. We send 0.
- **`thinking_level`.** 3.7-flash: low/medium(default)/high. ~5.8k thinking tokens
  per 95 output tokens today.
- **`media_processing`**: STATIC (fixed-rate extraction) vs AGENTIC (model-driven
  navigation). Undocumented beyond enum strings.
- **State-then-boundaries.** Strongest zero-shot grounding results (REZE: 51.5 vs
  6.5 mIoU for the same model) come from asking the model to classify frame state
  and deriving boundaries in code, not from asking for timestamps.
- **WGO-Bench's own finding**: windowing hurt them (22% of boundaries on sheet
  edges); one call per episode was best. Our windowing helped ±0.25 s recall only.
- **`gemini-robotics-er-2-preview`**: moment finding 91% / 0.96 s MAD claimed;
  $2/$10 per M tokens (vs $0.75/$3.75 for 3.7-flash). ER 1.6 scored below 3.5
  Flash on WGO-Bench in Macrodata's sweep.

## Experiment queue

Ordered by expected information per dollar. Each is one `run_eval.py --config`
run, paired against `g37_base` (fast) or `g37_balanced_v2` with `variance.py`.

1. Native video segmentation at 2 fps and 4 fps, fast mode. (`segment_input=video`)
2. Temperature 1.0 vs 0 on the sheet pipeline.
3. thinking_level low / high.
4. Pick/place decomposition rule in the segment prompt + "label the event that
   completes at the END of this segment" in the label prompt.
5. State-table segmentation: per-tile gripper/object state → boundaries in code.
6. Robotics-ER 2 on the best config.
7. Cross-dataset validation on a second benchmark (data agent pending).

## Results as they land

All `fast` (segmentation only, no subdivision, no labeling) unless stated, so the
coarse pass is measured in isolation. Paired bootstrap against `g37_base`.

| run | change | F1@0.5 | f1_wgo | ±0.25 | ±0.5 | ±1.0 | med err | R(1–2 s) | $/vid-h |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `g37_base` | sheets, 0.5 s, 60 s windows | 0.598 | 0.385 | 0.173 | 0.329 | 0.491 | 1.04 | 0.183 | 1.08 |
| `v_fps2` | native video, 2 fps, 60 s windows | **0.649** | **0.416** | 0.218 | 0.370 | 0.526 | 0.84 | 0.358 | **0.76** |

`v_fps2` vs `g37_base`, paired on 98: F1 +0.048 [+0.001, +0.093] significant;
±0.25 s +0.045 [+0.006, +0.080] significant; ±0.5/1.0/2.0 positive, not
significant. Output tokens halve (114k vs 216k: less thinking per call). Precision
0.80 / recall 0.55: it under-segments HomER (277 vs 470 gold). Costs above are at
the 2026 list price ($0.75 / $3.75 per M); the first-pass docs used superseded
rates, so their $/video-hour figures are ~2.5× too low.

### Failure taxonomy (`scripts/errors.py`)

Why each gold segment is missed, `g37_base` → `v_fps2`:

| slice | ok | merged | split | shifted | missing |
|---|---:|---:|---:|---:|---:|
| droid | 0.71 → 0.71 | 0.11 → 0.17 | 0.01 → 0.00 | 0.14 → 0.08 | 0.03 → 0.04 |
| galaxea | 0.71 → 0.76 | 0.24 → 0.15 | 0 | 0.02 → 0.07 | 0.03 → 0.02 |
| homer | 0.37 → 0.44 | **0.52 → 0.48** | 0.01 → 0 | 0.06 | 0.03 → 0.02 |
| gold < 2 s | 0.13 → 0.25 | **0.74 → 0.63** | 0 | 0.07 | 0.06 → 0.05 |

Misses are merges of adjacent events, almost never splits or shifts. The
problem is under-segmentation, not boundary placement; boundary error is a
symptom of a missing boundary, not a misplaced one. This is why subdivision (a
split pass) was the first pass's largest gain, and it is what the pick/place
decomposition prompt and the per-frame state backend attack directly.

| `v_fps4` | native video, 4 fps | 0.641 | 0.393 | 0.208 | 0.345 | 0.504 | 0.97 | 0.317 | 1.11 |

`v_fps4` vs `v_fps2`: −0.008 F1, every tolerance slightly negative, nothing
significant, 1.9× the input tokens. Denser native sampling does not help at
60 s windows; 2 fps is the setting. (Consistent with Moment-Video's plateau
above ~5 fps and WGO's finding that denser sheets did not help.)
| `s_tempdef` | sheets, temperature = API default (1.0) | 0.596 | 0.381 | 0.178 | 0.316 | 0.466 | 1.26 | 0.167 | 1.13 |

`s_tempdef` vs `g37_base`: nothing significant in either direction (F1 −0.003,
CI [−0.073, +0.064]). Google's warning that sub-1.0 temperature can degrade 3.x
output does not show up here; temperature is not a lever for this task.
| `s_thinklow` | sheets, thinking_level=low | 0.594 | 0.337 | 0.162 | 0.317 | 0.492 | 1.04 | 0.200 | **0.66** |

`s_thinklow` vs `g37_base`: F1 −0.005, CI [−0.037, +0.032]; boundary tolerances
all within noise; but `f1_wgo` −0.050, CI [−0.092, −0.010], **significant**.
Output tokens fall from 216k to 79k, cost from $1.08 to $0.66 per video-hour.
Low thinking keeps the loose-IoU F1 and loses tight-IoU F1: the model still finds
the events but places their edges less carefully. Not adopted.
| `s_state` | per-frame state rows, boundaries derived in code | 0.660 | 0.351 | **0.236** | **0.442** | **0.615** | **0.58** | **0.575** | 2.32 |

`s_state` vs `g37_base`: ±0.25 s +0.063 [+0.020, +0.104], ±0.5 s +0.112
[+0.040, +0.177], ±1.0 s +0.124 [+0.040, +0.197], all significant; F1 +0.062
just short (CI [−0.004, +0.119]). vs `v_fps2`: ±0.5 s +0.077 and ±1.0 s +0.098
significant, but `f1_wgo` −0.062 [−0.123, −0.009] significant. Reformulating
"emit timestamps" as "classify each frame" finds the events (HomER merges
0.52 → 0.29; sub-2 s recall 0.18 → 0.58) but places their edges on grid
midpoints, so it loses at IoU 0.75. Output tokens are 5× the sheet pipeline
(one row per frame), hence the cost. Next: state to find, video clip to place.
| `s_v2prompt` | sheets + pick/place decomposition prompt (`segment_v2.md`) | **0.695** | **0.434** | **0.272** | **0.508** | **0.699** | **0.49** | **0.650** | 1.24 |

`s_v2prompt` vs `g37_base`: F1 +0.097 [+0.025, +0.156]; ±0.25 s +0.098
[+0.045, +0.142]; ±0.5 s +0.178 [+0.109, +0.235]; ±1.0 s +0.207 [+0.129,
+0.274]; ±2.0 s +0.124 [+0.048, +0.198] — every metric significant, at 2–4× the
noise floor. vs `s_state`: `f1_wgo` +0.083 [+0.036, +0.129] significant. HomER
merges 0.52 → 0.25, sub-2 s recall 0.18 → 0.65. The prompt change is three
sentences: pick and place are two subtasks, every object gets its own, do not
cap the segment count. Largest single gain in the project, at no extra cost.
The diagnosis from the failure taxonomy (merging, not misplacement) was right.
| `v_fps2_v2` | native video 2 fps + decomposition prompt | **0.720** | **0.469** | 0.268 | 0.488 | 0.676 | 0.52 | **0.691** | **0.81** |

`v_fps2_v2` vs `v_fps2`: F1 +0.071 [+0.026, +0.118] significant — the prompt
replicates on a second input modality. vs `s_v2prompt`: F1 +0.025, `f1_wgo`
+0.033, boundary recall −0.005 to −0.026: nothing significant either way. With
the decomposition prompt, video and sheets are equivalent finders; video is 35%
cheaper with higher precision (0.785 vs 0.730), sheets have slightly better
boundary recall. Per family, video: DROID 0.746, Galaxea 0.886, HomER 0.666.
| `v_fps2_rep` | *identical to `v_fps2`* | 0.610 | 0.402 | 0.162 | 0.331 | 0.479 | 1.01 | — | 0.78 |

**Withdrawn: native video alone is not an improvement.** The replication of
`v_fps2` scored F1 0.610 vs 0.649 (Δ −0.039, CI [−0.097, +0.015]) and is
*significantly* below the first run at ±0.25 s (−0.056) and ±2.0 s (−0.072).
Against `g37_base` the replication is +0.009 F1, CI [−0.062, +0.064]. The
first run's "+0.048, significant" was a favourable draw; the CI touched zero
at +0.001. Two lessons: (1) native video's run-to-run variance is at least as
large as the sheet pipeline's; (2) a CI whose lower bound is +0.001 is not
evidence. Native video keeps its cost advantage only. The decomposition-prompt
effect (+0.07 to +0.10) is 2–3× this noise; its replications (`s_v2_rep`,
`v_v2_rep`) are queued and decide the default.
| `v_fps2_whole` | native video, one call per episode (no windows) | 0.620 | — | 0.202 | 0.330 | 0.495 | 1.01 | — | ~0.75 |

`v_fps2_whole` vs `v_fps2_rep` (same prompt, windows): F1 +0.010 [−0.058,
+0.084]; ±0.25 s +0.040 [+0.007, +0.079]; everything else within noise.
Macrodata found windowing hurt their sheet pipeline; here, for native video,
windowing vs one call makes no reliable difference. Windows stay (they bound
the request size for long HomER episodes).

## Additional evaluation data (`scripts/fetch_*.py`, all under gitignored `data/`)

| dataset | episodes | video | segs/ep | median seg | provenance | licence |
|---|---:|---:|---:|---:|---|---|
| `robocerebra` (sim, LIBERO-style; step boundaries from human teleop) | 39 | 30 min | 8.7 | 4.6 s | human-teleop | MIT |
| `behavior` (BEHAVIOR-1K 2025 demos, sim, head cam; skill frame ranges) | 32 | 87 min | 10.4 | 12.1 s | human-teleop | MIT |
| `agibot` (AgiBotWorld2026, real bimanual, head cam; human skill segments) | 11 | 16 min | 12.5 | 4.4 s | human | CC BY-NC-SA 4.0 |
| `holoassist` (egocentric human, fine-grained verb-noun actions, ≤180 s chunks) | 18 | 35 min | 23.8 | 1.9 s | human | CDLA-Permissive-2.0 |

Conventions differ from WGO-Bench and from each other: BEHAVIOR includes
navigation segments ("move to microwave") and short unannotated gaps; AgiBot
segments are per-arm skills; HoloAssist actions are ~2 s and include "inspect".
So these are a robustness check for the segmentation *mechanism* (does the
decomposition prompt over-split a sim scene? does the model track two arms?),
not a second leaderboard. The sim sets are the closest public analogue to
Manda's RoboLab clips. Caveats recorded per episode in `metadata.notes`
(RoboCerebra `coffee_table/case1` has a 35-frame count mismatch; AgiBot is
limited to 11 episodes because the next archives are ≥4.5 GB; gated AgiBot
Beta/Alpha and RoboInter need an HF account with accepted terms).

Planned use: run the final `fast` and `balanced` presets over all four with
`run_eval.py --dataset <name>`, report per-dataset F1@0.5 / boundary recall /
`errors.py` taxonomy, and inspect the worst episodes by eye.
