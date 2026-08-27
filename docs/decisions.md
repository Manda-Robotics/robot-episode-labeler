# Decisions

Dated ADRs for `robot-episode-labeler`. Context and consequences, not just the call.
Convention inherited from the `manda` product repo.

---

## ADR-001 — Ship the annotation primitive, not the annotation business (2026-08-27)

**Context.** Established annotation vendors in this space sell a bundle: customer
ontology alignment, manually labelled golden examples, customer-specific trained
models, human QA on low-confidence predictions, and a review console. That bundle
is what makes their boundaries trustworthy, and it is expensive and slow to build.

**Decision.** Build only the automatic primitive: `video + instruction -> timestamped
subtasks`. No human review loop, no per-customer training, no review console, no
onboarding call.

**Consequences.** We cannot match a human-in-the-loop vendor on boundary accuracy
and must not claim to. In exchange we can be tried in sixty seconds by anyone with
a video, which is a different product with a different distribution channel. Any
future moat comes from corrections users make to our output, not from the wrapper.

---

## ADR-002 — Own the frame-sampling grid (2026-08-27)

**Context.** Video-native model APIs accept an MP4 directly, but typically ingest at
around 1 fps by default. Manipulation boundaries — a gripper closing, an object
leaving a surface — routinely occur inside a single one-second frame. At 1 fps an
episode reads as "not holding" then "holding", which supports the label `pick up
cup` but cannot place the boundary.

**Decision.** Decode with ffmpeg on a grid we choose (0.5 s coarse), tile frames
into contact sheets with the timestamp burned into each tile, and pass images.

**Consequences.** More moving parts than uploading an MP4, and we own the
correctness of the time axis. Verified: sampling is exact against a synthetic
colour-coded timecode video (40/40 frames in the correct bucket), and windowed
sampling is pixel-identical to full-episode sampling on H.264 with only two
keyframes in 25 s, so boundary refinement can seek cheaply without drift.

---

## ADR-003 — WGO-Bench is the development benchmark, and is eval-only (2026-08-27)

**Context.** `macrodata/WGO-Bench` is 100 manually annotated robot and egocentric
episodes with 743 gold subtask segments across DROID (50), Galaxea (25) and HomER
egocentric video (25). It is the only public benchmark aimed squarely at this task.

**Decision.** Use it as the development benchmark. Score segmentation and labeling
separately. Report the boundary tolerance curve (±0.25 / 0.5 / 1 / 2 s) as the
headline, because that is the number a robotics user can interpret.

**Consequences.** The licence is CC-BY-NC-SA-4.0, so the data must never be
redistributed or bundled into a commercial artifact; `data/` is gitignored.
Publishing measured scores is unaffected. The corpus is unbalanced — HomER supplies
470 of 743 segments — so pooled metrics are dominated by egocentric video and every
result is also reported per family.

---

## ADR-004 — Telemetry boundary-snapping is a conditional feature, not the wedge (2026-08-27)

**Context.** Robot telemetry seemed like an obvious edge over a video-only
competitor: a gripper opening or closing *is* the world-state change that defines
most manipulation boundaries, and WGO-Bench ships synchronized gripper channels for
its 75 robot episodes. Before building on it, we measured whether debounced gripper
transitions actually coincide with gold boundaries.

**Measured** (Schmitt-trigger transitions, 0.3 s minimum dwell, both arms merged):

| family | gold boundaries | transitions | recall ±0.5 s | recall ±1.0 s |
|---|---:|---:|---:|---:|
| galaxea | 98 | 311 | **0.561** | **0.816** |
| droid | 100 | 154 | 0.130 | 0.210 |

**Decision.** Do not build the pipeline around telemetry. Keep it as an optional
snapping step for callers who supply gripper state, gated per robot family.

**Consequences.** The signal is real but narrow. It is strong on Galaxea, weak on
DROID, and absent on HomER, which is video-only and is the majority of the
benchmark's segments. Since the product's premise is that a caller can send a plain
video, a video-only path has to carry the accuracy on its own. Effort goes to VLM
boundary refinement instead.

*(An earlier pass of this analysis scored family-specific channels against the
whole corpus's boundaries and understated Galaxea badly; the table above uses
per-family denominators.)*

---

## ADR-005 — Invariants are enforced in code, not requested in prompts (2026-08-27)

**Context.** Ordering, in-bounds timestamps, contiguity, non-overlap, closed label
vocabularies and closed attribute rubrics are all things a prompt can ask for and a
model can silently violate.

**Decision.** `annotation/validate.py` enforces them deterministically after every
inference stage: clamp to the episode, drop sub-threshold segments, trim overlaps,
snap gaps, snap out-of-vocabulary labels to the caller's list, drop attributes
outside the rubric — each recorded as a caller-visible warning.

**Consequences.** "Schema mode" is a guarantee rather than a request. Every
correction is surfaced instead of hidden, which is also how we learn where prompts
are failing.

---

## ADR-006 — Default to gemini-3.5-flash, but treat the model as a measurement (2026-08-27)

**Context.** WGO-Bench's published pipeline used Gemini 3.5 Flash, so defaulting to
it makes our numbers comparable to theirs. But `gemini-3.6-flash` and
`gemini-3.7-flash` are both available on the same API.

**Decision.** Default to `gemini-3.5-flash` for comparability; keep the model a
single constructor argument and A/B the newer models on the benchmark before
changing the default.

**Consequences.** Our first number is interpretable against published work. The
model choice becomes a result we can show rather than an assumption we inherited.

---

## ADR-007 — Confidence is derived from disagreement, not self-report (2026-08-27)

**Context.** A model will happily emit `"confidence": 0.97`. That number looks
precise and is not calibrated.

**Decision.** Report coarse `high | medium | low` plus explicit `flags`, derived
from observable disagreement: how far refinement moved a boundary
(`boundary_moved_1.25s`), whether the labeling pass disagreed with the segmentation
pass (`label_disagreement`), whether a repeat pass found a different number of
events (`segment_count_unstable`), whether a label had to be snapped into the
caller's vocabulary (`label_outside_vocabulary`).

**Consequences.** Users can filter to unflagged segments without us pretending the
system is human-quality. Flags are also the natural place to trigger human review
if we ever add one.

---

## ADR-008 — Annotations cover the manipulation, not the whole video (2026-08-27)

**Context.** The first segmentation prompt required contiguous coverage ending at
the episode duration. It also forbade segments for retreat motion. When an episode
ends with the arm retreating — which most do — those two rules contradict, and the
model resolved the contradiction by inventing a trailing `retract_arm` segment.
Observed twice before it was traced to the prompt rather than the model.

**Measured** across all 100 gold episodes:

| family | median coverage | episodes with internal gaps | median unused tail |
|---|---:|---:|---:|
| droid | 0.963 | 2/50 | 0.98 s |
| galaxea | 0.865 | 0/25 | 2.05 s |
| homer | 0.992 | 0/25 | 0.51 s |

**Decision.** Gold annotations are contiguous *internally* and always start at 0.00,
but stop when the last manipulation completes. The prompt now says so explicitly,
and gap-snapping in `validate.py` stays (internal gaps are near-nonexistent) while
nothing extends the last segment to the episode end.

**Consequences.** On two spot-checked episodes the spurious trailing segment
disappeared and segment counts became correct, with boundary errors of 0.12 s and
0.46 s. Generalising from two episodes proves nothing; the corpus run is what
counts. The wider lesson is that the output contract should be derived from what
the gold data does, not assumed.

---

## ADR-009 — Prompts are frozen per process and fingerprinted (2026-08-27)

**Context.** Prompts were read from disk on every call. Editing one during a
benchmark run silently changed the system half way through, producing a result that
measured two different pipelines and was discarded.

**Decision.** `load_prompt` is cached per process, and a `prompts` fingerprint (a
short hash over all prompt files) is stamped into every response's metadata.

**Consequences.** A run cannot drift mid-flight, and any stored result can be tied
back to the exact prompts that produced it. Changing a prompt now requires a new
run, which is the correct cost.

---

## ADR-010 — Decoding is a property of the package, not the host (2026-08-27)

**Context.** Every Galaxea episode (25 of 100) failed to decode. Homebrew's macOS
ffmpeg 8.0.1 ships an AV1 decoder with `Supported hardware devices: videotoolbox`
and no software fallback, so on hardware without AV1 acceleration every AV1 file is
undecodable. `-hwaccel none` does not help, because the decoder itself is
hardware-only. A quarter of the benchmark — and any customer sending AV1 — was
silently unusable.

**Decision.** `video/decode.py` tries the host ffmpeg first and transparently falls
back to the static `imageio-ffmpeg` build, which carries libaom. The regression test
builds its own AV1 fixture with libsvtav1, so it needs no checked-in binary.

**Consequences.** One extra dependency (~20 MB) in exchange for decoding not
depending on how the deployment host's ffmpeg was compiled. This also matters for
the fal and Replicate images, where we do not control the base ffmpeg build.
