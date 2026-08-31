# Inputs for the model page examples

Replicate shows a featured example's exact inputs, so these are the values to use.
Both were run against the deployed model; prediction ids are recorded so they can
be featured in the web UI.

## 1. Schema mode with a rubric (recommended featured example)

Prediction `ay37j0p579rmw0d08r0ajq72kw`

| Field | Value |
|---|---|
| `video` | `droid_blue_block_in_green_bowl.mp4` |
| `prompt` | `Put the blue block in the green bowl` |
| `subtasks` | `Pick Up Block,Place Block In Bowl` |
| `attributes` | `retry,missed_grasp,dropped_object` |
| `quality` | `balanced` |

Output:

```
 0.00 - 11.00   fail   Place Block In Bowl
11.00 - 15.50   pass   Place Block In Bowl
```

Shows labels constrained to the caller's own vocabulary, and a failed attempt
located in time before the successful one.

## 2. Discovery mode (no vocabulary)

Prediction `knxezbv0a9rmr0d08qntx272cr`

| Field | Value |
|---|---|
| `video` | `droid_blue_block_in_green_bowl.mp4` |
| `prompt` | `Put the blue block in the green bowl` |
| `subtasks` | *(empty)* |
| `attributes` | *(empty)* |
| `quality` | `balanced` |

Output:

```
 0.00 - 11.50   fail   place_blue_block_in_green_bowl
11.50 - 15.50   pass   place_blue_block_in_green_bowl
```

The model invents its own snake_case label instead of using a supplied vocabulary.

## 3. A harder episode, if a second clip is wanted

| Field | Value |
|---|---|
| `video` | `droid_duvet_tip_left.mp4` |
| `prompt` | `Move the bottom right tip of the duvet to the left` |
| `subtasks` | `Grasp Duvet,Drag Duvet,Release Duvet` |
| `attributes` | `retry,missed_grasp,dropped_object` |

This one over-segments: eight subtasks with `Grasp Duvet` repeated, on a
deformable object where "one completed event" is ambiguous. A weaker example, so
not the recommended feature.

## Repeatability

The same inputs do not give byte-identical outputs. A local run of example 1
returned both segments as `pass` with different labels; the hosted run above
returned `fail` then `pass`. Corpus-level measurement of this variance is in
`docs/results.md`: the same configuration run twice moves corpus F1 by 0.03 to
0.08. Do not present any single example as a guaranteed result.
