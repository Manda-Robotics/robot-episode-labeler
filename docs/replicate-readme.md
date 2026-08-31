Turn a robot manipulation video into timestamped subtasks.

Send an episode and a one-sentence task description. The model returns a start and end time for each completed manipulation event, a label, a pass or fail result, and failure attributes such as `retry` or `missed_grasp`.

Robot teams hold thousands of teleoperation episodes that cannot be searched. Finding every failed grasp of the yellow block means watching all of them. With these labels attached, the episodes can be queried.

## Example

Input: a 17 second DROID episode with the instruction `Put the blue block in the green bowl`.

Output, unedited, from a real run:

```json
{
  "task": "Put the blue block in the green bowl",
  "duration_seconds": 17.73,
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 11.0,
      "label": "place_blue_block_in_green_bowl",
      "result": "fail",
      "attributes": [],
      "description": "The robot attempts to grasp the blue block to place it into the green bowl, but fails to pick it up from the table.",
      "confidence": "low",
      "flags": [
        "label_disagreement"
      ]
    },
    {
      "start_seconds": 11.0,
      "end_seconds": 15.5,
      "label": "place_blue_block_in_green_bowl",
      "result": "pass",
      "attributes": [],
      "description": "The robot carries the blue block over to the green bowl and successfully places it inside.",
      "confidence": "medium",
      "flags": []
    }
  ]
}
```

A failed attempt followed by a successful one, located in time.

## Inputs

| Input | What it does |
|---|---|
| `video` | The episode. mp4, mov or webm. AV1 and H.264 both work. |
| `prompt` | What the robot is doing. Optional: leave it blank and the episode is annotated without a task hint. |
| `subtasks` | Optional comma-separated vocabulary, for example `Pick Box,Fold Left,Fold Right`. |
| `attributes` | Optional comma-separated rubric, for example `retry,missed_grasp,dropped_object`. |
| `quality` | `fast`, `balanced` (default) or `strict`. |
| `gemini_api_key` | Your Gemini API key. Write only: it is scoped to the call and never returned. |

### Schema mode

Supplying `subtasks` switches on schema mode. Labels are constrained to your vocabulary and snapped to it in code, in addition to being requested in the prompt. If the model returns a label outside the list, it is mapped to the nearest allowed label and the segment is flagged.

Most robotics teams want this mode. The SOP already exists. The hard part is applying it consistently across thousands of episodes.

### Quality modes

| Mode | Pipeline | For |
|---|---|---|
| `fast` | windowed segmentation | dataset browsing, prototyping |
| `balanced` | plus context labeling | default |
| `strict` | plus subdivision of long segments, boundary refinement and disagreement flags | curation and QA |

## Accuracy

Measured on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), a public benchmark of 100 manually annotated robot and egocentric episodes with 743 gold subtask segments.

Current default pipeline, paired-bootstrap checked, replicated where marked:

| Metric | Value |
|---|---|
| Segmentation F1 (IoU 0.5), no vocabulary | 0.68–0.73 (replicated) |
| Segmentation F1 with a caller vocabulary (schema mode) | 0.77 |
| End-to-end schema mode (segmentation + labels) | 0.70 |
| F1 at WGO-Bench's own protocol (IoU 0.75; the benchmark authors publish 0.306) | 0.40–0.48 |
| True boundaries found within 0.5 s | 49–51% |
| Median boundary error | 0.44–0.53 s |
| Label accuracy on matched segments (model-judged) | 0.80 |

Schema-mode numbers use vocabularies derived from each episode's own gold labels, which is the best case; a coarser customer SOP list will land between the two F1 rows.

Scoring protocol: greedy one-to-one matching, pooled across the corpus. Boundary recall counts interior boundaries only, since the first start and last end are set by the episode rather than discovered. Every number was checked against run-to-run noise with a paired bootstrap. Details and negative results are in the repository's `docs/`.

## Limitations

**Short events.** Events lasting 1–2 s are found 65% of the time, events under 1 s 19%, longer events about 70–75%. A missed event is almost always merged into its neighbour rather than misplaced. Timestamps are reported at sub-second resolution. We do not claim sub-second accuracy.

**Over-splitting.** Precision is 0.68 in `balanced` (0.73 for the segmentation pass alone). Finding more real events costs some spurious boundaries. For search that is usually the right trade, and low-confidence splits carry flags you can filter on.

**Label accuracy varies by robot.** 0.89 on Galaxea, 0.84 on egocentric video, 0.59 on DROID, whose reference labels read like task instructions rather than event descriptions.

**No human review.** The labels are generated automatically. Treat them as a first pass over a corpus. They are not ground truth.

## Confidence and flags

`confidence` is derived from observable disagreement between pipeline stages. The model is not asked how sure it is. A self-reported 0.97 looks precise and is not calibrated.

Flags you may see:

- `label_disagreement`: the segmentation pass and the labeling pass named the event differently
- `boundary_moved_1.25s`: refinement moved this boundary a long way
- `segment_count_unstable`: a repeat pass found a different number of events
- `label_outside_vocabulary`: the label had to be snapped into your `subtasks` list

Filter to unflagged segments when you need higher precision.

## Bring your own key

The model calls the Gemini API. Replicate has no model-level secret store, so you supply `gemini_api_key` with each call. Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The key is write only, scoped to your call, and never written into the response.

Typical cost on your Gemini account at 2026 list prices: about 1.25 USD per hour of video for segmentation only (`fast`), about 4 USD per hour in `balanced` mode.

## Data handling

Robot footage often shows unreleased hardware, so here is what happens to it.

The model container keeps nothing. Frames are extracted to a temporary directory and deleted when the call returns. No video, frame or annotation is written to storage we control. The video is sent to the Gemini API as contact-sheet images under your own API key, so Google's terms for your account apply to it, and ours do not. The key is write only and is never returned in the response.

Replicate stores prediction inputs and outputs under its own retention policy. Review that policy before uploading sensitive footage.

## Example footage

The example episodes come from [DROID](https://droid-dataset.github.io/) via `lerobot/droid_1.0.1` (Apache 2.0; DROID itself CC BY 4.0), used with attribution.

## Built by

Built by [Manda Robotics](https://github.com/Manda-Robotics). Hugging Face: [huggingface.co/mandarobotics](https://huggingface.co/mandarobotics).
