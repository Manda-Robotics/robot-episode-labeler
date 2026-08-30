Turn a robot manipulation video into timestamped subtasks.

Send an episode and a sentence describing the task. Get back structured labels: start and end times, a name for each completed manipulation event, pass or fail, and failure attributes such as retry or missed grasp.

Robot teams hold thousands of teleoperation episodes that are effectively unsearchable. To find every failed grasp of the yellow block, someone has to watch them. This makes episodes queryable.

## Example

Input: a 17 second DROID episode, with the instruction `Put the blue block in the green bowl`.

Output, verbatim from a real run (not edited):

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

A failed attempt followed by a successful one, located in time. That is the shape of query most teams want and cannot run today.

## Inputs

| Input | What it does |
|---|---|
| `video` | The episode. mp4, mov or webm. AV1 and H.264 both work. |
| `prompt` | What the robot is doing. Optional: leave it blank and the episode is annotated without a task hint. |
| `subtasks` | Optional comma separated vocabulary, for example `Pick Box,Fold Left,Fold Right`. |
| `attributes` | Optional comma separated rubric, for example `retry,missed_grasp,dropped_object`. |
| `quality` | `fast`, `balanced` (default) or `strict`. |
| `gemini_api_key` | Your Gemini API key. Write only: it is scoped to the call and never returned. |

### Schema mode

Supplying `subtasks` switches on schema mode. Labels are then constrained to your vocabulary and snapped to it in code, not merely requested in a prompt. If the model returns something outside the list, it is mapped to the nearest allowed label and the segment is flagged.

This is the mode most robotics teams want. You already know your SOP. The hard part is applying it consistently across thousands of episodes.

### Quality modes

| Mode | Pipeline | For |
|---|---|---|
| `fast` | windowed segmentation | dataset browsing, prototyping |
| `balanced` | plus subdivision of long segments, plus context labeling | default |
| `strict` | plus boundary refinement and disagreement flags | curation and QA |

## How accurate is it

Measured on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), a public benchmark of 100 manually annotated robot and egocentric episodes with 743 gold subtask segments.

Segmentation pass (the current default prompt, `fast` mode):

| Metric | Value |
|---|---|
| Segmentation F1 (IoU 0.5) | 0.695 |
| Segmentation F1 at WGO-Bench's own protocol (IoU 0.75; the benchmark authors publish 0.306) | 0.434 |
| True boundaries found within 0.5 s | 51% |
| True boundaries found within 1.0 s | 70% |
| Median boundary error | 0.49 s |
| Events shorter than 2 s found | 65% |

The end-to-end `balanced` path (segmentation plus subdivision plus labeling) was last measured on the previous prompt: F1 0.629, 40.5% of boundaries within 0.5 s, label accuracy 0.72 on matched segments. It has not yet been re-measured on the current default, so treat the table above as the segmentation ceiling and those as the floor.

Scoring protocol: greedy one to one matching, pooled across the corpus. Boundary recall counts interior boundaries only, since the first start and last end are dictated by the episode rather than discovered. Every number was checked against the run to run noise with a paired bootstrap; details and negative results are in the repository's `docs/`.

## What it gets wrong

Read this before depending on it.

**Short events are the weak spot.** Events under two seconds are found about 65% of the time, longer ones about 80%. When one is missed it is almost always merged into its neighbour rather than misplaced. Timestamps have sub second resolution, but that is not the same as sub second accuracy, and we do not claim the latter.

**It over splits.** Precision is 0.68. Finding more real events costs some spurious boundaries. For search that is usually the right trade, and low confidence splits carry flags you can filter on.

**Label accuracy is uneven across robots.** 0.80 on some sources, 0.46 on DROID, whose reference labels read like task instructions rather than event descriptions.

These are automatically generated labels with no human review. Treat them as a first pass over a corpus, not as ground truth.

## Confidence and flags

`confidence` is derived from observable disagreement between pipeline stages, not from asking the model how sure it is. A self reported 0.97 looks precise and is not calibrated.

Flags you may see:

- `label_disagreement`: the segmentation pass and the labeling pass named the event differently
- `boundary_moved_1.25s`: refinement moved this boundary a long way
- `segment_count_unstable`: a repeat pass found a different number of events
- `label_outside_vocabulary`: the label had to be snapped into your `subtasks` list

Filter to unflagged segments when you need higher precision.

## Bring your own key

This model calls the Gemini API and Replicate has no model level secret store, so you supply `gemini_api_key` yourself. Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The key is write only, scoped to your call, and never written into the response.

Typical cost on your Gemini account at 2026 list prices is about 1.25 USD per hour of video for segmentation only (`fast`) and about 4 USD per hour in `balanced` mode.

## Data handling

Worth being explicit about, since robot footage is often unreleased hardware.

The model container keeps nothing. Frames are extracted to a temporary directory and deleted when the call returns, and no video, frame or annotation is written to storage we control. Your video is sent to the Gemini API as contact sheet images under your own API key, so Google's terms for your account apply, not ours. Your key is write only and is never returned in the response.

Replicate itself stores prediction inputs and outputs under its own retention policy. If your footage is sensitive, review that before uploading.

## Example footage

The example episodes come from [DROID](https://droid-dataset.github.io/) via `lerobot/droid_1.0.1` (Apache 2.0, DROID itself CC BY 4.0), used with attribution.

## Built by

[Manda](https://github.com/Manda-Robotics), a public benefit corporation building evaluation infrastructure for robot learning.
