## How accurate is it

Measured on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), 100 manually annotated robot and egocentric episodes with 743 gold subtask segments. Segmentation pass, current default:

| Metric | Value |
|---|---|
| Segmentation F1 (IoU 0.5) | 0.695 |
| F1 at WGO-Bench's own protocol (IoU 0.75; the benchmark authors publish 0.306) | 0.434 |
| True boundaries found within 0.5 s | 51% |
| True boundaries found within 1.0 s | 70% |
| Median boundary error | 0.49 s |
| Events shorter than 2 s found | 65% |

The end-to-end `balanced` path (segmentation + subdivision + labeling) was last measured on the previous prompt: F1 0.629, 40.5% of boundaries within 0.5 s, label accuracy 0.72 on matched segments. Treat the table as the segmentation ceiling and those as the floor. Every number was checked against run-to-run noise with a paired bootstrap; the full record, including what did not work, is in the repository's `docs/`.

## What it gets wrong

- **Short events.** Events under two seconds are found about 65% of the time; when one is missed it is almost always merged into its neighbour. Timestamps have sub-second *resolution*, not sub-second *accuracy*.
- **Labels are uneven across robots** — good on Galaxea and egocentric video, weaker on DROID, whose reference labels read like task instructions.
- These are automatically generated labels with no human review: a first pass over a corpus, not ground truth. `confidence` and `flags` come from disagreement between pipeline stages, not from asking the model; filter to unflagged segments for higher precision.

## Data handling

Your video is decoded here into frames, sent to the Gemini API under **your** key, and deleted when the call returns; this Space stores no video, frame, key or annotation. Google's terms for your account apply to what Gemini receives. Hugging Face may keep request logs under its own policy — review that before uploading unreleased footage.

## Elsewhere

- Source and evaluation code: [github.com/Manda-Robotics/robot-episode-labeler](https://github.com/Manda-Robotics/robot-episode-labeler) (Apache-2.0)
- Hosted API: [replicate.com/mandarobotics/robot-episode-labeler](https://replicate.com/mandarobotics/robot-episode-labeler)
- Example clips: DROID via `lerobot/droid_1.0.1` (Apache-2.0; DROID CC-BY-4.0).

Built by [Manda](https://github.com/Manda-Robotics), a public benefit corporation building evaluation infrastructure for robot learning.
