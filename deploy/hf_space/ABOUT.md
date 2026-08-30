## Evaluation

Measured on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), 100 manually annotated robot and egocentric episodes with 743 gold subtask segments. Segmentation pass, current default prompt:

| Metric | Value |
|---|---|
| Segmentation F1 (IoU 0.5) | 0.695 |
| F1 at WGO-Bench's own protocol (IoU 0.75; the benchmark authors publish 0.306) | 0.434 |
| True boundaries found within 0.5 s | 51% |
| True boundaries found within 1.0 s | 70% |
| Median boundary error | 0.49 s |
| Events lasting 1–2 s found | 65% |
| Events under 1 s found | 19% |

The end-to-end `balanced` path (segmentation + subdivision + labeling) was last measured on the previous prompt: F1 0.629, 40.5% of boundaries within 0.5 s, label accuracy 0.72 on matched segments. Treat the table as the segmentation ceiling and those figures as the floor. Every number was checked against run-to-run noise with a paired bootstrap. The full record, including what did not work, is in the repository's `docs/`.

## Limitations

- **Short events.** Events lasting 1–2 s are found 65% of the time, events under 1 s 19%. A missed event is almost always merged into its neighbour. Timestamps are reported at sub-second resolution; the median boundary error is 0.49 s, so they are not accurate to the sub-second.
- **Label accuracy varies by robot.** Good on Galaxea and egocentric video, weaker on DROID, whose reference labels read like task instructions.
- **No human review.** The labels are generated automatically. Treat them as a first pass over a corpus. They are not ground truth.
- **Confidence is derived from stage disagreement.** `confidence` and `flags` come from disagreement between pipeline stages. The model is not asked how sure it is. Filter to unflagged segments for higher precision.

## Data handling

The video is decoded into frames on this Space, sent to the Gemini API under **your** key, and deleted when the call returns. The Space stores no video, frame, key or annotation. Google's terms for your account apply to what Gemini receives. Hugging Face may keep request logs under its own policy; review that policy before uploading unreleased footage.

## Links

- Source and evaluation code: [github.com/Manda-Robotics/robot-episode-labeler](https://github.com/Manda-Robotics/robot-episode-labeler) (Apache-2.0)
- Hosted API: [replicate.com/mandarobotics/robot-episode-labeler](https://replicate.com/mandarobotics/robot-episode-labeler)
- Example clips: DROID via `lerobot/droid_1.0.1` (Apache-2.0; DROID CC-BY-4.0).

Built by [Manda Robotics](https://github.com/Manda-Robotics).
