## Evaluation

Measured on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), 100 manually annotated robot and egocentric episodes with 743 gold subtask segments. Current default pipeline; every number was checked against run-to-run noise with a paired bootstrap and replicated where marked.

| Metric | Value |
|---|---|
| Segmentation F1 (IoU 0.5), no vocabulary | 0.68–0.73 (replicated) |
| Segmentation F1 with a caller vocabulary (schema mode) | 0.77 |
| End-to-end schema mode (segmentation + labels) | 0.70 |
| F1 at WGO-Bench's own protocol (IoU 0.75; the benchmark authors publish 0.306) | 0.40–0.48 |
| True boundaries found within 0.5 s | 49–51% |
| Median boundary error | 0.44–0.53 s |
| Label accuracy on matched segments (model-judged) | 0.80 |

Schema-mode numbers use vocabularies derived from each episode's own gold labels, which is the best case; a coarser customer SOP list will land between the two F1 rows. The full record, including confidence intervals, negative results and withdrawn claims, is in the repository's `docs/`.

## Limitations

- **Short events.** Events lasting 1–2 s are found 65% of the time, events under 1 s 19%. A missed event is almost always merged into its neighbour. Timestamps are reported at sub-second resolution; the median boundary error is 0.49 s, so they are not accurate to the sub-second.
- **Label accuracy varies by robot.** 0.89 on Galaxea, 0.84 on egocentric video, 0.59 on DROID, whose reference labels read like task instructions.
- **No human review.** The labels are generated automatically. Treat them as a first pass over a corpus. They are not ground truth.
- **Confidence is derived from stage disagreement.** `confidence` and `flags` come from disagreement between pipeline stages. The model is not asked how sure it is. Filter to unflagged segments for higher precision.

## Data handling

The video is decoded into frames on this Space, sent to the Gemini API under **your** key, and deleted when the call returns. The Space stores no video, frame, key or annotation. Google's terms for your account apply to what Gemini receives. Hugging Face may keep request logs under its own policy; review that policy before uploading unreleased footage.

## Links

- Source and evaluation code: [github.com/Manda-Robotics/robot-episode-labeler](https://github.com/Manda-Robotics/robot-episode-labeler) (Apache-2.0)
- Hosted API: [replicate.com/mandarobotics/robot-episode-labeler](https://replicate.com/mandarobotics/robot-episode-labeler)
- Example clips: DROID via `lerobot/droid_1.0.1` (Apache-2.0; DROID CC-BY-4.0).

Built by [Manda Robotics](https://github.com/Manda-Robotics).
