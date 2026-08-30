# Demo assets for the public model page

| File | Purpose |
|---|---|
| `droid_blue_block_in_green_bowl.mp4` | Primary example. 17.7 s. A failed placement followed by a successful one, which is the query shape the product exists to serve. |
| `droid_duvet_tip_left.mp4` | Second example. 27.3 s, deformable object, four subtasks. |
| `cover.png` | 1200x470 model cover image. |

Licensing and provenance are in `ATTRIBUTION.md`. Both clips are Apache-2.0 /
CC-BY-4.0 and usable commercially with attribution. Benchmark footage from
WGO-Bench is **not** usable here: it is CC-BY-NC-SA-4.0.

The clips were cut at episode boundaries from the source dataset's own metadata,
so each file is exactly one episode with the dataset's own task string. The
source video is AV1, which the host ffmpeg on macOS cannot decode in software;
`src/rel/video/decode.py` falls back to a bundled static build, and the same
fallback was needed to cut these files.

The exact inputs and outputs used for the model page are in `EXAMPLES.md`. Sim
clips for trying the model, not cleared for public use, are in `robolab/`.
