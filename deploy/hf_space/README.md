---
title: Robot Episode Labeler
emoji: 🦾
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: "6.26.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Robot video → timestamped subtasks with pass/fail. Bring your own Gemini key.
tags:
  - robotics
  - video-understanding
  - temporal-segmentation
  - manipulation
  - gemini
---

# Robot Episode Labeler

Robot Episode Labeler turns a robot manipulation video into timestamped subtask annotations.

```
video + task description  ->  [{start, end, label, result, attributes, description}, ...]
```

Upload an episode, describe the task in one sentence, paste a Gemini API key,
and the app returns structured labels you can query. Supplying a **subtask
vocabulary** switches on schema mode: labels are constrained to your list and
snapped to it in code. Teams with an existing SOP usually want this mode.

## Bring your own key

The Space calls the Gemini API with the key you paste in. The key is used for
that call only and is never stored. Get one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey). At 2026 list
prices, expect about $1.25 per hour of video for segmentation only (`fast`)
and about $4 per hour for `balanced`, charged to your Gemini account.

## Limits of this demo

- Episodes up to 5 minutes and 200 MB.
- Free CPU hardware. The model calls run in Gemini, so the Space does no heavy work itself.
- For batch use, run the package locally or call the
  [Replicate model](https://replicate.com/mandarobotics/robot-episode-labeler).

## How it works

1. Frames are sampled on a fixed grid (0.5 s, 224 px) and tiled into contact
   sheets with the timestamp burned into each tile.
2. One model call per 60 s window proposes segments. The prompt defines a
   boundary as a completed world-state change and states that a pick and the
   following place are two subtasks. That rule produced the largest measured
   improvement.
3. In `balanced` and `strict`, segments longer than 3 s are re-read at 0.25 s
   to find short events that coarse sampling hides.
4. Each segment is named and judged pass or fail with its neighbours as context.
5. Ordering, bounds, contiguity and closed vocabularies are enforced in code.
   Every correction is reported as a warning.

Accuracy, known failure modes and data handling are described below the form
in the app, and in full in the
[repository](https://github.com/Manda-Robotics/robot-episode-labeler)
(`docs/results.md`, `docs/research-log.md`).

Built by [Manda Robotics](https://github.com/Manda-Robotics). Apache-2.0.
