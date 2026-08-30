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

Turn a robot manipulation video into timestamped subtask annotations:

```
video + task description  ->  [{start, end, label, result, attributes, description}, ...]
```

Upload an episode, describe the task in a sentence, add your Gemini API key,
and get back structured, queryable labels. Supplying a **subtask vocabulary**
switches on schema mode: labels are constrained to your list and snapped to it
in code, which is what robotics teams with an existing SOP usually want.

**Bring your own key.** This Space calls the Gemini API under the key you paste
in; it is used for that call only and never stored. Get one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey). Typical cost on
your account at 2026 list prices: about $1.25 per hour of video for segmentation
only (`fast`), about $4 per hour for `balanced`.

**Limits of this demo:** episodes up to 5 minutes and 200 MB, free CPU hardware
(the work happens in Gemini, not here). For batch use, run the package locally
or call the [Replicate model](https://replicate.com/mandarobotics/robot-episode-labeler).

## How it works

1. Frames are sampled on a grid we control (0.5 s, 224 px) and tiled into
   contact sheets with the timestamp burned into each tile.
2. One model call per 60 s window proposes segments, with a prompt that defines
   boundaries as *completed world-state changes* and states that a pick and the
   following place are two subtasks — the rule that produced the largest
   measured improvement.
3. Segments longer than 3 s are re-read at 0.25 s to find short events that
   hide at coarse sampling (`balanced` and `strict`).
4. Each segment is named and judged pass/fail with its neighbours as context.
5. Ordering, bounds, contiguity and closed vocabularies are enforced in code,
   and every correction is reported as a warning.

Accuracy, known failure modes and data handling are described below the form
in the app, and in full in the
[repository](https://github.com/Manda-Robotics/robot-episode-labeler)
(`docs/results.md`, `docs/research-log.md`).

Built by [Manda](https://github.com/Manda-Robotics), a public benefit corporation
building evaluation infrastructure for robot learning. Apache-2.0.
