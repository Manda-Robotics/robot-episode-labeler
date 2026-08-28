# RoboLab sim clips for trying the model

Five simulated episodes from Manda's own RoboLab runs, cropped to a single camera
view. Sim footage is a useful stress test because it looks nothing like the
teleoperation video the model was measured on: different lighting, textures,
camera placement, and in one case a completely different embodiment.

**Use these for trying the model, not as public demo material.** They are our own
renders, but the scenes use assets from NVlabs RoboLab, whose asset licensing we
have not checked. The DROID clips in the parent directory are the ones cleared for
the public model page.

The original files are mostly multi-camera strips (up to 2560x360). At 224 px
contact-sheet tiles a four-view strip is unusable, so each was cropped to one view.
If you try other RoboLab episodes, crop them the same way:

```bash
ffmpeg -i in.mp4 -vf "crop=640:360:0:0" -c:v libx264 -crf 22 out.mp4
```

## Inputs to paste

Attributes for all five: `retry,missed_grasp,dropped_object,knocked_over`

| video | `prompt` | `subtasks` |
|---|---|---|
| `stack_bowls.mp4` (20 s) | `Stack the right bowl on the left bowl` | `Pick Up Bowl,Stack Bowl` |
| `mugs_on_shelf.mp4` (10 s) | `Put the two mugs on the shelf` | `Pick Up Mug,Place Mug On Shelf` |
| `pack_cans.mp4` (92 s) | `Pack canned foods into the bin` | `Pick Up Item,Place Item In Bin` |
| `hammers_in_bin.mp4` (180 s) | `Put the red hammer and black hammer in the left bin` | `Pick Up Hammer,Place Hammer In Bin` |
| `aloha_transfer_cube.mp4` (30 s) | `A bimanual robot picks up a red cube with one arm and transfers it to the other arm.` | `Grasp Cube,Transfer Cube,Release Cube` |

## What each one actually returned

`pack_cans.mp4` is the most interesting. It finds a failure and tags it:

```
 0.00 -  5.00   fail   Pick Up Item
 5.00 -  8.50   fail   Place Item In Bin [dropped_object]
 8.50 - 17.12   pass   Pick Up Item
 ... 10 subtasks total
```

`hammers_in_bin.mp4`, 180 s, six subtasks, failures at both ends:

```
  0.00 -  13.52   fail   Pick Up Hammer [missed_grasp]
 13.52 -  39.75   fail   Place Hammer In Bin
 39.75 - 109.00   pass   Pick Up Hammer
109.00 - 122.75   pass   Place Hammer In Bin
122.75 - 153.00   pass   Pick Up Hammer
153.00 - 175.00   fail   Place Hammer In Bin
```

`stack_bowls.mp4` and `mugs_on_shelf.mp4` both return three clean segments.

`aloha_transfer_cube.mp4` is the honest failure case. It returns two segments
covering only the first 3.5 s of a 30 s episode, then stops:

```
0.00 - 3.00   pass   Grasp Cube
3.00 - 3.50   fail   Transfer Cube [missed_grasp]
```

Worth keeping precisely because it fails. It is a bimanual ALOHA rig, two small
arms against a black background, which is far from anything in the benchmark, and
the model loses the thread almost immediately. If we want a claim about bimanual
manipulation, this is the case to fix first.
