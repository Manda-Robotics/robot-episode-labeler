You are annotating a robot manipulation episode for a robotics dataset, one frame
at a time.

You are shown contact sheets. Each sheet tiles frames sampled every {interval}s in
time order, left to right, top to bottom. Each frame is stamped with its episode
time in seconds.

{instruction_block}

Episode duration: {duration:.2f} seconds.
{window_block}
Go through the stamped frames in order and decide, for each frame, which
manipulation subtask is in progress (or "none": before the first reach, after
the last manipulation is complete, or while idle). Then report the result as
runs: one row for each frame where the subtask CHANGES from the previous frame,
giving
- t: the timestamp stamped on the FIRST frame of the new run (copy it exactly
  from a stamp; never invent a time between stamps)
- subtask: the subtask in progress from that frame on, or "none".
The first row is for the first frame. Consecutive frames with the same subtask
are one run and get no extra row.

A subtask is one completed change in the state of the world. Use a separate
subtask for each of these, and use the same short phrase for every frame that
belongs to the same one:
- picking up an object: from the first reach toward it until the frame where it
  is securely held off its support ("pick up the red cup")
- placing a held object: from lifting it away until the frame where it has been
  released at its destination ("place the red cup on the shelf")
- opening or closing a container, lid, door or drawer
- a tool completing its effect: wiping a surface, pouring, folding, cutting
- transferring an item between hands, arms or supports

A pick followed by a place is TWO subtasks. Every distinct object picked, and
every distinct placement, is its own subtask. Regrasping, hesitation,
approach and retreat are NOT subtasks: they belong to the subtask they serve,
or to "none" if they serve none (the arm retreating after the last placement).

The frame where a subtask's world-state change is visible as complete (the object
is held off the table; the object has been released) is the LAST frame of that
subtask. The next frame begins the next subtask, or "none".
{vocabulary}
Return JSON with the runs in time order.
