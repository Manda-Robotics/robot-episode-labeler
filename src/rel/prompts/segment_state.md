You are annotating a robot manipulation episode for a robotics dataset, one frame
at a time.

You are shown contact sheets. Each sheet tiles frames sampled every {interval}s in
time order, left to right, top to bottom. Each frame is stamped with its episode
time in seconds.

{instruction_block}

Episode duration: {duration:.2f} seconds.
{window_block}
For EVERY stamped frame, in order, report one row:
- t: the timestamp stamped on that frame (copy it exactly)
- held: what the gripper or hand is holding in that frame ("none" if empty)
- subtask: the manipulation subtask in progress in that frame, or "none" if no
  subtask is in progress (before the first reach, after the last manipulation
  is complete, or while idle).

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
Return JSON with one row per frame, in time order.
