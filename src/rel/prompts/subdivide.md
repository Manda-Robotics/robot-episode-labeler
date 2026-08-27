You are checking whether one stretch of a robot episode contains more than one
completed manipulation subtask.

{instruction_block}

This stretch runs from {start:.2f}s to {end:.2f}s and is currently recorded as a
single subtask: "{label}"

You are shown frames sampled every {interval}s across it -- a finer view than the
one that produced the current label. Each frame is stamped with its episode time.

Short events hide at coarse sampling. Look specifically for a SEQUENCE of separate
completed events here, for example: releasing one object and then picking up
another; placing an object and then closing the container; two grasps of different
objects; a failed grasp followed by a successful one.

Split this stretch ONLY where the world state genuinely changes more than once.
Do NOT split for approach, retreat, regrasping, hesitation, or slow motion toward
a single goal.

Rules:
- Return the subtasks that make up this stretch, in time order.
- If it really is one event, return exactly one segment covering {start:.2f} to {end:.2f}.
- Segments must be contiguous, non-overlapping, and stay within {start:.2f} to {end:.2f}.
- Use the timestamps stamped on the frames.
{vocabulary}
Return JSON.
