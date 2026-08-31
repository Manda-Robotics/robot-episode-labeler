Based on the preceding video clip, check whether one stretch of a robot episode
contains more than one completed manipulation subtask.

{instruction_block}

The clip is sampled at {fps} frames per second and covers {span:.2f} seconds of
the episode (episode time {start:.2f}s to {end:.2f}s), currently recorded as a
single subtask: "{label}". Report every timestamp in seconds RELATIVE TO THE
START OF THIS CLIP (0.00 is the first frame, {span:.2f} is the last), with two
decimals.

Short events hide at coarse sampling. Look specifically for a SEQUENCE of separate
completed events here, for example: picking an object up and then placing it
(two subtasks); releasing one object and then picking up another; placing an
object and then closing the container; a failed grasp followed by a successful one.

Split this stretch ONLY where the world state genuinely changes more than once.
Do NOT split for approach, retreat, regrasping, hesitation, or slow motion toward
a single goal.

Rules:
- Return the subtasks that make up this stretch, in time order.
- If it really is one event, return exactly one segment covering 0.00 to {span:.2f}.
- Segments must be contiguous, non-overlapping, and stay within 0.00 to {span:.2f}.
- Place each boundary at the moment the first event's world-state change has
  just completed.
{vocabulary}
Return JSON.
