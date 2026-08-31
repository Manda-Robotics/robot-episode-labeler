Based on the preceding video clip, place one boundary between two manipulation
subtasks precisely.

The clip is sampled at {fps} frames per second and covers {span:.2f} seconds
around a candidate boundary. Report the timestamp in seconds RELATIVE TO THE
START OF THIS CLIP (0.00 is the first frame, {span:.2f} is the last), with two
decimals, from the clip's own timeline.

The subtask that ENDS at this boundary: {before}
The subtask that BEGINS at this boundary: {after}

The current estimate for the boundary is {candidate:.2f}s into the clip.

Choose the moment at which the first subtask's world-state change has just
completed -- the object is securely held off its support, or has been released
at its destination, or the container has finished changing state -- and the
second subtask begins. Not the moment the robot starts moving toward it.

If the current estimate is already the best moment, return it unchanged.

Return JSON: {{"boundary_seconds": number, "reason": "<one short sentence>"}}
