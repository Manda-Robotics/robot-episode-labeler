You are placing one boundary between two manipulation subtasks precisely.

You are shown a contact sheet of frames sampled every {interval}s around a
candidate boundary. Each frame is stamped with its episode time in seconds.

The subtask that ENDS at this boundary: {before}
The subtask that BEGINS at this boundary: {after}

The current estimate for the boundary is {candidate:.2f}s.

Choose the timestamp at which the first subtask is complete and the second
begins. The boundary is the moment the world-state change that defines the
first subtask has finished, not the moment the robot begins moving toward it.

You must choose one of the timestamps stamped on the frames shown, between
{low:.2f} and {high:.2f}. If the current estimate is already the best of them,
return it unchanged.

Return JSON: {{"boundary_seconds": number, "reason": "<one short sentence>"}}
