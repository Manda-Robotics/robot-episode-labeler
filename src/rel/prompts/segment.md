You are annotating a robot manipulation episode for a robotics dataset.

You are shown contact sheets. Each sheet tiles frames sampled every {interval}s in
time order, left to right, top to bottom. Each frame is stamped with its episode
time in seconds. Those stamps are the only valid source of timestamps.

TASK BEING PERFORMED: {instruction}

Episode duration: {duration:.2f} seconds.

Segment the episode into completed manipulation subtasks.

Start a new segment when an action produces a meaningful change in the state of
the world:
- an object becomes grasped, or is released
- an object arrives at a new task-relevant location
- a container, lid, door or drawer changes state
- a tool completes its effect on an object
- an item is transferred between hands, arms or supports

Do NOT start a new segment for motion that leaves the world unchanged:
- approaching an object, or retreating from one
- adjusting or re-seating a grasp on an object already held
- hesitation, pausing, or slow repositioning
- camera movement
- small corrective motions toward the same goal

Rules:
- Segments must be in time order, non-overlapping, and contiguous: each segment
  starts where the previous one ended.
- The first segment starts at 0.00.
- The LAST segment ends when the last meaningful manipulation is complete. This is
  usually BEFORE the end of the video. Video after the final manipulation --
  the arm retreating, returning home, or holding still -- belongs to no segment.
  Do not stretch the last segment to {duration:.2f} to fill the episode, and do
  not add a final segment for the retreat.
- Prefer few, meaningful segments over many small ones. A typical episode has
  between 2 and 12 segments.
- Every timestamp must be a real moment in the episode, between 0.00 and {duration:.2f}.
{vocabulary}
Return the segments as JSON.
