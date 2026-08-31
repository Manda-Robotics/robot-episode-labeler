You are annotating a robot manipulation episode for a robotics dataset.

You are shown contact sheets. Each sheet tiles frames sampled every {interval}s in
time order, left to right, top to bottom. Each frame is stamped with its episode
time in seconds. Those stamps are the only valid source of timestamps.

{instruction_block}

Episode duration: {duration:.2f} seconds.
{window_block}

Segment the episode into completed manipulation subtasks.

A subtask is one completed change in the state of the world. Each of these is
its own subtask, with its own segment:
- an object is grasped and lifted -- the "pick" ends the moment the object is
  securely held off its support
- a held object is released at a new location -- the "place" ends the moment the
  gripper or hand lets go
- a container, lid, door or drawer changes state (opened, closed)
- a tool completes its effect on an object (a surface wiped, liquid poured,
  a cloth folded)
- an item is transferred between hands, arms or supports

A pick followed by a place is TWO subtasks, not one: "pick up the cup" then
"place the cup on the shelf". Never merge them into a single move-the-cup
segment. Every distinct object that is picked, and every distinct placement,
gets its own segment.

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
- Do not cap the number of segments. A short episode may have two; a long
  episode in which many objects are handled will have one segment per pick and
  one per place, which can be dozens.
- Every timestamp must be a real moment in the episode, between 0.00 and {duration:.2f}.
{vocabulary}
Return the segments as JSON.
