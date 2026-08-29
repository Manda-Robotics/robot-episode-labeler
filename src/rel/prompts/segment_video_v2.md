Based on the preceding video clip, annotate a robot manipulation episode for a
robotics dataset.

The clip is sampled at {fps} frames per second. It covers episode time
{lo:.2f}s to {hi:.2f}s. Report every timestamp in seconds RELATIVE TO THE START
OF THIS CLIP (0.00 is the first frame, {span:.2f} is the last), with two
decimals, from the clip's own timeline.

{instruction_block}
{window_block}
Segment the clip into completed manipulation subtasks.

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
- Place each boundary at the frame where the state change has just completed
  (the gripper has closed on the object, the object has settled at its new place).
- The first segment starts at 0.00 unless the clip begins in the middle of a
  subtask that is still in progress, in which case it starts at 0.00 anyway and
  ends when that subtask completes.
- The LAST segment ends when the last meaningful manipulation is complete. Video
  after the final manipulation -- the arm retreating, returning home, or holding
  still -- belongs to no segment. Do not add a final segment for the retreat.
- Do not cap the number of segments. A short episode may have two; a long
  episode in which many objects are handled will have one segment per pick and
  one per place, which can be dozens.
{vocabulary}
Return the segments as JSON.
