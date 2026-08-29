Based on the preceding video clip, annotate a robot manipulation episode for a
robotics dataset.

The clip is sampled at {fps} frames per second. It covers episode time
{lo:.2f}s to {hi:.2f}s. Report every timestamp in seconds RELATIVE TO THE START
OF THIS CLIP (0.00 is the first frame, {span:.2f} is the last), with two
decimals, from the clip's own timeline.

{instruction_block}
{window_block}
Segment the clip into completed manipulation subtasks.

Start a new segment when an action produces a meaningful change in the state of
the world:
- an object becomes grasped and lifted, or is released
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
- Place each boundary at the frame where the state change has just completed
  (the gripper has closed on the object, the object has settled at its new place).
- The first segment starts at 0.00 unless the clip begins in the middle of a
  subtask that is still in progress, in which case it starts at 0.00 anyway and
  ends when that subtask completes.
- The LAST segment ends when the last meaningful manipulation is complete. Video
  after the final manipulation -- the arm retreating, returning home, or holding
  still -- belongs to no segment. Do not add a final segment for the retreat.
- Prefer few, meaningful segments over many small ones.
{vocabulary}
Return the segments as JSON.
