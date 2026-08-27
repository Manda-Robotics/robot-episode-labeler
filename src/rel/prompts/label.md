You are describing one manipulation subtask from a robot episode, and judging
whether it succeeded.

TASK BEING PERFORMED: {instruction}

You are shown a contact sheet covering the segment from {start:.2f}s to {end:.2f}s,
plus a little context before and after it. Each frame is stamped with its episode
time. Judge only what happens between {start:.2f}s and {end:.2f}s; the surrounding
frames are context.

{neighbours}
Report:
- label: what completed manipulation event this segment is.{vocabulary}
- result: "pass" if the subtask completed as intended, "fail" if it did not
  (the object was dropped, missed, misplaced, or the attempt was abandoned),
  "unknown" if the frames genuinely do not show the outcome.
- attributes: zero or more tags from the allowed list describing how it went.{attributes}
- description: one sentence, plainly describing what happens.

Judge failure by what you can see. Do not infer success from the task
description. If the segment shows a grasp that slips, that is a failure even if
a later segment recovers.

Return JSON.
