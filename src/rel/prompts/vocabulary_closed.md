- You MUST label each segment using exactly one of these allowed labels:
{labels}
  Use the label verbatim, and do not invent new ones.
- If a stretch of the episode contains NO completed subtask from that list -- the
  robot is fumbling, repeatedly failing to grasp, pushing an object around without
  placing it, or idle -- label that stretch exactly `no_completed_subtask`.
  Do NOT stretch a real subtask to cover it, and do not force it into the closest
  allowed label. A long stretch where nothing is achieved is a real and useful
  observation; report it as one.
