#!/bin/zsh
# Remaining experiment queue from the 2026-08-28 rework, in priority order.
# Each run is one full WGO-Bench pass (~$1-3 on gemini-3.7-flash at list price;
# the robotics-er-2 run ~$3). Re-runnable: completed tags are skipped.
cd "$(dirname "$0")/.."
set -a; source .env; set +a
run() { tag=$1; shift
  if [ -f results/$tag.json ]; then echo "skip $tag (done)"; return; fi
  echo "### $tag $(date +%H:%M:%S)"
  uv run python scripts/run_eval.py --tag $tag --workers 6 --resume "$@" 2>&1 | grep -v -i "AFC\|warning" | tail -14
}
S2="segment_prompt=segment_v2.md"
V2="segment_input=video,video_fps=2,segment_video_prompt=segment_video_v2.md"
# 1. replications of the two best coarse configs (decide the default)
run s_v2_rep      --quality fast --config $S2
run v_v2_rep      --quality fast --config $V2
# 2. what to stack on top of the decomposition prompt
run s_v2_sub      --quality fast --config $S2,subdivide=true
run s_v2_refv     --quality fast --config $S2,refine=true,refine_input=video
run v_v2_subv     --quality fast --config $V2,subdivide=true,subdivide_input=video
run v_v2_refv     --quality fast --config $V2,refine=true,refine_input=video
# 3. state backend follow-ups
run st_refv       --quality fast --config segment_input=state,refine=true,refine_input=video
run st_runs       --quality fast --config segment_input=state,state_output=runs,segment_state_prompt=segment_state_runs.md
# 4. remaining single-lever checks
run v_fps2_static --quality fast --config segment_input=video,video_fps=2,media_processing=static
run v_fps2_er2    --quality fast --config segment_input=video,video_fps=2,model=gemini-robotics-er-2-preview
# 5. labeling prompt A/B on fixed segments, judged
for v in v1:label.md v2:label_v2.md; do name=${v%%:*}; prompt=${v#*:}
  if [ ! -f results/lbl_$name.json ]; then
    uv run python scripts/relabel.py --source s_v2prompt --tag lbl_$name --config label_prompt=$prompt --workers 4 2>&1 | tail -1
  fi
  uv run python scripts/score_labels.py --tag lbl_$name 2>&1 | grep -v -i "AFC\|warning" | tail -5
done
echo "### RESUME QUEUE DONE $(date +%H:%M:%S)"
