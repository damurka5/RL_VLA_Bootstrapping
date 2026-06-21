#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-openvla-oft}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_openvla_grpo_complex_tasks.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/runs/cdpr_openvla_grpo_complex_tasks_20260610_141107/rl/step_0216000}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$REPO_ROOT/runs/cdpr_action_primitive_evaluations}"
EPISODES_PER_INSTRUCTION="${EPISODES_PER_INSTRUCTION:-20}"
MOVE_TO_OBJECT_EPISODES_PER_TARGET="${MOVE_TO_OBJECT_EPISODES_PER_TARGET:-20}"
MAX_RESET_ATTEMPTS="${MAX_RESET_ATTEMPTS:-10}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
checkpoint_name="$(basename "$CHECKPOINT_DIR")"
run_parent="$(basename "$(dirname "$(dirname "$CHECKPOINT_DIR")")")"
run_dir="${RUN_DIR:-$EVAL_OUTPUT_ROOT/${run_parent}_${checkpoint_name}_${timestamp}}"
mkdir -p "$run_dir"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO_ROOT"

cmd=(
  conda run --no-capture-output -n "$ENV_NAME"
  python3 -m rl_vla_bootstrapping.cli.validate_cdpr_policy
  --config "$CONFIG_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --run-dir "$run_dir"
  --instruction-types
  move_left move_right move_top move_bottom move_up move_down
  open_gripper close_gripper
  rotate_gripper_clockwise rotate_gripper_counterclockwise
  move_to_object
  --episodes-per-instruction "$EPISODES_PER_INSTRUCTION"
  --move-to-object-episodes-per-target "$MOVE_TO_OBJECT_EPISODES_PER_TARGET"
  --move-to-object-success-distance 0.025
  --directional-displacement-threshold 0.05
  --include-synonyms
  --synonyms-per-instruction 2
  --multi-object-scenes
  --min-scene-objects 3
  --max-scene-objects 4
  --stratify-move-to-object-targets
  --arbitrary-instructions-count 10
  --record-success-videos
  --record-failure-videos
  --no-record-all-success-videos
  --video-coverage instruction
  --video-action-overlay
  --max-reset-attempts "$MAX_RESET_ATTEMPTS"
  --strict-video-validation
  --no-require-complete-video-coverage
  --progress-only
)
cmd+=("$@")

{
  printf 'Primitive evaluation output: %s\n' "$run_dir"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
} 2>&1 | tee "$run_dir/evaluation.log"
