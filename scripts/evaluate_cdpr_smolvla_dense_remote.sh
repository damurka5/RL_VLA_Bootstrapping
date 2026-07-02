#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_dense_2gpu.yaml}"
ENV_NAME="${ENV_NAME:-smolvla}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$REPO_ROOT/runs/cdpr_smolvla_dense_evaluations}"
EPISODES_PER_INSTRUCTION="${EPISODES_PER_INSTRUCTION:-20}"
MOVE_TO_OBJECT_EPISODES_PER_TARGET="${MOVE_TO_OBJECT_EPISODES_PER_TARGET:-20}"
MAX_RESET_ATTEMPTS="${MAX_RESET_ATTEMPTS:-10}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  latest_file="$(find "$REPO_ROOT/runs" -path '*/cdpr_smolvla_dense_2gpu_*/rl/latest.pt' -print | sort | tail -n 1 || true)"
  if [[ -z "$latest_file" ]]; then
    latest_file="$(find "$REPO_ROOT/runs" -path '*/rl/latest.pt' -print | sort | tail -n 1 || true)"
  fi
  if [[ -z "$latest_file" ]]; then
    echo "Could not find a SmolVLA latest.pt under $REPO_ROOT/runs. Set CHECKPOINT_DIR explicitly." >&2
    exit 2
  fi
  CHECKPOINT_DIR="$(dirname "$latest_file")"
fi

if [[ ! -d "$CHECKPOINT_DIR" && ! -f "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint path does not exist: $CHECKPOINT_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
checkpoint_name="$(basename "$CHECKPOINT_DIR")"
run_parent="$(basename "$(dirname "$CHECKPOINT_DIR")")"
run_dir="${RUN_DIR:-$EVAL_OUTPUT_ROOT/${run_parent}_${checkpoint_name}_${timestamp}}"
mkdir -p "$run_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

cmd=(
  conda run --no-capture-output -n "$ENV_NAME"
  python3 -m rl_vla_bootstrapping.cli.validate_cdpr_smolvla_policy
  --config "$CONFIG_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --run-dir "$run_dir"
  --instruction-types
  move_left move_right move_top move_bottom move_up move_down
  move_to_object
  open_gripper close_gripper
  rotate_gripper_clockwise rotate_gripper_counterclockwise
  --episodes-per-instruction "$EPISODES_PER_INSTRUCTION"
  --move-to-object-episodes-per-target "$MOVE_TO_OBJECT_EPISODES_PER_TARGET"
  --move-to-object-success-distance 0.025
  --directional-displacement-threshold 0.05
  --multi-object-scenes
  --min-scene-objects 3
  --max-scene-objects 4
  --stratify-move-to-object-targets
  --max-reset-attempts "$MAX_RESET_ATTEMPTS"
  --progress-only
)
cmd+=("$@")

{
  printf 'SmolVLA primitive evaluation output: %s\n' "$run_dir"
  printf 'Checkpoint: %s\n' "$CHECKPOINT_DIR"
  printf 'Baseline to beat: overall simple success > 16.7%%, move_to_object > 9%%\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  fi
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
} 2>&1 | tee "$run_dir/evaluation.log"
