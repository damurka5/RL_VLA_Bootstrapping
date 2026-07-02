#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_octo_small_dense_simple.yaml}"
ENV_NAME="${ENV_NAME:-octo}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$REPO_ROOT/runs/cdpr_octo_small_dense_evaluations}"
EPISODES_PER_INSTRUCTION="${EPISODES_PER_INSTRUCTION:-20}"
MOVE_TO_OBJECT_EPISODES_PER_TARGET="${MOVE_TO_OBJECT_EPISODES_PER_TARGET:-20}"
MAX_RESET_ATTEMPTS="${MAX_RESET_ATTEMPTS:-10}"
VIDEO_ACTION_OVERLAY="${VIDEO_ACTION_OVERLAY:-1}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  latest_file="$(find "$REPO_ROOT/runs" -path '*/rl/latest.pt' -print | sort | tail -n 1 || true)"
  if [[ -z "$latest_file" ]]; then
    echo "Could not find an Octo latest.pt under $REPO_ROOT/runs. Set CHECKPOINT_DIR explicitly." >&2
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

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONUNBUFFERED=1
export OCTO_REPO_PATH="${OCTO_REPO_PATH:-/root/repo/octo}"
export PYTHONPATH="$OCTO_REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${JAX_CLEAR_LD_LIBRARY_PATH:-1}" == "1" ]]; then
  export RLVLA_ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  unset LD_LIBRARY_PATH
fi
if [[ "${JAX_PREPEND_PIP_NVIDIA_LIBS:-0}" == "1" ]]; then
  octo_site_packages="$(conda run --no-capture-output -n "$ENV_NAME" python3 -c 'import site; print(site.getsitepackages()[0])')"
  nvidia_ld_dirs=()
  for lib_dir in \
    "$octo_site_packages"/nvidia/cudnn/lib \
    "$octo_site_packages"/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
      nvidia_ld_dirs+=("$lib_dir")
    fi
  done
  if [[ "${#nvidia_ld_dirs[@]}" -gt 0 ]]; then
    nvidia_ld_path="$(IFS=:; printf '%s' "${nvidia_ld_dirs[*]}")"
    export LD_LIBRARY_PATH="$nvidia_ld_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  elif [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    unset LD_LIBRARY_PATH
  fi
fi

cd "$REPO_ROOT"

cmd=(
  conda run --no-capture-output -n "$ENV_NAME"
  python3 -m rl_vla_bootstrapping.cli.validate_cdpr_octo_policy
  --config "$CONFIG_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --run-dir "$run_dir"
  --instruction-types
  move_left move_right move_top move_bottom
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
if [[ "$VIDEO_ACTION_OVERLAY" == "1" ]]; then
  cmd+=(--video-action-overlay)
else
  cmd+=(--no-video-action-overlay)
fi
cmd+=("$@")

{
  printf 'Octo primitive evaluation output: %s\n' "$run_dir"
  printf 'Checkpoint: %s\n' "$CHECKPOINT_DIR"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
} 2>&1 | tee "$run_dir/evaluation.log"
