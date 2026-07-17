#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-smolvla}"
CHECKPOINT="${CHECKPOINT:-/root/repo/RL_VLA_Bootstrapping/runs/cdpr_smolvla_stage3_object_dense_complex_resume_step_5000000_to_10000000_20260710_193100/rl/step_6700000}"
START_STEP="${START_STEP:-6700000}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-$((START_STEP + TRAIN_STEPS))}"
GPU="${GPU:-0}"
SMOLVLA_NPROC_PER_NODE="${SMOLVLA_NPROC_PER_NODE:-1}"
STAGE="${STAGE:-rl}"
DRY_RUN="${DRY_RUN:-0}"
WALLTIME="${WALLTIME:-none}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_strict_dense_bridge.yaml}"
WRAPPER_CACHE_REFRESH_MODE="${RLVLA_CDPR_WRAPPER_CACHE_REFRESH:-force}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-cdpr_smolvla_strict_dense_bridge_step_${START_STEP}_to_${MAX_TRAIN_STEPS}_${timestamp}}"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
LOG_PATH="$RUN_DIR/train.log"
FINAL_CHECKPOINT="$RUN_DIR/rl/step_$(printf '%07d' "$MAX_TRAIN_STEPS")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export RLVLA_CDPR_OFFSCREEN_WIDTH="${RLVLA_CDPR_OFFSCREEN_WIDTH:-320}"
export RLVLA_CDPR_OFFSCREEN_HEIGHT="${RLVLA_CDPR_OFFSCREEN_HEIGHT:-240}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR"

cmd=()
if [[ -z "$ENV_NAME" || "$ENV_NAME" == "none" ]]; then
  cmd+=(python3)
else
  cmd+=(conda run --no-capture-output -n "$ENV_NAME" python3)
fi
cmd+=(
  -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG"
  --stage "$STAGE"
  --run-name "$RUN_NAME"
  --execute
)

{
  printf '[strict-dense] run directory: %s\n' "$RUN_DIR"
  printf '[strict-dense] config: %s\n' "$CONFIG"
  printf '[strict-dense] GPU: %s (single-process)\n' "$GPU"
  printf '[strict-dense] checkpoint: %s\n' "$CHECKPOINT"
  printf '[strict-dense] requested env steps: %s\n' "$TRAIN_STEPS"
  printf '[strict-dense] global step range: %s -> %s\n' "$START_STEP" "$MAX_TRAIN_STEPS"
  printf '[strict-dense] wrapper cache refresh: %s\n' "$WRAPPER_CACHE_REFRESH_MODE"
  printf '[strict-dense] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
  fi

  huggingface_public_models_preflight "$ENV_NAME"
  if [[ -z "$ENV_NAME" || "$ENV_NAME" == "none" ]]; then
    python3 scripts/refresh_cdpr_wrapper_cache.py \
      --repo-root "$REPO_ROOT" \
      --mode "$WRAPPER_CACHE_REFRESH_MODE"
  else
    conda run --no-capture-output -n "$ENV_NAME" \
      python3 scripts/refresh_cdpr_wrapper_cache.py \
      --repo-root "$REPO_ROOT" \
      --mode "$WRAPPER_CACHE_REFRESH_MODE"
  fi

  if [[ -n "$WALLTIME" && "$WALLTIME" != "none" ]] && command -v timeout >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$GPU" \
      RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
      RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT" \
      RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
      RLVLA_SMOLVLA_NOISE_SCHEDULE_START_STEP="$START_STEP" \
      timeout "$WALLTIME" "${cmd[@]}"
  else
    CUDA_VISIBLE_DEVICES="$GPU" \
      RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
      RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT" \
      RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
      RLVLA_SMOLVLA_NOISE_SCHEDULE_START_STEP="$START_STEP" \
      "${cmd[@]}"
  fi

  printf '[strict-dense] completed checkpoint: %s\n' "$FINAL_CHECKPOINT"
  printf '[strict-dense] TensorBoard: tensorboard --logdir %q --port 6006\n' "$RUN_DIR/rl/tensorboard"
  printf '[strict-dense] next GRPO command:\n'
  printf 'START_STEP=%q CHECKPOINT=%q bash scripts/train_cdpr_smolvla_complex_grpo_dual_remote.sh\n' \
    "$MAX_TRAIN_STEPS" "$FINAL_CHECKPOINT"
} 2>&1 | tee "$LOG_PATH"
