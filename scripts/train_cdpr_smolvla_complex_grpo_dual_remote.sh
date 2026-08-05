#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-smolvla}"
CHECKPOINT="${CHECKPOINT:-/root/repo/RL_VLA_Bootstrapping/runs/cdpr_smolvla_stage3_object_dense_complex_resume_step_5000000_to_10000000_20260710_193100/rl/step_6700000}"
START_STEP="${START_STEP:-6700000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000000}"
STAGE="${STAGE:-rl}"
DRY_RUN="${DRY_RUN:-0}"
WALLTIME="${WALLTIME:-none}"
RUN_REVERSE="${RUN_REVERSE:-1}"
RUN_LCHOL="${RUN_LCHOL:-0}"
REVERSE_GPU="${REVERSE_GPU:-0,1}"
LCHOL_GPU="${LCHOL_GPU:-1}"
SMOLVLA_NPROC_PER_NODE="${SMOLVLA_NPROC_PER_NODE:-2}"
WRAPPER_CACHE_REFRESH_MODE="${RLVLA_CDPR_WRAPPER_CACHE_REFRESH:-force}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
REVERSE_RUN_NAME="${REVERSE_RUN_NAME:-cdpr_smolvla_complex_reverse_frontier_step_${START_STEP}_to_${MAX_TRAIN_STEPS}_${timestamp}}"
LCHOL_RUN_NAME="${LCHOL_RUN_NAME:-cdpr_smolvla_complex_lchol_hindsight_step_${START_STEP}_to_${MAX_TRAIN_STEPS}_${timestamp}}"
REVERSE_CONFIG="${REVERSE_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_complex_reverse_frontier_grpo.yaml}"
LCHOL_CONFIG="${LCHOL_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_complex_lchol_hindsight_grpo.yaml}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_CDPR_OFFSCREEN_WIDTH="${RLVLA_CDPR_OFFSCREEN_WIDTH:-320}"
export RLVLA_CDPR_OFFSCREEN_HEIGHT="${RLVLA_CDPR_OFFSCREEN_HEIGHT:-240}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

python_command() {
  if [[ -z "$ENV_NAME" || "$ENV_NAME" == "none" ]]; then
    printf '%s\0' python3
  else
    printf '%s\0' conda run --no-capture-output -n "$ENV_NAME" python3
  fi
}

if [[ "$DRY_RUN" != "1" ]]; then
  huggingface_public_models_preflight "$ENV_NAME"
  printf '[smolvla-grpo] wrapper cache refresh: %s\n' "$WRAPPER_CACHE_REFRESH_MODE"
  refresh_cmd=()
  while IFS= read -r -d '' part; do
    refresh_cmd+=("$part")
  done < <(python_command)
  "${refresh_cmd[@]}" scripts/refresh_cdpr_wrapper_cache.py \
    --repo-root "$REPO_ROOT" \
    --mode "$WRAPPER_CACHE_REFRESH_MODE"
fi

run_experiment() {
  local label="$1"
  local gpu="$2"
  local config="$3"
  local run_name="$4"
  local run_dir="$REPO_ROOT/runs/$run_name"
  local log_path="$run_dir/train.log"
  local cmd=()

  mkdir -p "$run_dir"
  while IFS= read -r -d '' part; do
    cmd+=("$part")
  done < <(python_command)
  cmd+=(
    -m rl_vla_bootstrapping.cli.train
    --config "$config"
    --stage "$STAGE"
    --run-name "$run_name"
    --execute
  )

  {
    printf '[%s] run directory: %s\n' "$label" "$run_dir"
    printf '[%s] config: %s\n' "$label" "$config"
    printf '[%s] GPU: %s\n' "$label" "$gpu"
    printf '[%s] checkpoint: %s\n' "$label" "$CHECKPOINT"
    printf '[%s] target global step: %s\n' "$label" "$MAX_TRAIN_STEPS"
    printf '[%s] offscreen render size: %sx%s\n' "$label" "$RLVLA_CDPR_OFFSCREEN_WIDTH" "$RLVLA_CDPR_OFFSCREEN_HEIGHT"
    printf '[%s] command:' "$label"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == "1" ]]; then
      return 0
    fi

    if [[ -n "$WALLTIME" && "$WALLTIME" != "none" ]] && command -v timeout >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="$gpu" \
        RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
        RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT" \
        RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
        timeout "$WALLTIME" "${cmd[@]}"
    else
      CUDA_VISIBLE_DEVICES="$gpu" \
        RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
        RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT" \
        RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
        "${cmd[@]}"
    fi
  } 2>&1 | tee "$log_path"
}

pids=()
labels=()

stop_children() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_children INT TERM

if [[ "$RUN_REVERSE" == "1" ]]; then
  run_experiment reverse_frontier "$REVERSE_GPU" "$REVERSE_CONFIG" "$REVERSE_RUN_NAME" &
  pids+=("$!")
  labels+=("reverse_frontier")
fi

if [[ "$RUN_LCHOL" == "1" ]]; then
  run_experiment lchol_hindsight "$LCHOL_GPU" "$LCHOL_CONFIG" "$LCHOL_RUN_NAME" &
  pids+=("$!")
  labels+=("lchol_hindsight")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "Nothing selected: RUN_REVERSE=$RUN_REVERSE RUN_LCHOL=$RUN_LCHOL" >&2
  exit 2
fi

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '[%s] finished successfully\n' "${labels[$index]}"
  else
    code="$?"
    printf '[%s] failed with exit code %s\n' "${labels[$index]}" "$code" >&2
    status="$code"
  fi
done

printf 'TensorBoard: tensorboard --logdir %q --port 6006\n' "$REPO_ROOT/runs"
if [[ "$RUN_REVERSE" == "1" ]]; then
  printf 'Reverse logdir: %s\n' "$REPO_ROOT/runs/$REVERSE_RUN_NAME/rl/tensorboard"
fi
if [[ "$RUN_LCHOL" == "1" ]]; then
  printf 'LC-HOL++ logdir: %s\n' "$REPO_ROOT/runs/$LCHOL_RUN_NAME/rl/tensorboard"
fi
exit "$status"
