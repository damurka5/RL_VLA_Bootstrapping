#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-smolvla}"
STAGE="${STAGE:-rl}"
WALLTIME="${WALLTIME:-none}"
DRY_RUN="${DRY_RUN:-0}"

RUN_COMPLEX="${RUN_COMPLEX:-1}"
RUN_SMOOTH="${RUN_SMOOTH:-1}"
COMPLEX_GPU="${COMPLEX_GPU:-0}"
SMOOTH_GPU="${SMOOTH_GPU:-1}"
SMOLVLA_NPROC_PER_NODE="${SMOLVLA_NPROC_PER_NODE:-1}"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

checkpoint_step() {
  local checkpoint="${1%/}"
  local base
  while [[ -n "$checkpoint" && "$checkpoint" != "/" && "$checkpoint" != "." ]]; do
    base="${checkpoint##*/}"
    if [[ "$base" =~ ^step_([0-9]+)$ ]]; then
      printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
      return 0
    fi
    checkpoint="${checkpoint%/*}"
  done
  return 1
}

target_train_steps() {
  local checkpoint="$1"
  local extra_steps="$2"
  local explicit_target="$3"
  local checkpoint_step_value

  if [[ -n "$explicit_target" ]]; then
    printf '%s\n' "$explicit_target"
    return 0
  fi
  if ! checkpoint_step_value="$(checkpoint_step "$checkpoint")"; then
    echo "Could not infer checkpoint step from $checkpoint; set an explicit max train step target." >&2
    return 2
  fi
  printf '%d\n' "$((checkpoint_step_value + extra_steps))"
}

COMPLEX_RUN_NAME="${COMPLEX_RUN_NAME:-cdpr_smolvla_stage2_complex_strict_resume_step_3000000_${timestamp}}"
SMOOTH_RUN_NAME="${SMOOTH_RUN_NAME:-cdpr_smolvla_smooth_strict_resume_step_3500000_${timestamp}}"

COMPLEX_CONFIG_PATH="${COMPLEX_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_stage2_complex_resume_1500000.yaml}"
SMOOTH_CONFIG_PATH="${SMOOTH_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_smooth_refinement_resume_1500000.yaml}"

COMPLEX_RESUME_CHECKPOINT_DEFAULT="/root/repo/RL_VLA_Bootstrapping/runs/cdpr_smolvla_stage2_complex_resume_1500000_20260706_220107/rl/step_3000000/smolvla_cdpr_adapter.pt"
SMOOTH_RESUME_CHECKPOINT_DEFAULT="/root/repo/RL_VLA_Bootstrapping/runs/cdpr_smolvla_smooth_refinement_resume_1500000_20260706_220107/rl/step_3500000/smolvla_cdpr_adapter.pt"
COMPLEX_RESUME_CHECKPOINT="${COMPLEX_RESUME_CHECKPOINT:-${RESUME_CHECKPOINT:-$COMPLEX_RESUME_CHECKPOINT_DEFAULT}}"
SMOOTH_RESUME_CHECKPOINT="${SMOOTH_RESUME_CHECKPOINT:-${RESUME_CHECKPOINT:-$SMOOTH_RESUME_CHECKPOINT_DEFAULT}}"
COMPLEX_EXTRA_TRAIN_STEPS="${COMPLEX_EXTRA_TRAIN_STEPS:-2000000}"
SMOOTH_EXTRA_TRAIN_STEPS="${SMOOTH_EXTRA_TRAIN_STEPS:-2000000}"
COMPLEX_MAX_TRAIN_STEPS="$(target_train_steps "$COMPLEX_RESUME_CHECKPOINT" "$COMPLEX_EXTRA_TRAIN_STEPS" "${COMPLEX_MAX_TRAIN_STEPS:-}")"
SMOOTH_MAX_TRAIN_STEPS="$(target_train_steps "$SMOOTH_RESUME_CHECKPOINT" "$SMOOTH_EXTRA_TRAIN_STEPS" "${SMOOTH_MAX_TRAIN_STEPS:-}")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

if [[ "$RUN_COMPLEX" == "1" && "$RUN_SMOOTH" == "1" && "$COMPLEX_RUN_NAME" == "$SMOOTH_RUN_NAME" ]]; then
  echo "COMPLEX_RUN_NAME and SMOOTH_RUN_NAME must be different." >&2
  exit 2
fi

python_cmd() {
  if [[ "$ENV_NAME" == "none" || -z "$ENV_NAME" ]]; then
    printf '%s\0' python3
  else
    printf '%s\0' conda run --no-capture-output -n "$ENV_NAME" python3
  fi
}

run_experiment() {
  local label="$1"
  local gpu="$2"
  local config_path="$3"
  local run_name="$4"
  local resume_checkpoint="$5"
  local max_train_steps="$6"
  local run_dir="$REPO_ROOT/runs/$run_name"
  local log_path="$run_dir/train.log"
  local cmd=()
  shift 6

  mkdir -p "$run_dir"
  while IFS= read -r -d '' part; do
    cmd+=("$part")
  done < <(python_cmd)
  cmd+=(
    -m rl_vla_bootstrapping.cli.train
    --config "$config_path"
    --stage "$STAGE"
    --run-name "$run_name"
    --execute
  )
  cmd+=("$@")

  {
    printf '%s SmolVLA run directory: %s\n' "$label" "$run_dir"
    printf 'Config: %s\n' "$config_path"
    printf 'GPU: CUDA_VISIBLE_DEVICES=%s\n' "$gpu"
    printf 'Resume checkpoint: %s\n' "$resume_checkpoint"
    printf 'Max train steps: %s\n' "$max_train_steps"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      return 0
    fi
    if [[ -n "$WALLTIME" && "$WALLTIME" != "none" ]]; then
      if command -v timeout >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="$gpu" \
          RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
          RLVLA_SMOLVLA_RESUME_CHECKPOINT="$resume_checkpoint" \
          RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$max_train_steps" \
          timeout "$WALLTIME" "${cmd[@]}"
      else
        echo "timeout is unavailable; running without a walltime guard" >&2
        CUDA_VISIBLE_DEVICES="$gpu" \
          RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
          RLVLA_SMOLVLA_RESUME_CHECKPOINT="$resume_checkpoint" \
          RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$max_train_steps" \
          "${cmd[@]}"
      fi
    else
      CUDA_VISIBLE_DEVICES="$gpu" \
        RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
        RLVLA_SMOLVLA_RESUME_CHECKPOINT="$resume_checkpoint" \
        RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$max_train_steps" \
        "${cmd[@]}"
    fi
  } 2>&1 | tee "$log_path"
}

pids=()
names=()

stop_children() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_children INT TERM

if [[ "$RUN_COMPLEX" == "1" ]]; then
  run_experiment \
    complex \
    "$COMPLEX_GPU" \
    "$COMPLEX_CONFIG_PATH" \
    "$COMPLEX_RUN_NAME" \
    "$COMPLEX_RESUME_CHECKPOINT" \
    "$COMPLEX_MAX_TRAIN_STEPS" \
    "$@" &
  pids+=("$!")
  names+=("complex")
fi

if [[ "$RUN_SMOOTH" == "1" ]]; then
  run_experiment \
    smooth \
    "$SMOOTH_GPU" \
    "$SMOOTH_CONFIG_PATH" \
    "$SMOOTH_RUN_NAME" \
    "$SMOOTH_RESUME_CHECKPOINT" \
    "$SMOOTH_MAX_TRAIN_STEPS" \
    "$@" &
  pids+=("$!")
  names+=("smooth")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "Nothing to run: RUN_COMPLEX=$RUN_COMPLEX RUN_SMOOTH=$RUN_SMOOTH" >&2
  exit 2
fi

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished ${names[$index]} successfully."
  else
    code="$?"
    echo "${names[$index]} failed with exit code $code." >&2
    status="$code"
  fi
done

exit "$status"
