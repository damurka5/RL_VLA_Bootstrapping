#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_dense_2gpu.yaml}"
ENV_NAME="${ENV_NAME:-smolvla}"
STAGE="${STAGE:-rl}"
WALLTIME="${WALLTIME:-24h}"

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-cdpr_smolvla_dense_2gpu_${timestamp}}"
run_dir="$REPO_ROOT/runs/$RUN_NAME"
mkdir -p "$run_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
export PYTHONUNBUFFERED=1

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  export RLVLA_SMOLVLA_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
fi

cd "$REPO_ROOT"

cmd=(
  conda run --no-capture-output -n "$ENV_NAME"
  python3 -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG_PATH"
  --stage "$STAGE"
  --run-name "$RUN_NAME"
  --execute
)
cmd+=("$@")

{
  printf 'SmolVLA dense CDPR run directory: %s\n' "$run_dir"
  printf 'CUDA_VISIBLE_DEVICES=%s MUJOCO_GL=%s PYTORCH_CUDA_ALLOC_CONF=%s\n' \
    "${CUDA_VISIBLE_DEVICES:-}" "${MUJOCO_GL:-}" "${PYTORCH_CUDA_ALLOC_CONF:-}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  fi
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ -n "$WALLTIME" && "$WALLTIME" != "none" ]]; then
    if command -v timeout >/dev/null 2>&1; then
      timeout "$WALLTIME" "${cmd[@]}"
    else
      echo "timeout is unavailable; running without a walltime guard" >&2
      "${cmd[@]}"
    fi
  else
    "${cmd[@]}"
  fi
} 2>&1 | tee "$run_dir/train.log"
