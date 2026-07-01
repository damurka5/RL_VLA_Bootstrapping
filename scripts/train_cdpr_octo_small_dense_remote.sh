#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_octo_small_dense_simple.yaml}"
ENV_NAME="${ENV_NAME:-octo}"
STAGE="${STAGE:-rl}"
WALLTIME="${WALLTIME:-24h}"

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-cdpr_octo_small_dense_${timestamp}}"
run_dir="$REPO_ROOT/runs/$RUN_NAME"
mkdir -p "$run_dir"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  export RLVLA_OCTO_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
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
  printf 'Octo-Small dense CDPR run directory: %s\n' "$run_dir"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ -n "$WALLTIME" && "$WALLTIME" != "none" ]]; then
    timeout "$WALLTIME" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
} 2>&1 | tee "$run_dir/train.log"
