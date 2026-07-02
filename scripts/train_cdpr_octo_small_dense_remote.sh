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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
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
  printf 'CUDA_VISIBLE_DEVICES=%s XLA_PYTHON_CLIENT_PREALLOCATE=%s XLA_PYTHON_CLIENT_MEM_FRACTION=%s\n' \
    "${CUDA_VISIBLE_DEVICES:-}" "${XLA_PYTHON_CLIENT_PREALLOCATE:-}" "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ -n "$WALLTIME" && "$WALLTIME" != "none" ]]; then
    timeout "$WALLTIME" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
} 2>&1 | tee "$run_dir/train.log"
