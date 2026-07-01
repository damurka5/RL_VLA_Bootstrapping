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
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONUNBUFFERED=1
export OCTO_REPO_PATH="${OCTO_REPO_PATH:-/root/repo/octo}"
export PYTHONPATH="$OCTO_REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${JAX_CLEAR_LD_LIBRARY_PATH:-1}" == "1" ]]; then
  export RLVLA_ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  LD_LIBRARY_PATH=""
fi
if [[ "${JAX_PREPEND_PIP_NVIDIA_LIBS:-1}" == "1" ]]; then
  octo_site_packages="$(conda run --no-capture-output -n "$ENV_NAME" python3 -c 'import site; print(site.getsitepackages()[0])')"
  nvidia_ld_dirs=()
  for subdir in \
    nvidia/cudnn/lib \
    nvidia/cublas/lib \
    nvidia/cuda_runtime/lib \
    nvidia/cufft/lib \
    nvidia/cusolver/lib \
    nvidia/cusparse/lib \
    nvidia/nccl/lib \
    nvidia/cuda_cupti/lib; do
    if [[ -d "$octo_site_packages/$subdir" ]]; then
      nvidia_ld_dirs+=("$octo_site_packages/$subdir")
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
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ -n "$WALLTIME" && "$WALLTIME" != "none" ]]; then
    timeout "$WALLTIME" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
} 2>&1 | tee "$run_dir/train.log"
