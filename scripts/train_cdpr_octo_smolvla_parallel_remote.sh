#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
RUN_OCTO="${RUN_OCTO:-1}"
RUN_SMOLVLA="${RUN_SMOLVLA:-1}"
OCTO_GPU="${OCTO_GPU:-0}"
SMOLVLA_GPU="${SMOLVLA_GPU:-1}"
timestamp="$(date +%Y%m%d_%H%M%S)"

OCTO_RUN_NAME="${OCTO_RUN_NAME:-cdpr_octo_small_dense_staged_resume_step_1000000_${timestamp}}"
OCTO_CONFIG_PATH="${OCTO_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_octo_small_dense_simple.yaml}"
OCTO_ENV_NAME="${OCTO_ENV_NAME:-octo}"
OCTO_WALLTIME="${OCTO_WALLTIME:-${WALLTIME:-24h}}"
OCTO_NPROC_PER_NODE="${OCTO_NPROC_PER_NODE:-1}"
OCTO_RESUME_CHECKPOINT="${OCTO_RESUME_CHECKPOINT:-$REPO_ROOT/runs/cdpr_octo_small_dense_staged_20260704_135252/rl/step_1000000}"

SMOLVLA_RUN_NAME="${SMOLVLA_RUN_NAME:-cdpr_smolvla_dense_2gpu_staged_resume_step_0500000_${timestamp}}"
SMOLVLA_CONFIG_PATH="${SMOLVLA_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_dense_2gpu.yaml}"
SMOLVLA_ENV_NAME="${SMOLVLA_ENV_NAME:-smolvla}"
SMOLVLA_WALLTIME="${SMOLVLA_WALLTIME:-${WALLTIME:-24h}}"
SMOLVLA_NPROC_PER_NODE="${SMOLVLA_NPROC_PER_NODE:-1}"
SMOLVLA_RESUME_CHECKPOINT="${SMOLVLA_RESUME_CHECKPOINT:-$REPO_ROOT/runs/cdpr_smolvla_dense_2gpu_staged_20260704_135252/rl/step_0500000}"

cd "$REPO_ROOT"

if [[ "$RUN_OCTO" == "1" && "$RUN_SMOLVLA" == "1" && "$OCTO_RUN_NAME" == "$SMOLVLA_RUN_NAME" ]]; then
  echo "OCTO_RUN_NAME and SMOLVLA_RUN_NAME must be different for parallel training." >&2
  exit 2
fi

pids=()
names=()

stop_children() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_children INT TERM

if [[ "$RUN_OCTO" == "1" ]]; then
  echo "Starting Octo on CUDA_VISIBLE_DEVICES=$OCTO_GPU: $OCTO_RUN_NAME"
  echo "  run dir: $REPO_ROOT/runs/$OCTO_RUN_NAME"
  echo "  resume checkpoint: $OCTO_RESUME_CHECKPOINT"
  (
    CONFIG_PATH="$OCTO_CONFIG_PATH" \
      ENV_NAME="$OCTO_ENV_NAME" \
      RUN_NAME="$OCTO_RUN_NAME" \
      WALLTIME="$OCTO_WALLTIME" \
      CUDA_VISIBLE_DEVICES="$OCTO_GPU" \
      RLVLA_OCTO_NPROC_PER_NODE="$OCTO_NPROC_PER_NODE" \
      RESUME_CHECKPOINT="$OCTO_RESUME_CHECKPOINT" \
      bash "$REPO_ROOT/scripts/train_cdpr_octo_small_dense_remote.sh" "$@"
  ) &
  pids+=("$!")
  names+=("octo")
fi

if [[ "$RUN_SMOLVLA" == "1" ]]; then
  echo "Starting SmolVLA on CUDA_VISIBLE_DEVICES=$SMOLVLA_GPU: $SMOLVLA_RUN_NAME"
  echo "  run dir: $REPO_ROOT/runs/$SMOLVLA_RUN_NAME"
  echo "  resume checkpoint: $SMOLVLA_RESUME_CHECKPOINT"
  (
    CONFIG_PATH="$SMOLVLA_CONFIG_PATH" \
      ENV_NAME="$SMOLVLA_ENV_NAME" \
      RUN_NAME="$SMOLVLA_RUN_NAME" \
      WALLTIME="$SMOLVLA_WALLTIME" \
      CUDA_VISIBLE_DEVICES="$SMOLVLA_GPU" \
      RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
      RESUME_CHECKPOINT="$SMOLVLA_RESUME_CHECKPOINT" \
      bash "$REPO_ROOT/scripts/train_cdpr_smolvla_dense_2gpu_remote.sh" "$@"
  ) &
  pids+=("$!")
  names+=("smolvla")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "Nothing to run: RUN_OCTO=$RUN_OCTO RUN_SMOLVLA=$RUN_SMOLVLA" >&2
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
