#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
RUN_OCTO="${RUN_OCTO:-1}"
RUN_SMOLVLA="${RUN_SMOLVLA:-1}"
OCTO_GPU="${OCTO_GPU:-0}"
SMOLVLA_GPU="${SMOLVLA_GPU:-1}"
timestamp="$(date +%Y%m%d_%H%M%S)"

checkpoint_step() {
  local checkpoint="$1"
  local base="${checkpoint%/}"
  base="${base##*/}"
  if [[ "$base" =~ ^step_([0-9]+)$ ]]; then
    printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
    return 0
  fi
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

OCTO_RUN_NAME="${OCTO_RUN_NAME:-cdpr_octo_small_dense_staged_resume_step_1000000_${timestamp}}"
OCTO_CONFIG_PATH="${OCTO_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_octo_small_dense_simple.yaml}"
OCTO_ENV_NAME="${OCTO_ENV_NAME:-octo}"
OCTO_WALLTIME="${OCTO_WALLTIME:-${WALLTIME:-24h}}"
OCTO_NPROC_PER_NODE="${OCTO_NPROC_PER_NODE:-1}"
OCTO_RESUME_CHECKPOINT="${OCTO_RESUME_CHECKPOINT:-$REPO_ROOT/runs/cdpr_octo_small_dense_staged_20260704_135252/rl/step_1000000}"
OCTO_EXTRA_TRAIN_STEPS="${OCTO_EXTRA_TRAIN_STEPS:-1000000}"
OCTO_MAX_TRAIN_STEPS="$(target_train_steps "$OCTO_RESUME_CHECKPOINT" "$OCTO_EXTRA_TRAIN_STEPS" "${OCTO_MAX_TRAIN_STEPS:-}")"

SMOLVLA_RUN_NAME="${SMOLVLA_RUN_NAME:-cdpr_smolvla_dense_2gpu_staged_resume_step_0500000_${timestamp}}"
SMOLVLA_CONFIG_PATH="${SMOLVLA_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_dense_2gpu.yaml}"
SMOLVLA_ENV_NAME="${SMOLVLA_ENV_NAME:-smolvla}"
SMOLVLA_WALLTIME="${SMOLVLA_WALLTIME:-${WALLTIME:-24h}}"
SMOLVLA_NPROC_PER_NODE="${SMOLVLA_NPROC_PER_NODE:-1}"
SMOLVLA_RESUME_CHECKPOINT="${SMOLVLA_RESUME_CHECKPOINT:-$REPO_ROOT/runs/cdpr_smolvla_dense_2gpu_staged_20260704_135252/rl/step_0500000}"
SMOLVLA_EXTRA_TRAIN_STEPS="${SMOLVLA_EXTRA_TRAIN_STEPS:-1000000}"
SMOLVLA_MAX_TRAIN_STEPS="$(target_train_steps "$SMOLVLA_RESUME_CHECKPOINT" "$SMOLVLA_EXTRA_TRAIN_STEPS" "${SMOLVLA_MAX_TRAIN_STEPS:-}")"

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
  echo "  max train steps: $OCTO_MAX_TRAIN_STEPS"
  (
    CONFIG_PATH="$OCTO_CONFIG_PATH" \
      ENV_NAME="$OCTO_ENV_NAME" \
      RUN_NAME="$OCTO_RUN_NAME" \
      WALLTIME="$OCTO_WALLTIME" \
      CUDA_VISIBLE_DEVICES="$OCTO_GPU" \
      RLVLA_OCTO_NPROC_PER_NODE="$OCTO_NPROC_PER_NODE" \
      RLVLA_OCTO_MAX_TRAIN_STEPS="$OCTO_MAX_TRAIN_STEPS" \
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
  echo "  max train steps: $SMOLVLA_MAX_TRAIN_STEPS"
  (
    CONFIG_PATH="$SMOLVLA_CONFIG_PATH" \
      ENV_NAME="$SMOLVLA_ENV_NAME" \
      RUN_NAME="$SMOLVLA_RUN_NAME" \
      WALLTIME="$SMOLVLA_WALLTIME" \
      CUDA_VISIBLE_DEVICES="$SMOLVLA_GPU" \
      RLVLA_SMOLVLA_NPROC_PER_NODE="$SMOLVLA_NPROC_PER_NODE" \
      RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$SMOLVLA_MAX_TRAIN_STEPS" \
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
