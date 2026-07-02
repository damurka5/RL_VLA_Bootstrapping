#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
RUN_OCTO="${RUN_OCTO:-1}"
RUN_SMOLVLA="${RUN_SMOLVLA:-1}"
timestamp="$(date +%Y%m%d_%H%M%S)"

cd "$REPO_ROOT"

if [[ "$RUN_OCTO" == "1" ]]; then
  OCTO_RUN_NAME="${OCTO_RUN_NAME:-cdpr_octo_small_dense_staged_${timestamp}}"
  OCTO_CONFIG_PATH="${OCTO_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_octo_small_dense_simple.yaml}"
  OCTO_ENV_NAME="${OCTO_ENV_NAME:-octo}"
  OCTO_WALLTIME="${OCTO_WALLTIME:-${WALLTIME:-24h}}"
  echo "Starting Octo dense staged training: $OCTO_RUN_NAME"
  CONFIG_PATH="$OCTO_CONFIG_PATH" \
    ENV_NAME="$OCTO_ENV_NAME" \
    RUN_NAME="$OCTO_RUN_NAME" \
    WALLTIME="$OCTO_WALLTIME" \
    bash "$REPO_ROOT/scripts/train_cdpr_octo_small_dense_remote.sh" "$@"
fi

if [[ "$RUN_SMOLVLA" == "1" ]]; then
  SMOLVLA_RUN_NAME="${SMOLVLA_RUN_NAME:-cdpr_smolvla_dense_2gpu_staged_${timestamp}}"
  SMOLVLA_CONFIG_PATH="${SMOLVLA_CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_dense_2gpu.yaml}"
  SMOLVLA_ENV_NAME="${SMOLVLA_ENV_NAME:-smolvla}"
  SMOLVLA_WALLTIME="${SMOLVLA_WALLTIME:-${WALLTIME:-24h}}"
  echo "Starting SmolVLA dense staged training: $SMOLVLA_RUN_NAME"
  CONFIG_PATH="$SMOLVLA_CONFIG_PATH" \
    ENV_NAME="$SMOLVLA_ENV_NAME" \
    RUN_NAME="$SMOLVLA_RUN_NAME" \
    WALLTIME="$SMOLVLA_WALLTIME" \
    bash "$REPO_ROOT/scripts/train_cdpr_smolvla_dense_2gpu_remote.sh" "$@"
fi

echo "Finished requested dense staged training sequence."
