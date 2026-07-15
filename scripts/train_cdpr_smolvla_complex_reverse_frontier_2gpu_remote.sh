#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_REVERSE=1
export RUN_LCHOL=0
export REVERSE_GPU="${REVERSE_GPU:-0,1}"
export SMOLVLA_NPROC_PER_NODE="${SMOLVLA_NPROC_PER_NODE:-2}"

if [[ "$SMOLVLA_NPROC_PER_NODE" != "2" ]]; then
  echo "This launcher is tuned for two synchronized rollout ranks; got SMOLVLA_NPROC_PER_NODE=$SMOLVLA_NPROC_PER_NODE" >&2
fi

exec bash "$SCRIPT_DIR/train_cdpr_smolvla_complex_grpo_dual_remote.sh"
