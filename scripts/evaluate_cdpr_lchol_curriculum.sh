#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR [CONFIG] [extra validator args...]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
shift

CONFIG="configs/examples/cdpr_openvla_grpo_complex_tasks.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi

"${PYTHON:-python3}" -m rl_vla_bootstrapping.cli.validate_cdpr_policy \
  --config "$CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --record-success-videos \
  --record-all-success-videos \
  --record-failure-videos \
  --run-name cdpr_lchol_curriculum_eval \
  "$@"
