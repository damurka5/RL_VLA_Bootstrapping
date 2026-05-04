#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m rl_vla_bootstrapping.cli.train \
  --config configs/examples/cdpr_openvla_grpo_complex_tasks.yaml \
  --stage rl \
  --execute
