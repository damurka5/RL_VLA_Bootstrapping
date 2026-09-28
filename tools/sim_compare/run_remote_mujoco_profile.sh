#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-tools/sim_compare/out_remote_mujoco}"
: "${MUJOCO_GL:=egl}"

python tools/sim_compare/run_comparator.py \
  --backend mujoco_raw_cdpr \
  --render \
  --render-backend "${MUJOCO_GL}" \
  --camera-count 2 \
  --width 320 \
  --height 240 \
  --episodes 20 \
  --steps 1000 \
  --out "${OUT_DIR}"
