#!/usr/bin/env bash
# Complete bounded demo-guided GRPO pilot on the remote two-GPU machine.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
export ENV_NAME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MUJOCO_GL=egl
ARGS=(--steps "${MAX_TRAIN_STEPS:-1000000}" --demo-rounds "${DEMO_ROUNDS:-12}")
[[ -z "${WARMSTART_CHECKPOINT:-}" ]] || ARGS+=(--checkpoint "$WARMSTART_CHECKPOINT")
[[ -z "${CONFIG:-}" ]] || ARGS+=(--config "$CONFIG")
[[ -z "${PILOT_RUN_DIR:-}" ]] || ARGS+=(--run-dir "$PILOT_RUN_DIR")
[[ "${DRY_RUN:-0}" != 1 ]] || ARGS+=(--dry-run)
exec conda run --no-capture-output -n "$ENV_NAME" python3 tools/train/run_cdpr_demo_grpo_pilot.py "${ARGS[@]}"
