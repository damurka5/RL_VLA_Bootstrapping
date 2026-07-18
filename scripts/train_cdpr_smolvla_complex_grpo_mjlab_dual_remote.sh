#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000000}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-16}"
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ALLOW_LEGACY_SIMULATOR_CHECKPOINT="${ALLOW_LEGACY_SIMULATOR_CHECKPOINT:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-cdpr_smolvla_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

if [[ "$WORLDS_PER_RANK" -lt 8 || $((WORLDS_PER_RANK % 8)) -ne 0 ]]; then
  echo "WORLDS_PER_RANK must be a positive multiple of 8." >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "MJLab config not found: $CONFIG" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_SMOLVLA_NPROC_PER_NODE=2
export RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
export RLVLA_MJWARP_WORLDS_PER_RANK="$WORLDS_PER_RANK"
export RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE"
export RLVLA_SMOLVLA_ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT"
if [[ -n "$CHECKPOINT" ]]; then
  export RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT"
fi

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
train_cmd=(
  "${python_cmd[@]}"
  -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG"
  --stage rl
  --run-name "$RUN_NAME"
  --execute
)

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'config=%s\n' "$CONFIG"
printf 'worlds_per_rank=%s groups_per_rank=%s\n' \
  "$WORLDS_PER_RANK" "$((WORLDS_PER_RANK / 8))"
printf 'command:'
printf ' %q' "${train_cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

cd "$REPO_ROOT"
if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  "${python_cmd[@]}" scripts/preflight_cdpr_mjlab.py \
    --config "$CONFIG" \
    --require-gpus 2 \
    --worlds "$WORLDS_PER_RANK" \
    --output "$RUN_DIR/preflight.json"
fi

"${train_cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
