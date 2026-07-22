#!/usr/bin/env bash
# Resume an interrupted move-to phase from a numbered GRPO checkpoint.
#
# The scratch launcher deliberately refuses CHECKPOINT (its contract is "start
# at global step zero"). This variant is the counterpart for continuing a run
# that died mid-phase, e.g. after a Warp out-of-memory abort. MAX_TRAIN_STEPS
# stays the phase target (5,000,000): the trainer restores global_step from the
# checkpoint and trains the remainder.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000000}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
# 256, not 512: the 512-world SmolVLA inference activations are the
# dominant GPU peak, and at 512 the combined PyTorch+Warp footprint sat
# on the card limit and Warp OOM'd mid-run once the LoRA backward was
# added. 256 halves that activation peak at negligible throughput cost.
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ALLOW_LEGACY_SIMULATOR_CHECKPOINT="${ALLOW_LEGACY_SIMULATOR_CHECKPOINT:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"
CHECKPOINT="${CHECKPOINT:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-cdpr_smolvla_move_to_resume_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"

if [[ -z "$CHECKPOINT" ]]; then
  echo "CHECKPOINT is required; use the scratch launcher to start from step zero." >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Resume adapter not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ "$WORLDS_PER_RANK" -lt 8 || $((WORLDS_PER_RANK % 8)) -ne 0 ]]; then
  echo "WORLDS_PER_RANK must be a positive multiple of the GRPO group size (8)." >&2
  exit 2
fi
if [[ "$SMOLVLA_MICROBATCH_SIZE" -lt 1 || "$SMOLVLA_MICROBATCH_SIZE" -gt "$WORLDS_PER_RANK" ]]; then
  echo "SMOLVLA_MICROBATCH_SIZE must be in [1, WORLDS_PER_RANK]." >&2
  exit 2
fi
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "Exactly two CUDA devices are required; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "MJLab move-to config not found: $CONFIG" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
# garbage_collection_threshold lets the caching allocator hand blocks back
# before it starves Warp's separate CUDA allocator.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_SMOLVLA_NPROC_PER_NODE=2
export RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
export RLVLA_MJWARP_WORLDS_PER_RANK="$WORLDS_PER_RANK"
export RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE"
export RLVLA_SMOLVLA_ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT"
export RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
train_cmd=(
  "${python_cmd[@]}"
  -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG"
  --stage rl
  --run-name "$RUN_NAME"
  --execute
)

printf 'mode=resume\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'tensorboard_dir=%s\n' "$RUN_DIR/rl/tensorboard"
printf 'validation_metrics=%s\n' "$RUN_DIR/rl/validation.jsonl"
printf 'config=%s\n' "$CONFIG"
printf 'max_train_steps=%s (phase target; start step comes from the checkpoint)\n' "$MAX_TRAIN_STEPS"
printf 'cuda_visible_devices=%s ranks=2\n' "$CUDA_VISIBLE_DEVICES"
printf 'worlds_per_rank=%s groups_per_rank=%s server_worlds=%s\n' \
  "$WORLDS_PER_RANK" "$((WORLDS_PER_RANK / 8))" "$((2 * WORLDS_PER_RANK))"
printf 'smolvla_inference_microbatch_size=%s\n' "$SMOLVLA_MICROBATCH_SIZE"
printf 'command:'
printf ' %q' "${train_cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

cd "$REPO_ROOT"
if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  huggingface_public_models_preflight "$ENV_NAME"
  "${python_cmd[@]}" scripts/preflight_cdpr_mjlab.py \
    --config "$CONFIG" \
    --require-gpus 2 \
    --worlds "$WORLDS_PER_RANK" \
    --output "$RUN_DIR/preflight.json"
fi

"${train_cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
