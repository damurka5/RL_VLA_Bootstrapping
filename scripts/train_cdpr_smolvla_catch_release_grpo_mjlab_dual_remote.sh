#!/usr/bin/env bash
# Phase 2 of the manipulation curriculum: put_into_plate + put_into_bowl with
# pick_up kept in the mix as rehearsal, warm-started WEIGHTS-ONLY from the
# phase-1 pick_up adapter.
#
# Weights-only (not --resume-checkpoint) is deliberate. A full resume would also
# restore phase 1's per-instruction approach-curriculum caps and optimizer
# moments; phase 1 never trained the placement instructions, so their caps carry
# no information, and pick_up's cap was earned under a different scene
# distribution. Weights-only keeps the learned behaviour and restarts the
# schedule at step 0.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-15000000}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
# 256, not 512. The 512-world SmolVLA inference activations are the dominant GPU
# peak; at 512 the combined PyTorch+Warp footprint sat on the A40 limit and Warp
# OOM'd mid-run once the LoRA backward was added. This env var OVERRIDES the
# config value, so it must stay at 256 for as long as train_vla_lora is true.
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ALLOW_LEGACY_SIMULATOR_CHECKPOINT="${ALLOW_LEGACY_SIMULATOR_CHECKPOINT:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline

# shellcheck source=scripts/run_naming.sh
source "$SCRIPT_DIR/run_naming.sh"
RUN_NAME="$(cdpr_compose_run_name "cdpr_smolvla_catch_release_dense_mjwarp_w${WORLDS_PER_RANK}")"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
cdpr_guard_run_dir "$RUN_DIR"

if [[ -n "${CHECKPOINT:-}" || -n "${RLVLA_SMOLVLA_RESUME_CHECKPOINT:-}" ]]; then
  echo "This launcher warm-starts weights only; do not set CHECKPOINT/RLVLA_SMOLVLA_RESUME_CHECKPOINT." >&2
  echo "Pass the phase-1 pick_up adapter as WARMSTART_CHECKPOINT instead." >&2
  exit 2
fi
if [[ -z "$WARMSTART_CHECKPOINT" ]]; then
  echo "WARMSTART_CHECKPOINT is required: point it at the phase-1 pick_up smolvla_grpo_adapter.pt." >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" && ! -f "$WARMSTART_CHECKPOINT" ]]; then
  echo "WARMSTART_CHECKPOINT not found: $WARMSTART_CHECKPOINT" >&2
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
  echo "MJLab catch/release config not found: $CONFIG" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
unset RLVLA_SMOLVLA_RESUME_CHECKPOINT
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_SMOLVLA_NPROC_PER_NODE=2
export RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
export RLVLA_MJWARP_WORLDS_PER_RANK="$WORLDS_PER_RANK"
export RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE"
export RLVLA_SMOLVLA_ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT"
export RLVLA_SMOLVLA_WARMSTART_CHECKPOINT="$WARMSTART_CHECKPOINT"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
train_cmd=(
  "${python_cmd[@]}"
  -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG"
  --stage rl
  --run-name "$RUN_NAME"
  --execute
)

printf 'mode=catch_release_weights_only_warmstart\n'
printf 'warmstart_checkpoint=%s\n' "$WARMSTART_CHECKPOINT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'tensorboard_dir=%s\n' "$RUN_DIR/rl/tensorboard"
printf 'validation_metrics=%s\n' "$RUN_DIR/rl/validation.jsonl"
printf 'config=%s\n' "$CONFIG"
printf 'max_train_steps=%s\n' "$MAX_TRAIN_STEPS"
printf 'cuda_visible_devices=%s ranks=2\n' "$CUDA_VISIBLE_DEVICES"
printf 'worlds_per_rank=%s groups_per_rank=%s server_worlds=%s\n' \
  "$WORLDS_PER_RANK" "$((WORLDS_PER_RANK / 8))" "$((2 * WORLDS_PER_RANK))"
printf 'smolvla_inference_microbatch_size=%s\n' "$SMOLVLA_MICROBATCH_SIZE"
printf 'watch: validation/by_instruction/<name>/success_rate and curriculum/start_max_goal_distance_m/<name>\n'
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
