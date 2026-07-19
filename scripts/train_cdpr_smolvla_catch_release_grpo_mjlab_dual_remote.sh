#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml}"
CHECKPOINT="${CHECKPOINT:-/root/repo/RL_VLA_Bootstrapping/runs/cdpr_smolvla_move_to_scratch_mjwarp_w512_20260719_081705/rl/step_5000081/smolvla_grpo_adapter.pt}"
START_STEP="${START_STEP:-5000081}"
ADDITIONAL_TRAIN_STEPS="${ADDITIONAL_TRAIN_STEPS:-15000000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-$((START_STEP + ADDITIONAL_TRAIN_STEPS))}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
CHECKPOINT_WORLDS_PER_RANK="${CHECKPOINT_WORLDS_PER_RANK:-512}"
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ALLOW_LEGACY_SIMULATOR_CHECKPOINT="${ALLOW_LEGACY_SIMULATOR_CHECKPOINT:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-cdpr_smolvla_catch_release_dense_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"

if [[ "$ADDITIONAL_TRAIN_STEPS" -ne 15000000 ]]; then
  echo "This launcher requires ADDITIONAL_TRAIN_STEPS=15000000." >&2
  exit 2
fi
if [[ "$MAX_TRAIN_STEPS" -ne $((START_STEP + ADDITIONAL_TRAIN_STEPS)) ]]; then
  echo "MAX_TRAIN_STEPS must equal START_STEP + ADDITIONAL_TRAIN_STEPS." >&2
  exit 2
fi
if [[ "$WORLDS_PER_RANK" -lt 8 || $((WORLDS_PER_RANK % 8)) -ne 0 ]]; then
  echo "WORLDS_PER_RANK must be a positive multiple of the GRPO group size (8)." >&2
  exit 2
fi
if [[ "$WORLDS_PER_RANK" -ne "$CHECKPOINT_WORLDS_PER_RANK" ]]; then
  echo "WORLDS_PER_RANK must match checkpoint simulator metadata ($CHECKPOINT_WORLDS_PER_RANK)." >&2
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
if [[ "$DRY_RUN" != "1" && ! -f "$CHECKPOINT" ]]; then
  echo "Resume adapter not found: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_SMOLVLA_NPROC_PER_NODE=2
export RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT"
export RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
export RLVLA_MJWARP_WORLDS_PER_RANK="$WORLDS_PER_RANK"
export RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE"
export RLVLA_SMOLVLA_ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
train_cmd=(
  "${python_cmd[@]}"
  -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG"
  --stage rl
  --run-name "$RUN_NAME"
  --execute
)

printf 'mode=resume_catch_release_dense\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'start_step=%s additional_train_steps=%s max_train_steps=%s\n' \
  "$START_STEP" "$ADDITIONAL_TRAIN_STEPS" "$MAX_TRAIN_STEPS"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'tensorboard_dir=%s\n' "$RUN_DIR/rl/tensorboard"
printf 'validation_metrics=%s\n' "$RUN_DIR/rl/validation.jsonl"
printf 'config=%s\n' "$CONFIG"
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
