#!/usr/bin/env bash
# Resume placement RL from a checkpoint -- which for phase 4 means the SFT
# result the loop just produced.
#
# This is the step that CLOSES the loop, and until now it did not exist. The
# catch_release launcher warm-starts weights only and refuses
# CHECKPOINT/RLVLA_SMOLVLA_RESUME_CHECKPOINT outright, so an accepted SFT
# checkpoint had nowhere to go: the loop could harvest, train and reach a
# verdict, and then had no way to hand the result back to RL.
#
# A resume, not a warm start, and the distinction is the whole point.
# trainer.load restores extra_state, where the approach-curriculum caps live,
# so the run continues from the rung the previous iteration earned.
# load_weights_only discards them and would drop the cap back to the first rung,
# undoing the iteration that earned the promotion. sil_sft writes a payload with
# both optimizer states removed precisely so this resume rebuilds them fresh
# rather than inheriting moments taken from a supervised loss.
#
# The checkpoint's provenance is printed before launch, because "were the
# self-recorded demonstrations applied?" should be answerable from the launch
# itself and not from memory. NO SFT STAMP here means the loop has not closed.
#
# Usage::
#
#   CHECKPOINT=runs/phase4_iter1/sft/sil_sft_adapter.pt \
#   CONFIG=$PWD/configs/examples/cdpr_smolvla_phase4_placement_loop.yaml \
#   RUN_LABEL=phase4_placement_iter1 \
#     bash scripts/train_cdpr_smolvla_catch_release_grpo_mjlab_dual_remote_resume.sh
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_placement_loop.yaml}"
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
configure_huggingface_offline

# shellcheck source=scripts/run_naming.sh
source "$SCRIPT_DIR/run_naming.sh"
RUN_NAME="$(cdpr_compose_run_name "cdpr_smolvla_placement_resume_mjwarp_w${WORLDS_PER_RANK}")"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
# A resume writes a NEW run directory and reads the checkpoint by path,
# so a collision here is the same hazard as on a fresh start.
cdpr_guard_run_dir "$RUN_DIR"

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

printf 'mode=placement_resume\n'
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
# Absolute: this runs BEFORE the cd into REPO_ROOT, so a relative path here
# resolves against wherever the launcher happened to be invoked from and the
# whole point of printing it is lost -- silently, because of the || true.
printf 'provenance:\n'
"${python_cmd[@]}" "$REPO_ROOT/tools/audit/checkpoint_provenance.py" \
  "$CHECKPOINT" --brief || echo "  (could not read $CHECKPOINT)"
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
