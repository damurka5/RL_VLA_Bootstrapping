#!/usr/bin/env bash
# Train and validate a full-task put_into policy from the balanced strict bank.
#
# The source bank was produced by the same checkpoint used to initialize SFT.
# Training directly against its stored priors would therefore be almost an
# identity fit.  Refresh first: re-run SmolVLA on the exact saved frames and
# final put_into prompt, preserving the successful plant actions while drawing
# current priors.  The residual then learns the selected successful behaviour
# rather than simply copying its own cached output.
#
# Default remote usage after pulling this commit:
#   RUN_DIR=runs/strict_success_dataset_step_52791642 \
#     bash scripts/train_cdpr_strict_success_sft_remote.sh
#
# Resume individual phases without repeating completed work:
#   RUN_DIR=runs/strict_success_dataset_step_52791642 STEPS=train \
#     bash scripts/train_cdpr_strict_success_sft_remote.sh
#   RUN_DIR=runs/strict_success_dataset_step_52791642 STEPS=eval \
#     bash scripts/train_cdpr_strict_success_sft_remote.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

RUN_DIR="${RUN_DIR:-runs/strict_success_dataset_step_52791642}"
CHECKPOINT="${CHECKPOINT:-/root/repo/RL_VLA_Bootstrapping/runs/three_stage_sparse_grpo_20260918_210212/rl/step_52791642/smolvla_grpo_adapter.pt}"
CONFIG="${CONFIG:-configs/examples/cdpr_smolvla_three_stage_put_into.yaml}"
SCENES="${SCENES:-runs/three_stage/scenes_8192.json}"
DATASET="${DATASET:-$RUN_DIR/dataset/demonstrations.npz}"
WORK_DIR="${WORK_DIR:-$RUN_DIR/sft_from_step_52791642}"
REFRESHED_DIR="${REFRESHED_DIR:-$WORK_DIR/refreshed}"
MODEL_DIR="${MODEL_DIR:-$WORK_DIR/model}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-$MODEL_DIR/sil_sft_adapter.pt}"

# One process and one GPU: SmolVLA SFT is not a multi-rank trainer.
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-refresh train eval}"
REFRESH_BATCH_SIZE="${REFRESH_BATCH_SIZE:-32}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-20260922}"

# Action-expert LoRA is enabled by default.  Vision-tower LoRA remains opt-in,
# matching the source GRPO configuration.  LORA_ROWS is the host-RAM bound:
# 8192 rows hold about 3.8 GB of uint8 overview+wrist frames, not the full bank.
TRAIN_LORA="${TRAIN_LORA:-1}"
TRAIN_VISION_LORA="${TRAIN_VISION_LORA:-0}"
LORA_ROWS="${LORA_ROWS:-8192}"
LORA_EPOCHS="${LORA_EPOCHS:-8}"
LORA_MICROBATCH="${LORA_MICROBATCH:-4}"
LORA_LR="${LORA_LR:-0}"
LORA_ACTOR_LR="${LORA_ACTOR_LR:-0}"
LORA_KL_COEF="${LORA_KL_COEF:-0.1}"

# Optional original-label retention bank.  If supplied, it must already have
# been refreshed under CHECKPOINT; the strict put_into bank cannot serve as its
# own retention data because all of its rows carry the final put_into prompt.
RETENTION_DATASET="${RETENTION_DATASET:-}"
RETENTION_FRACTION="${RETENTION_FRACTION:-0.2}"

# The verdict is an unassisted rollout, not the held-out imitation loss.
EVAL_BASELINE="${EVAL_BASELINE:-1}"
EVAL_WORLDS="${EVAL_WORLDS:-64}"
EVAL_ROUNDS="${EVAL_ROUNDS:-4}"
EVAL_MICROBATCH="${EVAL_MICROBATCH:-16}"
EVAL_DECISIONS="${EVAL_DECISIONS:-128}"
EVAL_SPLIT="${EVAL_SPLIT:-student_validation}"

if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT="$CHECKPOINT/smolvla_grpo_adapter.pt"
fi
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$SCENES" ]] || { echo "Scene manifest not found: $SCENES" >&2; exit 2; }

for step in $STEPS; do
  case "$step" in
    refresh|train|eval) ;;
    *) echo "Unknown STEPS entry '$step'; use refresh, train, and/or eval." >&2; exit 2 ;;
  esac
done
for pair in \
  "REFRESH_BATCH_SIZE:$REFRESH_BATCH_SIZE" \
  "SFT_BATCH_SIZE:$SFT_BATCH_SIZE" \
  "EPOCHS:$EPOCHS" \
  "LORA_ROWS:$LORA_ROWS" \
  "LORA_EPOCHS:$LORA_EPOCHS" \
  "LORA_MICROBATCH:$LORA_MICROBATCH" \
  "EVAL_WORLDS:$EVAL_WORLDS" \
  "EVAL_ROUNDS:$EVAL_ROUNDS"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; got '$value'." >&2
    exit 2
  fi
done
if (( EVAL_WORLDS % 2 != 0 )); then
  echo "EVAL_WORLDS must be even." >&2
  exit 2
fi
for flag in TRAIN_LORA TRAIN_VISION_LORA EVAL_BASELINE; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$flag must be 0 or 1; got '$value'." >&2
    exit 2
  fi
done

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
run() {
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$ENV_NAME" \
    python3 "$@"
}

mkdir -p "$WORK_DIR"
RUN_LOG="${RUN_LOG:-$WORK_DIR/sft_$(date +%Y%m%d_%H%M%S)_$$.log}"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== strict-success SFT ==="
echo "source checkpoint: $CHECKPOINT"
echo "source dataset:    $DATASET"
echo "scene manifest:    $SCENES"
echo "work directory:    $WORK_DIR"
echo "steps:             $STEPS"
echo "device:            physical GPU $GPU -> $DEVICE"
echo "fit:               epochs=$EPOCHS batch=$SFT_BATCH_SIZE sampler=balanced split=scene"
echo "LoRA:              enabled=$TRAIN_LORA rows=$LORA_ROWS epochs=$LORA_EPOCHS microbatch=$LORA_MICROBATCH vision=$TRAIN_VISION_LORA"
if [[ -n "$RETENTION_DATASET" ]]; then
  echo "retention:         $RETENTION_DATASET fraction=$RETENTION_FRACTION"
else
  echo "retention:         none (put_into-only specialization)"
fi

shopt -s nullglob
FRAME_PATHS=("$RUN_DIR"/bank_shard*/frames_*.npz)
shopt -u nullglob

if has_step refresh; then
  [[ -f "$DATASET" ]] || { echo "Dataset not found: $DATASET" >&2; exit 2; }
  if [[ "${#FRAME_PATHS[@]}" -eq 0 ]]; then
    echo "No frame shards found under $RUN_DIR/bank_shard*." >&2
    exit 2
  fi
  echo "=== refresh priors from exact policy frames ==="
  run tools/audit/sil_refresh_priors.py \
    --dataset "$DATASET" \
    --frames "${FRAME_PATHS[@]}" \
    --checkpoint "$CHECKPOINT" \
    --output "$REFRESHED_DIR" \
    --device "$DEVICE" \
    --batch-size "$REFRESH_BATCH_SIZE"
fi

if has_step train; then
  REFRESHED_DATASET="$REFRESHED_DIR/demonstrations.npz"
  [[ -f "$REFRESHED_DATASET" ]] || {
    echo "Refreshed dataset not found: $REFRESHED_DATASET" >&2
    echo "Run with STEPS=refresh first (or include refresh in STEPS)." >&2
    exit 2
  }
  SFT_ARGS=(
    tools/audit/sil_sft.py
    --dataset "$REFRESHED_DATASET"
    --checkpoint "$CHECKPOINT"
    --output "$MODEL_DIR"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --batch-size "$SFT_BATCH_SIZE"
    --lr "$LR"
    --val-fraction "$VAL_FRACTION"
    --seed "$SEED"
    --split-by scene
    --sampler balanced
    --progress never
  )
  if [[ -n "$RETENTION_DATASET" ]]; then
    [[ -f "$RETENTION_DATASET" ]] || {
      echo "Retention dataset not found: $RETENTION_DATASET" >&2
      exit 2
    }
    SFT_ARGS+=(
      --retention-dataset "$RETENTION_DATASET"
      --retention-fraction "$RETENTION_FRACTION"
    )
  fi
  if [[ "$TRAIN_LORA" == "1" ]]; then
    if [[ "${#FRAME_PATHS[@]}" -eq 0 ]]; then
      echo "TRAIN_LORA=1 but no frame shards were found." >&2
      exit 2
    fi
    SFT_ARGS+=(
      --frames "${FRAME_PATHS[@]}"
      --lora-rows "$LORA_ROWS"
      --lora-epochs "$LORA_EPOCHS"
      --lora-microbatch "$LORA_MICROBATCH"
      --lora-lr "$LORA_LR"
      --lora-actor-lr "$LORA_ACTOR_LR"
      --lora-kl-coef "$LORA_KL_COEF"
    )
    [[ "$TRAIN_VISION_LORA" == "1" ]] && SFT_ARGS+=(--train-vision-lora)
  fi
  TRAIN_LABEL="residual only"
  [[ "$TRAIN_LORA" == "1" ]] && TRAIN_LABEL="residual and bounded action-expert LoRA"
  echo "=== train $TRAIN_LABEL ==="
  run "${SFT_ARGS[@]}"
  [[ -f "$SFT_CHECKPOINT" ]] || {
    echo "SFT finished without writing $SFT_CHECKPOINT" >&2
    exit 1
  }
fi

evaluate() {
  local name="$1"
  local checkpoint="$2"
  run tools/audit/evaluate_cdpr_full_put_into.py \
    --config "$CONFIG" \
    --checkpoint "$checkpoint" \
    --scene-manifest "$SCENES" \
    --split "$EVAL_SPLIT" \
    --worlds "$EVAL_WORLDS" \
    --rounds "$EVAL_ROUNDS" \
    --microbatch "$EVAL_MICROBATCH" \
    --device "$DEVICE" \
    --decisions "$EVAL_DECISIONS" \
    --settle-decisions 0 \
    --distinct-scene-rounds \
    --output "$WORK_DIR/eval_$name"
}

if has_step eval; then
  [[ -f "$SFT_CHECKPOINT" ]] || {
    echo "SFT checkpoint not found: $SFT_CHECKPOINT" >&2
    echo "Run with STEPS=train first (or include train in STEPS)." >&2
    exit 2
  }
  echo "=== unassisted matched validation: $((EVAL_WORLDS * EVAL_ROUNDS)) distinct scenes ==="
  if [[ "$EVAL_BASELINE" == "1" ]]; then
    evaluate baseline "$CHECKPOINT"
  fi
  evaluate sft "$SFT_CHECKPOINT"
fi

echo
echo "=== complete ==="
echo "refreshed dataset: $REFRESHED_DIR/demonstrations.npz"
echo "SFT checkpoint:    $SFT_CHECKPOINT"
echo "SFT report:        $MODEL_DIR/sft_report.json"
if has_step eval; then
  [[ "$EVAL_BASELINE" == "1" ]] && echo "baseline eval:     $WORK_DIR/eval_baseline/evaluation.json"
  echo "SFT eval:          $WORK_DIR/eval_sft/evaluation.json"
fi
echo "log:               $RUN_LOG"
