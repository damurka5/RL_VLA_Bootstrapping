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
#
# The bare form above ran with `retention: none` (a put_into-only
# specialization) and regressed step_52791642: standalone strict 35.55% ->
# 33.20%, overfit from epoch 0 (best-epoch val MSE already above the
# untrained baseline). RETENTION_DATASET/RETENTION_FRACTION exist precisely
# for this failure mode but the strict put_into bank cannot supply its own
# retention rows -- every one of them carries the final put_into prompt.
# Refresh an original-label bank (move_to/pick_up/placement, e.g. the
# runs/phase4_bank pool) under the SAME checkpoint first, then mix it in:
#
#   CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n cdpr-mjlab python3 \
#     tools/audit/sil_refresh_priors.py \
#     --dataset runs/phase4_bank/dataset/demonstrations.npz \
#     --frames runs/phase4_bank/*_demos/frames_*.npz \
#     --checkpoint runs/three_stage_sparse_grpo_20260918_210212/rl/step_52791642/smolvla_grpo_adapter.pt \
#     --output runs/strict_success_dataset_step_52791642/retention_refreshed \
#     --device cuda:0 --batch-size 32 \
#     --final-prompt-prefix '' \
#     --min-resolved-fraction 0.98
#
# --final-prompt-prefix '' is required: the default 'put ' expects every row
# to carry one put_into chain's final prompt, and phase4_bank's rows legitimately
# carry the original move_to/pick_up/placement labels (e.g. "move to banana"),
# not that prompt. Without the flag sil_refresh_priors.py refuses with exactly
# this error -- it is the tool correctly identifying an original-label
# retention bank, not a broken bank.
#
# --min-resolved-fraction 0.98 is needed too: confirmed on the remote,
# phase4_bank joins by position (no frame_uid, an older recorder), resolves
# 23422/23709 = 0.9879 of rows to a frame, and the default floor is 0.99.
# That is normal wear on an old, many-times-reharvested bank, not corruption
# -- dropping ~1.2% of a share that is itself only 20% of the SFT mix is not
# worth a physics re-replay (which the tool's own docstring warns destroyed
# most of a bank when tried under a different checkpoint). Note: this
# override only works because the join is positional; had the bank carried
# frame_uid (uid join), the tool forces the floor to 1.0 regardless of this
# flag, on the theory that a uid join should never legitimately miss a row.
#
#   RUN_DIR=runs/strict_success_dataset_step_52791642 \
#   WORK_DIR=runs/strict_success_dataset_step_52791642/sft_from_step_52791642_retention \
#   RETENTION_DATASET=runs/strict_success_dataset_step_52791642/retention_refreshed/demonstrations.npz \
#   RETENTION_FRACTION=0.2 \
#   EVAL_BASELINE=0 \
#     bash scripts/train_cdpr_strict_success_sft_remote.sh
#
# WORK_DIR is set explicitly so this does not overwrite the no-retention
# run's evidence (eval_baseline/eval_sft/model under sft_from_step_52791642).
# EVAL_BASELINE=0 skips re-evaluating step_52791642, whose baseline
# evaluation.json already exists from that run. sil_sft.py refuses a
# retention bank whose state width or action-slot count does not match this
# checkpoint's -- if it does, the bank predates a state/action contract
# change and needs a different source, not a forced retry.
#
# REPAIRED retention (2026-09-26). phase4_bank is not compatible with the
# three-stage residual (yaw), so the arm above does not measure retention. The
# harvest launcher now records a retention bank under the SAME checkpoint and
# config, in the same rollouts as the strict bank (see its header). Then:
#
#   RUN_DIR=runs/strict_success_dataset_step_56072006_<stamp> \
#   CHECKPOINT=runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006 \
#   RETENTION_SOURCE=runs/strict_success_dataset_step_56072006_<stamp>/retention_dataset/demonstrations.npz \
#   RETENTION_FRACTION=0.2 \
#   EVAL_BASELINE=0 BASELINE_EVAL_DIR=runs/three_stage_put_into_eval/<step_56072006 eval> \
#     bash scripts/train_cdpr_strict_success_sft_remote.sh
#
# RETENTION_SOURCE is refreshed under CHECKPOINT like the strict bank, and the
# refreshed copy becomes RETENTION_DATASET. Run the same command without
# RETENTION_SOURCE (and a different WORK_DIR) for the no-retention control.
#
# Selection and promotion. sil_sft.py treats the untouched checkpoint as a
# candidate: if no epoch beats it on the held-out put_into (and retention)
# rows, no adapter is written and eval is skipped. Otherwise the SFT adapter is
# evaluated on the baseline's exact scenes and promotion.json says PROMOTE only
# for a significant paired strict win (exact McNemar, PROMOTION_ALPHA).
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
if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT="$CHECKPOINT/smolvla_grpo_adapter.pt"
fi
SOURCE_STEP="$(basename "$(dirname "$CHECKPOINT")")"
WORK_DIR="${WORK_DIR:-$RUN_DIR/sft_from_${SOURCE_STEP}}"
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
RETENTION_SOURCE="${RETENTION_SOURCE:-}"
RETENTION_REFRESHED_DIR="${RETENTION_REFRESHED_DIR:-$WORK_DIR/retention_refreshed}"
if [[ -n "$RETENTION_SOURCE" ]]; then
  RETENTION_DATASET="${RETENTION_DATASET:-$RETENTION_REFRESHED_DIR/demonstrations.npz}"
fi
RETENTION_DATASET="${RETENTION_DATASET:-}"
RETENTION_FRACTION="${RETENTION_FRACTION:-0.2}"
BASELINE_EVAL_DIR="${BASELINE_EVAL_DIR:-$WORK_DIR/eval_baseline}"
PROMOTION_ALPHA="${PROMOTION_ALPHA:-0.05}"

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
  if [[ -n "$RETENTION_SOURCE" ]]; then
    [[ -f "$RETENTION_SOURCE" ]] || { echo "Retention source not found: $RETENTION_SOURCE" >&2; exit 2; }
    shopt -s nullglob
    RETENTION_FRAME_PATHS=("$RUN_DIR"/bank_shard*/frames_*.npz "$RUN_DIR"/bank_shard*/retention_frames_*.npz)
    shopt -u nullglob
    echo "=== refresh retention priors from exact policy frames ==="
    run tools/audit/sil_refresh_priors.py \
      --dataset "$RETENTION_SOURCE" \
      --frames "${RETENTION_FRAME_PATHS[@]}" \
      --checkpoint "$CHECKPOINT" \
      --output "$RETENTION_REFRESHED_DIR" \
      --device "$DEVICE" \
      --batch-size "$REFRESH_BATCH_SIZE"
  fi
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
  if [[ ! -f "$SFT_CHECKPOINT" ]]; then
    selected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("selected", "unknown"))' "$MODEL_DIR/sft_report.json" 2>/dev/null || echo unknown)"
    if [[ "$selected" == "initializer" ]]; then
      echo "SFT selected the untouched checkpoint: no epoch beat it on held-out rows. Nothing to evaluate or promote."
    else
      echo "SFT finished without writing $SFT_CHECKPOINT (selected=$selected)" >&2
      exit 1
    fi
  fi
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
  if [[ ! -f "$SFT_CHECKPOINT" ]]; then
    if [[ -f "$MODEL_DIR/sft_report.json" ]] && grep -q '"selected": "initializer"' "$MODEL_DIR/sft_report.json"; then
      echo "=== no SFT adapter: the untouched checkpoint was selected; skipping eval ==="
      printf '{"verdict": "keep_source", "reason": "sil_sft selected the initializer"}\n' > "$WORK_DIR/promotion.json"
    else
      echo "SFT checkpoint not found: $SFT_CHECKPOINT" >&2
      echo "Run with STEPS=train first (or include train in STEPS)." >&2
      exit 2
    fi
  else
    echo "=== unassisted matched validation: $((EVAL_WORLDS * EVAL_ROUNDS)) distinct scenes ==="
    if [[ "$EVAL_BASELINE" == "1" ]]; then
      evaluate baseline "$CHECKPOINT"
      BASELINE_EVAL_DIR="$WORK_DIR/eval_baseline"
    fi
    evaluate sft "$SFT_CHECKPOINT"
    [[ -f "$BASELINE_EVAL_DIR/evaluation.json" ]] || {
      echo "No baseline evaluation at $BASELINE_EVAL_DIR; set BASELINE_EVAL_DIR or EVAL_BASELINE=1." >&2
      exit 2
    }
    echo "=== closed-loop promotion gate: paired strict comparison on the same scenes ==="
    run tools/audit/compare_put_into_evaluations.py \
      "$BASELINE_EVAL_DIR" "$WORK_DIR/eval_sft" \
      --alpha "$PROMOTION_ALPHA" --output "$WORK_DIR/promotion_strict.json"
    if [[ -f "$BASELINE_EVAL_DIR/evaluation.json" ]] && grep -q '"episodes"' "$BASELINE_EVAL_DIR/evaluation.json"; then
      run tools/audit/compare_put_into_evaluations.py \
        "$BASELINE_EVAL_DIR" "$WORK_DIR/eval_sft" --metric native \
        --alpha "$PROMOTION_ALPHA" --output "$WORK_DIR/promotion_native.json"
    fi
    if grep -q '"verdict": "candidate_better"' "$WORK_DIR/promotion_strict.json"; then
      echo "PROMOTE: the SFT adapter wins the paired strict comparison."
      printf '{"verdict": "promote", "evidence": "promotion_strict.json"}\n' > "$WORK_DIR/promotion.json"
    else
      echo "KEEP SOURCE: no significant paired strict win; do not promote the SFT adapter."
      printf '{"verdict": "keep_source", "evidence": "promotion_strict.json"}\n' > "$WORK_DIR/promotion.json"
    fi
  fi
fi

echo
echo "=== complete ==="
echo "refreshed dataset: $REFRESHED_DIR/demonstrations.npz"
echo "SFT checkpoint:    $SFT_CHECKPOINT"
echo "SFT report:        $MODEL_DIR/sft_report.json"
if has_step eval; then
  [[ "$EVAL_BASELINE" == "1" ]] && echo "baseline eval:     $WORK_DIR/eval_baseline/evaluation.json"
  echo "SFT eval:          $WORK_DIR/eval_sft/evaluation.json"
  echo "promotion:         $WORK_DIR/promotion.json"
fi
echo "log:               $RUN_LOG"
