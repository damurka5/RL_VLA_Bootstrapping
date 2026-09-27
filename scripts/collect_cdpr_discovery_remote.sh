#!/usr/bin/env bash
# Step 4, stage 1: where does SAMPLING solve what the deterministic policy fails?
#
# Self-imitation on the policy's own deterministic successes had nothing to
# teach (2026-09-27: both SFT arms selected the untouched checkpoint). New
# supervision has to come from scenes the deterministic mean FAILS but the
# sampled GRPO behaviour policy sometimes solves. This measures that yield by
# failure mode before any bank is built.
#
#   select  classify the deterministic harvest's failures (no_grasp,
#           failed_lift, carry_slip, placement) into a stratified scene list
#   record  attempt every listed scene REPEATS times with the sampled residual,
#           keeping strict successes with frames (the discovered solutions)
#   yield   strict successes per attempt and scenes solved at least once, by
#           failure mode, destination and object
#
#   DET_RUN_DIR=runs/strict_success_dataset_step_56072006_20260926_223044 \
#   CHECKPOINT=runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006 \
#     bash scripts/collect_cdpr_discovery_remote.sh
#
# CHECKPOINT must be the one that produced DET_RUN_DIR: a "failure" is a
# failure of that checkpoint's deterministic mean.
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

DET_RUN_DIR="${DET_RUN_DIR:?Set DET_RUN_DIR to the deterministic strict harvest run.}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the checkpoint that produced DET_RUN_DIR.}"
CONFIG="${CONFIG:-configs/examples/cdpr_smolvla_three_stage_put_into.yaml}"
SCENES="${SCENES:-runs/three_stage/scenes_8192.json}"
GPUS="${GPUS:-0 1}"
WORLDS="${WORLDS:-32}"
REPEATS="${REPEATS:-4}"
MAX_SCENES="${MAX_SCENES:-1024}"
MODES="${MODES:-no_grasp failed_lift carry_slip placement}"
DECISIONS="${DECISIONS:-128}"
MICROBATCH="${MICROBATCH:-16}"
STOCHASTIC_SEED="${STOCHASTIC_SEED:-20260927}"
SEED_TORCH="${SEED_TORCH:-20260927}"
SELECTION_SEED="${SELECTION_SEED:-20260927}"
STEPS="${STEPS:-select record yield}"

if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT="$CHECKPOINT/smolvla_grpo_adapter.pt"
fi
STEP_NAME="$(basename "$(dirname "$CHECKPOINT")")"
RUN_DIR="${RUN_DIR:-runs/discovery_${STEP_NAME}_$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-disc_${STEP_NAME//_/}}"
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
for pair in "WORLDS:$WORLDS" "REPEATS:$REPEATS"; do
  [[ "${pair#*:}" =~ ^[1-9][0-9]*$ ]] || { echo "${pair%%:*} must be positive." >&2; exit 2; }
done

run() { conda run --no-capture-output -n "$ENV_NAME" python3 "$@"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }
read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS="${#GPU_LIST[@]}"
read -r -a MODE_LIST <<< "$MODES"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/discovery_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
echo "=== stochastic discovery on deterministic failures ==="
echo "checkpoint=$CHECKPOINT"
echo "deterministic harvest=$DET_RUN_DIR"
echo "max scenes=$MAX_SCENES x repeats=$REPEATS, modes=$MODES, stochastic seed=$STOCHASTIC_SEED"
echo "run_dir=$RUN_DIR"

if has_step select; then
  shopt -s nullglob
  ATTEMPTS=("$DET_RUN_DIR"/bank_shard*/attempts_*.npz)
  shopt -u nullglob
  [[ "${#ATTEMPTS[@]}" -gt 0 ]] || { echo "No attempts_*.npz under $DET_RUN_DIR." >&2; exit 1; }
  run tools/audit/discovery_scenes.py select \
    --attempts "${ATTEMPTS[@]}" --output "$RUN_DIR/hard_scenes.json" \
    --modes "${MODE_LIST[@]}" --max-scenes "$MAX_SCENES" --seed "$SELECTION_SEED"
fi

if has_step record; then
  [[ -f "$RUN_DIR/hard_scenes.json" ]] || { echo "Run STEPS=select first." >&2; exit 1; }
  PIDS=()
  SHARD=0
  for gpu in "${GPU_LIST[@]}"; do
    OUT="$RUN_DIR/bank_shard${SHARD}"
    mkdir -p "$OUT"
    (
      CUDA_VISIBLE_DEVICES="$gpu" run tools/audit/collect_cdpr_full_put_into.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --scene-manifest "$SCENES" \
        --split collection --scene-uids "$RUN_DIR/hard_scenes.json" \
        --repeats "$REPEATS" --rounds 0 --stochastic-seed "$STOCHASTIC_SEED" \
        --output "$OUT" --device cuda:0 --worlds "$WORLDS" \
        --decisions "$DECISIONS" --microbatch "$MICROBATCH" \
        --shard "$SHARD" --num-shards "$NUM_SHARDS" \
        --tag "$TAG" --seed-torch "$SEED_TORCH" 2>&1 | sed "s/^/[shard$SHARD] /"
    ) &
    PIDS+=("$!")
    SHARD=$((SHARD + 1))
  done
  FAILED=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
  [[ "$FAILED" -eq 0 ]] || { echo "A discovery shard failed." >&2; exit 1; }
fi

if has_step yield; then
  shopt -s nullglob
  ATTEMPTS=("$RUN_DIR"/bank_shard*/attempts_*.npz)
  shopt -u nullglob
  [[ "${#ATTEMPTS[@]}" -gt 0 ]] || { echo "No stochastic attempts under $RUN_DIR." >&2; exit 1; }
  run tools/audit/discovery_scenes.py yield \
    --selection "$RUN_DIR/hard_scenes.json" --attempts "${ATTEMPTS[@]}" \
    --output "$RUN_DIR/discovery_yield.json"
fi

echo "=== complete: $RUN_DIR/discovery_yield.json ==="
