#!/usr/bin/env bash
# Harvest 512 balanced, strict, policy-only full-task put_into demonstrations.
#
# Contract: collection split, distinct empty-start scenes, one final put_into
# prompt from action zero, one checkpoint continuously controlling approach ->
# pickup -> carry -> release, and no teacher/servo/restore/recovery/settle arm.
#
# Default workload: 2 GPUs x 64 rounds x 32 worlds = 4096 distinct attempts.
# The builder then takes exactly 64 successes from each of four objects x two
# destinations (512 total). It fails loudly if the weakest cell is short.
#
# Remote usage after pulling this commit:
#   bash scripts/collect_cdpr_strict_success_dataset_remote.sh
#
# Common overrides:
#   GPUS="0" ROUNDS=96 WORLDS=32 \
#     bash scripts/collect_cdpr_strict_success_dataset_remote.sh
#   STEPS=build RUN_DIR=runs/<existing-run> \
#     bash scripts/collect_cdpr_strict_success_dataset_remote.sh
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

CHECKPOINT="${CHECKPOINT:-/root/repo/RL_VLA_Bootstrapping/runs/three_stage_sparse_grpo_20260918_210212/rl/step_52791642/smolvla_grpo_adapter.pt}"
CONFIG="${CONFIG:-configs/examples/cdpr_smolvla_three_stage_put_into.yaml}"
SCENES="${SCENES:-runs/three_stage/scenes_8192.json}"
RUN_DIR="${RUN_DIR:-runs/strict_success_dataset_step_52791642_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1}"
WORLDS="${WORLDS:-32}"
ROUNDS="${ROUNDS:-64}"
DECISIONS="${DECISIONS:-128}"
MICROBATCH="${MICROBATCH:-16}"
SCENE_OFFSET="${SCENE_OFFSET:-0}"
SUCCESSES_PER_CELL="${SUCCESSES_PER_CELL:-64}"
SELECTION_SEED="${SELECTION_SEED:-20260922}"
SEED_TORCH="${SEED_TORCH:-20260922}"
RECORD_FRAMES="${RECORD_FRAMES:-1}"
STEPS="${STEPS:-record build}"
TAG="${TAG:-strict_step52791642}"

if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT="$CHECKPOINT/smolvla_grpo_adapter.pt"
fi
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$SCENES" ]] || { echo "Scene manifest not found: $SCENES" >&2; exit 2; }
if [[ ! "$WORLDS" =~ ^[0-9]+$ || "$WORLDS" -lt 2 || $((WORLDS % 2)) -ne 0 ]]; then
  echo "WORLDS must be an even integer >= 2." >&2
  exit 2
fi
if [[ ! "$SUCCESSES_PER_CELL" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUCCESSES_PER_CELL must be positive." >&2
  exit 2
fi

run() { conda run --no-capture-output -n "$ENV_NAME" python3 "$@"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS="${#GPU_LIST[@]}"
[[ "$NUM_SHARDS" -gt 0 ]] || { echo "GPUS is empty." >&2; exit 2; }

mkdir -p "$RUN_DIR"
RUN_LOG="${RUN_LOG:-$RUN_DIR/collection_$(date +%Y%m%d_%H%M%S)_$$.log}"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== strict full-task self-imitation harvest ==="
echo "checkpoint=$CHECKPOINT"
echo "scenes=$SCENES split=collection"
echo "attempts=$NUM_SHARDS x $ROUNDS x $WORLDS = $((NUM_SHARDS * ROUNDS * WORLDS)) distinct scenes"
echo "target=$SUCCESSES_PER_CELL x 4 objects x 2 destinations = $((SUCCESSES_PER_CELL * 8)) strict successes"
echo "frames=$RECORD_FRAMES run_dir=$RUN_DIR"
echo "assistance=none settle_decisions=0"

if has_step record; then
  echo "=== record policy-only trajectories ==="
  PIDS=()
  SHARD=0
  for gpu in "${GPU_LIST[@]}"; do
    OUT="$RUN_DIR/bank_shard${SHARD}"
    mkdir -p "$OUT"
    FRAME_ARGS=()
    [[ "$RECORD_FRAMES" == "1" ]] || FRAME_ARGS+=(--no-frames)
    (
      CUDA_VISIBLE_DEVICES="$gpu" run \
        tools/audit/collect_cdpr_full_put_into.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --scene-manifest "$SCENES" \
        --split collection \
        --output "$OUT" \
        --device cuda:0 \
        --worlds "$WORLDS" \
        --rounds "$ROUNDS" \
        --decisions "$DECISIONS" \
        --microbatch "$MICROBATCH" \
        --shard "$SHARD" \
        --num-shards "$NUM_SHARDS" \
        --scene-offset "$SCENE_OFFSET" \
        --tag "$TAG" \
        --seed-torch "$SEED_TORCH" \
        "${FRAME_ARGS[@]}" 2>&1 | sed "s/^/[shard$SHARD] /"
    ) &
    PIDS+=("$!")
    SHARD=$((SHARD + 1))
  done
  FAILED=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAILED=1; fi
  done
  if [[ "$FAILED" -ne 0 ]]; then
    echo "At least one collection shard failed; refusing to build a partial bank." >&2
    exit 1
  fi
fi

if has_step build; then
  echo "=== select exact object x destination quota and build dataset ==="
  shopt -s nullglob
  RECORD_PATHS=("$RUN_DIR"/bank_shard*/record_*.npz)
  FRAME_PATHS=("$RUN_DIR"/bank_shard*/frames_*.npz)
  shopt -u nullglob
  if [[ "${#RECORD_PATHS[@]}" -eq 0 ]]; then
    echo "No record shards found under $RUN_DIR/bank_shard*." >&2
    exit 1
  fi
  BUILD_ARGS=(
    tools/audit/build_cdpr_full_put_into_dataset.py
    --records "${RECORD_PATHS[@]}"
    --output "$RUN_DIR/dataset"
    --successes-per-cell "$SUCCESSES_PER_CELL"
    --selection-seed "$SELECTION_SEED"
  )
  if [[ "$RECORD_FRAMES" == "1" ]]; then
    if [[ "${#FRAME_PATHS[@]}" -eq 0 ]]; then
      echo "RECORD_FRAMES=1 but no frame shards were found." >&2
      exit 1
    fi
    BUILD_ARGS+=(--frames "${FRAME_PATHS[@]}")
  else
    BUILD_ARGS+=(--allow-missing-frames)
  fi
  run "${BUILD_ARGS[@]}"
fi

echo
echo "=== complete ==="
echo "run:       $RUN_DIR"
echo "records:   $RUN_DIR/bank_shard*/record_*.npz"
echo "frames:    $RUN_DIR/bank_shard*/frames_*.npz"
echo "dataset:   $RUN_DIR/dataset/demonstrations.npz"
echo "audit:     $RUN_DIR/dataset/dataset.json"
echo "selection: $RUN_DIR/dataset/selected_episodes.json"
echo "log:       $RUN_LOG"
