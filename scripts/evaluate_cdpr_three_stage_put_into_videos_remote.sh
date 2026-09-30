#!/usr/bin/env bash
# Unassisted full-task put_into validation of one checkpoint, with MP4s of the
# successful episodes.
#
# Empty start, one final prompt (plate or bowl), no teacher, servo or stage
# switch; the same FullTaskOutcome strict verdict and milestone machine as
# training. Only put_into_plate / put_into_bowl scenes exist in the manifest,
# so no other instruction is evaluated.
#
#   CHECKPOINT=runs/three_stage_sparse_grpo_20260914_195542/rl/step_28309431 \
#     bash scripts/evaluate_cdpr_three_stage_put_into_videos_remote.sh
#
# Outputs under $OUTPUT_DIR:
#   evaluation.json          strict/native per destination, phases, milestones
#   videos/*.mp4 + *.json    overview | wrist, one frame per executed action,
#                            with outcome and event times (grasp, lift, ...)
#   videos/videos.json       index of every kept video
#   eval.log                 durable transcript
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
export MUJOCO_GL="${MUJOCO_GL:-egl}"

CHECKPOINT="${CHECKPOINT:-runs/three_stage_sparse_grpo_20260914_195542/rl/step_28309431}"
CONFIG="${CONFIG:-configs/examples/cdpr_smolvla_three_stage_put_into.yaml}"
SCENES="${SCENES:-runs/three_stage/scenes_8192.json}"
SPLIT="${SPLIT:-student_validation}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORLDS="${WORLDS:-64}"
ROUNDS="${ROUNDS:-4}"
# 1: every round gets new scenes (WORLDS x ROUNDS distinct scenes).
# 0: repeat the first WORLDS scenes, the protocol of the 2026-09-12 SFT table.
DISTINCT_SCENES="${DISTINCT_SCENES:-1}"
DECISIONS="${DECISIONS:-128}"
SETTLE_DECISIONS="${SETTLE_DECISIONS:-0}"
MICROBATCH="${MICROBATCH:-32}"
VIDEO_OUTCOME="${VIDEO_OUTCOME:-strict}"   # strict | native | failed | all
VIDEO_FPS="${VIDEO_FPS:-20}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"              # 0 keeps every matching episode
STOCHASTIC_SEED="${STOCHASTIC_SEED:-}"     # empty: deterministic residual, as in training validation
# Scale on the SmolVLA prior's flow-matching start noise. Empty = LeRobot's
# unit-normal draw (every evaluation so far); 0 = a deterministic prior.
PRIOR_NOISE_SCALE="${PRIOR_NOISE_SCALE:-}"
# 0: no MP4s (per-scene outcomes are still in evaluation.json). For repeats.
VIDEOS="${VIDEOS:-1}"
# 1: drop the training run's fixed in-run validation panel (validation_seed,
# 512 worlds / group 8 per rank, 2 ranks, 1 round) from the split first.
EXCLUDE_VALIDATION_PANEL="${EXCLUDE_VALIDATION_PANEL:-0}"
PANEL_VALIDATION_SEED="${PANEL_VALIDATION_SEED:-2000000}"
PANEL_GROUPS_PER_RANK="${PANEL_GROUPS_PER_RANK:-64}"
PANEL_RANKS="${PANEL_RANKS:-2}"

if [[ -d "$CHECKPOINT" ]]; then
  CHECKPOINT="$CHECKPOINT/smolvla_grpo_adapter.pt"
fi
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$SCENES" ]] || { echo "Scene manifest not found: $SCENES" >&2; exit 2; }
if [[ "$VIDEOS" == "1" ]]; then
  command -v ffmpeg >/dev/null || { echo "ffmpeg is required for the videos." >&2; exit 2; }
fi
case "$VIDEO_OUTCOME" in strict|native|failed|all) ;; *)
  echo "VIDEO_OUTCOME must be strict, native, failed or all." >&2; exit 2 ;;
esac
if [[ ! "$WORLDS" =~ ^[0-9]+$ || "$WORLDS" -lt 2 || $((WORLDS % 2)) -ne 0 ]]; then
  echo "WORLDS must be an even integer >= 2." >&2; exit 2
fi

step_name="$(basename "$(dirname "$CHECKPOINT")")"
run_name="$(basename "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")")"
noise_tag=""
[[ -n "$PRIOR_NOISE_SCALE" ]] && noise_tag="_priornoise${PRIOR_NOISE_SCALE}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/three_stage_put_into_eval/${run_name}_${step_name}_${SPLIT}${noise_tag}_$(date +%Y%m%d_%H%M%S)}"
unset RLVLA_SMOLVLA_PRIOR_NOISE_SCALE
[[ -n "$PRIOR_NOISE_SCALE" ]] && export RLVLA_SMOLVLA_PRIOR_NOISE_SCALE="$PRIOR_NOISE_SCALE"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/eval.log") 2>&1
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

args=(
  tools/audit/evaluate_cdpr_full_put_into.py
  --config "$CONFIG" --checkpoint "$CHECKPOINT"
  --scene-manifest "$SCENES" --split "$SPLIT"
  --worlds "$WORLDS" --rounds "$ROUNDS" --microbatch "$MICROBATCH"
  --device "$DEVICE" --decisions "$DECISIONS" --settle-decisions "$SETTLE_DECISIONS"
  --output "$OUTPUT_DIR"
)
if [[ "$VIDEOS" == "1" ]]; then
  args+=(--video-dir "$OUTPUT_DIR/videos" --video-outcome "$VIDEO_OUTCOME"
    --video-fps "$VIDEO_FPS" --max-videos "$MAX_VIDEOS")
fi
if [[ "$EXCLUDE_VALIDATION_PANEL" == "1" ]]; then
  args+=(--exclude-validation-panel --panel-validation-seed "$PANEL_VALIDATION_SEED"
    --panel-groups-per-rank "$PANEL_GROUPS_PER_RANK" --panel-ranks "$PANEL_RANKS")
fi
[[ "$DISTINCT_SCENES" == "1" ]] && args+=(--distinct-scene-rounds)
[[ -n "$STOCHASTIC_SEED" ]] && args+=(--stochastic-seed "$STOCHASTIC_SEED")

echo "checkpoint=$CHECKPOINT"
echo "prior noise scale=${PRIOR_NOISE_SCALE:-1 (default)}"
echo "split=$SPLIT worlds=$WORLDS rounds=$ROUNDS distinct_scenes=$DISTINCT_SCENES decisions=$DECISIONS"
if [[ "$VIDEOS" == "1" ]]; then
  echo "videos: outcome=$VIDEO_OUTCOME fps=$VIDEO_FPS max=$MAX_VIDEOS -> $OUTPUT_DIR/videos"
else
  echo "videos: off"
fi
echo "exclude in-run validation panel=$EXCLUDE_VALIDATION_PANEL"
[[ "$SPLIT" == "final_test" ]] && echo "WARNING: final_test is the locked split; run it once, on the selected checkpoint."
sha256sum "$CHECKPOINT" | tee "$OUTPUT_DIR/checkpoint.sha256"
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"

conda run --no-capture-output -n "$ENV_NAME" python3 "${args[@]}"
if [[ "$VIDEOS" == "1" ]]; then
  echo "=== done: $OUTPUT_DIR/evaluation.json, $(find "$OUTPUT_DIR/videos" -name '*.mp4' | wc -l) videos ==="
else
  echo "=== done: $OUTPUT_DIR/evaluation.json ==="
fi
