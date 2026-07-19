#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/runs/cdpr_smolvla_move_to_scratch_mjwarp_w512_20260719_081705/rl/step_5000081/smolvla_grpo_adapter.pt}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$REPO_ROOT/runs/cdpr_smolvla_move_to_qualitative}"
EPISODES_PER_TARGET="${EPISODES_PER_TARGET:-3}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-0.02}"
TOLERANCES="${TOLERANCES:-0.02 0.025 0.05 0.10 0.15}"
VALIDATION_SEED="${VALIDATION_SEED:-1000000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-8}"
VIDEO_FPS="${VIDEO_FPS:-10}"
TERMINAL_HOLD_SECONDS="${TERMINAL_HOLD_SECONDS:-1}"
CURRICULUM_SHELL="${CURRICULUM_SHELL:-}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Evaluation config not found: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "SmolVLA adapter checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ "$EPISODES_PER_TARGET" -lt 1 ]]; then
  echo "EPISODES_PER_TARGET must be positive." >&2
  exit 2
fi
if [[ "$SMOLVLA_MICROBATCH_SIZE" -lt 1 || "$SMOLVLA_MICROBATCH_SIZE" -gt 8 ]]; then
  echo "SMOLVLA_MICROBATCH_SIZE must be in [1, 8]." >&2
  exit 2
fi

checkpoint_step="$(basename "$(dirname "$CHECKPOINT")")"
training_run="$(basename "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")")"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$EVAL_OUTPUT_ROOT/${training_run}_${checkpoint_step}_${timestamp}}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
export PYTHONUNBUFFERED=1

cmd=(
  conda run --no-capture-output -n "$ENV_NAME"
  python3 -m rl_vla_bootstrapping.cli.evaluate_cdpr_smolvla_mjwarp_videos
  --config "$CONFIG_PATH"
  --checkpoint "$CHECKPOINT"
  --run-dir "$RUN_DIR"
  --episodes-per-target "$EPISODES_PER_TARGET"
  --success-distance "$SUCCESS_DISTANCE"
  --validation-seed "$VALIDATION_SEED"
  --device "$DEVICE"
  --smolvla-microbatch-size "$SMOLVLA_MICROBATCH_SIZE"
  --fps "$VIDEO_FPS"
  --terminal-hold-seconds "$TERMINAL_HOLD_SECONDS"
)

read -r -a tolerance_values <<< "$TOLERANCES"
cmd+=(--tolerances "${tolerance_values[@]}")
if [[ -n "$CURRICULUM_SHELL" ]]; then
  cmd+=(--curriculum-shell "$CURRICULUM_SHELL")
fi
cmd+=("$@")

printf 'mode=qualitative_move_to_object_video_review\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'config=%s\n' "$CONFIG_PATH"
printf 'output=%s\n' "$RUN_DIR"
printf 'backend=mjlab_mjwarp exact_training_backend=true\n'
printf 'targets=all_configured episodes_per_target=%s\n' "$EPISODES_PER_TARGET"
printf 'strict_success_xy_distance_m=%s\n' "$SUCCESS_DISTANCE"
printf 'validation_seed=%s curriculum_shell=%s\n' \
  "$VALIDATION_SEED" "${CURRICULUM_SHELL:-checkpoint}"
printf 'cuda_visible_devices=%s device=%s worlds=8 groups=1 group_size=8\n' \
  "$CUDA_VISIBLE_DEVICES" "$DEVICE"
printf 'smolvla_microbatch_size=%s video_fps=%s\n' \
  "$SMOLVLA_MICROBATCH_SIZE" "$VIDEO_FPS"
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
{
  "${cmd[@]}"
  printf '\nReview videos: %s/videos\n' "$RUN_DIR"
  printf 'Per-episode telemetry: %s/episode_results.csv\n' "$RUN_DIR"
  printf 'Per-action telemetry: %s/action_telemetry.csv\n' "$RUN_DIR"
  printf 'Tolerance sweep: %s/move_to_object_threshold_sweep.csv\n' "$RUN_DIR"
  printf 'Summary report: %s/validation_report.md\n' "$RUN_DIR"
} 2>&1 | tee "$RUN_DIR/evaluation.log"
