#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-openvla-oft}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_openvla_grpo_complex_tasks.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/runs/cdpr_openvla_grpo_complex_tasks_20260618_221123/rl/step_0048000}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$REPO_ROOT/runs/cdpr_openvla_grpo_complex_tasks_20260618_221123/evaluation}"
EPISODES_PER_CASE="${EPISODES_PER_CASE:-20}"
MOVE_TO_OBJECT_EPISODES_PER_TARGET="${MOVE_TO_OBJECT_EPISODES_PER_TARGET:-20}"
VIDEO_SEARCH_EXTRA_EPISODES="${VIDEO_SEARCH_EXTRA_EPISODES:-40}"
ARBITRARY_INSTRUCTIONS_COUNT="${ARBITRARY_INSTRUCTIONS_COUNT:-8}"
REQUIRE_COMPLETE_VIDEO_COVERAGE="${REQUIRE_COMPLETE_VIDEO_COVERAGE:-true}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
checkpoint_name="$(basename "$CHECKPOINT_DIR")"
run_dir="${RUN_DIR:-$EVAL_OUTPUT_ROOT/${checkpoint_name}_${timestamp}}"
mkdir -p "$run_dir"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO_ROOT"

cmd=(
  conda run --no-capture-output -n "$ENV_NAME"
  python3 -m rl_vla_bootstrapping.cli.validate_cdpr_policy
  --config "$CONFIG_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --run-dir "$run_dir"
  --episodes-per-instruction "$EPISODES_PER_CASE"
  --move-to-object-episodes-per-target "$MOVE_TO_OBJECT_EPISODES_PER_TARGET"
  --evaluate-reverse-shells
  --include-synonyms
  --synonyms-per-instruction 2
  --synonym-shells normal
  --multi-object-scenes
  --min-scene-objects 3
  --max-scene-objects 4
  --stratify-move-to-object-targets
  --arbitrary-instructions-count "$ARBITRARY_INSTRUCTIONS_COUNT"
  --record-success-videos
  --record-failure-videos
  --no-record-all-success-videos
  --video-coverage instruction
  --video-search-extra-episodes "$VIDEO_SEARCH_EXTRA_EPISODES"
  --strict-video-validation
  --progress-only
)

if [[ "$REQUIRE_COMPLETE_VIDEO_COVERAGE" == "true" ]]; then
  cmd+=(--require-complete-video-coverage)
else
  cmd+=(--no-require-complete-video-coverage)
fi

cmd+=("$@")

{
  printf 'Evaluation output: %s\n' "$run_dir"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
} 2>&1 | tee "$run_dir/evaluation.log"
