#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
MOVE_TO_CONFIG="${MOVE_TO_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml}"
CATCH_RELEASE_CONFIG="${CATCH_RELEASE_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml}"
MOVE_TO_TRAIN_STEPS="${MOVE_TO_TRAIN_STEPS:-10000000}"
CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS="${CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS:-15000000}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
MOVE_TO_SMOLVLA_MICROBATCH_SIZE="${MOVE_TO_SMOLVLA_MICROBATCH_SIZE:-${SMOLVLA_MICROBATCH_SIZE:-512}}"
CATCH_RELEASE_SMOLVLA_MICROBATCH_SIZE="${CATCH_RELEASE_SMOLVLA_MICROBATCH_SIZE:-${SMOLVLA_MICROBATCH_SIZE:-512}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ALLOW_LEGACY_SIMULATOR_CHECKPOINT="${ALLOW_LEGACY_SIMULATOR_CHECKPOINT:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOVE_TO_LAUNCHER="$SCRIPT_DIR/train_cdpr_smolvla_move_to_grpo_mjlab_dual_remote.sh"
CATCH_RELEASE_LAUNCHER="$SCRIPT_DIR/train_cdpr_smolvla_catch_release_grpo_mjlab_dual_remote.sh"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MOVE_TO_RUN_NAME="${MOVE_TO_RUN_NAME:-cdpr_smolvla_move_to_scratch_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
CATCH_RELEASE_RUN_NAME="${CATCH_RELEASE_RUN_NAME:-cdpr_smolvla_catch_release_dense_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
MOVE_TO_RUN_DIR="$REPO_ROOT/runs/$MOVE_TO_RUN_NAME"
CATCH_RELEASE_RUN_DIR="$REPO_ROOT/runs/$CATCH_RELEASE_RUN_NAME"

if [[ -n "${CHECKPOINT:-}" || -n "${RLVLA_SMOLVLA_RESUME_CHECKPOINT:-}" ]]; then
  echo "The two-stage launcher always starts from scratch; do not set CHECKPOINT/RLVLA_SMOLVLA_RESUME_CHECKPOINT." >&2
  exit 2
fi
if [[ "$MOVE_TO_TRAIN_STEPS" -ne 10000000 ]]; then
  echo "Phase 1 requires MOVE_TO_TRAIN_STEPS=10000000." >&2
  exit 2
fi
if [[ "$CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS" -ne 15000000 ]]; then
  echo "Phase 2 requires CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS=15000000." >&2
  exit 2
fi
if [[ "$MOVE_TO_RUN_NAME" == "$CATCH_RELEASE_RUN_NAME" ]]; then
  echo "MOVE_TO_RUN_NAME and CATCH_RELEASE_RUN_NAME must be different." >&2
  exit 2
fi
for required_file in \
  "$MOVE_TO_CONFIG" \
  "$CATCH_RELEASE_CONFIG" \
  "$MOVE_TO_LAUNCHER" \
  "$CATCH_RELEASE_LAUNCHER"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required two-stage training file not found: $required_file" >&2
    exit 2
  fi
done

assert_plain_grpo_config() {
  local config="$1"
  if ! grep -Eq '^[[:space:]]+algorithm:[[:space:]]+[^#]*grpo' "$config"; then
    echo "Two-stage training requires a GRPO config: $config" >&2
    return 2
  fi
  if grep -Eq '^[[:space:]]+algorithm:[[:space:]]+[^#]*(reverse_frontier|lchol)' "$config" \
    || grep -Eq '^[[:space:]]+reverse_frontier_profile:' "$config" \
    || grep -Eq '^[[:space:]]+lchol_' "$config"; then
    echo "Reverse Frontier shells and LC-HOL are forbidden in this pipeline: $config" >&2
    return 2
  fi
  if ! grep -Eq '^[[:space:]]+complex_training_approach:[[:space:]]+none([[:space:]]*(#.*)?)?$' "$config"; then
    echo "Two-stage training requires complex_training_approach: none: $config" >&2
    return 2
  fi
}

assert_plain_grpo_config "$MOVE_TO_CONFIG"
assert_plain_grpo_config "$CATCH_RELEASE_CONFIG"
if grep -Eq '^[[:space:]]+resume_checkpoint:' "$MOVE_TO_CONFIG"; then
  echo "Phase 1 must not contain a resume_checkpoint: $MOVE_TO_CONFIG" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" && ( -e "$MOVE_TO_RUN_DIR" || -e "$CATCH_RELEASE_RUN_DIR" ) ]]; then
  echo "Refusing to reuse an existing two-stage run directory; choose new run names or RUN_TIMESTAMP." >&2
  exit 2
fi

checkpoint_step() {
  local checkpoint="${1%/}"
  local base
  while [[ -n "$checkpoint" && "$checkpoint" != "/" && "$checkpoint" != "." ]]; do
    base="${checkpoint##*/}"
    if [[ "$base" =~ ^step_([0-9]+)$ ]]; then
      printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
      return 0
    fi
    checkpoint="${checkpoint%/*}"
  done
  return 1
}

latest_move_to_checkpoint() {
  local candidate
  local candidate_step
  local best_checkpoint=""
  local best_step=-1
  for candidate in "$MOVE_TO_RUN_DIR"/rl/step_*/smolvla_grpo_adapter.pt; do
    [[ -f "$candidate" ]] || continue
    if ! candidate_step="$(checkpoint_step "$candidate")"; then
      continue
    fi
    if [[ "$candidate_step" -gt "$best_step" ]]; then
      best_checkpoint="$candidate"
      best_step="$candidate_step"
    fi
  done
  if [[ -z "$best_checkpoint" ]]; then
    echo "Phase 1 finished without a numbered GRPO checkpoint under $MOVE_TO_RUN_DIR/rl." >&2
    return 2
  fi
  printf '%s\n' "$best_checkpoint"
}

printf 'pipeline=move_to_then_catch_release_plain_grpo\n'
printf 'phase_1_run_dir=%s\n' "$MOVE_TO_RUN_DIR"
printf 'phase_2_run_dir=%s\n' "$CATCH_RELEASE_RUN_DIR"
printf 'phase_1_steps=%s phase_2_additional_steps=%s\n' \
  "$MOVE_TO_TRAIN_STEPS" "$CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS"
printf 'cuda_visible_devices=%s ranks=2 worlds_per_rank=%s\n' \
  "$CUDA_VISIBLE_DEVICES" "$WORLDS_PER_RANK"
printf 'move_to_microbatch_size=%s catch_release_microbatch_size=%s\n' \
  "$MOVE_TO_SMOLVLA_MICROBATCH_SIZE" \
  "$CATCH_RELEASE_SMOLVLA_MICROBATCH_SIZE"
printf 'curriculum=none lchol=disabled checkpoint_handoff=full_grpo_state\n'

printf '\n=== Phase 1/2: fresh GRPO on move-to-object instructions ===\n'
(
  unset CHECKPOINT
  unset RLVLA_SMOLVLA_RESUME_CHECKPOINT
  REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$ENV_NAME" \
    CONFIG="$MOVE_TO_CONFIG" \
    MAX_TRAIN_STEPS="$MOVE_TO_TRAIN_STEPS" \
    WORLDS_PER_RANK="$WORLDS_PER_RANK" \
    SMOLVLA_MICROBATCH_SIZE="$MOVE_TO_SMOLVLA_MICROBATCH_SIZE" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT" \
    RUN_PREFLIGHT="$RUN_PREFLIGHT" \
    DRY_RUN="$DRY_RUN" \
    RUN_NAME="$MOVE_TO_RUN_NAME" \
    bash "$MOVE_TO_LAUNCHER"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '\n=== Process boundary: phase 1 exits; phase 2 starts a new GRPO process ===\n'
  printf 'phase_2_checkpoint=<highest numbered phase-1 GRPO checkpoint, resolved after phase 1>\n'
  CHECKPOINT="$MOVE_TO_RUN_DIR/rl/latest.pt" \
    START_STEP="$MOVE_TO_TRAIN_STEPS" \
    ADDITIONAL_TRAIN_STEPS="$CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS" \
    MAX_TRAIN_STEPS="$((MOVE_TO_TRAIN_STEPS + CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS))" \
    REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$ENV_NAME" \
    CONFIG="$CATCH_RELEASE_CONFIG" \
    WORLDS_PER_RANK="$WORLDS_PER_RANK" \
    CHECKPOINT_WORLDS_PER_RANK="$WORLDS_PER_RANK" \
    SMOLVLA_MICROBATCH_SIZE="$CATCH_RELEASE_SMOLVLA_MICROBATCH_SIZE" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT" \
    RUN_PREFLIGHT="$RUN_PREFLIGHT" \
    DRY_RUN=1 \
    RUN_NAME="$CATCH_RELEASE_RUN_NAME" \
    bash "$CATCH_RELEASE_LAUNCHER"
  exit 0
fi

MOVE_TO_CHECKPOINT="$(latest_move_to_checkpoint)"
START_STEP="$(checkpoint_step "$MOVE_TO_CHECKPOINT")"
if [[ "$START_STEP" -lt "$MOVE_TO_TRAIN_STEPS" ]]; then
  echo "Phase 1 checkpoint step $START_STEP is below the required $MOVE_TO_TRAIN_STEPS actions." >&2
  exit 2
fi
MAX_TRAIN_STEPS="$((START_STEP + CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS))"

printf '\n=== Process boundary: phase 1 exited; reloading GRPO for phase 2 ===\n'
printf 'phase_2_checkpoint=%s\n' "$MOVE_TO_CHECKPOINT"
printf 'phase_2_start_step=%s phase_2_max_train_steps=%s\n' \
  "$START_STEP" "$MAX_TRAIN_STEPS"
printf '\n=== Phase 2/2: resumed plain GRPO on catch/release instructions ===\n'

CHECKPOINT="$MOVE_TO_CHECKPOINT" \
  START_STEP="$START_STEP" \
  ADDITIONAL_TRAIN_STEPS="$CATCH_RELEASE_ADDITIONAL_TRAIN_STEPS" \
  MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
  REPO_ROOT="$REPO_ROOT" \
  ENV_NAME="$ENV_NAME" \
  CONFIG="$CATCH_RELEASE_CONFIG" \
  WORLDS_PER_RANK="$WORLDS_PER_RANK" \
  CHECKPOINT_WORLDS_PER_RANK="$WORLDS_PER_RANK" \
  SMOLVLA_MICROBATCH_SIZE="$CATCH_RELEASE_SMOLVLA_MICROBATCH_SIZE" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT" \
  RUN_PREFLIGHT="$RUN_PREFLIGHT" \
  DRY_RUN=0 \
  RUN_NAME="$CATCH_RELEASE_RUN_NAME" \
  bash "$CATCH_RELEASE_LAUNCHER"

printf '\nTwo-stage plain-GRPO training completed successfully.\n'
printf 'final_run_dir=%s\n' "$CATCH_RELEASE_RUN_DIR"
