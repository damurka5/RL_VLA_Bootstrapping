#!/usr/bin/env bash
# Full manipulation curriculum, three phases, each a fresh torchrun process
# warm-started WEIGHTS-ONLY from the previous phase's adapter:
#
#   1. move_to_object    -- learn to servo the gripper to a named object
#   2. pick_up           -- learn descend / close / lift at that same hover point
#   3. put_into_{plate,bowl} + pick_up rehearsal
#
# Weights-only handoff, not --resume-checkpoint: each phase changes the task, so
# carrying the previous phase's approach-curriculum caps and optimizer moments
# would start the new task at a difficulty it never earned. The learned
# behaviour transfers; the schedule restarts at step 0.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
MOVE_TO_CONFIG="${MOVE_TO_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml}"
PICK_UP_CONFIG="${PICK_UP_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml}"
CATCH_RELEASE_CONFIG="${CATCH_RELEASE_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml}"
MOVE_TO_TRAIN_STEPS="${MOVE_TO_TRAIN_STEPS:-10000000}"
PICK_UP_TRAIN_STEPS="${PICK_UP_TRAIN_STEPS:-10000000}"
CATCH_RELEASE_TRAIN_STEPS="${CATCH_RELEASE_TRAIN_STEPS:-15000000}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
# 256 everywhere: at 512 the combined PyTorch+Warp footprint sits on the A40
# limit and Warp OOMs once the LoRA backward is added.
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ALLOW_LEGACY_SIMULATOR_CHECKPOINT="${ALLOW_LEGACY_SIMULATOR_CHECKPOINT:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOVE_TO_LAUNCHER="$SCRIPT_DIR/train_cdpr_smolvla_move_to_grpo_mjlab_dual_remote.sh"
PICK_UP_LAUNCHER="$SCRIPT_DIR/train_cdpr_smolvla_pick_up_grpo_mjlab_dual_remote.sh"
CATCH_RELEASE_LAUNCHER="$SCRIPT_DIR/train_cdpr_smolvla_catch_release_grpo_mjlab_dual_remote.sh"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MOVE_TO_RUN_NAME="${MOVE_TO_RUN_NAME:-cdpr_smolvla_move_to_scratch_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
PICK_UP_RUN_NAME="${PICK_UP_RUN_NAME:-cdpr_smolvla_pick_up_warmstart_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
CATCH_RELEASE_RUN_NAME="${CATCH_RELEASE_RUN_NAME:-cdpr_smolvla_catch_release_dense_mjwarp_w${WORLDS_PER_RANK}_${timestamp}}"
MOVE_TO_RUN_DIR="$REPO_ROOT/runs/$MOVE_TO_RUN_NAME"
PICK_UP_RUN_DIR="$REPO_ROOT/runs/$PICK_UP_RUN_NAME"
CATCH_RELEASE_RUN_DIR="$REPO_ROOT/runs/$CATCH_RELEASE_RUN_NAME"

if [[ -n "${CHECKPOINT:-}" || -n "${RLVLA_SMOLVLA_RESUME_CHECKPOINT:-}" ]]; then
  echo "The staged launcher always starts from scratch; do not set CHECKPOINT/RLVLA_SMOLVLA_RESUME_CHECKPOINT." >&2
  exit 2
fi
if [[ "$MOVE_TO_TRAIN_STEPS" -ne 10000000 ]]; then
  echo "Phase 1 requires MOVE_TO_TRAIN_STEPS=10000000." >&2
  exit 2
fi
distinct_names="$(printf '%s\n' "$MOVE_TO_RUN_NAME" "$PICK_UP_RUN_NAME" "$CATCH_RELEASE_RUN_NAME" | sort -u | wc -l | tr -d ' ')"
if [[ "$distinct_names" -ne 3 ]]; then
  echo "The three phase run names must be distinct." >&2
  exit 2
fi
for required_file in \
  "$MOVE_TO_CONFIG" \
  "$PICK_UP_CONFIG" \
  "$CATCH_RELEASE_CONFIG" \
  "$MOVE_TO_LAUNCHER" \
  "$PICK_UP_LAUNCHER" \
  "$CATCH_RELEASE_LAUNCHER"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required staged training file not found: $required_file" >&2
    exit 2
  fi
done

assert_plain_grpo_config() {
  local config="$1"
  if ! grep -Eq '^[[:space:]]+algorithm:[[:space:]]+[^#]*grpo' "$config"; then
    echo "Staged training requires a GRPO config: $config" >&2
    return 2
  fi
  if grep -Eq '^[[:space:]]+algorithm:[[:space:]]+[^#]*(reverse_frontier|lchol)' "$config" \
    || grep -Eq '^[[:space:]]+reverse_frontier_profile:' "$config" \
    || grep -Eq '^[[:space:]]+lchol_' "$config"; then
    echo "Reverse Frontier shells and LC-HOL are forbidden in this pipeline: $config" >&2
    return 2
  fi
  if ! grep -Eq '^[[:space:]]+complex_training_approach:[[:space:]]+none([[:space:]]*(#.*)?)?$' "$config"; then
    echo "Staged training requires complex_training_approach: none: $config" >&2
    return 2
  fi
  # Every phase hands its weights to the next through a strict load_state_dict,
  # so the residual architecture must be identical throughout. Omitting the
  # vision features silently builds a 6-wide residual where the checkpoint holds
  # a 518-wide one, and the handoff dies at load time.
  if ! grep -Eq '^[[:space:]]+residual_vision_features:[[:space:]]+true' "$config"; then
    echo "Staged training requires residual_vision_features: true: $config" >&2
    return 2
  fi
  if ! grep -Eq '^[[:space:]]+residual_vision_dim:[[:space:]]+512' "$config"; then
    echo "Staged training requires residual_vision_dim: 512: $config" >&2
    return 2
  fi
  if ! grep -Eq '^[[:space:]]+train_vla_lora:[[:space:]]+true' "$config"; then
    echo "Staged training requires train_vla_lora: true: $config" >&2
    return 2
  fi
}

assert_plain_grpo_config "$MOVE_TO_CONFIG"
assert_plain_grpo_config "$PICK_UP_CONFIG"
assert_plain_grpo_config "$CATCH_RELEASE_CONFIG"
for config in "$MOVE_TO_CONFIG" "$PICK_UP_CONFIG" "$CATCH_RELEASE_CONFIG"; do
  if grep -Eq '^[[:space:]]+resume_checkpoint:' "$config"; then
    echo "Staged phases hand off weights only; remove resume_checkpoint from $config" >&2
    exit 2
  fi
done
if [[ "$DRY_RUN" != "1" ]]; then
  for run_dir in "$MOVE_TO_RUN_DIR" "$PICK_UP_RUN_DIR" "$CATCH_RELEASE_RUN_DIR"; do
    if [[ -e "$run_dir" ]]; then
      echo "Refusing to reuse an existing staged run directory: $run_dir" >&2
      exit 2
    fi
  done
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

latest_checkpoint() {
  local run_dir="$1"
  local minimum_step="$2"
  local candidate candidate_step
  local best_checkpoint=""
  local best_step=-1
  for candidate in "$run_dir"/rl/step_*/smolvla_grpo_adapter.pt; do
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
    echo "No numbered GRPO checkpoint under $run_dir/rl." >&2
    return 2
  fi
  if [[ "$best_step" -lt "$minimum_step" ]]; then
    echo "Checkpoint step $best_step under $run_dir is below the required $minimum_step actions." >&2
    return 2
  fi
  printf '%s\n' "$best_checkpoint"
}

run_phase() {
  local label="$1"
  local launcher="$2"
  local config="$3"
  local run_name="$4"
  local steps="$5"
  local warmstart="$6"

  printf '\n=== %s ===\n' "$label"
  printf 'warmstart_checkpoint=%s\n' "$warmstart"
  (
    unset CHECKPOINT
    unset RLVLA_SMOLVLA_RESUME_CHECKPOINT
    WARMSTART_CHECKPOINT="$warmstart" \
      REPO_ROOT="$REPO_ROOT" \
      ENV_NAME="$ENV_NAME" \
      CONFIG="$config" \
      MAX_TRAIN_STEPS="$steps" \
      WORLDS_PER_RANK="$WORLDS_PER_RANK" \
      SMOLVLA_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT" \
      RUN_PREFLIGHT="$RUN_PREFLIGHT" \
      DRY_RUN="$DRY_RUN" \
      RUN_NAME="$run_name" \
      bash "$launcher"
  )
}

printf 'pipeline=move_to_then_pick_up_then_catch_release_plain_grpo\n'
printf 'phase_1_run_dir=%s\n' "$MOVE_TO_RUN_DIR"
printf 'phase_2_run_dir=%s\n' "$PICK_UP_RUN_DIR"
printf 'phase_3_run_dir=%s\n' "$CATCH_RELEASE_RUN_DIR"
printf 'phase_steps=%s/%s/%s\n' \
  "$MOVE_TO_TRAIN_STEPS" "$PICK_UP_TRAIN_STEPS" "$CATCH_RELEASE_TRAIN_STEPS"
printf 'cuda_visible_devices=%s ranks=2 worlds_per_rank=%s microbatch=%s\n' \
  "$CUDA_VISIBLE_DEVICES" "$WORLDS_PER_RANK" "$SMOLVLA_MICROBATCH_SIZE"
printf 'curriculum=per_instruction_success_gated lchol=disabled handoff=weights_only\n'

printf '\n=== Phase 1/3: fresh GRPO on move-to-object ===\n'
(
  unset CHECKPOINT
  unset RLVLA_SMOLVLA_RESUME_CHECKPOINT
  unset WARMSTART_CHECKPOINT
  REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$ENV_NAME" \
    CONFIG="$MOVE_TO_CONFIG" \
    MAX_TRAIN_STEPS="$MOVE_TO_TRAIN_STEPS" \
    WORLDS_PER_RANK="$WORLDS_PER_RANK" \
    SMOLVLA_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    ALLOW_LEGACY_SIMULATOR_CHECKPOINT="$ALLOW_LEGACY_SIMULATOR_CHECKPOINT" \
    RUN_PREFLIGHT="$RUN_PREFLIGHT" \
    DRY_RUN="$DRY_RUN" \
    RUN_NAME="$MOVE_TO_RUN_NAME" \
    bash "$MOVE_TO_LAUNCHER"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '\n=== Process boundary: each phase exits before the next starts ===\n'
  run_phase "Phase 2/3 (dry run): pick_up" "$PICK_UP_LAUNCHER" \
    "$PICK_UP_CONFIG" "$PICK_UP_RUN_NAME" "$PICK_UP_TRAIN_STEPS" \
    "$MOVE_TO_RUN_DIR/rl/latest.pt"
  run_phase "Phase 3/3 (dry run): catch/release" "$CATCH_RELEASE_LAUNCHER" \
    "$CATCH_RELEASE_CONFIG" "$CATCH_RELEASE_RUN_NAME" \
    "$CATCH_RELEASE_TRAIN_STEPS" "$PICK_UP_RUN_DIR/rl/latest.pt"
  exit 0
fi

MOVE_TO_CHECKPOINT="$(latest_checkpoint "$MOVE_TO_RUN_DIR" "$MOVE_TO_TRAIN_STEPS")"
run_phase "Phase 2/3: pick_up, warm-started from move-to" "$PICK_UP_LAUNCHER" \
  "$PICK_UP_CONFIG" "$PICK_UP_RUN_NAME" "$PICK_UP_TRAIN_STEPS" \
  "$MOVE_TO_CHECKPOINT"

PICK_UP_CHECKPOINT="$(latest_checkpoint "$PICK_UP_RUN_DIR" "$PICK_UP_TRAIN_STEPS")"
run_phase "Phase 3/3: catch/release, warm-started from pick_up" \
  "$CATCH_RELEASE_LAUNCHER" "$CATCH_RELEASE_CONFIG" \
  "$CATCH_RELEASE_RUN_NAME" "$CATCH_RELEASE_TRAIN_STEPS" \
  "$PICK_UP_CHECKPOINT"

printf '\nStaged plain-GRPO training completed successfully.\n'
printf 'final_run_dir=%s\n' "$CATCH_RELEASE_RUN_DIR"
