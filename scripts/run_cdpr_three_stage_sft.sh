#!/usr/bin/env bash
# Refresh, train and evaluate the three-stage put_into student.
#
# The bounded comparison from the design, run at declared intervals rather than
# as one long unattended job:
#
#   refresh   re-derive state/prior from the stored frames under the STUDENT's
#             initialization and the final prompt; clears the stale marker
#   arm A     no training. The full-task baseline of the chosen initializer.
#   arm B     residual only, on the refreshed inputs, with independent fits
#             and matched rollouts at CHECK_EPOCHS plus the canonical EPOCHS
#   arm C     arm B's residual, then the action-expert LoRA from the pictures
#   eval      each arm on student_validation, unassisted, one prompt from step 0
#
# ARMS selects a subset. Arm C requires --frames and is skipped without them.
#
# Nothing here touches the final_test split. Lock the selection first, then run
# the held-out number once.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
export MUJOCO_GL="${MUJOCO_GL:-egl}"

CONFIG="${CONFIG:-configs/examples/cdpr_smolvla_three_stage_put_into.yaml}"
: "${RUN_DIR:?set RUN_DIR to the collection run directory}"
: "${STUDENT_INIT:?set STUDENT_INIT to the checkpoint the student starts from}"

SCENES="${SCENES:-$RUN_DIR/scenes.json}"
# The first bank measured 20 verified move transitions, 17 pickups and only
# five complete chains. Train the explicitly gated transition view by default;
# DATASET_VIEW=demonstrations keeps the strict end-to-end ablation available.
DATASET_VIEW="${DATASET_VIEW:-stage_transitions}"
case "$DATASET_VIEW" in
  stage_transitions|demonstrations) ;;
  *) echo "DATASET_VIEW must be stage_transitions or demonstrations" >&2; exit 2 ;;
esac
DATASET="${DATASET:-$RUN_DIR/dataset/$DATASET_VIEW.npz}"
FRAMES_GLOB="${FRAMES_GLOB:-$RUN_DIR/bank_shard*/frames_*.npz}"
REFRESHED="${REFRESHED:-$RUN_DIR/dataset_refreshed}"
DEVICE="${DEVICE:-cuda:0}"
WORLDS="${WORLDS:-64}"
EVAL_ROUNDS="${EVAL_ROUNDS:-2}"
EPOCHS="${EPOCHS:-20}"
# Small saved intervals with matched validation rollouts, not one long run.
# The historical broad residual-SFT composition regression is why: a validation
# imitation loss alone does not establish task success.
CHECK_EPOCHS="${CHECK_EPOCHS:-1 2 4}"
RETENTION_DATASET="${RETENTION_DATASET:-}"
RETENTION_FRACTION="${RETENTION_FRACTION:-0.0}"
ARMS="${ARMS:-A B C}"

run() { conda run --no-capture-output -n "$ENV_NAME" python3 "$@"; }
has_arm() { [[ " $ARMS " == *" $1 "* ]]; }

# sil_sft.py keeps the best validation-loss checkpoint from one invocation; it
# does not emit intermediate rollout checkpoints.  CHECK_EPOCHS used to be a
# dead variable, so a request such as CHECK_EPOCHS="1 2 4" silently trained and
# evaluated only the default 20-epoch arm.  Run each declared depth from the
# SAME initialization instead.  Residual fitting is cheap compared with the
# rollouts, and independent fits make every checkpoint's optimizer budget
# explicit in its own report.
read -r -a CHECK_EPOCH_LIST <<< "$CHECK_EPOCHS"
for checkpoint_epochs in "${CHECK_EPOCH_LIST[@]}" "$EPOCHS"; do
  if [[ ! "$checkpoint_epochs" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHECK_EPOCHS and EPOCHS must contain positive integers; got '$checkpoint_epochs'" >&2
    exit 2
  fi
done

EVALUATED_REPORTS=()

echo "=== refresh: re-derive state/prior under the student initialization ==="
echo "dataset view: $DATASET_VIEW ($DATASET)"
# NOT a replay. Replaying the bank under a different checkpoint was tried and
# destroyed it: 30 of 3072 episodes survived. What is wanted is a forward pass
# over the stored pictures, which is what this does.
run tools/audit/sil_refresh_priors.py \
  --dataset "$DATASET" \
  --frames $FRAMES_GLOB \
  --checkpoint "$STUDENT_INIT" \
  --device "$DEVICE" \
  --output "$REFRESHED"

RETENTION_ARGS=()
if [[ -n "$RETENTION_DATASET" ]]; then
  RETENTION_ARGS=(--retention-dataset "$RETENTION_DATASET"
                  --retention-fraction "$RETENTION_FRACTION")
fi

evaluate() {  # evaluate <name> <checkpoint>
  local name="$1" checkpoint="$2"
  run tools/audit/evaluate_cdpr_full_put_into.py \
    --config "$CONFIG" --checkpoint "$checkpoint" \
    --scene-manifest "$SCENES" --split student_validation \
    --worlds "$WORLDS" --rounds "$EVAL_ROUNDS" --device "$DEVICE" \
    --decisions "${EVAL_DECISIONS:-128}" \
    --settle-decisions "${SETTLE_DECISIONS:-0}" \
    --output "$RUN_DIR/eval_$name"
  EVALUATED_REPORTS+=("$RUN_DIR/eval_$name/evaluation.json")
}

if has_arm A; then
  echo "=== arm A: baseline, nothing trained ==="
  # The null. If the initializer already does the task, every later number is
  # measured against the wrong zero.
  evaluate armA "$STUDENT_INIT"
fi

if has_arm B; then
  echo "=== arm B: residual only ==="
  # Include EPOCHS as the canonical/final arm even when it is absent from the
  # requested checks.  De-duplicate values while preserving the user's order.
  ARM_B_EPOCHS=()
  for checkpoint_epochs in "${CHECK_EPOCH_LIST[@]}" "$EPOCHS"; do
    [[ " ${ARM_B_EPOCHS[*]} " == *" $checkpoint_epochs "* ]] || \
      ARM_B_EPOCHS+=("$checkpoint_epochs")
  done
  for checkpoint_epochs in "${ARM_B_EPOCHS[@]}"; do
    if [[ "$checkpoint_epochs" == "$EPOCHS" ]]; then
      arm_name="armB"
      arm_output="$RUN_DIR/sft_armB"
    else
      arm_name="armB_e${checkpoint_epochs}"
      arm_output="$RUN_DIR/sft_armB_e${checkpoint_epochs}"
    fi
    echo "--- arm B checkpoint: $checkpoint_epochs effective epochs ---"
    run tools/audit/sil_sft.py \
      --dataset "$REFRESHED/demonstrations.npz" \
      --checkpoint "$STUDENT_INIT" \
      --output "$arm_output" \
      --device "$DEVICE" --epochs "$checkpoint_epochs" \
      --split-by scene --sampler balanced \
      --val-fraction "${VAL_FRACTION:-0.1}" \
      "${RETENTION_ARGS[@]}"
    evaluate "$arm_name" "$arm_output/sil_sft_adapter.pt"
  done
fi

if has_arm C; then
  echo "=== arm C: residual, then action-expert LoRA from the pictures ==="
  # The LoRA stage anchors on the REFRESHED priors -- the ones drawn under the
  # final prompt -- not the teachers'. Same sampler as the residual stage, so
  # the two are not drawing from different distributions.
  run tools/audit/sil_sft.py \
    --dataset "$REFRESHED/demonstrations.npz" \
    --checkpoint "$STUDENT_INIT" \
    --output "$RUN_DIR/sft_armC" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --split-by scene --sampler balanced \
    --val-fraction "${VAL_FRACTION:-0.1}" \
    --frames $FRAMES_GLOB \
    --lora-epochs "${LORA_EPOCHS:-8}" \
    --lora-rows "${LORA_ROWS:-8192}" \
    "${RETENTION_ARGS[@]}"
  evaluate armC "$RUN_DIR/sft_armC/sil_sft_adapter.pt"
fi

echo
echo "Arms compared on student_validation, unassisted, one prompt from step 0:"
for report in "${EVALUATED_REPORTS[@]}"; do
  python3 - "$report" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
results = data["results"]
print(
    f"  {sys.argv[1].split('/')[-2]:12s} native={results['native']['rate']} "
    f"strict={results['strict']['rate']} "
    f"plate={results.get('strict_plate', {}).get('rate')} "
    f"bowl={results.get('strict_bowl', {}).get('rate')} "
    f"ladder={results['conditional_ladder']}"
)
PY
done
echo
echo "Choose a checkpoint on UNASSISTED strict full-chain success, with plate"
echo "and bowl reported separately and retention as a declared constraint."
echo "Lock that choice before running anything on the final_test split."
