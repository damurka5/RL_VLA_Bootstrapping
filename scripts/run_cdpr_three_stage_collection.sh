#!/usr/bin/env bash
# Collect the three-stage put_into demonstration bank. Trains nothing.
#
# Five bounded steps, each resumable and each writing an artefact the next one
# verifies rather than assumes:
#
#   1 yaw       solve the fixed pickup yaw from the MJCF's camera extrinsics
#   2 scenes    generate and audit the full-task scene manifest
#   3 select    screen the teacher candidates on the teacher_selection split
#   4 record    run the chains on the collection split, one shard per GPU
#   5 dataset   assemble strict chains and verified stage-transition SFT views
#
# STEPS is a space-separated subset, so a re-run can skip what already
# succeeded: STEPS="record dataset" ./scripts/run_cdpr_three_stage_collection.sh
#
# Step 3 is skipped unless CANDIDATES is set. With teachers already chosen,
# pass TEACHER_MOVE_TO / TEACHER_PICK_UP / TEACHER_PLACEMENT directly.
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
RUN_DIR="${RUN_DIR:-runs/three_stage_$(date +%Y%m%d_%H%M%S)}"
SCENES="${SCENES:-$RUN_DIR/scenes.json}"
YAW="${YAW:-$RUN_DIR/yaw_calibration.json}"
SCENE_COUNT="${SCENE_COUNT:-1024}"
SCENE_SEED="${SCENE_SEED:-20260910}"
WORLDS="${WORLDS:-64}"
ROUNDS="${ROUNDS:-4}"
MICROBATCH="${MICROBATCH:-32}"
GPUS="${GPUS:-0 1}"
STEPS="${STEPS:-yaw scenes select record dataset}"
# These are one protocol, shared by teacher screening and collection. The
# defaults are the remotely verified handoff: an explicit XY bridge, a 48
# decision alignment budget, and the final destination prompt during pickup.
# Setting one only for selection used to produce a manifest that collection
# could not reproduce.
ALIGN_XY_CENTRING="${ALIGN_XY_CENTRING:-1}"
ALIGN_DECISIONS="${ALIGN_DECISIONS:-48}"
ALIGN_XY_DEADBAND="${ALIGN_XY_DEADBAND:-0.005}"
ALIGN_XY_ABORT="${ALIGN_XY_ABORT:-0.009}"
ALIGN_YAW_SERVO_GAIN="${ALIGN_YAW_SERVO_GAIN:-0.35}"
ALIGN_DESCENT_GAIN="${ALIGN_DESCENT_GAIN:-1.0}"
ALIGN_HANDOFF_AT_CLEARANCE="${ALIGN_HANDOFF_AT_CLEARANCE:-0}"
PICKUP_PROMPT="${PICKUP_PROMPT:-destination}"
# How many consecutive decision boundaries the alignment conjunction must hold
# before the pickup teacher takes over. Empty means "whatever the calibration
# file says" (2), which is the shipped protocol. Screen a change before
# collecting with it: it alters the recorded handoff distribution, so two banks
# collected at different values are not the same dataset.
ALIGN_CONSECUTIVE_DECISIONS="${ALIGN_CONSECUTIVE_DECISIONS:-}"
# Reject target presentations the open fingers cannot bracket at the calibrated
# pickup yaw. On by default: roughly half of potato draws are ungraspable at a
# pinned gripper yaw, and the generator resamples them into graspable
# orientations rather than dropping the stratum, so the census is unchanged and
# the chains are not spent on impossible work.
SCENE_CLEARANCE_FILTER="${SCENE_CLEARANCE_FILTER:-1}"

mkdir -p "$RUN_DIR"
run() { conda run --no-capture-output -n "$ENV_NAME" python3 "$@"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

STAGED_PROTOCOL_ARGS=(
  --align-decisions "$ALIGN_DECISIONS"
  --align-xy-deadband "$ALIGN_XY_DEADBAND"
  --align-xy-abort "$ALIGN_XY_ABORT"
  --align-yaw-servo-gain "$ALIGN_YAW_SERVO_GAIN"
  --align-descent-gain "$ALIGN_DESCENT_GAIN"
  --pickup-prompt "$PICKUP_PROMPT"
)
if [[ "$ALIGN_XY_CENTRING" == "1" ]]; then
  STAGED_PROTOCOL_ARGS+=(--align-xy-centring)
fi
if [[ "$ALIGN_HANDOFF_AT_CLEARANCE" == "1" ]]; then
  STAGED_PROTOCOL_ARGS+=(--align-handoff-at-clearance)
fi
if [[ -n "$ALIGN_CONSECUTIVE_DECISIONS" ]]; then
  STAGED_PROTOCOL_ARGS+=(--align-consecutive-decisions "$ALIGN_CONSECUTIVE_DECISIONS")
fi

if has_step yaw; then
  echo "=== 1/5 yaw calibration ==="
  # Kinematics from the MJCF: no GPU, no checkpoint. The reported workspace
  # residual is the cost of the fixed-angle choice and belongs in the record.
  MUJOCO_GL=disable run tools/audit/calibrate_cdpr_pickup_yaw.py \
    --output "$YAW" \
    --tolerance-degrees "${YAW_TOLERANCE_DEG:-5.0}" \
    --calibration-z "${YAW_SAFE_Z:-0.26}"
fi

if has_step scenes; then
  echo "=== 2/5 scene manifest ==="
  # Success radii come from the config; this never widens them. The audit at
  # the end refuses a manifest with overlapping splits or with any scene whose
  # object starts inside its destination's success radius.
  SCENE_ARGS=()
  if [[ "$SCENE_CLEARANCE_FILTER" == "1" ]]; then
    # The filter needs the calibrated yaw, so the yaw step has to have run.
    # Reading it from the file rather than taking a number keeps the manifest
    # and the collector agreeing on ONE measured angle.
    if [[ ! -f "$YAW" ]]; then
      echo "SCENE_CLEARANCE_FILTER=1 needs $YAW; run the yaw step first or" \
           "set SCENE_CLEARANCE_FILTER=0." >&2
      exit 1
    fi
    SCENE_ARGS+=(--yaw-calibration "$YAW")
  fi
  run tools/audit/build_cdpr_composition_scenes.py \
    --config "$CONFIG" --count "$SCENE_COUNT" --seed "$SCENE_SEED" \
    "${SCENE_ARGS[@]}" \
    --output "$SCENES"
fi

if has_step select && [[ -n "${CANDIDATES:-}" ]]; then
  echo "=== 3/5 teacher screening ==="
  CANDIDATE_ARGS=()
  for spec in $CANDIDATES; do CANDIDATE_ARGS+=(--candidate "$spec"); done
  run tools/audit/select_cdpr_stage_teachers.py \
    --config "$CONFIG" --scene-manifest "$SCENES" --yaw-calibration "$YAW" \
    "${CANDIDATE_ARGS[@]}" \
    "${STAGED_PROTOCOL_ARGS[@]}" \
    --worlds "$WORLDS" --rounds "${SELECT_ROUNDS:-1}" \
    --microbatch "$MICROBATCH" --device "cuda:${GPUS%% *}" \
    --output "$RUN_DIR/teacher_selection"
  SELECTED="$RUN_DIR/teacher_selection/selected_teachers.json"
  TEACHER_MOVE_TO="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['teachers']['move_to']['checkpoint'])" "$SELECTED")"
  TEACHER_PICK_UP="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['teachers']['pick_up']['checkpoint'])" "$SELECTED")"
  TEACHER_PLACEMENT="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['teachers']['placement']['checkpoint'])" "$SELECTED")"
elif has_step select; then
  echo "=== 3/5 teacher screening SKIPPED (CANDIDATES unset) ==="
fi

: "${TEACHER_MOVE_TO:?set TEACHER_MOVE_TO or run the select step with CANDIDATES}"
: "${TEACHER_PICK_UP:?set TEACHER_PICK_UP or run the select step with CANDIDATES}"
: "${TEACHER_PLACEMENT:?set TEACHER_PLACEMENT or run the select step with CANDIDATES}"

if has_step record; then
  echo "=== 4/5 collection ==="
  # One worker per GPU with DISJOINT scenes. Two shards sharing a scene would
  # produce two episodes with one scene_uid, which the split treats as one unit
  # and the uncertainty analysis as one cluster -- they would look independent
  # and be counted twice.
  SHARD=0
  NUM_SHARDS="$(echo "$GPUS" | wc -w | tr -d ' ')"
  PIDS=()
  for gpu in $GPUS; do
    OUT="$RUN_DIR/bank_shard${SHARD}"
    CUDA_VISIBLE_DEVICES="$gpu" run tools/audit/record_cdpr_staged_put_into.py \
      --config "$CONFIG" --scene-manifest "$SCENES" --split collection \
      --teacher "move_to=$TEACHER_MOVE_TO" \
      --teacher "pick_up=$TEACHER_PICK_UP" \
      --teacher "placement=$TEACHER_PLACEMENT" \
      --yaw-calibration "$YAW" \
      "${STAGED_PROTOCOL_ARGS[@]}" \
      --worlds "$WORLDS" --rounds "$ROUNDS" \
      --shard "$SHARD" --num-shards "$NUM_SHARDS" \
      --microbatch "$MICROBATCH" --device cuda:0 \
      --move-decisions "${MOVE_DECISIONS:-32}" \
      --pickup-decisions "${PICKUP_DECISIONS:-32}" \
      --placement-decisions "${PLACEMENT_DECISIONS:-64}" \
      --settle-decisions "${SETTLE_DECISIONS:-0}" \
      --output "$OUT" 2>&1 | sed "s/^/[shard$SHARD] /" &
    PIDS+=($!)
    SHARD=$((SHARD + 1))
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done
fi

if has_step dataset; then
  echo "=== 5/5 dataset ==="
  # Frame coverage must be complete. A bank whose resolvable rows are a subset
  # is selected by "whose episodes happened to keep pictures", which is a
  # selection nobody chose.
  run tools/audit/build_cdpr_staged_sft_dataset.py \
    --records "$RUN_DIR"/bank_shard*/staged_*.npz \
    --frames "$RUN_DIR"/bank_shard*/frames_*.npz \
    --output "$RUN_DIR/dataset"
fi

echo
echo "Collection artefacts in $RUN_DIR"
echo "  scenes:     $SCENES"
echo "  yaw:        $YAW"
echo "  bank:       $RUN_DIR/bank_shard*/collection.json"
echo "  dataset:    $RUN_DIR/dataset/dataset.json"
echo "  strict:     $RUN_DIR/dataset/demonstrations.npz"
echo "  transitions:$RUN_DIR/dataset/stage_transitions.npz"
echo
echo "The dataset's priors are STALE by construction: its rows were relabelled"
echo "to the student prompt but state/prior were computed under the teachers'."
echo "Run scripts/run_cdpr_three_stage_sft.sh, which refreshes them first."
