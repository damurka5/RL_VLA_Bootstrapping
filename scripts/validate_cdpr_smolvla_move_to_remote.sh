#!/usr/bin/env bash
# move_to validation for one SmolVLA GRPO checkpoint, in three legs.
#
#   A. train_config             -- the metric. The checkpoint's own earned
#                                  approach cap, the config's 1-2 objects, the
#                                  trainer's validate_round. Nothing overridden,
#                                  so counts_toward_validation_metric is true.
#   B. multi_object             -- 2-3 objects on the desk, everything else as
#                                  in A. Prices the distractors alone.
#   C. multi_object_wrist_blind -- 2-3 objects AND a start cap past the trained
#                                  ladder top (0.19 m), which is the only way an
#                                  object leaves a camera here: the object grid
#                                  is fixed and sits inside the overview frustum
#                                  at every cell, so "invisible in one camera"
#                                  means the WRIST, and it is the EE-to-target
#                                  geometry that hides it. Videos are filtered to
#                                  episodes where the named object is certainly
#                                  outside the wrist frame at reset.
#
# B and C are diagnostics. Their manifests carry
# counts_toward_validation_metric: false and their episodes are in their own
# CSVs -- do not pool them with A.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/runs/phase4_move_to_iter0_resume_20260818_080928/rl/step_11009573/smolvla_grpo_adapter.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/move_to_validation}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
WORLDS="${WORLDS:-512}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SMOLVLA_MICROBATCH="${SMOLVLA_MICROBATCH:-256}"
FPS="${FPS:-6}"

# Leg A. 512 x 2 = 1024 episodes, which is validation_episodes_per_instruction.
TRAIN_ROUNDS="${TRAIN_ROUNDS:-2}"
TRAIN_TRACK_WORLDS="${TRAIN_TRACK_WORLDS:-32}"
TRAIN_VIDEOS="${TRAIN_VIDEOS:-5}"

# Legs B and C.
HARD_ROUNDS="${HARD_ROUNDS:-2}"
HARD_TRACK_WORLDS="${HARD_TRACK_WORLDS:-32}"
HARD_VIDEOS="${HARD_VIDEOS:-4}"
BLIND_ROUNDS="${BLIND_ROUNDS:-3}"
# The tool points its frame budget at the qualifying episodes of each reset
# (--video-filter), so a tracked world in leg C is a wrist-blind world rather
# than a 1-in-7 chance of being one. 32 is therefore ~96 wrist-blind recorded
# episodes over three rounds, not ~14.
BLIND_TRACK_WORLDS="${BLIND_TRACK_WORLDS:-32}"
BLIND_VIDEOS="${BLIND_VIDEOS:-4}"
BLIND_NEAR_MISS_VIDEOS="${BLIND_NEAR_MISS_VIDEOS:-3}"
MIN_SCENE_OBJECTS="${MIN_SCENE_OBJECTS:-2}"
MAX_SCENE_OBJECTS="${MAX_SCENE_OBJECTS:-3}"
# 0.33 m ~ the 0.24*sqrt(2) workspace diagonal, and the cap is what decides how
# often the wrist loses the object. Measured on THIS config's reset distribution
# with the 2-3 object override, 2048 resets per rung against the real resetter
# (tools/audit/start_distance_probe.py --backend fake), fraction of episodes
# whose target is CERTAINLY outside the wrist frame at reset:
#
#   cap 0.19 (trained top)  0.000    <- leg C cannot exist at the training cap
#   cap 0.25                0.027
#   cap 0.29                0.078-0.125
#   cap 0.33                0.145-0.172
#   cap 0.40                0.254
#
# The whole scene (every active object) out of the wrist frame is rarer:
# 0.012 / 0.027 / 0.043 / 0.066 at the same rungs -- use
# --video-filter all_objects_out_of_wrist and a higher cap if that is the case
# you want. The target stays inside the OVERVIEW frame at 1.0000 on every rung,
# which is what keeps the episode well posed rather than impossible.
BLIND_START_CAP="${BLIND_START_CAP:-0.33}"

SKIP_LEGS="${SKIP_LEGS:-}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

checkpoint_step="$(basename "$(dirname "$CHECKPOINT")")"
training_run="$(basename "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")")"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/${training_run}_${checkpoint_step}_${timestamp}}"

export CUDA_VISIBLE_DEVICES
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export RLVLA_CDPR_WRAPPER_LOG="${RLVLA_CDPR_WRAPPER_LOG:-0}"
export PYTHONUNBUFFERED=1

TOOL=(
  conda run --no-capture-output -n "$ENV_NAME"
  python3 "$REPO_ROOT/tools/audit/move_to_validation_videos.py"
  --checkpoint "$CHECKPOINT"
  --config "$CONFIG_PATH"
  --device "$DEVICE"
  --worlds "$WORLDS"
  --group-size "$GROUP_SIZE"
  --smolvla-microbatch "$SMOLVLA_MICROBATCH"
  --fps "$FPS"
)

leg_skipped() {
  [[ ",$SKIP_LEGS," == *",$1,"* ]]
}

run_leg() {
  local label="$1"
  shift
  if leg_skipped "$label"; then
    printf '[validate-move-to] skipping leg %s\n' "$label"
    return 0
  fi
  local cmd=("${TOOL[@]}" --label "$label" --output "$RUN_DIR/$label" "$@")
  printf '[validate-move-to] leg %s:' "$label"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

printf 'mode=move_to_object_validation_plus_videos\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'config=%s\n' "$CONFIG_PATH"
printf 'output=%s\n' "$RUN_DIR"
printf 'backend=mjlab_mjwarp harness=collector.validate_round (training validation)\n'
printf 'start_distance_cap=checkpoint_earned (legs A,B) %s (leg C)\n' "$BLIND_START_CAP"
printf 'scene_objects=config (leg A) %s-%s (legs B,C)\n' \
  "$MIN_SCENE_OBJECTS" "$MAX_SCENE_OBJECTS"
printf 'episodes=%s x %s (A) / %s (B) / %s (C)\n' \
  "$WORLDS" "$TRAIN_ROUNDS" "$HARD_ROUNDS" "$BLIND_ROUNDS"
printf 'metric_leg=train_config diagnostic_legs=multi_object,multi_object_wrist_blind\n'

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_DIR"
fi
cd "$REPO_ROOT"

{
  run_leg train_config \
    --rounds "$TRAIN_ROUNDS" \
    --track-worlds "$TRAIN_TRACK_WORLDS" \
    --max-videos "$TRAIN_VIDEOS" \
    --video-filter any

  run_leg multi_object \
    --rounds "$HARD_ROUNDS" \
    --track-worlds "$HARD_TRACK_WORLDS" \
    --max-videos "$HARD_VIDEOS" \
    --video-filter any \
    --metadata-override "min_scene_objects=$MIN_SCENE_OBJECTS" \
    --metadata-override "max_scene_objects=$MAX_SCENE_OBJECTS"

  run_leg multi_object_wrist_blind \
    --rounds "$BLIND_ROUNDS" \
    --track-worlds "$BLIND_TRACK_WORLDS" \
    --max-videos "$BLIND_VIDEOS" \
    --failure-videos "$BLIND_NEAR_MISS_VIDEOS" \
    --video-filter target_out_of_wrist \
    --start-distance-cap "$BLIND_START_CAP" \
    --metadata-override "min_scene_objects=$MIN_SCENE_OBJECTS" \
    --metadata-override "max_scene_objects=$MAX_SCENE_OBJECTS"

  if [[ "$DRY_RUN" != "1" ]]; then
    python3 - "$RUN_DIR" <<'PY'
"""One table across the legs, with the metric leg marked as such."""
import csv, json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for summary in sorted(run_dir.glob("*/validation_summary.csv")):
    leg = summary.parent.name
    manifest = summary.parent / "manifest.json"
    meta = json.loads(manifest.read_text()) if manifest.is_file() else {}
    with summary.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "leg": leg,
                    "counts_toward_validation_metric": meta.get(
                        "counts_toward_validation_metric", ""
                    ),
                    "start_distance_cap_m": json.dumps(
                        meta.get("start_distance_cap_m", "")
                    ),
                    "scene_object_range": json.dumps(
                        meta.get("scene_object_range", "")
                    ),
                    **row,
                }
            )
if rows:
    out = run_dir / "all_legs_validation_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[validate-move-to] combined table: {out}")
else:
    print("[validate-move-to] no per-leg summaries found to combine")
PY
    printf '\nVALIDATION METRIC (training configuration):\n'
    printf '  %s/train_config/validation_summary.csv\n' "$RUN_DIR"
    printf '  %s/train_config/episodes.csv\n' "$RUN_DIR"
    printf 'SUCCESS VIDEOS:\n'
    printf '  %s/train_config/videos\n' "$RUN_DIR"
    printf '  %s/multi_object/videos\n' "$RUN_DIR"
    printf '  %s/multi_object_wrist_blind/videos\n' "$RUN_DIR"
    printf 'DIAGNOSTIC CSVs (NOT part of the metric):\n'
    printf '  %s/multi_object/{episodes,validation_summary}.csv\n' "$RUN_DIR"
    printf '  %s/multi_object_wrist_blind/{episodes,validation_summary}.csv\n' "$RUN_DIR"
  fi
} 2>&1 | { if [[ "$DRY_RUN" == "1" ]]; then cat; else tee "$RUN_DIR/validation.log"; fi; }
