#!/usr/bin/env bash
# Re-run the oracle reference-episode harness on the remote box.
#
# What this proves
# ----------------
# The reset, the reward, the success predicate and the grasp detector are the
# real training code. PHYSICS defaults to `auto`, which takes the production
# MJWarp engine whenever CUDA and the MJWarp runtime are present -- so on the
# A40 box this is the training stack end to end. It falls back to the MuJoCo CPU
# reference elsewhere and says so; PHYSICS=mjlab_mjwarp makes an unavailable
# runtime an error rather than a silent downgrade, and PHYSICS=mujoco_cpu pins
# the CPU reference for comparison against older records.
#
# A CPU number and a GPU number are not comparable: different precision,
# different solver iteration order, GPU nondeterminism. The manifest records
# `physics_backend` and `exact_production_backend` for exactly this reason --
# check them before comparing a run against anything.
#
# Why two passes by default
# -------------------------
# A single pass says nothing. The oracle is a fixed script, so whether a given
# episode succeeds depends on which object it drew, and the resetter's RNG
# stream shifts whenever a reset-shaping knob is added. COMPARE_BASELINE=1 runs
# the config as written and then again with pick_up_prelifted_group_fraction
# forced to 0 -- the reset the run had before that knob existed -- into separate
# output directories, and prints both. Read the pair, never one number.
#
# On the default object pool one episode reliably fails: the scripted oracle
# opens the gripper before closing it, and a banana slips out. That is a
# property of the oracle, not of the reward. The 3/3 recorded under
# runs/cdpr_task_reference_episodes/ used the four-catalog set in
# REFERENCE_TARGET_CATALOGS below; set TARGET_CATALOGS to it to compare against
# that record.
#
# Usage (on the remote box):
#   cd /root/repo/RL_VLA_Bootstrapping
#   REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
#     bash scripts/render_cdpr_task_reference_episodes_remote.sh
#
# Any extra arguments are forwarded verbatim to the Python harness.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/cdpr_task_reference_episodes_remote}"
INSTRUCTIONS="${INSTRUCTIONS:-pick_up}"
EPISODES_PER_INSTRUCTION="${EPISODES_PER_INSTRUCTION:-3}"
# Empty means "whatever the config's target_object_pool holds".
TARGET_CATALOGS="${TARGET_CATALOGS:-}"
REFERENCE_TARGET_CATALOGS="robocasa_apple robocasa_tomato robocasa_orange robocasa_potato"
SEED="${SEED:-20260728}"
# Blank keeps the harness default, which is the config's
# random_workspace_start_distance_initial, i.e. step 0 of the run you are about
# to launch. Pass 0 to disable the approach-curriculum cap entirely.
START_DISTANCE_CAP="${START_DISTANCE_CAP:-}"
COMPARE_BASELINE="${COMPARE_BASELINE:-1}"
VIDEO="${VIDEO:-1}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
# auto | mjlab_mjwarp | mujoco_cpu. See the header.
PHYSICS="${PHYSICS:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Index within CUDA_VISIBLE_DEVICES, so cuda:0 is correct for a single-GPU mask.
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi
if [[ "$EPISODES_PER_INSTRUCTION" -lt 1 ]]; then
  echo "EPISODES_PER_INSTRUCTION must be positive." >&2
  exit 2
fi
if [[ "$COMPARE_BASELINE" != "0" && "$COMPARE_BASELINE" != "1" ]]; then
  echo "COMPARE_BASELINE must be 0 or 1." >&2
  exit 2
fi
if [[ "$VIDEO" != "0" && "$VIDEO" != "1" ]]; then
  echo "VIDEO must be 0 or 1." >&2
  exit 2
fi
if [[ "$VIDEO" == "1" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; re-run with VIDEO=0 for telemetry only." >&2
  exit 2
fi
case "$PHYSICS" in
  auto|mjlab_mjwarp|mujoco_cpu) ;;
  *)
    echo "PHYSICS must be auto, mjlab_mjwarp or mujoco_cpu." >&2
    exit 2
    ;;
esac

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$timestamp}"

export CUDA_VISIBLE_DEVICES
export MUJOCO_GL
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export RLVLA_CDPR_QUIET="${RLVLA_CDPR_QUIET:-1}"
export PYTHONUNBUFFERED=1

read -r -a instruction_values <<< "$INSTRUCTIONS"
# Forwarded verbatim to the harness. Guarded on length everywhere it is
# expanded: "${empty[@]}" is an unbound-variable error under `set -u` on the
# bash 4.x still shipped by several of the remote images.
script_args=("$@")

# $1 = output subdirectory, remaining args = extra harness flags.
build_cmd() {
  local pass_dir="$1"
  shift
  cmd=(
    conda run --no-capture-output -n "$ENV_NAME"
    python3 "$REPO_ROOT/scripts/render_cdpr_task_reference_episodes.py"
    --config "$CONFIG_PATH"
    --output "$pass_dir"
    --instructions "${instruction_values[@]}"
    --episodes-per-instruction "$EPISODES_PER_INSTRUCTION"
    --seed "$SEED"
    --physics "$PHYSICS"
    --device "$DEVICE"
  )
  if [[ -n "$TARGET_CATALOGS" ]]; then
    local catalog_values
    read -r -a catalog_values <<< "$TARGET_CATALOGS"
    cmd+=(--target-catalogs "${catalog_values[@]}")
  fi
  if [[ -n "$START_DISTANCE_CAP" ]]; then
    cmd+=(--start-distance-cap "$START_DISTANCE_CAP")
  fi
  if [[ "$VIDEO" == "0" ]]; then
    cmd+=(--no-video)
  fi
  if (( $# > 0 )); then
    cmd+=("$@")
  fi
  if (( ${#script_args[@]} > 0 )); then
    cmd+=("${script_args[@]}")
  fi
}

# Reads the manifest the harness wrote and prints one line per episode plus a
# success tally, so the two passes can be compared without opening the CSV.
# Deliberately the system python3 rather than `conda run`: this reads json from
# the stdlib only, and piping a heredoc through `conda run` is not reliable.
summarize() {
  local label="$1"
  local pass_dir="$2"
  python3 - "$label" "$pass_dir" <<'PY'
import json
import sys
from pathlib import Path

label, root = sys.argv[1], Path(sys.argv[2])
manifests = sorted(root.rglob("manifest.json"))
if not manifests:
    print(f"[{label}] no manifest.json under {root}")
    raise SystemExit(1)
successes = total = 0
for manifest_path in manifests:
    manifest = json.loads(manifest_path.read_text())
    overrides = manifest.get("metadata_overrides") or ["<config as written>"]
    print(f"[{label}] metadata_overrides={' '.join(overrides)}")
    print(
        f"[{label}] physics={manifest.get('physics_backend', '?')} "
        f"({manifest.get('physics_backend_selection', '?')}) "
        f"device={manifest.get('physics_device', '?')} "
        f"exact_production_backend={manifest.get('exact_production_backend')}"
    )
    for episode in manifest.get("episodes", []):
        total += 1
        successes += bool(episode.get("success"))
        print(
            f"[{label}] episode {episode.get('episode')} "
            f"{episode.get('instruction_type')} "
            f"{episode.get('target_catalog')}: "
            f"success={episode.get('success')} "
            f"reward={float(episode.get('final_reward', 0.0)):+.3f} "
            f"env_steps={episode.get('env_steps')}"
        )
print(f"[{label}] {successes}/{total} successful")
PY
}

configured_dir="$RUN_DIR/as_configured"
baseline_dir="$RUN_DIR/prelifted_fraction_0"

printf 'mode=oracle_reference_episodes\n'
printf 'config=%s\n' "$CONFIG_PATH"
printf 'output=%s\n' "$RUN_DIR"
printf 'physics_request=%s device=%s (resolved by the harness; see its [physics] line)\n' \
  "$PHYSICS" "$DEVICE"
printf 'instructions=%s episodes_per_instruction=%s seed=%s\n' \
  "$INSTRUCTIONS" "$EPISODES_PER_INSTRUCTION" "$SEED"
printf 'target_catalogs=%s\n' "${TARGET_CATALOGS:-config_pool}"
printf 'reference_target_catalogs=%s\n' "$REFERENCE_TARGET_CATALOGS"
printf 'start_distance_cap=%s video=%s mujoco_gl=%s\n' \
  "${START_DISTANCE_CAP:-config_initial}" "$VIDEO" "$MUJOCO_GL"
printf 'compare_baseline=%s\n' "$COMPARE_BASELINE"

build_cmd "$configured_dir"
printf 'command[as_configured]:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$COMPARE_BASELINE" == "1" ]]; then
  build_cmd "$baseline_dir" --metadata-override pick_up_prelifted_group_fraction=0
  printf 'command[prelifted_fraction_0]:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
fi

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
{
  build_cmd "$configured_dir"
  "${cmd[@]}"
  if [[ "$COMPARE_BASELINE" == "1" ]]; then
    build_cmd "$baseline_dir" --metadata-override pick_up_prelifted_group_fraction=0
    "${cmd[@]}"
  fi

  printf '\n===== summary =====\n'
  summarize as_configured "$configured_dir"
  if [[ "$COMPARE_BASELINE" == "1" ]]; then
    summarize prelifted_fraction_0 "$baseline_dir"
    printf '\nThe two passes are not step-for-step comparable: the pre-grasped\n'
    printf 'draw shifts everything sampled after it, so the starts differ even\n'
    printf 'where the catalogs match. Compare the success tallies and the\n'
    printf 'terminal rewards of the successes (expect ~5.7).\n'
  fi
  printf '\nTelemetry: %s/*/telemetry.csv\n' "$RUN_DIR"
  printf 'Manifests: %s/*/manifest.json\n' "$RUN_DIR"
  if [[ "$VIDEO" == "1" ]]; then
    printf 'Videos: %s/*/<instruction>/\n' "$RUN_DIR"
  fi
} 2>&1 | tee "$RUN_DIR/oracle_reference_episodes.log"
