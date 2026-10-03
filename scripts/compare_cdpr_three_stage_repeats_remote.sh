#!/usr/bin/env bash
# Repeated, matched standalone put_into evaluations of several checkpoints,
# then a scene-clustered comparison against the first (the baseline).
#
# Every checkpoint is evaluated REPEATS times on one fixed scene set. By default
# that set excludes the training run's in-run validation panel, because a
# checkpoint selected as that panel's best must not be re-scored on the scenes
# that selected it. Jobs are ordered repeat-major (A1 B1 C1 A2 B2 C2 ...) and
# dealt round-robin over GPUS, so time and device are balanced across
# checkpoints. Finished evaluations (evaluation.json present) are skipped, so
# the script can be re-run after an interruption.
#
#   CHECKPOINTS="runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006 step_63525522 step_66086572" \
#     bash scripts/compare_cdpr_three_stage_repeats_remote.sh
#
# A bare step_<N> is looked up as runs/*/rl/step_<N>; it must match exactly one.
# LABEL=PATH names a checkpoint explicitly, for runs whose step directories
# share a name (e.g. matched pilot arms: candidate=runs/A/rl/step_N control=...).
#
# Outputs under $OUT_ROOT:
#   <label>/rep<i>/          one evaluate_cdpr_three_stage_put_into_videos_remote.sh run
#   comparison.json          compare_put_into_repeats.py result
#   comparison.log           its printed summary
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
cd "$REPO_ROOT"

CHECKPOINTS="${CHECKPOINTS:-runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006 step_63525522 step_66086572}"
REPEATS="${REPEATS:-4}"
GPUS="${GPUS:-0 1}"
WORLDS="${WORLDS:-64}"
ROUNDS="${ROUNDS:-8}"                   # WORLDS x ROUNDS distinct scenes
EXCLUDE_VALIDATION_PANEL="${EXCLUDE_VALIDATION_PANEL:-1}"
OUT_ROOT="${OUT_ROOT:-runs/three_stage_put_into_repeats/$(date +%Y%m%d_%H%M%S)}"
# Optional: the pilot promotion rule (grasp/lift CI lower bound > -margin, e.g.
# 0.03) and a paired strict breakdown by target-y quartile from the manifest.
RETENTION_MARGIN="${RETENTION_MARGIN:-}"
SCENE_MANIFEST="${SCENE_MANIFEST:-}"

resolve_checkpoint() {
  local spec="$1" matches
  if [[ -e "$spec" ]]; then
    echo "$spec"; return
  fi
  if [[ "$spec" == step_* ]]; then
    mapfile -t matches < <(compgen -G "runs/*/rl/$spec" || true)
    if [[ "${#matches[@]}" -eq 1 ]]; then
      echo "${matches[0]}"; return
    fi
    echo "$spec: ${#matches[@]} matches under runs/*/rl/ (need exactly 1): ${matches[*]:-}" >&2
    exit 2
  fi
  echo "Checkpoint not found: $spec" >&2
  exit 2
}

labels=()
paths=()
for spec in $CHECKPOINTS; do
  explicit_label=""
  if [[ "$spec" == *=* ]]; then
    explicit_label="${spec%%=*}"
    spec="${spec#*=}"
  fi
  path="$(resolve_checkpoint "$spec")"
  [[ -d "$path" ]] && path="$path/smolvla_grpo_adapter.pt"
  [[ -f "$path" ]] || { echo "Adapter not found: $path" >&2; exit 2; }
  label="${explicit_label:-$(basename "$(dirname "$path")")}"
  for existing in "${labels[@]:-}"; do
    [[ "$existing" == "$label" ]] && { echo "Duplicate label $label." >&2; exit 2; }
  done
  labels+=("$label")
  paths+=("$path")
done
read -r -a gpu_list <<< "$GPUS"
[[ "${#gpu_list[@]}" -ge 1 ]] || { echo "GPUS is empty." >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -ge 2 ]] || { echo "REPEATS must be >= 2 (the noise floor needs two)." >&2; exit 2; }

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$OUT_ROOT/run.log") 2>&1
echo "out=$OUT_ROOT repeats=$REPEATS gpus=${gpu_list[*]} scenes=$((WORLDS * ROUNDS)) exclude_panel=$EXCLUDE_VALIDATION_PANEL"
for i in "${!labels[@]}"; do
  echo "  ${labels[$i]} = ${paths[$i]}$([[ $i -eq 0 ]] && echo '  (baseline)')"
done
git rev-parse HEAD | tee "$OUT_ROOT/git_commit.txt"

# Repeat-major job list, dealt round-robin over the GPUs.
declare -a gpu_jobs
job=0
for ((rep = 1; rep <= REPEATS; rep++)); do
  for i in "${!labels[@]}"; do
    slot=$((job % ${#gpu_list[@]}))
    gpu_jobs[$slot]+="${i}:${rep} "
    job=$((job + 1))
  done
done

run_gpu_queue() {
  local gpu="$1" queue="$2" item i rep dir
  for item in $queue; do
    i="${item%%:*}"; rep="${item##*:}"
    dir="$OUT_ROOT/${labels[$i]}/rep${rep}"
    if [[ -f "$dir/evaluation.json" ]]; then
      echo "[gpu $gpu] skip ${labels[$i]} rep$rep (done)"; continue
    fi
    rm -rf "$dir"
    echo "[gpu $gpu] start ${labels[$i]} rep$rep"
    CHECKPOINT="${paths[$i]}" OUTPUT_DIR="$dir" \
      CUDA_VISIBLE_DEVICES="$gpu" DEVICE=cuda:0 \
      WORLDS="$WORLDS" ROUNDS="$ROUNDS" DISTINCT_SCENES=1 VIDEOS=0 \
      EXCLUDE_VALIDATION_PANEL="$EXCLUDE_VALIDATION_PANEL" \
      bash "$SCRIPT_DIR/evaluate_cdpr_three_stage_put_into_videos_remote.sh" \
      > "$dir.stdout" 2>&1 \
      || { echo "[gpu $gpu] FAILED ${labels[$i]} rep$rep, see $dir.stdout" >&2; return 1; }
    echo "[gpu $gpu] done ${labels[$i]} rep$rep: $(grep '\] unassisted' "$dir/eval.log" | tail -1 || true)"
  done
}

mkdir -p "${labels[@]/#/$OUT_ROOT/}"
pids=()
for slot in "${!gpu_list[@]}"; do
  run_gpu_queue "${gpu_list[$slot]}" "${gpu_jobs[$slot]:-}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "Some evaluations failed; re-run the same command to resume." >&2; exit 1; }

compare_args=()
for i in "${!labels[@]}"; do
  dirs=""
  for ((rep = 1; rep <= REPEATS; rep++)); do dirs+="$OUT_ROOT/${labels[$i]}/rep${rep},"; done
  compare_args+=(--checkpoint "${labels[$i]}=${dirs%,}")
done
[[ -n "$RETENTION_MARGIN" ]] && compare_args+=(--retention-margin "$RETENTION_MARGIN")
[[ -n "$SCENE_MANIFEST" ]] && compare_args+=(--scene-manifest "$SCENE_MANIFEST")
conda run --no-capture-output -n "$ENV_NAME" python3 tools/audit/compare_put_into_repeats.py \
  "${compare_args[@]}" --output "$OUT_ROOT/comparison.json" \
  | tee "$OUT_ROOT/comparison.log" | sed -n '/^=== /,$p'
echo "=== wrote $OUT_ROOT/comparison.json ==="
