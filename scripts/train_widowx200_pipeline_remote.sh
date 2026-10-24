#!/usr/bin/env bash
# The whole WidowX-200 SmolVLA pipeline, one command, on the 2x A40 box.
#
#   1. GPU preflight of the arm's MJWarp backend against every stage config
#      (reachable resets, caught starts holding, scripted grasp+lift, cameras)
#   2. two production-size memory stress updates (40 decisions, max refill)
#   3. a toy-scale SMOKE pass over every stage and one SIL cycle, so a broken
#      handoff surfaces in about an hour instead of days in
#   4. the real pipeline (tools/train/run_widowx200_pipeline.py):
#        s1 move_to dense -> s2 pick_up dense -> s3 put_into dense
#        s4 move_to binary -> s5 pick_up binary -> s6 put_into binary
#           (binary = shaping off, fresh GRPO state, log_std re-initialized)
#        s7 all four instructions jointly, staged milestone reward
#      each stage stopping itself on its validation, each handing its best
#      checkpoint to the next weights-only, with SIL retention cycles whenever
#      an earlier instruction degrades.
#
# Resume: re-run with PIPELINE_DIR=<the same directory>; finished steps are
# skipped and an interrupted training stage continues from its latest.pt.
#
#   bash scripts/train_widowx200_pipeline_remote.sh
#   PIPELINE_DIR=runs/widowx200_pipeline_<ts> bash scripts/train_widowx200_pipeline_remote.sh
#
# Knobs: MODE=full|smoke|preflight (default full), SMOKE_FIRST=0 to skip the
# smoke pass, DRY_RUN=1 to write stage configs and print commands only,
# MEMORY_CHECK_FIRST=0 to skip the production-size memory stress check,
# RLVLA_HF_OFFLINE=1 to use the local Hugging Face cache only.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/widowx200_smolvla_pipeline.yaml}"
MODE="${MODE:-full}"
SMOKE_FIRST="${SMOKE_FIRST:-1}"
MEMORY_CHECK_FIRST="${MEMORY_CHECK_FIRST:-1}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

case "$MODE" in full|smoke|preflight) ;; *) echo "MODE must be full, smoke or preflight." >&2; exit 2;; esac
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "The pipeline trains on exactly two GPUs (CUDA_VISIBLE_DEVICES=0,1)." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline

timestamp="$(date +%Y%m%d_%H%M%S)"
PIPELINE_DIR="${PIPELINE_DIR:-$REPO_ROOT/runs/widowx200_pipeline_$timestamp}"
PIPELINE_DIR="$(mkdir -p "$PIPELINE_DIR" && cd "$PIPELINE_DIR" && pwd)"
# Never reuse a stress/smoke pass across changes to the code or source config.
validation_id="$(
  { git rev-parse HEAD; git diff HEAD -- .; git hash-object "$CONFIG" tools/audit/sil_artifacts.py; } |
    git hash-object --stdin
)"
SMOKE_DIR="${SMOKE_DIR:-${PIPELINE_DIR}_smoke_${validation_id:0:12}}"
MEMORY_DIR="${MEMORY_DIR:-${PIPELINE_DIR}_memory_${validation_id:0:12}}"
exec > >(tee -a "$PIPELINE_DIR/launcher.log") 2>&1

export RLVLA_EMBODIMENT=widowx200
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)

printf '[widowx200] repo %s @ %s\n' "$REPO_ROOT" "$(git rev-parse --short HEAD)"
printf '[widowx200] config %s\n' "$CONFIG"
printf '[widowx200] pipeline dir %s (mode %s, smoke first %s)\n' "$PIPELINE_DIR" "$MODE" "$SMOKE_FIRST"

# The RoboCasa visual meshes are gitignored and staged per machine; the CDPR
# runs on this box already staged them, so this is normally a no-op check.
if [[ "$DRY_RUN" != "1" ]]; then
if ! "${python_cmd[@]}" - "$REPO_ROOT/robots/widowx200/widowx200_mujoco/wx200_mjwarp_scene.xml" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import validate_object_assets
validate_object_assets(Path(sys.argv[1]))
print("[widowx200] RoboCasa object assets staged")
PYEOF
then
  echo "[widowx200] staging RoboCasa assets" >&2
  "${python_cmd[@]}" scripts/stage_cdpr_robocasa_assets.py
fi
huggingface_public_models_preflight "$ENV_NAME"
fi

driver=("${python_cmd[@]}" tools/train/run_widowx200_pipeline.py --config "$CONFIG" --devices cuda:0,cuda:1)
[[ "$DRY_RUN" == "1" ]] && driver+=(--dry-run)

# 1. GPU preflight (minutes). Skipped once it has passed for this directory.
if [[ "$DRY_RUN" != "1" && ( ! -f "$PIPELINE_DIR/preflight/PASSED" || "$(cat "$PIPELINE_DIR/preflight/PASSED")" != "$validation_id" ) ]]; then
  "${python_cmd[@]}" scripts/preflight_widowx200_pipeline.py \
    --config "$CONFIG" --device cuda:0 --worlds 512 --rounds 2 \
    --output "$PIPELINE_DIR/preflight"
  printf '%s\n' "$validation_id" > "$PIPELINE_DIR/preflight/PASSED"
fi
[[ "$MODE" == "preflight" ]] && exit 0

# The stress pass keeps the production world count, compilation, inference
# microbatch and LoRA backward. The toy pass alone cannot establish GPU fit.
if [[ "$MEMORY_CHECK_FIRST" == "1" && ! -f "$MEMORY_DIR/PASSED" ]]; then
  "${driver[@]}" --pipeline-dir "$MEMORY_DIR" --memory-check
  [[ "$DRY_RUN" == "1" ]] || touch "$MEMORY_DIR/PASSED"
fi

# 3. Smoke pass over every stage (about an hour), unless disabled or done.
if [[ "$MODE" == "smoke" || ( "$SMOKE_FIRST" == "1" && ! -f "$SMOKE_DIR/PASSED" ) ]]; then
  "${driver[@]}" --pipeline-dir "$SMOKE_DIR" --smoke
  [[ "$DRY_RUN" == "1" ]] || touch "$SMOKE_DIR/PASSED"
  printf '[widowx200] smoke pass complete: %s\n' "$SMOKE_DIR"
fi
[[ "$MODE" == "smoke" ]] && exit 0

# 4. The pipeline.
"${driver[@]}" --pipeline-dir "$PIPELINE_DIR"
printf '[widowx200] pipeline complete: %s/final_report.md\n' "$PIPELINE_DIR"
