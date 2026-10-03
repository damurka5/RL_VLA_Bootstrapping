#!/usr/bin/env bash
# GPU preflight for the latent-likelihood / zero-init correction pilot.
# Runs tools/audit/latent_correction_preflight.py on one GPU with the same
# Hugging Face and MuJoCo setup as the evaluation launchers. No optimization.
#
#   SOURCE_CHECKPOINT=runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006 \
#   EXPECTED_SOURCE_SHA256=<sha256sum of the adapter> \
#     bash scripts/preflight_cdpr_latent_correction_remote.sh
#
# After a one-update smoke run, add CANDIDATE_CHECKPOINT=<its step dir or .pt>
# to confirm evaluation loads and uses the learned correction.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006}"
CONFIG="${CONFIG:-configs/examples/cdpr_smolvla_three_stage_put_into_latent_correction.yaml}"
SCENES="${SCENES:-runs/three_stage/scenes_8192.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/latent_pilot_preflight_$(date +%Y%m%d_%H%M%S)}"
WORLDS="${WORLDS:-64}"
LR="${LR:-1e-5}"
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-}"

[[ -d "$SOURCE_CHECKPOINT" ]] && SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT/smolvla_grpo_adapter.pt"
[[ -f "$SOURCE_CHECKPOINT" ]] || { echo "Source adapter not found: $SOURCE_CHECKPOINT" >&2; exit 2; }
[[ -n "$CANDIDATE_CHECKPOINT" && -d "$CANDIDATE_CHECKPOINT" ]] && \
  CANDIDATE_CHECKPOINT="$CANDIDATE_CHECKPOINT/smolvla_grpo_adapter.pt"
echo "source=$(realpath "$SOURCE_CHECKPOINT") sha256=$(sha256sum "$SOURCE_CHECKPOINT" | cut -d' ' -f1)"

args=(--config "$CONFIG" --checkpoint "$SOURCE_CHECKPOINT" --scene-manifest "$SCENES"
  --output "$OUTPUT_DIR" --device cuda:0 --worlds "$WORLDS" --lr "$LR")
[[ -n "$EXPECTED_SOURCE_SHA256" ]] && args+=(--expected-source-sha256 "$EXPECTED_SOURCE_SHA256")
[[ -n "$CANDIDATE_CHECKPOINT" ]] && args+=(--candidate-checkpoint "$CANDIDATE_CHECKPOINT")
mkdir -p "$OUTPUT_DIR"
conda run --no-capture-output -n "$ENV_NAME" python3 tools/audit/latent_correction_preflight.py \
  "${args[@]}" 2>&1 | tee "$OUTPUT_DIR/preflight.log"
exit "${PIPESTATUS[0]}"
