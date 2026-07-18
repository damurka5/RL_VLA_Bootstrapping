#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-8}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-lerobot/smolvla_base}"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-cdpr_mjwarp_two_gpu_smoke_${timestamp}}"
RUN_ROOT="$REPO_ROOT/runs/$RUN_NAME"
XML="$REPO_ROOT/robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
"${python_cmd[@]}" scripts/preflight_cdpr_mjlab.py \
  --config "$CONFIG" \
  --require-gpus 2 \
  --worlds "$WORLDS_PER_RANK" \
  --output "$RUN_ROOT/preflight.json"

common=(
  --standalone
  --nproc-per-node 2
  -m rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr
  --config "$CONFIG"
  --simulator-backend mjlab_mjwarp
  --mjwarp-xml-path "$XML"
  --worlds-per-rank "$WORLDS_PER_RANK"
  --groups-per-rank "$((WORLDS_PER_RANK / 8))"
  --grpo-group-size 8
  --grpo-trajectory-groups
  --grpo-dynamic-sampling
  --grpo-target-records-per-update 0
  --grpo-max-groups-per-update "$((WORLDS_PER_RANK / 8))"
  --base-checkpoint "$BASE_CHECKPOINT"
  --run-root-dir "$RUN_ROOT"
  --run-id rl
  --device cuda
  --distributed
  --mixed-precision bf16
  --chunk-size 8
  --replan-every 4
  --no-lock-non-commanded-axes
  --smolvla-model-image-size 256
  --smolvla-inference-microbatch-size "$WORLDS_PER_RANK"
  --hidden-dim 1024
  --ppo-epochs 1
  --minibatch-size 64
  --microbatch-size 64
  --save-every-steps 1
  --no-progress
  --progress-only
)

conda run --no-capture-output -n "$ENV_NAME" \
  torchrun "${common[@]}" --max-train-steps 1 \
  2>&1 | tee "$RUN_ROOT/first_update.log"

LATEST="$RUN_ROOT/rl/latest.pt"
if [[ ! -f "$LATEST" ]]; then
  echo "First smoke update did not create $LATEST" >&2
  exit 1
fi
resume_target="$("${python_cmd[@]}" -c \
  'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(int(p["global_step"])+1)' \
  "$LATEST")"

conda run --no-capture-output -n "$ENV_NAME" \
  torchrun "${common[@]}" \
  --resume-checkpoint "$LATEST" \
  --max-train-steps "$resume_target" \
  2>&1 | tee "$RUN_ROOT/resume_update.log"

"${python_cmd[@]}" -c \
  'import json, pathlib, sys, torch; p=pathlib.Path(sys.argv[1]); x=torch.load(p, map_location="cpu", weights_only=False); assert x["global_step"] >= int(sys.argv[2]); assert x["simulator_metadata"]["backend"] == "mjlab_mjwarp"; print(json.dumps({"ok": True, "checkpoint": str(p), "global_step": x["global_step"], "gradient_step": x["gradient_step"], "simulator_metadata": x["simulator_metadata"]}, indent=2, default=str))' \
  "$LATEST" "$resume_target" | tee "$RUN_ROOT/resume_report.json"

echo "two_gpu_smoke_dir=$RUN_ROOT"
