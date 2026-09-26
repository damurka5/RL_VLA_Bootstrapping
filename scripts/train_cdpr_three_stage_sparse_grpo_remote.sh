#!/usr/bin/env bash
# End-to-end put_into GRPO with ordered approach / pickup / placement credit.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_three_stage_put_into.yaml}"
SCENES="${SCENES:-$REPO_ROOT/runs/three_stage/scenes_8192.json}"
# No default initializer. The September 13 run silently started from the RL
# step_2117145 adapter instead of the SFT result it was meant to continue
# (provenance/started_from_sft=0). Choose one after the matched evaluation.
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-}"
# Continue a three-stage run: restores weights, Adam moments, global step,
# update index and validation state. MAX_TRAIN_STEPS is then the TOTAL step
# budget, including the steps already in the checkpoint. Manifest resets carry
# no curriculum, so the weights-only rule for phase handoffs does not apply.
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000000}"
MAX_UPDATES="${MAX_UPDATES:-10}"
# Optional single-knob overrides; empty keeps the config's value.
# PPO_EPOCHS replaces ppo_epochs. LR_OVERRIDE sets the residual optimizer's rate
# AFTER the checkpoint loads -- a resume otherwise keeps the saved rate and
# ignores learning_rate in the YAML. The active rate is logged per update.
PPO_EPOCHS="${PPO_EPOCHS:-}"
LR_OVERRIDE="${LR_OVERRIDE:-}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
source "$SCRIPT_DIR/run_naming.sh"
RUN_NAME="$(cdpr_compose_run_name "three_stage_sparse_grpo")"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
cdpr_guard_run_dir "$RUN_DIR"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$SCENES" ]] || { echo "Scene manifest not found: $SCENES" >&2; exit 2; }
if [[ -n "$WARMSTART_CHECKPOINT" && -n "$RESUME_CHECKPOINT" ]]; then
  echo "Set WARMSTART_CHECKPOINT or RESUME_CHECKPOINT, not both." >&2; exit 2
fi
if [[ -z "$WARMSTART_CHECKPOINT" && -z "$RESUME_CHECKPOINT" ]]; then
  cat >&2 <<'MSG'
An initializer is required. Pick one explicitly, e.g.
  WARMSTART_CHECKPOINT=runs/<three_stage_sft_run>/sft_armC/sil_sft_adapter.pt   (fresh run)
  RESUME_CHECKPOINT=runs/<three_stage_sparse_grpo_run>/rl/latest.pt             (continue)
MSG
  exit 2
fi
INIT_CHECKPOINT="${RESUME_CHECKPOINT:-$WARMSTART_CHECKPOINT}"
INIT_MODE="$([[ -n "$RESUME_CHECKPOINT" ]] && echo resume || echo warm_start)"
[[ -f "$INIT_CHECKPOINT" ]] || {
  echo "Initializer checkpoint not found: $INIT_CHECKPOINT" >&2; exit 2;
}
if [[ "$INIT_MODE" == "resume" ]]; then
  resumed_step="$(conda run --no-capture-output -n "$ENV_NAME" python3 -c \
    'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("global_step", 0)))' \
    "$INIT_CHECKPOINT" | tail -1)"
  if [[ ! "$resumed_step" =~ ^[0-9]+$ || "$MAX_TRAIN_STEPS" -le "$resumed_step" ]]; then
    echo "MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS is the TOTAL budget and must exceed the checkpoint's global_step=$resumed_step." >&2
    exit 2
  fi
  echo "resume: checkpoint global_step=$resumed_step, training $((MAX_TRAIN_STEPS - resumed_step)) more steps"
fi
if [[ -n "$PPO_EPOCHS" && ! "$PPO_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PPO_EPOCHS must be a positive integer." >&2; exit 2
fi
if [[ ! "$MAX_UPDATES" =~ ^[0-9]+$ ]]; then
  echo "MAX_UPDATES must be a nonnegative integer (0 removes the diagnostic update cap)." >&2; exit 2
fi
if [[ "$WORLDS_PER_RANK" -lt 8 || $((WORLDS_PER_RANK % 8)) -ne 0 ]]; then
  echo "WORLDS_PER_RANK must be a positive multiple of 8." >&2; exit 2
fi
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "Exactly two CUDA devices are required." >&2; exit 2
fi

# Refuse the common silent protocol changes before allocating either GPU.
conda run --no-capture-output -n "$ENV_NAME" python3 - \
  "$CONFIG" "$SCENES" "$PPO_EPOCHS" "$LR_OVERRIDE" <<'PYEOF'
import sys
import yaml
sys.path.insert(0, ".")
from rl_vla_bootstrapping.core.commands import append_cli_arg
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import sparse_binary_reward_requested
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import read_manifest, select_split

config_path, manifest_path, ppo_epochs, lr_override = sys.argv[1:]
project = load_project_config(config_path)
metadata = dict(project.task.metadata or {})
raw = yaml.safe_load(open(config_path, encoding="utf-8"))
section = raw["training"]["rl"]["args"]
argv = []
for key, value in section.items():
    append_cli_arg(argv, key, value)
if ppo_epochs:
    append_cli_arg(argv, "ppo_epochs", int(ppo_epochs))
if lr_override:
    append_cli_arg(argv, "optimizer_lr_override", float(lr_override))
args = parse_args(argv)
print(f"[three-stage] optimization: ppo_epochs={args.ppo_epochs} "
      f"(config {section.get('ppo_epochs')}), learning_rate={args.learning_rate:g}, "
      f"optimizer_lr_override={args.optimizer_lr_override}")
scenes, _ = read_manifest(manifest_path)
collection = select_split(scenes, "collection")
validation = select_split(scenes, "student_validation")
final_test = select_split(scenes, "final_test")
failures = []
if not args.three_stage_sparse_credit:
    failures.append("three_stage_sparse_credit is off")
if args.split_credit_at_grasp:
    failures.append("legacy two-stage split_credit_at_grasp is also on")
if not sparse_binary_reward_requested(metadata):
    failures.append("sparse_binary_reward is off")
if args.three_stage_train_split != "collection":
    failures.append(f"training split is {args.three_stage_train_split!r}, not collection")
if args.three_stage_validation_split != "student_validation":
    failures.append("validation split is not student_validation")
if args.three_stage_horizon_decisions != 128:
    failures.append(f"horizon is {args.three_stage_horizon_decisions}, not 128 decisions")
if not bool(getattr(args, "train_vla_lora", False)):
    failures.append("train_vla_lora is off; attach/load the initializer LoRA")
if bool(getattr(args, "vla_lora_updates_enabled", True)):
    failures.append("VLA LoRA updates are on; decision-zero capture cannot balance all three stages")
if not bool(metadata.get("placement_wrong_drop_requires_lift", False)):
    failures.append("placement_wrong_drop_requires_lift is off; failed tabletop grasps would end episodes")
if not collection or not validation:
    failures.append("collection or student_validation split is empty")
print(f"[three-stage] scenes: collection={len(collection)} "
      f"student_validation={len(validation)} final_test_locked={len(final_test)}")
mass = args.three_stage_stage_loss_weights or [1 / 3, 1 / 3, 1 / 3]
print("[three-stage] rewards: approach=1 pickup=1 placement=1; independent group "
      "advantages; stage loss mass approach/pickup/placement = "
      + "/".join(f"{value:.3f}" for value in mass)
      + f"; full-task bonus on approach/pickup = {args.three_stage_full_task_bonus:g}"
      + "; negative bonus on an achieved milestone x"
      + f"{args.three_stage_full_task_bonus_achieved_negative_scale:g}")
if failures:
    for failure in failures:
        print(f"[three-stage] REFUSING: {failure}", file=sys.stderr)
    raise SystemExit(2)
print("[three-stage] preflight clean; final_test is not selected by this runner")
PYEOF

mkdir -p "$RUN_DIR"
unset RLVLA_SMOLVLA_RESUME_CHECKPOINT RLVLA_SMOLVLA_WARMSTART_CHECKPOINT RLVLA_CDPR_DEMO_BANK
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_SMOLVLA_NPROC_PER_NODE=2
export RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
export RLVLA_SMOLVLA_MJWARP_MAX_UPDATES="$MAX_UPDATES"
export RLVLA_MJWARP_WORLDS_PER_RANK="$WORLDS_PER_RANK"
export RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE"
unset RLVLA_SMOLVLA_PPO_EPOCHS RLVLA_SMOLVLA_OPTIMIZER_LR_OVERRIDE
[[ -n "$PPO_EPOCHS" ]] && export RLVLA_SMOLVLA_PPO_EPOCHS="$PPO_EPOCHS"
[[ -n "$LR_OVERRIDE" ]] && export RLVLA_SMOLVLA_OPTIMIZER_LR_OVERRIDE="$LR_OVERRIDE"
if [[ "$INIT_MODE" == "resume" ]]; then
  export RLVLA_SMOLVLA_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
else
  export RLVLA_SMOLVLA_WARMSTART_CHECKPOINT="$WARMSTART_CHECKPOINT"
fi
export RLVLA_CDPR_THREE_STAGE_SCENE_MANIFEST="$SCENES"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
train_cmd=("${python_cmd[@]}" -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG" --stage rl --run-name "$RUN_NAME" --execute)

printf 'run_dir=%s\nmanifest=%s\n%s=%s\n' \
  "$RUN_DIR" "$SCENES" "$INIT_MODE" "$INIT_CHECKPOINT"
printf 'max_train_steps=%s worlds_per_rank=%s groups_per_rank=%s\n' \
  "$MAX_TRAIN_STEPS" "$WORLDS_PER_RANK" "$((WORLDS_PER_RANK / 8))"
printf 'max_updates=%s; globally empty update stop: 3 consecutive cycles\n' "$MAX_UPDATES"
printf 'selection metric: independent strict placement; native and milestone diagnostics separately\n'
printf 'trainable component: residual actor; initializer LoRA is loaded and frozen\n'
printf 'command:'; printf ' %q' "${train_cmd[@]}"; printf '\n'
[[ "$DRY_RUN" == "1" ]] && exit 0

"${python_cmd[@]}" - "$INIT_CHECKPOINT" "$SCENES" "$RUN_DIR/launch_provenance.json" "$MAX_UPDATES" "$MAX_TRAIN_STEPS" "$INIT_MODE" "$CONFIG" "$PPO_EPOCHS" "$LR_OVERRIDE" <<'PYPROVENANCE'
import hashlib, json, pathlib, subprocess, sys
checkpoint, scenes, output, updates, steps, mode, config, ppo_epochs, lr_override = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
record = {"init_mode": mode, "checkpoint": str(pathlib.Path(checkpoint).resolve()),
          "checkpoint_sha256": digest(checkpoint), "scene_manifest": str(pathlib.Path(scenes).resolve()),
          "scene_manifest_sha256": digest(scenes),
          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "max_updates": int(updates), "max_train_steps": int(steps),
          "reward_protocol": "three_stage_accessible_v5_full_task_bonus", "termination_protocol": "wrong_place_requires_held_lift", "outcome_protocol": "independent_strict_full_task_v1",
          "lora_updates_enabled": False}
try:
    import yaml
    rl_args = yaml.safe_load(open(config, encoding="utf-8"))["training"]["rl"]["args"]
    record["stage_loss_weights"] = rl_args.get("three_stage_stage_loss_weights")
    record["full_task_bonus"] = rl_args.get("three_stage_full_task_bonus", 0.0)
    record["full_task_bonus_achieved_negative_scale"] = rl_args.get(
        "three_stage_full_task_bonus_achieved_negative_scale", 1.0)
    record["ppo_epochs"] = int(ppo_epochs) if ppo_epochs else rl_args.get("ppo_epochs")
    record["ppo_epochs_overridden"] = bool(ppo_epochs)
    record["learning_rate_yaml"] = rl_args.get("learning_rate")
    record["optimizer_lr_override"] = float(lr_override) if lr_override else None
except Exception as error:  # never block a launch on provenance
    record["stage_loss_weights"] = f"unavailable: {error}"
try:
    sys.path.insert(0, ".")
    from tools.audit.checkpoint_provenance import read_provenance
    record["started_from_sft"] = read_provenance(pathlib.Path(checkpoint))["sil_sft"] is not None
except Exception as error:  # never block a launch on the stamp reader
    record["started_from_sft"] = f"unavailable: {error}"
pathlib.Path(output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PYPROVENANCE

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  "${python_cmd[@]}" -m unittest discover -s tests -p test_grpo_three_stage_sparse_credit.py
  "${python_cmd[@]}" -m unittest discover -s tests -p test_three_stage_repairs.py
  huggingface_public_models_preflight "$ENV_NAME"
  "${python_cmd[@]}" scripts/preflight_cdpr_mjlab.py \
    --config "$CONFIG" --require-gpus 2 --worlds "$WORLDS_PER_RANK" \
    --output "$RUN_DIR/preflight.json"
fi
"${train_cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
