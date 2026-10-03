#!/usr/bin/env bash
# Latent-likelihood / zero-init correction pilot over a legacy three-stage
# checkpoint. Spec: docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_IMPLEMENTATION.md
#
# Two matched arms, identical except for the actor architecture:
#   ARM=candidate  frozen_reference_logit_correction_v1: the source residual is
#                  frozen as a reference; a zero-initialized unbounded
#                  correction (and log_std) trains.
#   ARM=control    bounded_residual_v0: the source residual itself trains.
# Both score the unclipped latent Gaussian sample conditioned on the realized
# episode offset (latent_gaussian_conditional_offset_v1) with a fresh optimizer.
#
# Exactly one initializer:
#   LEGACY_INIT_CHECKPOINT=<legacy adapter>  weights-only conversion; pilot step 0
#   RESUME_CHECKPOINT=<pilot checkpoint>     continue a pilot run of the same arm
#
# MAX_TRAIN_STEPS (pilot selected actions, TOTAL including a resumed run's) and
# MAX_UPDATES (0 = no update cap) have no defaults: the historical launcher's
# 10-update cap must never silently truncate a budgeted pilot.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_three_stage_put_into_latent_correction.yaml}"
SCENES="${SCENES:-$REPO_ROOT/runs/three_stage/scenes_8192.json}"
ARM="${ARM:-}"
LEGACY_INIT_CHECKPOINT="${LEGACY_INIT_CHECKPOINT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
# Optional: refuse to start unless the initializer has exactly this SHA-256.
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_UPDATES="${MAX_UPDATES:-}"
# Pilot optimization, identical in both arms. Deviations from the YAML
# (ppo_epochs 4, learning_rate 1e-4) and recorded as such in provenance.
PPO_EPOCHS="${PPO_EPOCHS:-1}"
LR_OVERRIDE="${LR_OVERRIDE:-1e-5}"
PRIOR_NOISE_SCALE="${PRIOR_NOISE_SCALE:-}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "$ARM" in
  candidate) POLICY_ARCHITECTURE=frozen_reference_logit_correction_v1 ;;
  control) POLICY_ARCHITECTURE=bounded_residual_v0 ;;
  *) echo "ARM must be 'candidate' or 'control' (got '$ARM')." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
source "$SCRIPT_DIR/run_naming.sh"
RUN_NAME="$(cdpr_compose_run_name "latent_pilot_${ARM}")"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
cdpr_guard_run_dir "$RUN_DIR"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$SCENES" ]] || { echo "Scene manifest not found: $SCENES" >&2; exit 2; }
if [[ -n "$LEGACY_INIT_CHECKPOINT" && -n "$RESUME_CHECKPOINT" ]]; then
  echo "Set LEGACY_INIT_CHECKPOINT (conversion) or RESUME_CHECKPOINT (continue), not both." >&2; exit 2
fi
if [[ -z "$LEGACY_INIT_CHECKPOINT" && -z "$RESUME_CHECKPOINT" ]]; then
  echo "An initializer is required: LEGACY_INIT_CHECKPOINT=<legacy adapter.pt> or RESUME_CHECKPOINT=<pilot checkpoint>." >&2
  exit 2
fi
if [[ -n "${WARMSTART_CHECKPOINT:-}" ]]; then
  echo "WARMSTART_CHECKPOINT is not used by this launcher; use LEGACY_INIT_CHECKPOINT." >&2; exit 2
fi
INIT_CHECKPOINT="${RESUME_CHECKPOINT:-$LEGACY_INIT_CHECKPOINT}"
INIT_MODE="$([[ -n "$RESUME_CHECKPOINT" ]] && echo resume || echo legacy_conversion)"
[[ -d "$INIT_CHECKPOINT" ]] && INIT_CHECKPOINT="$INIT_CHECKPOINT/smolvla_grpo_adapter.pt"
[[ -f "$INIT_CHECKPOINT" ]] || { echo "Initializer checkpoint not found: $INIT_CHECKPOINT" >&2; exit 2; }
INIT_CHECKPOINT="$(realpath "$INIT_CHECKPOINT")"
if [[ ! "$MAX_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TRAIN_STEPS must be set explicitly (pilot selected actions, e.g. 1000000 or 2000000)." >&2; exit 2
fi
if [[ ! "$MAX_UPDATES" =~ ^[0-9]+$ ]]; then
  echo "MAX_UPDATES must be set explicitly: 10 for the diagnostic, 0 (no cap) for a budgeted pilot." >&2; exit 2
fi
if [[ ! "$PPO_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PPO_EPOCHS must be a positive integer." >&2; exit 2
fi
if [[ "$WORLDS_PER_RANK" -lt 8 || $((WORLDS_PER_RANK % 8)) -ne 0 ]]; then
  echo "WORLDS_PER_RANK must be a positive multiple of 8." >&2; exit 2
fi
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "Exactly two CUDA devices are required." >&2; exit 2
fi

# Refuse protocol drift and wrong initializers before allocating a GPU.
conda run --no-capture-output -n "$ENV_NAME" python3 - \
  "$CONFIG" "$SCENES" "$PPO_EPOCHS" "$LR_OVERRIDE" "$POLICY_ARCHITECTURE" \
  "$INIT_MODE" "$INIT_CHECKPOINT" "$MAX_TRAIN_STEPS" "$EXPECTED_SOURCE_SHA256" <<'PYEOF'
import hashlib
import sys
import torch
import yaml
sys.path.insert(0, ".")
from rl_vla_bootstrapping.core.commands import append_cli_arg
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.latent_correction_policy import (
    ACTION_LIKELIHOOD_LATENT, checkpoint_action_likelihood, checkpoint_policy_architecture,
)
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import sparse_binary_reward_requested
from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import read_manifest, select_split

(config_path, manifest_path, ppo_epochs, lr_override, architecture, init_mode,
 checkpoint, max_steps, expected_sha) = sys.argv[1:]
project = load_project_config(config_path)
metadata = dict(project.task.metadata or {})
section = yaml.safe_load(open(config_path, encoding="utf-8"))["training"]["rl"]["args"]
argv = []
for key, value in section.items():
    append_cli_arg(argv, key, value)
append_cli_arg(argv, "ppo_epochs", int(ppo_epochs))
append_cli_arg(argv, "optimizer_lr_override", float(lr_override))
append_cli_arg(argv, "policy_architecture", architecture)
append_cli_arg(argv, "legacy_init_checkpoint" if init_mode == "legacy_conversion" else "resume_checkpoint", checkpoint)
args = parse_args(argv)
failures = []
if args.action_likelihood != ACTION_LIKELIHOOD_LATENT:
    failures.append(f"action_likelihood is {args.action_likelihood!r}")
if args.policy_architecture != architecture:
    failures.append(f"policy_architecture resolved to {args.policy_architecture!r}, not {architecture!r}")
if not args.three_stage_sparse_credit or args.split_credit_at_grasp:
    failures.append("three-stage sparse credit is not the only credit assignment")
if not sparse_binary_reward_requested(metadata):
    failures.append("sparse_binary_reward is off")
if len(args.three_stage_stage_loss_weights or []) != 3 or max(
        abs(float(v) - 1 / 3) for v in args.three_stage_stage_loss_weights) > 1e-9:
    failures.append(f"stage loss weights {args.three_stage_stage_loss_weights} are not [1,1,1]")
if args.three_stage_full_task_bonus != 1.0 or args.three_stage_full_task_bonus_achieved_negative_scale != 0.0:
    failures.append("full-task bonus protocol changed")
if (args.three_stage_train_split, args.three_stage_validation_split, args.three_stage_horizon_decisions) != (
        "collection", "student_validation", 128):
    failures.append("split/horizon protocol changed")
if args.grpo_group_size != 8 or (args.clip_range_low, args.clip_range_high) != (0.20, 0.28):
    failures.append("group size or PPO clip bounds changed")
if [float(v) for v in args.episode_offset_std] != [0.0, 0.0, 0.0, 0.0, 0.15]:
    failures.append(f"episode_offset_std is {args.episode_offset_std}, expected gripper-only 0.15")
if not args.train_vla_lora or args.vla_lora_updates_enabled or args.train_vla_vision_lora:
    failures.append("LoRA must be attached, frozen, and vision LoRA off")
if args.reference_anchor_bank:
    failures.append("reference anchor is on")
if not bool(metadata.get("placement_wrong_drop_requires_lift", False)):
    failures.append("placement_wrong_drop_requires_lift is off")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
digest = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
if expected_sha and digest != expected_sha:
    failures.append(f"initializer sha256 {digest} != EXPECTED_SOURCE_SHA256 {expected_sha}")
source_arch, source_lik = checkpoint_policy_architecture(payload), checkpoint_action_likelihood(payload)
if init_mode == "legacy_conversion":
    if (source_arch, source_lik) != ("bounded_residual_v0", "clipped_action_v0"):
        failures.append(f"conversion source is ({source_arch}, {source_lik}), not a legacy checkpoint")
else:
    if (source_arch, source_lik) != (architecture, ACTION_LIKELIHOOD_LATENT):
        failures.append(f"resume checkpoint is ({source_arch}, {source_lik}); this arm is ({architecture}, {ACTION_LIKELIHOOD_LATENT})")
    if int(max_steps) <= int(payload.get("global_step", 0)):
        failures.append(f"MAX_TRAIN_STEPS={max_steps} must exceed the pilot checkpoint's global_step={payload.get('global_step')}")
scenes, _ = read_manifest(manifest_path)
print(f"[latent-pilot] arm architecture={architecture} likelihood={args.action_likelihood} "
      f"init={init_mode} source=({source_arch}, {source_lik}) global_step={payload.get('global_step')} "
      f"sha256={digest}")
print(f"[latent-pilot] ppo_epochs={args.ppo_epochs} (yaml {section.get('ppo_epochs')}) "
      f"lr_override={args.optimizer_lr_override} (yaml {section.get('learning_rate')}) "
      f"episode_offset_std={args.episode_offset_std} collection={len(select_split(scenes, 'collection'))}")
if failures:
    for failure in failures:
        print(f"[latent-pilot] REFUSING: {failure}", file=sys.stderr)
    raise SystemExit(2)
print("[latent-pilot] preflight clean; final_test is not selected by this runner")
PYEOF

mkdir -p "$RUN_DIR"
unset RLVLA_SMOLVLA_RESUME_CHECKPOINT RLVLA_SMOLVLA_WARMSTART_CHECKPOINT RLVLA_CDPR_DEMO_BANK
unset RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT RLVLA_SMOLVLA_POLICY_ARCHITECTURE
unset RLVLA_SMOLVLA_REFERENCE_ANCHOR_BANK RLVLA_SMOLVLA_REFERENCE_ANCHOR_CHECKPOINT RLVLA_SMOLVLA_REFERENCE_ANCHOR_COEF
unset RLVLA_SMOLVLA_PPO_EPOCHS RLVLA_SMOLVLA_OPTIMIZER_LR_OVERRIDE RLVLA_SMOLVLA_PRIOR_NOISE_SCALE
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
export RLVLA_SMOLVLA_PPO_EPOCHS="$PPO_EPOCHS"
export RLVLA_SMOLVLA_OPTIMIZER_LR_OVERRIDE="$LR_OVERRIDE"
export RLVLA_SMOLVLA_POLICY_ARCHITECTURE="$POLICY_ARCHITECTURE"
[[ -n "$PRIOR_NOISE_SCALE" ]] && export RLVLA_SMOLVLA_PRIOR_NOISE_SCALE="$PRIOR_NOISE_SCALE"
if [[ "$INIT_MODE" == "resume" ]]; then
  export RLVLA_SMOLVLA_RESUME_CHECKPOINT="$INIT_CHECKPOINT"
else
  export RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT="$INIT_CHECKPOINT"
fi
export RLVLA_CDPR_THREE_STAGE_SCENE_MANIFEST="$SCENES"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
train_cmd=("${python_cmd[@]}" -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG" --stage rl --run-name "$RUN_NAME" --execute)

printf 'run_dir=%s\narm=%s architecture=%s likelihood=latent_gaussian_conditional_offset_v1\n' \
  "$RUN_DIR" "$ARM" "$POLICY_ARCHITECTURE"
printf '%s=%s\n' "$INIT_MODE" "$INIT_CHECKPOINT"
printf 'max_train_steps=%s (pilot selected actions) max_updates=%s worlds_per_rank=%s\n' \
  "$MAX_TRAIN_STEPS" "$MAX_UPDATES" "$WORLDS_PER_RANK"
printf 'ppo_epochs=%s lr_override=%s prior noise scale=%s\n' \
  "$PPO_EPOCHS" "$LR_OVERRIDE" "${PRIOR_NOISE_SCALE:-1 (LeRobot default)}"
printf 'command:'; printf ' %q' "${train_cmd[@]}"; printf '\n'
[[ "$DRY_RUN" == "1" ]] && exit 0

"${python_cmd[@]}" - "$INIT_CHECKPOINT" "$SCENES" "$RUN_DIR/launch_provenance.json" "$MAX_UPDATES" \
  "$MAX_TRAIN_STEPS" "$INIT_MODE" "$CONFIG" "$PPO_EPOCHS" "$LR_OVERRIDE" "$PRIOR_NOISE_SCALE" \
  "$ARM" "$POLICY_ARCHITECTURE" <<'PYPROVENANCE'
import hashlib, json, pathlib, platform, subprocess, sys
(checkpoint, scenes, output, updates, steps, mode, config, ppo_epochs, lr_override, prior_noise,
 arm, architecture) = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
def version(module):
    try:
        return __import__(module).__version__
    except Exception as error:  # never block a launch on a version probe
        return f"unavailable: {type(error).__name__}"
import torch, yaml
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
rl_args = yaml.safe_load(open(config, encoding="utf-8"))["training"]["rl"]["args"]
record = {
    "arm": arm, "policy_architecture": architecture,
    "action_likelihood": "latent_gaussian_conditional_offset_v1",
    "init_mode": mode, "checkpoint": str(pathlib.Path(checkpoint).resolve()),
    "checkpoint_sha256": digest(checkpoint),
    "checkpoint_global_step": int(payload.get("global_step", 0)),
    "checkpoint_lineage": payload.get("lineage"),
    "config": str(pathlib.Path(config).resolve()), "config_sha256": digest(config),
    "scene_manifest": str(pathlib.Path(scenes).resolve()), "scene_manifest_sha256": digest(scenes),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
    "max_updates": int(updates), "max_train_steps": int(steps),
    "max_train_steps_definition": "pilot selected environment actions (pilot_global_step), excluding source lineage",
    "ppo_epochs": int(ppo_epochs), "optimizer_lr_override": float(lr_override),
    "deviations_from_yaml": {
        "ppo_epochs": {"yaml": rl_args.get("ppo_epochs"), "pilot": int(ppo_epochs)},
        "learning_rate": {"yaml": rl_args.get("learning_rate"), "pilot": float(lr_override)},
        "mjwarp_max_updates": {"yaml": rl_args.get("mjwarp_max_updates"), "pilot": int(updates)},
        "max_train_steps": {"yaml": rl_args.get("max_train_steps"), "pilot": int(steps)},
    },
    "prior_noise_scale": float(prior_noise) if prior_noise else None,
    "episode_offset_std": rl_args.get("episode_offset_std"),
    "seed": rl_args.get("seed", 0),
    "lora_updates_enabled": False, "reference_anchor": None,
    "reward_protocol": "three_stage_accessible_v5_full_task_bonus",
    "termination_protocol": "wrong_place_requires_held_lift",
    "outcome_protocol": "independent_strict_full_task_v1",
    "runtime_versions": {"python": platform.python_version(), "torch": version("torch"),
                         "mujoco": version("mujoco"), "warp": version("warp"),
                         "cuda": torch.version.cuda},
}
pathlib.Path(output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PYPROVENANCE

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  "${python_cmd[@]}" -m unittest discover -s tests -p test_grpo_three_stage_sparse_credit.py
  "${python_cmd[@]}" -m unittest discover -s tests -p test_three_stage_repairs.py
  "${python_cmd[@]}" -m unittest discover -s tests -p 'test_latent_correction_*.py'
  huggingface_public_models_preflight "$ENV_NAME"
  "${python_cmd[@]}" scripts/preflight_cdpr_mjlab.py \
    --config "$CONFIG" --require-gpus 2 --worlds "$WORLDS_PER_RANK" \
    --output "$RUN_DIR/preflight.json"
fi
"${train_cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
