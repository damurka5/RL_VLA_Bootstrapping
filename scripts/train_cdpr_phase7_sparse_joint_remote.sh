#!/usr/bin/env bash
# Phase 7: ONE sparse binary reward, all four instructions, one RL run.
#
# THE HYPOTHESIS. Per-family DENSE rewards are why RL had to be run one family
# at a time, and per-family RL is why the campaign has a sawtooth -- the family
# most recently trained is strongest while the others are partially rebuilt by
# retention SFT. A binary outcome reward is instruction-agnostic, so all four
# families share one return stream, one advantage normalisation and one batch.
# That attacks the sawtooth's cause instead of compensating for it.
#
# WHY IT IS RUNNABLE NOW AND WAS NOT BEFORE. Binary GRPO gives gradient only
# where a group of eight holds both a success and a failure; the informative
# fraction is 1 - p^8 - (1-p)^8. On sft_phase7: move_to 0.99, pick_up 0.74,
# caught plate 0.95, caught bowl 0.99, composed plate 0.64, composed bowl 0.29.
# Every leg fits the 4-round refill budget. On sft_phase6 composed bowl was
# 0.19 and needed 5.17 rounds -- outside it. The horizon fix and the o7
# re-harvest are what brought it inside; sparse RL was blocked on a task fix,
# not on a sampling knob.
#
# WHAT TO STEER ON. validation_composed/, NOT validation/. phase6_compose_iter0
# spent 2.25M steps and fired its stop rule on a metric that was 80-90%
# carry-only episodes; the same checkpoint measured under an explicit uncaught
# protocol had not improved on its seed. The composed leg in the config is that
# measurement, run in-loop.
#
# WHAT TO WATCH BESIDES SUCCESS. usable_groups_collected / groups_collected is
# the realised informative fraction and rounds_collected the oversample paid to
# reach it. A run sitting near 0.11 is spending ~9 rollouts an update on one
# family; that is a signal to fix the task, not to raise the budget.
#
# AND THE ONE THAT WILL DECIDE THIS. The policy DROPS the object in ~20% of
# composed episodes where the oracle drops it in ~4% -- 305 of 1536 measured on
# sft_phase6, and the horizon fix made dropping a LARGER share of what is left
# (plate 71% -> 81% of no_release). That is the gap sparse RL has to close, it
# is ~5x and so will not hide inside the ~0.04 evaluation noise floor, and it
# is the reason to read the composed decomposition and not only the rate.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml}"
BANK="${BANK:-$REPO_ROOT/runs/phase4_bank}"
# Weights-only warm start. This is SimpleVLA-RL's stage-1 cold start: its
# LIBERO-Long 17.3 -> 91.7 begins from an SFT-supplied 17.3, not from zero, and
# under a binary reward a family at zero can never bootstrap because a carry
# that never opens the gripper scores exactly what one that never moved scores.
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-$BANK/sft_phase7/sil_sft_adapter.pt}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-6000000}"
WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"
SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models
configure_huggingface_offline
# shellcheck source=scripts/run_naming.sh
source "$SCRIPT_DIR/run_naming.sh"
RUN_NAME="$(cdpr_compose_run_name "phase7_sparse_joint")"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
cdpr_guard_run_dir "$RUN_DIR"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
if [[ -n "$WARMSTART_CHECKPOINT" && ! -f "$WARMSTART_CHECKPOINT" ]]; then
  echo "WARMSTART_CHECKPOINT not found: $WARMSTART_CHECKPOINT" >&2
  exit 2
fi
if [[ "$WORLDS_PER_RANK" -lt 8 || $((WORLDS_PER_RANK % 8)) -ne 0 ]]; then
  echo "WORLDS_PER_RANK must be a positive multiple of the GRPO group size (8)." >&2
  exit 2
fi
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "Exactly two CUDA devices are required; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi

# --- preflight: is this config actually the sparse joint one? ------------
# Each of these was a real failure mode earlier in the campaign. A sparse run
# that silently kept a dense reward, a refill loop pinned at one round by
# arithmetic, and a stop rule reading the caught-dominated metric all look
# exactly like a working run until the result is read.
conda run --no-capture-output -n "$ENV_NAME" python3 - \
  "$CONFIG" "$WORLDS_PER_RANK" <<'PYEOF'
import sys
sys.path.insert(0, ".")
import yaml
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.core.commands import append_cli_arg
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    BatchedCatchReleaseDenseReward, sparse_binary_reward_requested,
)

config_path, worlds = sys.argv[1], int(sys.argv[2])
project = load_project_config(config_path)
metadata = dict(project.task.metadata or {})
raw = yaml.safe_load(open(config_path))
section = next(v["args"] for v in raw["training"].values()
               if isinstance(v, dict) and "args" in v)
argv = []
for key, value in section.items():
    append_cli_arg(argv, key, value)
args = parse_args(argv)

failures = []

if not sparse_binary_reward_requested(metadata):
    failures.append("sparse_binary_reward is not set; this would run the DENSE "
                    "reward under a sparse run's name")
else:
    reward = BatchedCatchReleaseDenseReward.from_metadata(metadata)
    if reward.distance_reward_weight or reward.pick_lift_reward_weight:
        failures.append("shaping weights survived the sparse flag")
    # The geometry must NOT have moved, or the arms score a different task.
    if abs(reward.plate_radius - 0.091) > 1e-6 or abs(reward.bowl_radius - 0.057) > 1e-6:
        failures.append(f"success radii moved: plate {reward.plate_radius}, "
                        f"bowl {reward.bowl_radius} -- expected 0.091 / 0.057")

instructions = set(project.task.instruction_types or ())
expected = {"move_to_object", "pick_up", "put_into_plate", "put_into_bowl"}
if instructions != expected:
    failures.append(f"instruction_types is {sorted(instructions)}, not all four")

groups = worlds // int(args.grpo_group_size)
rounds = max(1, -(-int(args.grpo_max_groups_per_update) // groups))
if int(args.grpo_target_informative_groups) <= 0:
    failures.append("grpo_target_informative_groups is 0; dynamic sampling "
                    "would MASK degenerate groups without refilling")
elif rounds <= 1:
    failures.append(f"grpo_max_groups_per_update {args.grpo_max_groups_per_update} "
                    f"against {groups} groups/rank pins max_rounds at {rounds}; "
                    "the refill cannot take a second round")

if int(args.composed_validation_episodes_per_instruction) <= 0:
    failures.append("composed_validation_episodes_per_instruction is 0; the "
                    "run would steer on the caught-dominated metric, which is "
                    "what phase 6 did")

horizon = int(metadata.get("placement_grasp_horizon_min_decisions", 32))
if horizon < 40:
    failures.append(f"placement_grasp_horizon_min_decisions is {horizon}; the "
                    "composed episode does not fit in 32")

# THE CHECK THAT WOULD HAVE SAVED 13 HOURS. A sparse run with every exploration
# channel zeroed has nothing that pays for keeping a behaviour once it starts
# to drift: under a binary reward a carry that never opens the gripper scores
# exactly what one that never moved scores. Phase 7's first attempt ran 2.1M
# steps that way -- physical_release_rate 0.0771 -> 0.0535, grasp 0.2633 ->
# 0.2084, entropy and log_std flat to four decimals -- and this file's own note
# claimed episode_offset was on while the value read [0,0,0,0,0].
offsets = list(getattr(args, "episode_offset_std", []) or [])
if not any(float(v) > 0.0 for v in offsets):
    failures.append(
        f"episode_offset_std is {offsets or 'unset'} -- every exploration "
        "channel is zero. Under a sparse reward nothing then pays for keeping "
        "the release, and it erodes. Set the gripper channel (index 4)."
    )
elif len(offsets) >= 5 and float(offsets[4]) <= 0.0:
    failures.append(
        f"episode_offset_std is {offsets}: exploration is on, but not on the "
        "GRIPPER channel (index 4), which is the one the release needs."
    )

# The run must train the task it is scored on. Phase 7 measured 100% composed
# and trained 10-20% of it, which is the phase-6 error with a better metric
# bolted on.
caught = float(metadata.get("placement_caught_object_fraction", 1.0))
if caught > 0.75:
    failures.append(
        f"placement_caught_object_fraction is {caught}: {caught:.0%} of "
        "container episodes start already holding the object, while "
        "validation_composed scores 100% composed. That is what phase 7 did."
    )

# And the gate must read the composed task rather than a blend whose mixture is
# the caught knob.
gate_uncaught = str(
    metadata.get("approach_gate_uncaught_only", False)
).strip().lower() in {"1", "true", "yes", "on"}
if not gate_uncaught:
    failures.append(
        "approach_gate_uncaught_only is off, so the approach curriculum "
        "promotes on a blend of caught carries and composed episodes. On "
        "phase 7 that gate read 0.42 while the composed validation read 0.013."
    )

print(f"[phase7] sparse={sparse_binary_reward_requested(metadata)} "
      f"instructions={len(instructions)} groups/rank={groups} "
      f"max_rounds={rounds} target_groups={args.grpo_target_informative_groups} "
      f"composed_val={args.composed_validation_episodes_per_instruction} "
      f"horizon={horizon}")
print(f"[phase7] caught_fraction={caught} gate_uncaught_only={gate_uncaught} "
      f"episode_offset_std={offsets}")
if failures:
    for line in failures:
        print(f"[phase7] REFUSING: {line}", file=sys.stderr)
    raise SystemExit(2)
print("[phase7] preflight clean")
PYEOF

mkdir -p "$RUN_DIR"
unset RLVLA_SMOLVLA_RESUME_CHECKPOINT
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RLVLA_SMOLVLA_NPROC_PER_NODE=2
export RLVLA_SMOLVLA_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
export RLVLA_MJWARP_WORLDS_PER_RANK="$WORLDS_PER_RANK"
export RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE="$SMOLVLA_MICROBATCH_SIZE"
export RLVLA_SMOLVLA_WARMSTART_CHECKPOINT="$WARMSTART_CHECKPOINT"

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
train_cmd=(
  "${python_cmd[@]}"
  -m rl_vla_bootstrapping.cli.train
  --config "$CONFIG"
  --stage rl
  --run-name "$RUN_NAME"
  --execute
)

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'tensorboard_dir=%s\n' "$RUN_DIR/rl/tensorboard"
printf 'validation_metrics=%s\n' "$RUN_DIR/rl/validation.jsonl"
printf 'warm_start=%s (weights only, fresh optimizer and curriculum)\n' "$WARMSTART_CHECKPOINT"
printf 'max_train_steps=%s worlds_per_rank=%s groups_per_rank=%s\n' \
  "$MAX_TRAIN_STEPS" "$WORLDS_PER_RANK" "$((WORLDS_PER_RANK / 8))"
printf 'STEER ON validation_composed/, not validation/\n'
printf 'WATCH usable_groups_collected / groups_collected, and rounds_collected\n'
printf 'command:'
printf ' %q' "${train_cmd[@]}"
printf '\n'
[[ "$DRY_RUN" == "1" ]] && exit 0

cd "$REPO_ROOT"
if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  huggingface_public_models_preflight "$ENV_NAME"
  "${python_cmd[@]}" scripts/preflight_cdpr_mjlab.py \
    --config "$CONFIG" \
    --require-gpus 2 \
    --worlds "$WORLDS_PER_RANK" \
    --output "$RUN_DIR/preflight.json"
fi

"${train_cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
