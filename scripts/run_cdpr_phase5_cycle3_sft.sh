#!/usr/bin/env bash
# Cycle 3 of the phase-4/5 retention loop, end to end and unattended.
#
# Harvest placement from the checkpoint that holds it, pool it with the banked
# pick_up and move_to, refresh the priors against the same checkpoint, run the
# supervised pass, and score all four instructions. Composition is deliberately
# NOT here: it is a different task and wants its own run and its own attention.
#
# Two things this does that a hand-typed sequence does not.
#
# It BALANCES THE QUOTA FROM MEASUREMENT. The dataset is built twice: once with
# the quota effectively disabled to read how many decisions each instruction
# actually has, then again at the smallest of those. Section 5 of the phase-4
# report is the reason -- episode lengths differ by a factor of four across
# families, so a quota that reads as balanced is not, and the biggest slice
# takes the most gradient. Hard-coding 26000 was right for cycle 2 and is a
# guess for cycle 3.
#
# It SKIPS STAGES THAT ALREADY PRODUCED THEIR OUTPUT, so a run that dies at
# hour six resumes at hour six. Delete the directory of a stage to force it.
#
# The old placement demos are deliberately excluded from the pool. They came
# from checkpoints scoring 0.437-0.490 on plate; this harvest's checkpoint
# scores 0.798, and pick_up binds the quota either way, so keeping them would
# only dilute the slice with weaker trajectories.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
BANK="${BANK:-$REPO_ROOT/runs/phase4_bank}"
# The placement policy: validation 0.6211 overall, plate 0.7982, bowl 0.4073.
PLACEMENT_CHECKPOINT="${PLACEMENT_CHECKPOINT:-$REPO_ROOT/runs/phase5_placement_iter3_20260828_224948/rl/step_2754052/smolvla_grpo_adapter.pt}"
PLACEMENT_CONFIG="${PLACEMENT_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_placement_loop.yaml}"
PICKUP_CONFIG="${PICKUP_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_pick_up_loop.yaml}"
MOVE_TO_CONFIG="${MOVE_TO_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml}"
WORLDS="${WORLDS:-2048}"
ROUNDS_PER_CAP="${ROUNDS_PER_CAP:-4}"
CAPS="${CAPS:-0.10 0.15 0.20}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
EPOCHS="${EPOCHS:-45}"
# 0: the LoRA stage has failed to beat its own baseline three cycles running
# and cost 84 minutes on the last one. --lora-epochs 4 to reinstate it.
LORA_EPOCHS="${LORA_EPOCHS:-0}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
# The evals stay at the default 512 worlds: matching the measurement that
# produced the numbers they are compared against matters more than tighter
# error bars.
EVAL_WORLDS="${EVAL_WORLDS:-512}"
DRY_RUN="${DRY_RUN:-0}"

PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
LOG_DIR="$BANK/cycle3_logs"

say() { printf '\n[cycle3 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [[ ! -f "$PLACEMENT_CHECKPOINT" ]]; then
  echo "Placement checkpoint not found: $PLACEMENT_CHECKPOINT" >&2
  exit 2
fi
for cfg in "$PLACEMENT_CONFIG" "$PICKUP_CONFIG" "$MOVE_TO_CONFIG"; do
  [[ -f "$cfg" ]] || { echo "Config not found: $cfg" >&2; exit 2; }
done
if [[ $((WORLDS % 8)) -ne 0 ]]; then
  echo "WORLDS must be a multiple of the GRPO group size (8)." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

printf 'repo_root=%s\n' "$REPO_ROOT"
printf 'placement_checkpoint=%s\n' "$PLACEMENT_CHECKPOINT"
printf 'bank=%s\n' "$BANK"
printf 'caps=%s rounds_per_cap=%s worlds=%s devices=%s\n' \
  "$CAPS" "$ROUNDS_PER_CAP" "$WORLDS" "$DEVICES"
printf 'epochs=%s lora_epochs=%s\n' "$EPOCHS" "$LORA_EPOCHS"
printf 'logs=%s\n' "$LOG_DIR"
if [[ "$DRY_RUN" == "1" ]]; then
  say "DRY_RUN=1, stopping before the first GPU stage."
  exit 0
fi

# --- 1. harvest placement -------------------------------------------------
for cap in $CAPS; do
  out="$BANK/p3_${cap}"
  last=$(printf 'record_%02d.npz' $((ROUNDS_PER_CAP - 1)))
  if [[ -f "$out/$last" ]]; then
    say "harvest cap $cap already complete, skipping"
    continue
  fi
  say "harvest cap $cap -> $out"
  "${PY[@]}" tools/audit/sil_record.py --mode record \
    --rounds "$ROUNDS_PER_CAP" --worlds "$WORLDS" --devices "$DEVICES" \
    --seed-torch 0 --start-distance-cap "$cap" \
    --checkpoint "$PLACEMENT_CHECKPOINT" --config "$PLACEMENT_CONFIG" \
    --output "$out" 2>&1 | tee "$LOG_DIR/harvest_${cap}.log"
done

# --- 2. replay, two lanes per cap ----------------------------------------
# Replay is single-device, so the parallelism is two processes. They share one
# output directory safely: every replay is named after its SOURCE stem, which
# carries the rung and the round, so no two can collide.
IFS=',' read -r -a device_list <<< "$DEVICES"
lanes=${#device_list[@]}
for cap in $CAPS; do
  say "replay cap $cap -> $BANK/p3_demos"
  pids=()
  for lane in $(seq 0 $((lanes - 1))); do
    (
      for index in $(seq 0 $((ROUNDS_PER_CAP - 1))); do
        [[ $((index % lanes)) -eq "$lane" ]] || continue
        stem=$(printf 'record_%02d' "$index")
        if [[ -f "$BANK/p3_demos/replay_p3_${cap}_${stem}.npz" ]]; then
          echo "[cycle3] $cap $stem already replayed, skipping"
          continue
        fi
        "${PY[@]}" tools/audit/sil_record.py --mode replay \
          --smooth moving_average --smooth-window 5 \
          --actions "$BANK/p3_${cap}/${stem}.npz" \
          --worlds "$WORLDS" --device "${device_list[$lane]}" \
          --seed-torch 0 --start-distance-cap "$cap" \
          --checkpoint "$PLACEMENT_CHECKPOINT" --config "$PLACEMENT_CONFIG" \
          --record-frames --frame-worlds 0 --output "$BANK/p3_demos"
      done
    ) > "$LOG_DIR/replay_${cap}_lane${lane}.log" 2>&1 &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  tail -n 3 "$LOG_DIR/replay_${cap}"_lane*.log || true
  if [[ "$status" -ne 0 ]]; then
    echo "A replay lane failed at cap $cap; see $LOG_DIR/replay_${cap}_lane*.log" >&2
    exit 1
  fi
done

# --- 3. read availability, then build the balanced dataset ----------------
INPUTS=(
  "$BANK"/pick_up_demos/replay_*.npz
  "$BANK"/pick_up_iter2_demos/replay_*.npz
  "$BANK"/move_to_demos/replay_*.npz
  "$BANK"/m2_demos/replay_*.npz
  "$BANK"/p3_demos/replay_*.npz
)
if [[ ! -f "$BANK/dataset3_probe/dataset.json" ]]; then
  say "reading availability (quota disabled)"
  "${PY[@]}" tools/audit/sil_record.py --mode dataset \
    --inputs "${INPUTS[@]}" --rows-per-instruction 0 \
    --output "$BANK/dataset3_probe" 2>&1 | tee "$LOG_DIR/dataset_probe.log"
fi

QUOTA="$(
  "${PY[@]}" - "$BANK/dataset3_probe/dataset.json" <<'PYEOF'
import json, sys
stats = json.load(open(sys.argv[1]))
by = stats["by_instruction"]
counts = {name: int(entry["decisions"]) for name, entry in by.items()}
for name, value in sorted(counts.items()):
    print(f"[cycle3] available {name}: {value} decisions", file=sys.stderr)
# The balanced size is the smallest slice: anything larger leaves that
# instruction short while the others sit at the cap, which is the imbalance
# the quota exists to prevent.
print(min(counts.values()))
PYEOF
)"
say "balanced quota = $QUOTA decisions per instruction"

if [[ ! -f "$BANK/dataset3/demonstrations.npz" ]]; then
  say "building the balanced dataset"
  "${PY[@]}" tools/audit/sil_record.py --mode dataset \
    --inputs "${INPUTS[@]}" --rows-per-instruction "$QUOTA" \
    --output "$BANK/dataset3" 2>&1 | tee "$LOG_DIR/dataset_build.log"
fi

# --- 4. refresh priors against the checkpoint the SFT starts from ---------
if [[ ! -f "$BANK/refreshed3/demonstrations.npz" ]]; then
  say "refreshing priors"
  "${PY[@]}" tools/audit/sil_refresh_priors.py \
    --dataset "$BANK/dataset3/demonstrations.npz" \
    --frames "$BANK"/*_demos/frames_*.npz \
    --checkpoint "$PLACEMENT_CHECKPOINT" \
    --min-resolved-fraction 0.98 \
    --output "$BANK/refreshed3" 2>&1 | tee "$LOG_DIR/refresh.log"
fi

# --- 5. the supervised pass ----------------------------------------------
if [[ ! -f "$BANK/sft_cycle3/sil_sft_adapter.pt" ]]; then
  say "supervised fine-tune, $EPOCHS epochs"
  "${PY[@]}" tools/audit/sil_sft.py \
    --dataset "$BANK/refreshed3/demonstrations.npz" \
    --checkpoint "$PLACEMENT_CHECKPOINT" \
    --frames "$BANK"/*_demos/frames_*.npz \
    --epochs "$EPOCHS" --lora-epochs "$LORA_EPOCHS" \
    --progress never \
    --output "$BANK/sft_cycle3" 2>&1 | tee "$LOG_DIR/sft.log"
fi
ADAPTER="$BANK/sft_cycle3/sil_sft_adapter.pt"

# --- 6. score every instruction at the cap its baseline was measured at ---
run_eval() {
  local name="$1" cap="$2" config="$3"
  if [[ -f "$BANK/eval/cycle3_${name}/summary.json" ]]; then
    say "eval $name already done, skipping"
    return 0
  fi
  say "eval $name at cap $cap"
  "${PY[@]}" tools/audit/sil_record.py --mode record \
    --rounds "$EVAL_ROUNDS" --worlds "$EVAL_WORLDS" --seed-torch 0 \
    --start-distance-cap "$cap" --checkpoint "$ADAPTER" --config "$config" \
    --output "$BANK/eval/cycle3_${name}" 2>&1 \
    | tee "$LOG_DIR/eval_${name}.log"
}
run_eval pick_up 0.06 "$PICKUP_CONFIG"
run_eval move_to 0.19 "$MOVE_TO_CONFIG"
run_eval placement 0.20 "$PLACEMENT_CONFIG"

# --- 7. the table --------------------------------------------------------
say "results"
"${PY[@]}" - "$BANK/eval" <<'PYEOF'
import json, pathlib, sys

# Weighted by episodes, never averaged across rounds: the placement rounds draw
# uneven instruction counts and a small round would otherwise carry the same
# weight as a large one.
baseline = {
    "pick_up": 0.1738,
    "move_to_object": 0.4316,
    "put_into_plate": 0.5736,
    "put_into_bowl": 0.2721,
}
root = pathlib.Path(sys.argv[1])
totals: dict[str, list[int]] = {}
for summary_path in sorted(root.glob("cycle3_*/summary.json")):
    summary = json.loads(summary_path.read_text())
    for key, entry in summary.items():
        if not key.startswith("run_") or not isinstance(entry, dict):
            continue
        for name, stats in (entry.get("by_instruction") or {}).items():
            bucket = totals.setdefault(name, [0, 0])
            bucket[0] += int(stats.get("successes", 0))
            bucket[1] += int(stats.get("episodes", 0))

print(f"{'instruction':18s} {'rate':>8s} {'counts':>14s} {'cycle2':>8s} {'delta':>8s}")
for name in sorted(totals):
    successes, episodes = totals[name]
    if not episodes:
        continue
    rate = successes / episodes
    was = baseline.get(name)
    delta = f"{rate - was:+.4f}" if was is not None else "-"
    was_text = f"{was:.4f}" if was is not None else "-"
    print(
        f"{name:18s} {rate:8.4f} {successes:6d}/{episodes:<7d} "
        f"{was_text:>8s} {delta:>8s}"
    )
PYEOF

say "done. adapter: $ADAPTER"
