#!/usr/bin/env bash
# Seed the composed put_into from the scripted oracle, end to end.
#
# The composed task -- object on the desk, grasp it, carry it, release it --
# cannot be seeded from any policy this campaign has produced. Measured on
# sft_cycle3: 0.005 (plate) and 0.011 (bowl). Relabelled free-scene grasps do
# not substitute, because they miss the join: their median gripper-to-
# receptacle distance at the grasp is 0.2471 m against the 0.19-0.20 m a
# placement demonstration starts within, so only ~20% land in covered territory.
#
# The oracle does it: 1.000 (plate) and 0.427 (bowl) on the same task and the
# same success predicate. So the seed is scripted, and then RL takes over --
# the same shape as the pick_up seed in section 3 of the phase-5 report, which
# is the one intervention in that phase that clearly worked.
#
# --smooth none, unlike every other harvest here. Smoothing exists to fix
# POLICY jitter; the oracle is a P-D controller and its actions are already
# smooth. Measured on one file: survival 0.9971 unsmoothed against 0.9798
# smoothed, with the filter changing the step delta 0.0327 -> 0.0255. Filtering
# a trajectory that did not need it only makes the replay diverge from what was
# recorded.
#
# Composition is NOT trained here. This produces the adapter the composition RL
# starts from; the RL run is a separate, deliberate step.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
BANK="${BANK:-$REPO_ROOT/runs/phase4_bank}"
# The current single policy. Its actions are discarded in oracle mode, but its
# states and priors are what the demonstrations are conditioned on, and it is
# what the SFT starts from.
CHECKPOINT="${CHECKPOINT:-$BANK/sft_cycle3/sil_sft_adapter.pt}"
COMPOSE_CONFIG="${COMPOSE_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase5_compose_loop.yaml}"
PLACEMENT_CONFIG="${PLACEMENT_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_placement_loop.yaml}"
PICKUP_CONFIG="${PICKUP_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_pick_up_loop.yaml}"
MOVE_TO_CONFIG="${MOVE_TO_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml}"
WORLDS="${WORLDS:-2048}"
ROUNDS_PER_CAP="${ROUNDS_PER_CAP:-4}"
CAPS="${CAPS:-0.10 0.15 0.20}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
EPOCHS="${EPOCHS:-45}"
LORA_EPOCHS="${LORA_EPOCHS:-0}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EVAL_WORLDS="${EVAL_WORLDS:-512}"
# Frames per replay, 0 = every surviving world. The oracle survives far more
# worlds than a policy does -- 1378 of 2048 against the 600-900 a placement
# harvest kept -- and frames are ~48 KB per world-decision across two cameras,
# so a round can run to several GB and twelve of them to tens. Cap it if the
# disk is tight; the quota subsamples episodes anyway.
FRAME_WORLDS="${FRAME_WORLDS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Every container episode starts UNCAUGHT, and the curriculum is off so nothing
# restores the caught fraction part way through a harvest.
COMPOSED=(--metadata-override placement_caught_object_fraction=0.0
          placement_caught_curriculum_enabled=false)

PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
LOG_DIR="$BANK/phase6_logs"
say() { printf '\n[phase6 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
for cfg in "$COMPOSE_CONFIG" "$PLACEMENT_CONFIG" "$PICKUP_CONFIG" "$MOVE_TO_CONFIG"; do
  [[ -f "$cfg" ]] || { echo "Config not found: $cfg" >&2; exit 2; }
done
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'caps=%s rounds_per_cap=%s worlds=%s devices=%s\n' \
  "$CAPS" "$ROUNDS_PER_CAP" "$WORLDS" "$DEVICES"
printf 'epochs=%s lora_epochs=%s frame_worlds=%s logs=%s\n' "$EPOCHS" "$LORA_EPOCHS" "$FRAME_WORLDS" "$LOG_DIR"
printf 'free space: %s\n' "$(df -h "$BANK" | tail -1)"
[[ "$DRY_RUN" == "1" ]] && { say "DRY_RUN=1, stopping before the first GPU stage."; exit 0; }

# --- 1. oracle demonstrations of the composed task ------------------------
for cap in $CAPS; do
  out="$BANK/o6_${cap}"
  last=$(printf 'record_%02d.npz' $((ROUNDS_PER_CAP - 1)))
  if [[ -f "$out/$last" ]]; then say "oracle cap $cap already harvested"; continue; fi
  say "oracle harvest cap $cap"
  "${PY[@]}" tools/audit/sil_record.py --mode oracle \
    --rounds "$ROUNDS_PER_CAP" --worlds "$WORLDS" --devices "$DEVICES" \
    --seed-torch 0 --start-distance-cap "$cap" "${COMPOSED[@]}" \
    --checkpoint "$CHECKPOINT" --config "$COMPOSE_CONFIG" \
    --output "$out" 2>&1 | tee "$LOG_DIR/oracle_${cap}.log"
done

# --- 2. replay for frames, unsmoothed ------------------------------------
IFS=',' read -r -a device_list <<< "$DEVICES"
lanes=${#device_list[@]}
for cap in $CAPS; do
  say "replay cap $cap -> $BANK/o6_demos"
  pids=()
  for lane in $(seq 0 $((lanes - 1))); do
    (
      for index in $(seq 0 $((ROUNDS_PER_CAP - 1))); do
        [[ $((index % lanes)) -eq "$lane" ]] || continue
        stem=$(printf 'record_%02d' "$index")
        [[ -f "$BANK/o6_demos/replay_o6_${cap}_${stem}.npz" ]] && continue
        "${PY[@]}" tools/audit/sil_record.py --mode replay --smooth none \
          --actions "$BANK/o6_${cap}/${stem}.npz" \
          --worlds "$WORLDS" --device "${device_list[$lane]}" \
          --seed-torch 0 --start-distance-cap "$cap" "${COMPOSED[@]}" \
          --checkpoint "$CHECKPOINT" --config "$COMPOSE_CONFIG" \
          --record-frames --frame-worlds "$FRAME_WORLDS" \
          --output "$BANK/o6_demos"
      done
    ) > "$LOG_DIR/replay_${cap}_lane${lane}.log" 2>&1 &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  grep -h "survived" "$LOG_DIR/replay_${cap}"_lane*.log || true
  if [[ "$status" -ne 0 ]]; then
    # The lane's own output is the only place the reason exists, because the
    # subshell redirects it. Printing "a lane failed" and nothing else -- as
    # this did on its first run -- makes the operator go and find the log by
    # hand, which is work the script already had the path for.
    echo "=== replay lane failed at cap $cap; tail of each lane log ===" >&2
    for log in "$LOG_DIR/replay_${cap}"_lane*.log; do
      echo "--- $log" >&2
      tail -n 25 "$log" >&2
    done
    df -h "$BANK" >&2 || true
    exit 1
  fi
done

# --- 3. pool, balanced from a measured availability read -----------------
# The oracle rows join the CAUGHT-start placement rows rather than replacing
# them. Both are put_into, and the composition curriculum presents both: it
# anneals the caught fraction from 1.0 down to 0.25, so a policy that only ever
# saw composed starts would lose the carry it already does well.
INPUTS=(
  "$BANK"/pick_up_demos/replay_*.npz
  "$BANK"/pick_up_iter2_demos/replay_*.npz
  "$BANK"/move_to_demos/replay_*.npz
  "$BANK"/m2_demos/replay_*.npz
  "$BANK"/p3_demos/replay_*.npz
  "$BANK"/o6_demos/replay_*.npz
)
if [[ ! -f "$BANK/dataset6_probe/dataset.json" ]]; then
  say "reading availability"
  "${PY[@]}" tools/audit/sil_record.py --mode dataset --inputs "${INPUTS[@]}" \
    --rows-per-instruction 0 --output "$BANK/dataset6_probe" 2>&1 \
    | tee "$LOG_DIR/dataset_probe.log"
fi
QUOTA="$("${PY[@]}" - "$BANK/dataset6_probe/dataset.json" <<'PYEOF'
import json, sys
by = json.load(open(sys.argv[1]))["by_instruction"]
counts = {k: int(v["decisions"]) for k, v in by.items()}
for k, v in sorted(counts.items()):
    print(f"[phase6] available {k}: {v} decisions", file=sys.stderr)
print(min(counts.values()))
PYEOF
)"
say "balanced quota = $QUOTA"
if [[ ! -f "$BANK/dataset6/demonstrations.npz" ]]; then
  "${PY[@]}" tools/audit/sil_record.py --mode dataset --inputs "${INPUTS[@]}" \
    --rows-per-instruction "$QUOTA" --output "$BANK/dataset6" 2>&1 \
    | tee "$LOG_DIR/dataset_build.log"
fi

# --- 4. refresh and train ------------------------------------------------
if [[ ! -f "$BANK/refreshed6/demonstrations.npz" ]]; then
  say "refreshing priors"
  "${PY[@]}" tools/audit/sil_refresh_priors.py \
    --dataset "$BANK/dataset6/demonstrations.npz" \
    --frames "$BANK"/*_demos/frames_*.npz --checkpoint "$CHECKPOINT" \
    --min-resolved-fraction 0.98 --output "$BANK/refreshed6" 2>&1 \
    | tee "$LOG_DIR/refresh.log"
fi
if [[ ! -f "$BANK/sft_phase6/sil_sft_adapter.pt" ]]; then
  say "supervised fine-tune, $EPOCHS epochs"
  # WATCH `reachable`. It has been 0.905-0.914 on policy demonstrations, and it
  # is the fraction of targets inside tanh(prior +/- residual_scale). Oracle
  # actions were never produced by that parameterisation, so if they sit
  # outside it more often the loss floors in a way that reads as underfitting
  # and is not.
  "${PY[@]}" tools/audit/sil_sft.py \
    --dataset "$BANK/refreshed6/demonstrations.npz" --checkpoint "$CHECKPOINT" \
    --frames "$BANK"/*_demos/frames_*.npz --epochs "$EPOCHS" \
    --lora-epochs "$LORA_EPOCHS" --progress never \
    --output "$BANK/sft_phase6" 2>&1 | tee "$LOG_DIR/sft.log"
fi
ADAPTER="$BANK/sft_phase6/sil_sft_adapter.pt"

# --- 5. score, including the composed task the seed is FOR ---------------
run_eval() {
  local name="$1" cap="$2" config="$3"; shift 3
  [[ -f "$BANK/eval/phase6_${name}/summary.json" ]] && { say "eval $name done"; return 0; }
  say "eval $name at cap $cap"
  "${PY[@]}" tools/audit/sil_record.py --mode record --rounds "$EVAL_ROUNDS" \
    --worlds "$EVAL_WORLDS" --seed-torch 0 --start-distance-cap "$cap" \
    --checkpoint "$ADAPTER" --config "$config" "$@" \
    --output "$BANK/eval/phase6_${name}" 2>&1 | tee "$LOG_DIR/eval_${name}.log"
}
run_eval pick_up 0.06 "$PICKUP_CONFIG"
run_eval move_to 0.19 "$MOVE_TO_CONFIG"
run_eval placement_caught 0.20 "$PLACEMENT_CONFIG"
# The one this phase exists for. sft_cycle3 scored 0.005 / 0.011 here.
run_eval composed 0.20 "$COMPOSE_CONFIG" "${COMPOSED[@]}"

say "results"
"${PY[@]}" - "$BANK/eval" <<'PYEOF'
import json, pathlib, sys
baseline = {
    "pick_up": 0.1191, "move_to_object": 0.4779,
    "put_into_plate": 0.7383, "put_into_bowl": 0.5353,
}
root = pathlib.Path(sys.argv[1])
for directory in sorted(root.glob("phase6_*")):
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        continue
    totals: dict[str, list[int]] = {}
    summary = json.loads(summary_path.read_text())
    for key, entry in summary.items():
        if not key.startswith("run_") or not isinstance(entry, dict):
            continue
        for name, stats in (entry.get("by_instruction") or {}).items():
            bucket = totals.setdefault(name, [0, 0])
            bucket[0] += int(stats.get("successes", 0))
            bucket[1] += int(stats.get("episodes", 0))
    print(f"\n{directory.name}")
    for name in sorted(totals):
        good, total = totals[name]
        if not total:
            continue
        was = baseline.get(name)
        # The composed eval shares instruction names with the caught one but is
        # a different task, so cycle 3's numbers are not its baseline.
        tail = ""
        if was is not None and "composed" not in directory.name:
            tail = f"   cycle3 {was:.4f}   {good / total - was:+.4f}"
        print(f"  {name:18s} {good / total:.4f}  {good:5d}/{total:<6d}{tail}")
PYEOF
say "done. adapter: $ADAPTER"
