#!/usr/bin/env bash
# Harvest CAUGHT-start put_into episodes, so the composed fraction has room.
#
# WHY. The composed-fraction sweep could not vary its own variable: three arms
# at 0.2 / 0.4 / 0.6 all realised 0.981. put_into's pool is ~98% composed BY
# DECISION COUNT, so asking for 20% composed spends every caught decision the
# bank holds and spills the rest straight back to composed. At the ~26k-decision
# balanced budget, a realised 0.981 means caught supplied about 500 decisions --
# roughly 100 episodes at ~5 each.
#
# THE ASYMMETRY IS THE POINT. A composed episode is ~32 decisions and a caught
# carry ~5, and the quota binds on DECISIONS. So a pool that looks a third
# composed by episode is three quarters composed by decision, and the composed
# side got plentiful while the caught side did not. This is the opposite of the
# o7 re-harvest: that one added composed episodes, this one adds caught ones.
#
# THE ARITHMETIC THIS IS SIZED ON. The lowest composed fraction the quota can
# realise is (budget - caught_decisions) / budget. To reach a floor of 0.2 at a
# 26k budget the bank needs ~20.8k caught decisions, i.e. ~4200 episodes; to
# reach 0.0 it needs the whole budget, ~5200. The defaults below aim past that
# and the script reports the realised floor at the end, so the answer is
# measured rather than assumed.
#
# CAUGHT EPISODES ARE CHEAP. They run ~20 env steps against a composed
# episode's 160, so FRAME_WORLDS can be far higher here than in the composed
# harvest at the same host-RAM cost -- the frame tap builds the array in host
# memory before compressing, and that array is per step.
#
# SMOOTHED, unlike the composed harvest. These are POLICY actions and smoothing
# exists for policy jitter; the composed harvest used --smooth none because the
# oracle is a P-D controller whose actions are already smooth (survival 0.9971
# unsmoothed against 0.9798 smoothed there).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
BANK="${BANK:-$REPO_ROOT/runs/phase4_bank}"
# The current single policy, and the one whose caught behaviour is worth
# banking: it scores caught plate 0.6822 and bowl 0.4985 under sil_record.
# sft_phase6 is slightly better on plate (0.7150) and worse on bowl (0.4794);
# either is defensible, and this is the current lineage.
CHECKPOINT="${CHECKPOINT:-$BANK/sft_phase7/sil_sft_adapter.pt}"
# The CAUGHT config. placement_start_with_caught_object is true here and the
# composed overrides are deliberately absent -- that is the whole difference.
# It also keeps placement_grasp_horizon_min_decisions at its default 32, which
# is correct: a caught episode performs no grasp and must not pay the composed
# task's longer budget.
PLACEMENT_CONFIG="${PLACEMENT_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_placement_loop.yaml}"
WORLDS="${WORLDS:-2048}"
ROUNDS_PER_CAP="${ROUNDS_PER_CAP:-2}"
CAPS="${CAPS:-0.10 0.15 0.20}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
# 1024, against the composed harvest's 320. A caught episode is ~20 env steps
# where a composed one is 160, so this is roughly a third of that harvest's
# frame memory per round despite three times the worlds.
FRAME_WORLDS="${FRAME_WORLDS:-1024}"
REPLAY_LANES="${REPLAY_LANES:-0}"
SMOOTH="${SMOOTH:-moving_average}"
# The composed floor this harvest is trying to reach, used only to report
# whether it got there.
TARGET_FLOOR="${TARGET_FLOOR:-0.2}"
DRY_RUN="${DRY_RUN:-0}"

PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
LOG_DIR="$BANK/phase7_caught_logs"
say() { printf '\n[caught %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f "$PLACEMENT_CONFIG" ]] || { echo "Config not found: $PLACEMENT_CONFIG" >&2; exit 2; }
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# The config must NOT be the composed one. Harvesting caught episodes against a
# config whose placement_caught_object_fraction has been annealed away would
# produce composed episodes under a caught name, which is the one mistake that
# would make this whole run worthless and would not show up until the pool read
# at the end.
"${PY[@]}" - "$PLACEMENT_CONFIG" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from rl_vla_bootstrapping.core.config import load_project_config
metadata = dict(load_project_config(sys.argv[1]).task.metadata or {})
caught = float(metadata.get("placement_caught_object_fraction", 1.0))
enabled = str(metadata.get("placement_caught_curriculum_enabled", False)).lower()
print(f"[caught] placement_caught_object_fraction={caught} "
      f"curriculum_enabled={enabled}")
if caught < 0.99:
    raise SystemExit(
        f"This config starts only {caught:.0%} of container episodes caught. "
        "It would bank composed episodes under a caught name. Point "
        "PLACEMENT_CONFIG at the caught placement config."
    )
if enabled in {"1", "true", "yes", "on"}:
    raise SystemExit(
        "placement_caught_curriculum_enabled is on, so the caught fraction "
        "would anneal DOWN part way through the harvest and the later rounds "
        "would not be caught episodes at all."
    )
PYEOF

printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'caps=%s rounds_per_cap=%s worlds=%s frame_worlds=%s smooth=%s\n' \
  "$CAPS" "$ROUNDS_PER_CAP" "$WORLDS" "$FRAME_WORLDS" "$SMOOTH"
printf 'free RAM: %s\n' "$(free -g 2>/dev/null | awk '/^Mem:/{print $7" GiB available"}' || echo unknown)"
printf 'free space: %s\n' "$(df -h "$BANK" | tail -1)"
[[ "$DRY_RUN" == "1" ]] && { say "DRY_RUN=1, stopping before the first GPU stage."; exit 0; }

# --- 1. record the policy's own caught placements ------------------------
for cap in $CAPS; do
  out="$BANK/c7_${cap}"
  last=$(printf 'record_%02d.npz' $((ROUNDS_PER_CAP - 1)))
  if [[ -f "$out/$last" ]]; then say "cap $cap already harvested"; continue; fi
  say "harvest cap $cap"
  "${PY[@]}" tools/audit/sil_record.py --mode record \
    --rounds "$ROUNDS_PER_CAP" --worlds "$WORLDS" --devices "$DEVICES" \
    --seed-torch 0 --start-distance-cap "$cap" \
    --checkpoint "$CHECKPOINT" --config "$PLACEMENT_CONFIG" \
    --output "$out" 2>&1 | tee "$LOG_DIR/harvest_${cap}.log"
done

# --- 2. replay for frames, smoothed --------------------------------------
IFS=',' read -r -a device_list <<< "$DEVICES"
lanes=${#device_list[@]}
if [[ "$REPLAY_LANES" -gt 0 && "$REPLAY_LANES" -lt "$lanes" ]]; then
  lanes="$REPLAY_LANES"
fi
for cap in $CAPS; do
  say "replay cap $cap on $lanes lane(s) -> $BANK/c7_demos"
  pids=()
  for lane in $(seq 0 $((lanes - 1))); do
    (
      for index in $(seq 0 $((ROUNDS_PER_CAP - 1))); do
        [[ $((index % lanes)) -eq "$lane" ]] || continue
        stem=$(printf 'record_%02d' "$index")
        # Guard on the FRAMES file: sil_record writes the replay npz first and
        # the frames npz last, so a kill during frame compression leaves a
        # complete replay beside a truncated frames file, and a guard on the
        # replay would skip the round and leave the corruption to surface as
        # BadZipFile in sil_refresh_priors after the harvest was paid for.
        [[ -f "$BANK/c7_demos/frames_c7_${cap}_${stem}.npz" ]] && continue
        "${PY[@]}" tools/audit/sil_record.py --mode replay --smooth "$SMOOTH" \
          --actions "$BANK/c7_${cap}/${stem}.npz" \
          --worlds "$WORLDS" --device "${device_list[$lane]}" \
          --seed-torch 0 --start-distance-cap "$cap" \
          --checkpoint "$CHECKPOINT" --config "$PLACEMENT_CONFIG" \
          --record-frames --frame-worlds "$FRAME_WORLDS" \
          --output "$BANK/c7_demos"
      done
    ) > "$LOG_DIR/replay_${cap}_lane${lane}.log" 2>&1 &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  grep -h "survived" "$LOG_DIR/replay_${cap}"_lane*.log || true
  if [[ "$status" -ne 0 ]]; then
    echo "=== replay lane failed at cap $cap; tail of each lane log ===" >&2
    for log in "$LOG_DIR/replay_${cap}"_lane*.log; do
      echo "--- $log" >&2; tail -n 25 "$log" >&2
    done
    df -h "$BANK" >&2 || true
    free -g >&2 2>/dev/null || true
    echo "A lane that says only 'Killed' is the host OOM killer: lower" \
         "FRAME_WORLDS or set REPLAY_LANES=1." >&2
    exit 1
  fi
done

# --- 3. did it buy the range? --------------------------------------------
# The whole point, measured rather than assumed. Rebuilt from scratch because
# the pool has changed and a stale probe would report the old composition.
say "re-reading pool composition with the new caught rounds"
rm -rf "$BANK/dataset8_probe"
INPUTS=(
  "$BANK"/pick_up_demos/replay_*.npz
  "$BANK"/pick_up_iter2_demos/replay_*.npz
  "$BANK"/move_to_demos/replay_*.npz
  "$BANK"/m2_demos/replay_*.npz
  "$BANK"/p3_demos/replay_*.npz
  "$BANK"/o6_demos/replay_*.npz
  "$BANK"/o7_demos/replay_*.npz
  "$BANK"/c7_demos/replay_*.npz
)
"${PY[@]}" tools/audit/sil_record.py --mode dataset --inputs "${INPUTS[@]}" \
  --require-frames "$BANK"/*_demos/frames_*.npz \
  --rows-per-instruction 0 --output "$BANK/dataset8_probe" 2>&1 \
  | tee "$LOG_DIR/dataset_probe.log"

"${PY[@]}" - "$BANK/dataset8_probe/dataset.json" "$TARGET_FLOOR" <<'PYEOF'
import json, sys
by = json.load(open(sys.argv[1]))["by_instruction"]
target = float(sys.argv[2])
quota = min(int(e["decisions"]) for e in by.values())
print(f"\n[caught] balanced quota is now {quota} decisions per instruction")
ok = True
for name, entry in sorted(by.items()):
    pool = entry.get("pool_strata") or {}
    caught = int(pool.get("caught_decisions", 0))
    composed = int(pool.get("composed_decisions", 0))
    if not caught or not composed:
        continue
    floor = max(0.0, (quota - caught) / quota)
    flag = "" if floor <= target + 1e-9 else "   <-- STILL ABOVE TARGET"
    print(f"[caught] {name}: composed {composed} / caught {caught} decisions"
          f"  -> reachable composed floor {floor:.3f}{flag}")
    if floor > target + 1e-9:
        ok = False
        short = int((1.0 - target) * quota) - caught
        print(f"[caught]   short by ~{short} caught decisions "
              f"(~{short // 5} episodes at ~5 each); raise ROUNDS_PER_CAP and "
              "re-run -- every stage is guarded, so finished rounds are kept")
if ok:
    print(f"\n[caught] the composed fraction can now be swept down to {target}.")
PYEOF
say "done -- add c7_demos to the INPUTS list of the sweep and re-run it"
