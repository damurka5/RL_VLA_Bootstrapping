#!/usr/bin/env bash
# Re-harvest the composed oracle demonstrations on the corrected horizon.
#
# WHY THIS EXISTS. The o6_* harvest under runs/phase4_bank was collected with
# placement_grasp_horizon_min_decisions at its default 32, which is 128 env
# steps, and the composed episode does not fit in it. Decomposed over the 24 576
# episodes of that harvest (tools/audit/placement_failure_decomposition.py):
#
#   plate no_release   291 episodes, 291 ran the WHOLE budget  (1.000)
#   bowl  no_release  2095 episodes, 1735 ran the whole budget (0.828)
#   first grasp        plate p50 90  p90 116  |  bowl p50 81  p90 112
#   ended still holding  plate 279 of 291  |  bowl 1549 of 2095
#
# Those episodes never reached the release. Nothing about them is a placement
# failure: under the oracle, success|settled is 1.0000 for both receptacles and
# the object does not bounce out (bowl bounce mean 0.00018 m -- 0.18 mm). The
# budget ran out mid-carry.
#
# WHAT THE RE-HARVEST BUYS. Converting the timed-out episodes at the settle rate
# they would have met moves the ORACLE from 0.9088 -> 0.9315 on plate (against a
# 0.9317 grasp ceiling) and 0.4565 -> 0.6009 on bowl (against 0.6459). Bowl is
# the point: +32% relative, and every one of those is a demonstration the bank
# does not currently have. The composed slice is what the SFT is starved of.
#
# WHY NOT JUST RE-RUN THE SEED SCRIPT. Because the existing o6_* rounds are
# still valid demonstrations -- they were produced by the same oracle against
# the same predicate, and a successful 128-step episode is not made wrong by a
# longer budget being available. This harvests INTO A NEW DIRECTORY and pools
# both, so the bank grows rather than churns. It also does not re-derive the
# caught-start placement, pick_up or move_to slices at all; those are untouched
# on disk and are pooled as they are.
#
# THIS TRAINS NOTHING BEYOND THE SFT. The composition RL is a separate,
# deliberate step, as it was in phase 6.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
BANK="${BANK:-$REPO_ROOT/runs/phase4_bank}"
# sft_phase6, not sft_cycle3. In oracle mode the checkpoint's ACTIONS are
# discarded, but its states and priors are what the demonstrations are
# conditioned on and it is what the SFT starts from -- and sft_phase6 is the
# current composition seed. Deliberately NOT the phase6_compose_iter0 peak:
# that checkpoint is the better CAUGHT one and the worse COMPOSED one
# (composed plate 0.0689 against the seed's 0.0935), and composition is the
# thing this is for.
CHECKPOINT="${CHECKPOINT:-$BANK/sft_phase6/sil_sft_adapter.pt}"
COMPOSE_CONFIG="${COMPOSE_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase5_compose_loop.yaml}"
PLACEMENT_CONFIG="${PLACEMENT_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_placement_loop.yaml}"
PICKUP_CONFIG="${PICKUP_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_pick_up_loop.yaml}"
MOVE_TO_CONFIG="${MOVE_TO_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml}"
WORLDS="${WORLDS:-2048}"
ROUNDS_PER_CAP="${ROUNDS_PER_CAP:-4}"
CAPS="${CAPS:-0.10 0.15 0.20}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
EPOCHS="${EPOCHS:-60}"
LORA_EPOCHS="${LORA_EPOCHS:-0}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EVAL_WORLDS="${EVAL_WORLDS:-512}"
# Same reasoning as the seed script: this is a HOST RAM bound, not a disk one.
# The frame tap builds the whole array in host memory before compressing, and
# an uncapped round measured 2993 MB with two lanes holding one each beside two
# SmolVLA models and two 2048-world MJWarp instances. A LONGER horizon makes
# each surviving episode carry more frames, so if a lane is killed here and was
# not in phase 6, this is the knob -- lower it, or set REPLAY_LANES=1.
FRAME_WORLDS="${FRAME_WORLDS:-320}"
REPLAY_LANES="${REPLAY_LANES:-0}"
# Share of each instruction's row budget spent on episodes that did NOT start
# with the object between the fingers. Negative is OFF and reproduces the flat
# quota exactly, which is what phase 6 and the first phase-7 build ran.
#
# WHY IT IS OFF BY DEFAULT HERE. Turning it on changes the mix at the same time
# as the harvest changes, and then a moved number has two causes. Run this
# script once at -1 to get the re-harvest effect alone; set it afterwards to
# choose the trade deliberately. Measured on the first phase-7 build, the
# UNCHOSEN share was 0.16, and the o7 harvest raising it is what moved composed
# plate 0.0935 -> 0.1203 while caught plate fell 0.7150 -> 0.6822.
COMPOSED_FRACTION="${COMPOSED_FRACTION:--1}"
DRY_RUN="${DRY_RUN:-0}"

# The composed protocol: every container episode starts UNCAUGHT and the
# curriculum is off, so nothing restores the caught fraction mid-harvest.
COMPOSED=(--metadata-override placement_caught_object_fraction=0.0
          placement_caught_curriculum_enabled=false)

PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
LOG_DIR="$BANK/phase7_logs"
say() { printf '\n[phase7 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
for cfg in "$COMPOSE_CONFIG" "$PLACEMENT_CONFIG" "$PICKUP_CONFIG" "$MOVE_TO_CONFIG"; do
  [[ -f "$cfg" ]] || { echo "Config not found: $cfg" >&2; exit 2; }
done
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# THE PRECONDITION. The whole point is the longer budget; harvesting again on
# 32 would burn hours of GPU reproducing the data that already exists. Read the
# resolved value rather than trusting that the config was edited.
HORIZON="$("${PY[@]}" - "$COMPOSE_CONFIG" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from rl_vla_bootstrapping.core.config import load_project_config
metadata = dict(load_project_config(sys.argv[1]).task.metadata or {})
print(int(metadata.get("placement_grasp_horizon_min_decisions", 32)))
PYEOF
)"
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'composed horizon floor = %s decisions = %s env steps\n' \
  "$HORIZON" "$((HORIZON * 4))"
if [[ "$HORIZON" -lt 40 ]]; then
  echo "placement_grasp_horizon_min_decisions is $HORIZON in $COMPOSE_CONFIG." >&2
  echo "This script exists to re-harvest on a LONGER budget; at 32 it would" >&2
  echo "reproduce the harvest already under $BANK/o6_* for nothing. Set it to" >&2
  echo "40 (p90 first grasp 116 env steps + ~40 for the carry) and re-run." >&2
  exit 2
fi
printf 'caps=%s rounds_per_cap=%s worlds=%s devices=%s\n' \
  "$CAPS" "$ROUNDS_PER_CAP" "$WORLDS" "$DEVICES"
printf 'epochs=%s frame_worlds=%s replay_lanes=%s logs=%s\n' \
  "$EPOCHS" "$FRAME_WORLDS" "$REPLAY_LANES" "$LOG_DIR"
printf 'free RAM: %s\n' "$(free -g 2>/dev/null | awk '/^Mem:/{print $7" GiB available"}' || echo unknown)"
printf 'free space: %s\n' "$(df -h "$BANK" | tail -1)"
[[ "$DRY_RUN" == "1" ]] && { say "DRY_RUN=1, stopping before the first GPU stage."; exit 0; }

# --- 1. oracle demonstrations, longer budget -----------------------------
for cap in $CAPS; do
  out="$BANK/o7_${cap}"
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
# --smooth none, as in phase 6: smoothing exists to fix POLICY jitter and the
# oracle is a P-D controller whose actions are already smooth. Measured there,
# survival 0.9971 unsmoothed against 0.9798 smoothed.
IFS=',' read -r -a device_list <<< "$DEVICES"
lanes=${#device_list[@]}
if [[ "$REPLAY_LANES" -gt 0 && "$REPLAY_LANES" -lt "$lanes" ]]; then
  lanes="$REPLAY_LANES"
fi
for cap in $CAPS; do
  say "replay cap $cap on $lanes lane(s) -> $BANK/o7_demos"
  pids=()
  for lane in $(seq 0 $((lanes - 1))); do
    (
      for index in $(seq 0 $((ROUNDS_PER_CAP - 1))); do
        [[ $((index % lanes)) -eq "$lane" ]] || continue
        stem=$(printf 'record_%02d' "$index")
        # Guard on the FRAMES file, not the replay one: sil_record writes the
        # replay npz FIRST and the frames npz LAST, so a process killed while
        # compressing frames leaves a complete replay beside a truncated frames
        # file. A guard on the replay would skip the round and leave the
        # corruption to surface as BadZipFile in sil_refresh_priors, after the
        # whole harvest had been paid for. That happened in phase 6.
        [[ -f "$BANK/o7_demos/frames_o7_${cap}_${stem}.npz" ]] && continue
        "${PY[@]}" tools/audit/sil_record.py --mode replay --smooth none \
          --actions "$BANK/o7_${cap}/${stem}.npz" \
          --worlds "$WORLDS" --device "${device_list[$lane]}" \
          --seed-torch 0 --start-distance-cap "$cap" "${COMPOSED[@]}" \
          --checkpoint "$CHECKPOINT" --config "$COMPOSE_CONFIG" \
          --record-frames --frame-worlds "$FRAME_WORLDS" \
          --output "$BANK/o7_demos"
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
      echo "--- $log" >&2
      tail -n 25 "$log" >&2
    done
    df -h "$BANK" >&2 || true
    free -g >&2 2>/dev/null || true
    echo "If a lane says only 'Killed', that is the host OOM killer and not" \
         "this tool. The longer horizon makes each episode carry more frames" \
         "than phase 6 did, so lower FRAME_WORLDS or set REPLAY_LANES=1." >&2
    exit 1
  fi
done

# --- 2b. did the horizon actually buy anything? --------------------------
# Before paying for the SFT. This is the falsifiable claim the whole re-harvest
# rests on, it costs seconds on the CPU, and it reads the same tool that
# produced the finding.
say "decomposing the new harvest against the old"
for tag in o6 o7; do
  [[ -d "$BANK/${tag}_demos" ]] || continue
  "${PY[@]}" tools/audit/placement_failure_decomposition.py \
    --recordings "$BANK/${tag}_"*"/record_"*.npz \
    --config "$COMPOSE_CONFIG" \
    --output "$BANK/eval/${tag}_decomp" 2>&1 \
    | tee "$LOG_DIR/decomp_${tag}.log" || true
done
say "compare: o6 is the 128-step harvest, o7 the $((HORIZON * 4))-step one"
"${PY[@]}" - "$BANK/eval" <<'PYEOF' || true
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = {}
for tag in ("o6", "o7"):
    path = root / f"{tag}_decomp" / "failure_decomposition.json"
    if path.is_file():
        rows[tag] = json.loads(path.read_text())["by_instruction"]
if len(rows) == 2:
    print(f"\n{'instruction':<18}{'o6 (128)':>12}{'o7 (new)':>12}{'timed out o6':>15}{'timed out o7':>15}")
    for name in sorted(set(rows["o6"]) | set(rows["o7"])):
        cells = []
        for tag in ("o6", "o7"):
            entry = rows[tag].get(name)
            cells.append(entry["success_rate"] if entry else None)
        outs = []
        for tag in ("o6", "o7"):
            entry = rows[tag].get(name)
            diag = (entry or {}).get("no_release_diagnosis") or {}
            outs.append(diag.get("timed_out_fraction"))
        print(f"{name:<18}{str(cells[0]):>12}{str(cells[1]):>12}"
              f"{str(outs[0]):>15}{str(outs[1]):>15}")
    print("\nIf o7's success is not clearly above o6's, STOP. The horizon was "
          "not the binding constraint and the SFT below will not fix it.")
PYEOF

# --- 3. pool, balanced from a measured availability read -----------------
# o7 JOINS o6 rather than replacing it: a successful 128-step composed episode
# is not made wrong by a longer budget existing. Caught-start placement,
# pick_up and move_to are pooled untouched.
INPUTS=(
  "$BANK"/pick_up_demos/replay_*.npz
  "$BANK"/pick_up_iter2_demos/replay_*.npz
  "$BANK"/move_to_demos/replay_*.npz
  "$BANK"/m2_demos/replay_*.npz
  "$BANK"/p3_demos/replay_*.npz
  "$BANK"/o6_demos/replay_*.npz
  "$BANK"/o7_demos/replay_*.npz
)
if [[ ! -f "$BANK/dataset7_probe/dataset.json" ]]; then
  say "reading availability (frames-filtered)"
  "${PY[@]}" tools/audit/sil_record.py --mode dataset --inputs "${INPUTS[@]}" \
    --require-frames "$BANK"/*_demos/frames_*.npz \
    --rows-per-instruction 0 --output "$BANK/dataset7_probe" 2>&1 \
    | tee "$LOG_DIR/dataset_probe.log"
fi
QUOTA="$("${PY[@]}" - "$BANK/dataset7_probe/dataset.json" <<'PYEOF'
import json, sys
by = json.load(open(sys.argv[1]))["by_instruction"]
counts = {k: int(v["decisions"]) for k, v in by.items()}
for k, v in sorted(counts.items()):
    print(f"[phase7] available {k}: {v} decisions", file=sys.stderr)
print(min(counts.values()))
PYEOF
)"
say "balanced quota = $QUOTA decisions per instruction"
if [[ ! -f "$BANK/dataset7/demonstrations.npz" ]]; then
  "${PY[@]}" tools/audit/sil_record.py --mode dataset --inputs "${INPUTS[@]}" \
    --require-frames "$BANK"/*_demos/frames_*.npz \
    --rows-per-instruction "$QUOTA" --composed-fraction "$COMPOSED_FRACTION" \
    --output "$BANK/dataset7" 2>&1 \
    | tee "$LOG_DIR/dataset_build.log"
fi
# What the mix actually came out as. READ realized_composed_fraction, not the
# requested one: they differ whenever a stratum ran out, and that difference is
# a harvest finding rather than a mix finding -- no reweighting fixes a bank
# that does not hold the episodes.
"${PY[@]}" - "$BANK/dataset7/dataset.json" <<'PYEOF' || true
import json, sys
quota = (json.load(open(sys.argv[1])).get("quota") or {})
per = quota.get("by_instruction") or {}
if not per:
    print("[phase7] composed-fraction OFF; the mix is whatever the bank holds.")
for name, entry in sorted(per.items()):
    print(f"[phase7] {name}: composed {entry['composed_decisions']} / "
          f"{entry['decisions']} decisions = "
          f"{entry['realized_composed_fraction']} "
          f"(requested {entry['requested_composed_fraction']}, "
          f"{entry['available_composed_episodes']} composed episodes available)")
PYEOF

# --- 4. refresh and train ------------------------------------------------
if [[ ! -f "$BANK/refreshed7/demonstrations.npz" ]]; then
  say "refreshing priors against $CHECKPOINT"
  "${PY[@]}" tools/audit/sil_refresh_priors.py \
    --dataset "$BANK/dataset7/demonstrations.npz" \
    --frames "$BANK"/*_demos/frames_*.npz --checkpoint "$CHECKPOINT" \
    --min-resolved-fraction 0.98 --output "$BANK/refreshed7" 2>&1 \
    | tee "$LOG_DIR/refresh.log"
fi
if [[ ! -f "$BANK/sft_phase7/sil_sft_adapter.pt" ]]; then
  say "supervised fine-tune, $EPOCHS epochs"
  # 60 epochs, not phase 6's 45. Its val_mse was still falling at epoch 44 of
  # 45, so that mix was stopped short rather than converged, and this one is
  # larger. Best-validation checkpointing means the extra epochs cost wall
  # clock and cannot cost quality.
  #
  # WATCH `reachable` -- the fraction of targets inside
  # tanh(prior +/- residual_scale). Phase 6 read 0.91111 on oracle actions
  # against 0.905-0.914 on policy ones. Oracle actions were never produced by
  # that parameterisation, so a drop here makes the loss floor in a way that
  # reads as underfitting and is not.
  "${PY[@]}" tools/audit/sil_sft.py \
    --dataset "$BANK/refreshed7/demonstrations.npz" --checkpoint "$CHECKPOINT" \
    --frames "$BANK"/*_demos/frames_*.npz --epochs "$EPOCHS" \
    --lora-epochs "$LORA_EPOCHS" --progress never \
    --output "$BANK/sft_phase7" 2>&1 | tee "$LOG_DIR/sft.log"
fi
ADAPTER="$BANK/sft_phase7/sil_sft_adapter.pt"

# --- 5. score, on the same four protocols phase 6 used -------------------
run_eval() {
  local name="$1" cap="$2" config="$3"; shift 3
  [[ -f "$BANK/eval/phase7_${name}/summary.json" ]] && { say "eval $name done"; return 0; }
  say "eval $name at cap $cap"
  "${PY[@]}" tools/audit/sil_record.py --mode record --rounds "$EVAL_ROUNDS" \
    --worlds "$EVAL_WORLDS" --seed-torch 0 --start-distance-cap "$cap" \
    --checkpoint "$ADAPTER" --config "$config" "$@" \
    --output "$BANK/eval/phase7_${name}" 2>&1 | tee "$LOG_DIR/eval_${name}.log"
}
run_eval pick_up 0.06 "$PICKUP_CONFIG"
run_eval move_to 0.19 "$MOVE_TO_CONFIG"
run_eval placement_caught 0.20 "$PLACEMENT_CONFIG"
run_eval composed 0.20 "$COMPOSE_CONFIG" "${COMPOSED[@]}"

# --- 6. and decompose the policy, not just score it ----------------------
# The composed number alone cannot say WHY it moved. sft_phase6 lost 204 of its
# 287 no_release plate episodes to dropping the object mid-carry rather than to
# the clock, and a longer horizon does nothing for that failure -- so this is
# where it becomes visible whether the grip-retention problem survived.
say "decomposing the new policy on the composed protocol"
"${PY[@]}" tools/audit/placement_failure_decomposition.py \
  --recordings "$BANK/eval/phase7_composed/record_"*.npz \
  --config "$COMPOSE_CONFIG" \
  --output "$BANK/eval/phase7_composed_decomp" 2>&1 \
  | tee "$LOG_DIR/decomp_policy.log" || true

say "results, against sft_phase6 under the identical protocol"
"${PY[@]}" - "$BANK/eval" <<'PYEOF'
import json, pathlib, sys
# sft_phase6, three rounds of 512 worlds, same seeds and caps.
baseline = {
    "phase7_pick_up": {"pick_up": (229, 1536)},
    "phase7_move_to": {"move_to_object": (737, 1536)},
    "phase7_placement_caught": {
        "put_into_plate": (612, 856), "put_into_bowl": (326, 680)
    },
    "phase7_composed": {
        "put_into_plate": (80, 856), "put_into_bowl": (18, 680)
    },
}
root = pathlib.Path(sys.argv[1])
for directory in sorted(root.glob("phase7_*")):
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
        prior = (baseline.get(directory.name) or {}).get(name)
        if prior:
            was = prior[0] / prior[1]
            delta = good / total - was
            print(f"  {name:<18} {good:>5}/{total:<5} = {good / total:.4f}"
                  f"   sft_phase6 {prior[0]}/{prior[1]} = {was:.4f}"
                  f"   {delta:+.4f}")
        else:
            print(f"  {name:<18} {good:>5}/{total:<5} = {good / total:.4f}")
print("\nThe composed row is the one this phase is for. Read the decomposition "
      "in eval/phase7_composed_decomp beside it: a composed number that moved "
      "because the clock stopped binding and one that moved because the policy "
      "stopped dropping the object are different results.")
PYEOF
say "done"
