#!/usr/bin/env bash
# Price the composed/caught trade instead of inheriting it.
#
# WHAT IT ANSWERS. --rows-per-instruction balances by instruction id only, so
# inside put_into_* the caught-start carries and the composed grasp-carry-
# releases compete purely by how many of each the bank holds. The o7 re-harvest
# moved that mix as a SIDE EFFECT -- composed plate 0.0935 -> 0.1203 while
# caught plate fell 0.7150 -> 0.6822 -- and nothing chose the trade. This runs
# the same pipeline at several deliberate fractions so the trade can be read
# off a curve and picked.
#
# THE REFERENCE ARM IS FREE. sft_phase7 already exists and was built with the
# knob OFF, at a realized composed share of about 0.16. It is the "unchosen"
# point on the curve and this script reports it beside the swept ones rather
# than rebuilding it.
#
# ONLY put_into MOVES. `starts_grasped` comes from physical_grasp_at_reset, and
# harvests run through validate_round, which passes allow_prelifted=False -- so
# no pick_up demonstration in the bank ever started pre-grasped, and move_to has
# no caught stage at all. Both instructions therefore hold a single stratum, the
# quota spills to 100%, and their slices are identical across every arm. They
# are still EVALUATED, because the residual is trained jointly and a put_into
# mix can perturb them through shared weights even when their data does not
# change. That is the retention check and it is the reason not to skip it.
#
# COST. Each arm is one refresh, one 60-epoch SFT and four evaluations. Budget
# several hours per arm. Every stage is guarded on its output file, so the
# script is resumable, and ARMS lets it be run one fraction at a time.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
BANK="${BANK:-$REPO_ROOT/runs/phase4_bank}"
CHECKPOINT="${CHECKPOINT:-$BANK/sft_phase6/sil_sft_adapter.pt}"
COMPOSE_CONFIG="${COMPOSE_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase5_compose_loop.yaml}"
PLACEMENT_CONFIG="${PLACEMENT_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_placement_loop.yaml}"
PICKUP_CONFIG="${PICKUP_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_pick_up_loop.yaml}"
MOVE_TO_CONFIG="${MOVE_TO_CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml}"
ARMS="${ARMS:-0.2 0.4 0.6}"
EPOCHS="${EPOCHS:-60}"
LORA_EPOCHS="${LORA_EPOCHS:-0}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EVAL_WORLDS="${EVAL_WORLDS:-512}"
DRY_RUN="${DRY_RUN:-0}"

COMPOSED=(--metadata-override placement_caught_object_fraction=0.0
          placement_caught_curriculum_enabled=false)
PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
LOG_DIR="$BANK/phase7_sweep_logs"
say() { printf '\n[sweep %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# The same pool the phase-7 build used, so an arm differs from sft_phase7 in
# the MIX and in nothing else.
INPUTS=(
  "$BANK"/pick_up_demos/replay_*.npz
  "$BANK"/pick_up_iter2_demos/replay_*.npz
  "$BANK"/move_to_demos/replay_*.npz
  "$BANK"/m2_demos/replay_*.npz
  "$BANK"/p3_demos/replay_*.npz
  "$BANK"/o6_demos/replay_*.npz
  "$BANK"/o7_demos/replay_*.npz
)

# Reuse the availability read rather than repeating it: the pool has not
# changed, so the balanced quota has not either. Every arm must spend the SAME
# budget or the comparison is between slice sizes rather than between mixes.
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
print(min(int(v["decisions"]) for v in by.values()))
PYEOF
)"
say "balanced quota = $QUOTA decisions per instruction, arms: $ARMS"

# CAN THE POOL EVEN SUPPLY THESE ARMS? The first run of this script could not:
# three arms at 0.2/0.4/0.6 and every one realized 0.981, because put_into's
# pool is ~98% composed BY DECISION and the spill had nowhere else to go. Hours
# of GPU for one mix under three names. The probe now reports the composition,
# so this is checkable for free before any of it is spent.
say "pool composition, per instruction"
"${PY[@]}" - "$BANK/dataset7_probe/dataset.json" "$QUOTA" "$ARMS" <<'PYEOF'
import json, sys
by = json.load(open(sys.argv[1]))["by_instruction"]
quota, arms = int(sys.argv[2]), [float(a) for a in sys.argv[3].split()]
floors = []
for name, entry in sorted(by.items()):
    pool = entry.get("pool_strata")
    if not pool:
        print(f"[sweep]   {name}: no pool_strata -- the probe predates this "
              "read; delete dataset7_probe and let it rebuild")
        continue
    caught, composed = pool["caught_decisions"], pool["composed_decisions"]
    if not caught or not composed:
        print(f"[sweep]   {name}: single stratum (composed {composed}, "
              f"caught {caught}) -- the knob is vacuous here, spills to 100%")
        continue
    # The lowest composed fraction the pool can realise at this budget: spend
    # every caught decision it has, and composed for the remainder.
    floor = max(0.0, (quota - caught) / quota)
    print(f"[sweep]   {name}: composed {composed} / caught {caught} decisions "
          f"({pool['composed_decision_fraction']:.3f} composed), "
          f"{pool['composed_decisions_per_episode']} vs "
          f"{pool['caught_decisions_per_episode']} decisions per episode")
    print(f"[sweep]     at a {quota}-decision budget the reachable composed "
          f"range is {floor:.3f} - 1.000")
    floors.append(floor)
if floors:
    worst = max(floors)
    blocked = [a for a in arms if a < worst - 0.02]
    if blocked:
        print(f"[sweep] WARNING: arms {blocked} sit BELOW the reachable floor "
              f"{worst:.3f} and will every one realise about {worst:.3f} -- "
              "the same mix under different names. Harvest more caught-start "
              "put_into episodes, or sweep inside the reachable range.")
PYEOF

[[ "$DRY_RUN" == "1" ]] && { say "DRY_RUN=1, stopping before the first GPU stage."; exit 0; }

slug() { printf 'f%s' "${1/./}"; }

# --- per arm: build, refresh, train --------------------------------------
for frac in $ARMS; do
  tag="$(slug "$frac")"
  say "=== arm composed_fraction=$frac (tag $tag) ==="

  if [[ ! -f "$BANK/dataset7_$tag/demonstrations.npz" ]]; then
    "${PY[@]}" tools/audit/sil_record.py --mode dataset --inputs "${INPUTS[@]}" \
      --require-frames "$BANK"/*_demos/frames_*.npz \
      --rows-per-instruction "$QUOTA" --composed-fraction "$frac" \
      --output "$BANK/dataset7_$tag" 2>&1 \
      | tee "$LOG_DIR/dataset_$tag.log"
  fi
  # REALIZED, not requested. They part company as soon as a stratum runs out,
  # and an arm that could not reach its fraction is a point on a different
  # curve -- it says the bank is the constraint, not the mix.
  "${PY[@]}" - "$BANK/dataset7_$tag/dataset.json" <<'PYEOF' || true
import json, sys
per = (json.load(open(sys.argv[1])).get("quota") or {}).get("by_instruction") or {}
for name, e in sorted(per.items()):
    flag = "" if abs(e["realized_composed_fraction"] - e["requested_composed_fraction"]) < 0.02 else "  <-- SHORT"
    print(f"[sweep]   {name}: composed {e['composed_decisions']}/{e['decisions']}"
          f" = {e['realized_composed_fraction']} (asked {e['requested_composed_fraction']},"
          f" {e['available_composed_episodes']} composed eps available){flag}")
PYEOF

  if [[ ! -f "$BANK/refreshed7_$tag/demonstrations.npz" ]]; then
    say "refreshing priors ($tag)"
    "${PY[@]}" tools/audit/sil_refresh_priors.py \
      --dataset "$BANK/dataset7_$tag/demonstrations.npz" \
      --frames "$BANK"/*_demos/frames_*.npz --checkpoint "$CHECKPOINT" \
      --min-resolved-fraction 0.98 --output "$BANK/refreshed7_$tag" 2>&1 \
      | tee "$LOG_DIR/refresh_$tag.log"
  fi

  if [[ ! -f "$BANK/sft_phase7_$tag/sil_sft_adapter.pt" ]]; then
    say "supervised fine-tune ($tag), $EPOCHS epochs"
    "${PY[@]}" tools/audit/sil_sft.py \
      --dataset "$BANK/refreshed7_$tag/demonstrations.npz" \
      --checkpoint "$CHECKPOINT" \
      --frames "$BANK"/*_demos/frames_*.npz --epochs "$EPOCHS" \
      --lora-epochs "$LORA_EPOCHS" --progress never \
      --output "$BANK/sft_phase7_$tag" 2>&1 | tee "$LOG_DIR/sft_$tag.log"
  fi
done

# --- evaluate: the deciding pair first, then retention -------------------
# composed and placement_caught are the two sides of the trade, so they run for
# every arm before any retention leg does. An interrupted sweep then still
# leaves a complete trade curve.
run_eval() {
  local tag="$1" name="$2" cap="$3" config="$4"; shift 4
  local out="$BANK/eval/sweep_${tag}_${name}"
  [[ -f "$out/summary.json" ]] && return 0
  say "eval $tag/$name at cap $cap"
  "${PY[@]}" tools/audit/sil_record.py --mode record --rounds "$EVAL_ROUNDS" \
    --worlds "$EVAL_WORLDS" --seed-torch 0 --start-distance-cap "$cap" \
    --checkpoint "$BANK/sft_phase7_$tag/sil_sft_adapter.pt" \
    --config "$config" "$@" --output "$out" 2>&1 \
    | tee "$LOG_DIR/eval_${tag}_${name}.log"
}
for frac in $ARMS; do
  tag="$(slug "$frac")"
  run_eval "$tag" composed 0.20 "$COMPOSE_CONFIG" "${COMPOSED[@]}"
  run_eval "$tag" placement_caught 0.20 "$PLACEMENT_CONFIG"
done
for frac in $ARMS; do
  tag="$(slug "$frac")"
  run_eval "$tag" pick_up 0.06 "$PICKUP_CONFIG"
  run_eval "$tag" move_to 0.19 "$MOVE_TO_CONFIG"
done

# --- compare -------------------------------------------------------------
say "the curve"
"${PY[@]}" - "$BANK" "$ARMS" <<'PYEOF'
import json, pathlib, sys

bank = pathlib.Path(sys.argv[1])
arms = sys.argv[2].split()

def totals(directory):
    path = directory / "summary.json"
    if not path.is_file():
        return {}
    out = {}
    for key, entry in json.loads(path.read_text()).items():
        if not key.startswith("run_") or not isinstance(entry, dict):
            continue
        for name, stats in (entry.get("by_instruction") or {}).items():
            bucket = out.setdefault(name, [0, 0])
            bucket[0] += int(stats.get("successes", 0))
            bucket[1] += int(stats.get("episodes", 0))
    return out

# The reference arm: the phase-7 build, knob OFF.
points = [("off (~0.16)", "phase7", None)]
for frac in arms:
    points.append((frac, f"sweep_f{frac.replace('.', '')}", frac))

rows = []
for label, prefix, frac in points:
    entry = {"label": label}
    for leg, directory in (
        ("composed", "composed"), ("caught", "placement_caught"),
        ("pick_up", "pick_up"), ("move_to", "move_to"),
    ):
        name = (f"{prefix}_{directory}" if frac is None
                else f"{prefix}_{directory}")
        entry[leg] = totals(bank / "eval" / name)
    if frac is not None:
        stats = bank / f"dataset7_f{frac.replace('.', '')}" / "dataset.json"
        if stats.is_file():
            per = (json.loads(stats.read_text()).get("quota") or {}).get("by_instruction") or {}
            got = [e["realized_composed_fraction"] for e in per.values()
                   if e["available_caught_episodes"] and e["available_composed_episodes"]]
            entry["realized"] = round(sum(got) / len(got), 3) if got else None
    rows.append(entry)

def rate(bucket, key):
    got = bucket.get(key)
    return None if not got or not got[1] else got[0] / got[1]

COLUMNS = (
    ("comp plate", "composed", "put_into_plate"),
    ("comp bowl", "composed", "put_into_bowl"),
    ("caught plate", "caught", "put_into_plate"),
    ("caught bowl", "caught", "put_into_bowl"),
    ("pick_up", "pick_up", "pick_up"),
    ("move_to", "move_to", "move_to_object"),
)
WIDTH = 14
header = f"{'fraction':<14}{'realized':>10}" + "".join(
    f"{title:>{WIDTH}}" for title, _, _ in COLUMNS
)
print("\n" + header)
print("-" * len(header))
for entry in rows:
    realized = entry.get("realized")
    line = f"{entry['label']:<14}{('-' if realized is None else realized):>10}"
    for _, leg, key in COLUMNS:
        value = rate(entry[leg], key)
        line += f"{('-' if value is None else f'{value:.4f}'):>{WIDTH}}"
    print(line)
print("""
Read the two placement columns together: they are the trade. A fraction that
lifts composed while caught holds is a free gain; one that lifts composed by
the same amount caught loses is a choice about which task matters.

pick_up and move_to are the retention check. Their SLICES are identical across
every arm -- neither has two strata -- so any movement there is the residual
being perturbed through shared weights, and a large one is a reason to reject
the fraction whatever it did for placement.

realized < asked means the bank ran out of composed episodes, not that the mix
was rejected. That is a harvest finding: no reweighting fixes it, another
oracle harvest does.
""")
PYEOF
say "done"
