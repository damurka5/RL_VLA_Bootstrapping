#!/usr/bin/env bash
# One bounded joint-RL pilot, with the same four-family evaluation before/after.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
CONFIG="$REPO_ROOT/configs/examples/cdpr_smolvla_release_recovery_pilot.yaml"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-$REPO_ROOT/runs/phase7_sparse_joint_20260904_212930/rl/step_2017690/smolvla_grpo_adapter.pt}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500000}"
DRY_RUN="${DRY_RUN:-0}"
PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
source "$SCRIPT_DIR/run_naming.sh"
RUN_NAME="$(cdpr_compose_run_name release_recovery_pilot)"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
cdpr_guard_run_dir "$RUN_DIR"
# A partial pilot must not overwrite its earlier baseline or final evaluation.
[[ ! -e "$RUN_DIR/pilot_manifest.json" ]] || {
  echo "Pilot already exists: $RUN_DIR. Use a new RUN_LABEL." >&2; exit 2;
}
[[ -f "$WARMSTART_CHECKPOINT" ]] || {
  echo "Missing warm-start checkpoint: $WARMSTART_CHECKPOINT" >&2; exit 2;
}
[[ "$MAX_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_TRAIN_STEPS must be a positive integer." >&2; exit 2;
}
printf 'pilot=%s\nsource=%s\nconfig=%s\nselected_action_budget=%s\n' \
  "$RUN_DIR" "$WARMSTART_CHECKPOINT" "$CONFIG" "$MAX_TRAIN_STEPS"
printf 'Sequence: CPU checks -> 3-round baseline -> joint RL -> 3-round final evaluation\n'
printf 'Caps: move_to=0.08 pick_up=0.06; containers uncaught, original geometry, 40 decisions\n'
[[ "$DRY_RUN" == "1" ]] && exit 0
cd "$REPO_ROOT"

# Fail before allocating GPU time if caps drift or the termination fix is absent.
"${PY[@]}" -c 'import torch'
"${PY[@]}" -m unittest discover -s tests -p test_fixed_approach_pilot.py
"${PY[@]}" -m unittest discover -s tests -p test_wrong_place_settled.py
mkdir -p "$RUN_DIR"
"${PY[@]}" - "$CONFIG" "$WARMSTART_CHECKPOINT" "$RUN_DIR" "$MAX_TRAIN_STEPS" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
config, source, run = map(Path, sys.argv[1:4])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
manifest = dict(config=str(config), config_sha256=sha(config),
                source=str(source), source_sha256=sha(source),
                max_selected_actions=int(sys.argv[4]), evaluation_seed_torch=0,
                evaluation_rounds=[0, 1, 2], evaluation_worlds_per_round=512,
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
(run / 'pilot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(run / 'pilot_config_snapshot.yaml').write_bytes(config.read_bytes())
(run / 'tracked_changes_at_launch.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD']))
PY

evaluate() {
  # No --start-distance-cap: one scalar would overwrite the four fixed caps.
  "${PY[@]}" tools/audit/sil_record.py --mode record \
    --checkpoint "$1" --config "$CONFIG" \
    --worlds 512 --group-size 8 --rounds 3 --round-index 0 \
    --devices cuda:0,cuda:1 --seed-torch 0 --output "$2"
}
evaluate "$WARMSTART_CHECKPOINT" "$RUN_DIR/baseline" 2>&1 | tee "$RUN_DIR/baseline.log"

REPO_ROOT="$REPO_ROOT" ENV_NAME="$ENV_NAME" CONFIG="$CONFIG" \
  WARMSTART_CHECKPOINT="$WARMSTART_CHECKPOINT" MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
  RUN_NAME="$RUN_NAME" DRY_RUN=0 \
  bash "$SCRIPT_DIR/train_cdpr_phase7_sparse_joint_remote.sh"

FINAL_CHECKPOINT="$("${PY[@]}" - "$RUN_DIR" "$CONFIG" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, config = map(Path, sys.argv[1:])
m = json.loads((run / 'pilot_manifest.json').read_text())
assert hashlib.sha256(config.read_bytes()).hexdigest() == m['config_sha256'], 'Config changed during pilot'
paths = list((run / 'rl').glob('step_*/smolvla_grpo_adapter.pt'))
assert paths, 'Training produced no step checkpoint'
chosen = max(paths, key=lambda p: int(p.parent.name.removeprefix('step_')))
print(chosen)
PY
)"
evaluate "$FINAL_CHECKPOINT" "$RUN_DIR/final_eval" 2>&1 | tee "$RUN_DIR/final_eval.log"

"${PY[@]}" - "$RUN_DIR" "$FINAL_CHECKPOINT" <<'PY'
import hashlib, json, sys
from pathlib import Path
from tools.audit.sil_record import _Recording, _instruction_name
run, checkpoint = map(Path, sys.argv[1:])
report = {'final_checkpoint': str(checkpoint), 'arms': {}}
h = hashlib.sha256()
with checkpoint.open('rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(chunk)
report['final_checkpoint_sha256'] = h.hexdigest()
for arm in ('baseline', 'final_eval'):
    counts = {}
    paths = sorted((run / arm).glob('record_*.npz'))
    assert len(paths) == 3, (arm, len(paths))
    for path in paths:
        r = _Recording.from_npz(path)
        for w, i in enumerate(r.instruction_ids):
            name = _instruction_name(i)
            if name.startswith('put_into_'):
                assert not r.starts_grasped[w] and r.horizons[w] == 40
            c = counts.setdefault(name, {'successes': 0, 'episodes': 0})
            c['episodes'] += 1
            c['successes'] += int(r.episode_success[w])
    for c in counts.values():
        c['rate'] = c['successes'] / c['episodes']
    report['arms'][arm] = counts
for name, before in report['arms']['baseline'].items():
    after = report['arms']['final_eval'][name]
    print(f"[pilot] {name}: {before['successes']}/{before['episodes']}={before['rate']:.4f} -> "
          f"{after['successes']}/{after['episodes']}={after['rate']:.4f}; delta={after['rate']-before['rate']:+.4f}")
(run / 'pilot_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(f"[pilot] wrote {run / 'pilot_comparison.json'}")
print('[pilot] All checkpoints retained. No automatic promotion or further training.')
PY
