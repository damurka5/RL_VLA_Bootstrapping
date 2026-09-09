#!/usr/bin/env bash
# Two independent handoff audits, no optimizer updates or checkpoint promotion.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
PILOT_RUN="${PILOT_RUN:-runs/release_recovery_pilot_20260909_130003}"
DEMO_MANIFEST="${DEMO_MANIFEST:-runs/grpo_demo_prototype_20260909_170619/manifest.json}"
SOURCE_ROUND="${SOURCE_ROUND:-0}"
MAX_BOUNDARIES="${MAX_BOUNDARIES:-2}"
MAX_GROUPS="${MAX_GROUPS:-8}"
source "$SCRIPT_DIR/run_naming.sh"
PROBE_NAME="$(cdpr_compose_run_name demo_handoff_probe)"
PROBE_DIR="$REPO_ROOT/runs/$PROBE_NAME"
[[ ! -e "$PROBE_DIR" ]] || { echo "Output exists: $PROBE_DIR" >&2; exit 2; }
PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
COMMON=(tools/audit/probe_cdpr_demonstration_handoff.py
  --manifest "$DEMO_MANIFEST" --pilot-run "$PILOT_RUN"
  --source-round "$SOURCE_ROUND" --max-boundaries "$MAX_BOUNDARIES" --max-groups "$MAX_GROUPS")
# CPU-only provenance and landmark planning for both arms, before GPU work.
for task in pick_up placement; do
  "${PY[@]}" "${COMMON[@]}" --task "$task" --output "$PROBE_DIR/$task" --dry-run
done
printf 'Output: %s\nTwo handoff audits; optimizer updates=0\n' "$PROBE_DIR"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
"${PY[@]}" -m unittest tests.test_cdpr_demonstration_handoff
mkdir -p "$PROBE_DIR"
git rev-parse HEAD > "$PROBE_DIR/git_head.txt"
git diff HEAD > "$PROBE_DIR/tracked_changes.patch"
"${PY[@]}" "${COMMON[@]}" --task pick_up --device cuda:0 --output "$PROBE_DIR/pick_up" \
  > "$PROBE_DIR/pick_up.log" 2>&1 &
pickup_pid=$!
"${PY[@]}" "${COMMON[@]}" --task placement --device cuda:1 --output "$PROBE_DIR/placement" \
  > "$PROBE_DIR/placement.log" 2>&1 &
placement_pid=$!
printf 'Live logs: %s/pick_up.log and %s/placement.log\n' "$PROBE_DIR" "$PROBE_DIR"
pickup_status=0
placement_status=0
wait "$pickup_pid" || pickup_status=$?
wait "$placement_pid" || placement_status=$?
"${PY[@]}" - "$PROBE_DIR" "$pickup_status" "$placement_status" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
summary = {}
for task, code in zip(('pick_up', 'placement'), map(int, sys.argv[2:])):
    path = root / task / 'report.json'
    report = json.loads(path.read_text()) if path.exists() else None
    summary[task] = {'exit_code': code, 'report': str(path), 'batches': []}
    for result in (report or {}).get('results', []):
        groups = result.get('groups', [])
        clean = [g for g in groups if not g['contains_divergence']]
        summary[task]['batches'].append({
            'prefix_steps': result['prefix_steps'], 'status': result['status'],
            'planned_groups': len(result['episodes']), 'suffix_groups': len(groups),
            'clean_suffix_groups': len(clean),
            'clean_groups_with_reward_variation': sum(g['reward_std'] > 1e-6 for g in clean),
            'clean_suffix_successes': sum(g['successes'] for g in clean),
            'clean_suffix_candidates': sum(g['candidates'] for g in clean),
            'rejections': [e for e in result['episodes'] if e['rejected']]})
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print(f'[handoff] {root}/summary.json; no training or promotion')
PY
[[ "$pickup_status" == 0 && "$placement_status" == 0 ]]
