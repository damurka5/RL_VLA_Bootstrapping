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
BOUNDARY_BACKOFF="${BOUNDARY_BACKOFF:-0}"
PICKUP_BOUNDARY_BACKOFF="${PICKUP_BOUNDARY_BACKOFF:-$BOUNDARY_BACKOFF}"
PLACEMENT_BOUNDARY_BACKOFF="${PLACEMENT_BOUNDARY_BACKOFF:-$BOUNDARY_BACKOFF}"
PICKUP_SOURCE_ROUND="${PICKUP_SOURCE_ROUND:-$SOURCE_ROUND}"
PLACEMENT_SOURCE_ROUND="${PLACEMENT_SOURCE_ROUND:-$SOURCE_ROUND}"
source "$SCRIPT_DIR/run_naming.sh"
PROBE_NAME="$(cdpr_compose_run_name demo_handoff_probe)"
PROBE_DIR="$REPO_ROOT/runs/$PROBE_NAME"
[[ ! -e "$PROBE_DIR" ]] || { echo "Output exists: $PROBE_DIR" >&2; exit 2; }
PY=(conda run --no-capture-output -n "$ENV_NAME" python3)
COMMON=(tools/audit/probe_cdpr_demonstration_handoff.py
  --manifest "$DEMO_MANIFEST" --pilot-run "$PILOT_RUN"
  --max-boundaries "$MAX_BOUNDARIES" --max-groups "$MAX_GROUPS")
PICKUP_ARGS=(--source-round "$PICKUP_SOURCE_ROUND" --boundary-backoff "$PICKUP_BOUNDARY_BACKOFF")
PLACEMENT_ARGS=(--source-round "$PLACEMENT_SOURCE_ROUND" --boundary-backoff "$PLACEMENT_BOUNDARY_BACKOFF")
[[ -z "${PLACEMENT_SOURCE_INSTRUCTION:-}" ]] || PLACEMENT_ARGS+=(--source-instruction "$PLACEMENT_SOURCE_INSTRUCTION")
# CPU-only provenance and landmark planning for both arms, before GPU work.
for task in pick_up placement; do
  if [[ "$task" == pick_up ]]; then ARM_ARGS=("${PICKUP_ARGS[@]}"); else ARM_ARGS=("${PLACEMENT_ARGS[@]}"); fi
  "${PY[@]}" "${COMMON[@]}" "${ARM_ARGS[@]}" --task "$task" --output "$PROBE_DIR/$task" --dry-run
done
printf 'Output: %s\nTwo handoff audits; optimizer updates=0\n' "$PROBE_DIR"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
# tests/ is not a Python package. Discover by directory so an installed
# package named "tests" cannot shadow this repository's preflight. Discovery
# reports "OK" on a pattern that matches nothing, so a checkout without the
# preflight would reach GPU work unverified; require the file first.
PREFLIGHT=tests/test_cdpr_demonstration_handoff.py
[[ -f "$PREFLIGHT" ]] || { echo "Missing preflight: $PREFLIGHT (git pull --ff-only)" >&2; exit 2; }
"${PY[@]}" -m unittest discover -s tests -p "$(basename "$PREFLIGHT")"
mkdir -p "$PROBE_DIR"
git rev-parse HEAD > "$PROBE_DIR/git_head.txt"
git diff HEAD > "$PROBE_DIR/tracked_changes.patch"
"${PY[@]}" "${COMMON[@]}" "${PICKUP_ARGS[@]}" --task pick_up --device cuda:0 --output "$PROBE_DIR/pick_up" \
  > "$PROBE_DIR/pick_up.log" 2>&1 &
pickup_pid=$!
"${PY[@]}" "${COMMON[@]}" "${PLACEMENT_ARGS[@]}" --task placement --device cuda:1 --output "$PROBE_DIR/placement" \
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
            'clean_target_instructions': sorted({g.get('target_instruction', 'unknown') for g in clean}),
            'suffix_loss_rows': result.get('suffix_loss_rows'),
            'suffix_vla_record_rows': result.get('suffix_vla_record_rows'),
            'suffix_vla_nonzero_advantage_rows': result.get('suffix_vla_nonzero_advantage_rows'),
            'terminal_groups': result.get('destination_terminal_groups', []),
            'earliest_dropped_live_datum_lift_step': result.get('earliest_dropped_live_datum_lift_step'),
            'rejections': [e for e in result['episodes'] if e['rejected']]})
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print(f'[handoff] {root}/summary.json; no training or promotion')
PY
[[ "$pickup_status" == 0 && "$placement_status" == 0 ]]
