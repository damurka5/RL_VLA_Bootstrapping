#!/usr/bin/env bash
# Repeat the CDPR projection pilot on a fresh panel of reset groups.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
pilot=${AUDIT_PILOT:-runs/smolvla_audit_hVRwbw}
manifest="$pilot/swaps/manifest.json"
if [[ ! -r "$manifest" ]]; then
    echo "Pilot manifest not found: $manifest" >&2
    exit 1
fi

export RLVLA_EMBODIMENT=cdpr
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=1
output=$(mktemp -d "$PWD/runs/smolvla_confirmatory_XXXXXX")
echo "Audit output: $output"

device=(--device cuda:0)
if [[ -n "${AUDIT_DEVICES:-}" ]]; then
    device=(--devices "$AUDIT_DEVICES")
fi
common=(--manifest "$manifest" --round-index 200 --rounds 8 --worlds 128
        --group-size 2 --seed 1729 --microbatch 16 "${device[@]}")
audit=tools/audit/smolvla_skill_interventions.py
py() { conda run --no-capture-output -n cdpr-mjlab python "$@"; }

py "$audit" evaluate "${common[@]}" \
    --config configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml \
    --controller-workspace-from-checkpoint --cap 0.19 --output "$output/move_to"
py "$audit" evaluate "${common[@]}" \
    --config configs/examples/cdpr_smolvla_phase4_placement_loop.yaml \
    --cap 0.20 --metadata-override placement_caught_object_fraction=1.0 \
    --output "$output/placement"

printf '%s\n' "$output/move_to/paired.json" "$output/placement/paired.json"
