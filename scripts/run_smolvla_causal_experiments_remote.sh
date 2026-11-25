#!/usr/bin/env bash
# Closed-loop depth interventions and a fixed-budget SIL scene-diversity test.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export RLVLA_EMBODIMENT=cdpr MUJOCO_GL=${MUJOCO_GL:-egl} PYTHONUNBUFFERED=1
py() { conda run --no-capture-output -n cdpr-mjlab python "$@"; }

A=${AUDIT_BEFORE_FORGETTING:-runs/phase4_move_to_iter0_resume_20260818_080928/rl/step_11009573/smolvla_grpo_adapter.pt}
B=${AUDIT_INITIALIZER:-runs/phase4_placement_iter1_20260823_165011/rl/step_1504301/smolvla_grpo_adapter.pt}
C=${AUDIT_AFTER_SIL:-runs/phase4_bank/sft_retention_e60/sil_sft_adapter.pt}
audit=tools/audit/smolvla_skill_interventions.py
design=tools/audit/smolvla_sil_diversity.py

evaluate() {
    local manifest=$1 destination=$2 round=$3
    shift 3
    local device=(--device cuda:0)
    if [[ -n "${AUDIT_DEVICES:-}" ]]; then
        device=(--devices "$AUDIT_DEVICES")
    fi
    local common=(--manifest "$manifest" --round-index "$round" --rounds "${AUDIT_ROUNDS:-8}"
                  --worlds "${AUDIT_WORLDS:-128}" --group-size 2 --microbatch 16 --seed 1729 "${device[@]}")
    py "$audit" evaluate "${common[@]}" --config configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml \
        --controller-workspace-from-checkpoint --cap 0.19 --output "$destination/move_to" "$@"
    py "$audit" evaluate "${common[@]}" --config configs/examples/cdpr_smolvla_phase4_placement_loop.yaml \
        --cap 0.20 --metadata-override placement_caught_object_fraction=1.0 \
        --output "$destination/placement" "$@"
}

case "${1:-}" in
    depth-forgetting|depth-recovery)
        donor=$A
        if [[ "$1" == depth-recovery ]]; then
            donor=$C
        fi
        test -r "$donor" && test -r "$B" || { echo "Depth inputs missing: $B or $donor" >&2; exit 1; }
        output=$(mktemp -d "$PWD/runs/smolvla_${1}_XXXXXX")
        echo "Audit output: $output"
        py "$audit" make --recipient "$B" --donor "$donor" --experiment depth --depth 1 2 4 8 \
            --output "$output/swaps"
        evaluate "$output/swaps/manifest.json" "$output" "${AUDIT_ROUND_INDEX:-300}"
        ;;
    diversity-fit)
        : "${AUDIT_RECORDINGS:?Set AUDIT_RECORDINGS to the confirmatory rollout folder}"
        : "${AUDIT_SOURCE_MANIFEST:?Set AUDIT_SOURCE_MANIFEST to the original pilot swaps/manifest.json}"
        output=$(mktemp -d "$PWD/runs/smolvla_diversity_XXXXXX")
        echo "Audit output: $output"
        py "$design" build --recordings "$AUDIT_RECORDINGS" --source-manifest "$AUDIT_SOURCE_MANIFEST" \
            --initializer "$B" --output "$output/data"
        for arm in restricted diverse; do
            py tools/audit/sil_sft.py --dataset "$output/data/$arm.npz" --checkpoint "$B" \
                --split-by provided --save-final-residual --epochs 20 --lr 1e-4 --batch-size 128 \
                --seed "${AUDIT_SFT_SEED:-1729}" --device "${AUDIT_SFT_DEVICE:-cpu}" --output "$output/$arm"
        done
        py "$design" selection --training "$output" --initializer "$B" --output "$output/selection.csv"
        py "$audit" make-series --series sil_diversity --selection "$output/selection.csv" \
            --reference B --output "$output/eval_inputs"
        echo "SIL evaluation manifest: $output/eval_inputs/manifest.json"
        ;;
    diversity-eval)
        manifest="${2:?Pass the path to eval_inputs/manifest.json from diversity-fit}"
        output=$(mktemp -d "$PWD/runs/smolvla_diversity_eval_XXXXXX")
        echo "Audit output: $output"
        evaluate "$manifest" "$output" "${AUDIT_ROUND_INDEX:-400}" --holdout-quadrant NE \
            --holdout-pair robocasa_apple:plate --holdout-pair robocasa_orange:bowl
        ;;
    *)
        echo "Usage: $0 {depth-forgetting|depth-recovery|diversity-fit|diversity-eval MANIFEST}" >&2
        exit 2
        ;;
esac
printf 'Results: %s\n' "$output"
