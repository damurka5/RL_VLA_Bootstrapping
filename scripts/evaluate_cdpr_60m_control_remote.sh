#!/usr/bin/env bash
# Matched development evaluation of the 60M endpoint against promoted 40M.
# The existing comparison launcher resolves a bare step only when it is unique.
# Optional: PEAK_CHECKPOINT=step_59268127 adds the panel-selected alternative.
# Reuse OUT_ROOT to resume. No training or final_test evaluation is launched.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-runs/latent_control_to40m_20261012_230538/rl/step_40056193}"
ENDPOINT_CHECKPOINT="${ENDPOINT_CHECKPOINT:-step_60025885}"
PEAK_CHECKPOINT="${PEAK_CHECKPOINT:-}"
OUT_ROOT="${OUT_ROOT:-runs/latent_control_60m_comparison_$(date +%Y%m%d_%H%M%S)}"

baseline_adapter="$BASELINE_CHECKPOINT"
[[ -d "$baseline_adapter" ]] && baseline_adapter="$baseline_adapter/smolvla_grpo_adapter.pt"
[[ -f "$baseline_adapter" ]] || { echo "40M baseline not found: $baseline_adapter" >&2; exit 2; }
expected=63aff785cc0202ecea2f9c1e713f58648f14a0695a593975fa3022b4f2505e91
actual="$(sha256sum "$baseline_adapter")"
[[ "${actual%% *}" == "$expected" ]] || { echo "40M baseline SHA-256 does not match the promoted adapter." >&2; exit 2; }

checkpoints="control40m=$BASELINE_CHECKPOINT final60m=$ENDPOINT_CHECKPOINT"
[[ -n "$PEAK_CHECKPOINT" ]] && checkpoints+=" peak59m=$PEAK_CHECKPOINT"

# Clear diagnostic interventions and select the CDPR object geometry explicitly.
# The evaluator records adapter hashes, policy provenance and per-scene outcomes.
env -u SCENE_LIST -u SCENE_CLASS -u CONTROLLER_Z_MAX -u STOCHASTIC_SEED \
  RLVLA_EMBODIMENT=cdpr CHECKPOINTS="$checkpoints" OUT_ROOT="$OUT_ROOT" \
  REPEATS=4 GPUS="${GPUS:-0 1}" WORLDS=64 ROUNDS=8 \
  CONFIG=configs/examples/cdpr_smolvla_three_stage_put_into.yaml \
  SCENES=runs/three_stage/scenes_8192.json SPLIT=student_validation \
  SCENE_MANIFEST=runs/three_stage/scenes_8192.json RETENTION_MARGIN=0.03 \
  DECISIONS=128 SETTLE_DECISIONS=0 PRIOR_NOISE_SCALE=1 \
  EXCLUDE_VALIDATION_PANEL=1 PANEL_VALIDATION_SEED=2000000 \
  PANEL_GROUPS_PER_RANK=64 PANEL_RANKS=2 \
  bash "$SCRIPT_DIR/compare_cdpr_three_stage_repeats_remote.sh"

if [[ -n "$PEAK_CHECKPOINT" ]]; then
  conda run --no-capture-output -n "${ENV_NAME:-cdpr-mjlab}" python3 tools/audit/compare_put_into_repeats.py \
    --checkpoint "final60m=$OUT_ROOT/final60m/rep1,$OUT_ROOT/final60m/rep2,$OUT_ROOT/final60m/rep3,$OUT_ROOT/final60m/rep4" \
    --checkpoint "peak59m=$OUT_ROOT/peak59m/rep1,$OUT_ROOT/peak59m/rep2,$OUT_ROOT/peak59m/rep3,$OUT_ROOT/peak59m/rep4" \
    --retention-margin 0.03 --scene-manifest runs/three_stage/scenes_8192.json \
    --output "$OUT_ROOT/peak_vs_final.json"
fi
echo "Return $OUT_ROOT/comparison.json and, if requested, $OUT_ROOT/peak_vs_final.json"
