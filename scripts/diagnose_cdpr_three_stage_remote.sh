#!/usr/bin/env bash
# Re-screen the existing scenes after the floor/teacher-routing fixes. No SFT.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
SCENES="${SCENES:-runs/three_stage/scenes.json}"
YAW="${YAW:-runs/three_stage/yaw_calibration.json}"
OUTPUT="${OUTPUT:-runs/three_stage/selection_fixed_$(date +%Y%m%d_%H%M%S)}"
WORLDS="${WORLDS:-64}"
run() { conda run --no-capture-output -n "$ENV_NAME" python3 "$@"; }
run -m unittest discover -s tests -p test_cdpr_stage_selection_regressions.py
run -m unittest discover -s tests -p test_cdpr_staged_put_into.py
export RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl
RECENT=runs/release_recovery_continue_3m_20260908_102004/rl
printf 'Selection output: %s\n' "$OUTPUT"
run tools/audit/select_cdpr_stage_teachers.py \
  --config configs/examples/cdpr_smolvla_three_stage_put_into.yaml \
  --scene-manifest "$SCENES" --yaw-calibration "$YAW" \
  --candidate "move_to=$RECENT/step_3416645/smolvla_grpo_adapter.pt" \
  --candidate move_to=runs/phase4_move_to_iter0_resume_20260818_080928/rl/step_11009573/smolvla_grpo_adapter.pt \
  --candidate "pick_up=$RECENT/step_3540208/smolvla_grpo_adapter.pt" \
  --candidate "placement=$RECENT/step_2117145/smolvla_grpo_adapter.pt" \
  --candidate placement=runs/phase5_placement_iter3_20260828_224948/rl/step_2754052/smolvla_grpo_adapter.pt \
  --worlds "$WORLDS" --dump-rounds --output "$OUTPUT"
