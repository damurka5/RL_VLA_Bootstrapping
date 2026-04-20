#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_openvla_grpo_step_390_to_490_movetoobj.yaml}"
ENV_NAME="${ENV_NAME:-openvla-oft}"
STAGE="${STAGE:-rl}"
RUN_NAME="${RUN_NAME:-step_390_to_490_movetoobj}"

cd "$REPO_ROOT"

CMD=(conda run -n "$ENV_NAME" python3 -m rl_vla_bootstrapping.cli.train --config "$CONFIG_PATH" --stage "$STAGE" --execute --run-name "$RUN_NAME")
CMD+=("$@")

"${CMD[@]}"
