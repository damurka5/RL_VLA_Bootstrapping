#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_openvla_grpo_resume_380_to_480_move_to_object.yaml}"
ENV_NAME="${ENV_NAME:-openvla-oft}"
STAGE="${STAGE:-rl}"
RUN_NAME="${RUN_NAME:-}"

cd "$REPO_ROOT"

CMD=(conda run -n "$ENV_NAME" python3 -m rl_vla_bootstrapping.cli.train --config "$CONFIG_PATH" --stage "$STAGE" --execute)
if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run-name "$RUN_NAME")
fi
CMD+=("$@")

"${CMD[@]}"
