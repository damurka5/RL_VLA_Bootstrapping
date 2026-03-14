#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/examples/cdpr_openvla_bootstrap.yaml}"
ENV_FILE="${2:-environments/openvla-oft-remote.yaml}"
ENV_NAME="${3:-openvla-oft}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda run -n "$ENV_NAME" python3 -m rl_vla_bootstrapping.cli.assets --config "$CONFIG_PATH" --stage
conda run -n "$ENV_NAME" python3 -m rl_vla_bootstrapping.cli.doctor --config "$CONFIG_PATH"
