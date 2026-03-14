#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/examples/cdpr_openvla_bootstrap.yaml}"
shift || true

RUN_NAME="preview_$(date +%Y%m%d_%H%M%S)"
python3 -m rl_vla_bootstrapping.cli.train --config "$CONFIG_PATH" --stage preview --run-name "$RUN_NAME" --execute "$@"
