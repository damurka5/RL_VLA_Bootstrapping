#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/examples/cdpr_openvla_bootstrap.yaml}"
shift || true

python3 -m rl_vla_bootstrapping.cli.doctor --config "$CONFIG_PATH" "$@"
