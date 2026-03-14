from __future__ import annotations

import argparse
import sys

from rl_vla_bootstrapping.assets import collect_asset_status, export_asset_manifest, stage_asset_bundles
from rl_vla_bootstrapping.core.config import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-vla-bootstrap-assets",
        description="Verify or stage external asset and benchmark bundles into repo-local paths.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--stage", action="store_true", help="Create links/copies for configured bundles.")
    parser.add_argument("--force", action="store_true", help="Replace existing targets.")
    parser.add_argument("--manifest", default=None, help="Optional JSON path for asset status manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)

    statuses = (
        stage_asset_bundles(config, mode=args.mode, force=args.force)
        if args.stage
        else collect_asset_status(config)
    )

    missing_required = []
    for status in statuses:
        prefix = "[ok]" if status.exists else "[missing]"
        print(
            f"{prefix} {status.name} ({status.kind}) -> {status.target_path}"
            + (f" <- {status.source_path}" if status.source_path else "")
        )
        if status.description:
            print(f"  {status.description}")
        if status.required and not status.exists:
            missing_required.append(status.name)

    if args.manifest:
        manifest = export_asset_manifest(config, output_path=config.resolve_path(args.manifest))
        print(f"Manifest: {manifest}")

    if missing_required:
        print(
            "Missing required asset bundles: " + ", ".join(missing_required),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
