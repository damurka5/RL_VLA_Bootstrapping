from __future__ import annotations

import argparse

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.simulation.preview import render_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-vla-bootstrap-preview",
        description="Render a preview of the configured embodiment and cameras.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON/TOML project config.")
    parser.add_argument("--run-dir", required=True, help="Output directory for preview files.")
    parser.add_argument("--scene", default=None, help="Optional preview scene override.")
    parser.add_argument(
        "--object",
        dest="objects",
        action="append",
        default=None,
        help="Optional preview object override. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    metadata = render_preview(
        config,
        output_dir=args.run_dir,
        scene_name=args.scene,
        object_names=args.objects,
    )
    print(f"Preview XML: {metadata['xml_path']}")
    for slot, image_path in metadata.get("images", {}).items():
        print(f"{slot}: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
