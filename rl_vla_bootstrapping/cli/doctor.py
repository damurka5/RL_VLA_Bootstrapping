from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

from rl_vla_bootstrapping.assets import collect_asset_status
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.embodiments.mujoco import MujocoEmbodiment
from rl_vla_bootstrapping.simulation.preview import render_preview


DEFAULT_IMPORTS = ("cv2", "mujoco", "numpy", "torch", "yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-vla-bootstrap-doctor",
        description="Validate runtime imports, staged assets, robot XML, and optional headless preview.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-preview", action="store_true")
    parser.add_argument("--preview-dir", default="runs/doctor_preview")
    parser.add_argument("--import", dest="imports", action="append", default=[])
    return parser.parse_args()


def _check_imports(names: list[str]) -> list[str]:
    errors: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
            print(f"[ok] import {name}")
        except Exception as exc:
            print(f"[missing] import {name}: {exc}")
            errors.append(f"Missing import `{name}`: {exc}")
    return errors


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    errors: list[str] = []

    imports = list(DEFAULT_IMPORTS)
    imports.extend(args.imports)
    errors.extend(_check_imports(imports))

    for status in collect_asset_status(config):
        state = "ok" if status.exists else "missing"
        print(f"[{state}] asset {status.name} -> {status.target_path}")
        if status.required and not status.exists:
            errors.append(f"Required asset bundle `{status.name}` is missing at {status.target_path}")

    embodiment = MujocoEmbodiment(config=config, spec=config.embodiment)
    xml_path = embodiment.resolved_xml_path()
    print(f"[info] xml {xml_path}")
    if not xml_path.exists():
        errors.append(f"Robot XML is missing: {xml_path}")

    try:
        controller_cls = embodiment.load_controller_class()
        print(f"[ok] controller import {controller_cls.__name__}")
    except Exception as exc:
        traceback.print_exc()
        errors.append(f"Could not import controller: {exc}")

    try:
        import mujoco as mj

        mj.MjModel.from_xml_path(str(xml_path))
        print("[ok] mujoco XML parse")
    except Exception as exc:
        traceback.print_exc()
        errors.append(f"MuJoCo could not parse the robot XML: {exc}")

    if args.check_preview and not errors:
        try:
            preview_dir = config.resolve_path(args.preview_dir) or Path(args.preview_dir)
            metadata = render_preview(config, preview_dir)
            print(f"[ok] preview images: {metadata.get('images', {})}")
        except Exception as exc:
            traceback.print_exc()
            errors.append(f"Preview failed: {exc}")

    if errors:
        print("\nDoctor found problems:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nDoctor checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
