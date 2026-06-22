#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import types
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CDPR_ROOT = ROOT / "robots" / "cdpr"
if str(CDPR_ROOT) not in sys.path:
    sys.path.insert(0, str(CDPR_ROOT))

from rl_vla_bootstrapping.cli.validate_cdpr_policy import (
    _build_validation_env,
    _instruction_validation_task_metadata,
    _reset_validation_env_with_retries,
    _validation_env_vars,
)
from rl_vla_bootstrapping.core.config import load_project_config


DEFAULT_CONFIG = ROOT / "configs" / "examples" / "cdpr_openvla_grpo_complex_tasks.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_prebuilt_reset_smoke"


@contextmanager
def _temporary_environment(overrides: dict[str, str]):
    import os

    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update({key: str(value) for key, value in overrides.items()})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise randomized CDPR resets while deliberately reusing wrappers "
            "built at one fixed EE pose, matching the training prebuild path."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instruction-type", default="move_to_object")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--scene-pool-size", type=int, default=4)
    parser.add_argument("--max-reset-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--pose-tolerance", type=float, default=0.015)
    args = parser.parse_args()

    config = load_project_config(args.config.resolve())
    validator_args = SimpleNamespace(
        success_distance=0.05,
        directional_displacement_threshold=0.05,
        move_to_object_success_distance=0.10,
        multi_object_scenes=True,
        min_scene_objects=3,
        max_scene_objects=4,
        reuse_existing_wrapper_variants=False,
    )
    task_metadata = _instruction_validation_task_metadata(
        config,
        validator_args,
        instruction_type=str(args.instruction_type),
    )
    env_vars = _validation_env_vars(
        config,
        validator_args,
        instruction_type=str(args.instruction_type),
        task_metadata_override=task_metadata,
    )
    env_vars.pop("MUJOCO_GL", None)
    env_vars.pop("PYOPENGL_PLATFORM", None)

    rows: list[dict[str, object]] = []
    with _temporary_environment(env_vars):
        env = _build_validation_env(
            config=config,
            instruction_type=str(args.instruction_type),
            capture_frames=False,
            max_steps=120,
            hold_steps=None,
            seed=int(args.seed),
            args=validator_args,
            wrapper_dir=None,
        )
        try:
            scenes = list(getattr(env, "scenes", ()) or ())
            if not scenes:
                raise RuntimeError("No CDPR scenes are available for prebuilt reset smoke.")
            scenes = scenes[: max(1, min(int(args.scene_pool_size), len(scenes)))]
            env.scenes = list(scenes)

            original_build_wrapper = env._build_wrapper
            center_start = np.asarray(env._default_ee_start(), dtype=np.float32).reshape(3)
            wrapper_by_scene: dict[tuple[str, tuple[str, ...]], Path] = {}
            for scene in scenes:
                key = (
                    str(getattr(scene, "name", "")),
                    tuple(str(item) for item in (getattr(scene, "objects", ()) or ())),
                )
                wrapper_by_scene[key] = Path(
                    original_build_wrapper(scene, ee_start=center_start)
                ).resolve()

            def _build_prebuilt(this, scene, ee_start=None):
                del ee_start
                key = (
                    str(getattr(scene, "name", "")),
                    tuple(str(item) for item in (getattr(scene, "objects", ()) or ())),
                )
                this._last_wrapper_reused_from_cache = True
                return wrapper_by_scene[key]

            env._build_wrapper_original = original_build_wrapper
            env._build_wrapper = types.MethodType(_build_prebuilt, env)

            for episode in range(max(1, int(args.episodes))):
                try:
                    _obs, info, attempts = _reset_validation_env_with_retries(
                        env=env,
                        seed=int(args.seed) + episode,
                        reset_options={"instruction_type": str(args.instruction_type)},
                        max_attempts=max(1, int(args.max_reset_attempts)),
                        quiet=False,
                    )
                    requested = np.asarray(info.get("ee_start"), dtype=np.float64).reshape(3)
                    actual = np.asarray(info.get("ee_position"), dtype=np.float64).reshape(3)
                    pose_error = float(np.linalg.norm(actual - requested))
                    rows.append(
                        {
                            "episode": episode,
                            "success": pose_error <= float(args.pose_tolerance),
                            "attempts": int(attempts),
                            "pose_error": pose_error,
                            "requested_ee_start": requested.tolist(),
                            "actual_ee": actual.tolist(),
                            "wrapper_xml": str(info.get("wrapper_xml", "")),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "episode": episode,
                            "success": False,
                            "attempts": int(args.max_reset_attempts),
                            "pose_error": None,
                            "reason": str(exc),
                        }
                    )
        finally:
            env.close()

    failures = [row for row in rows if not bool(row["success"])]
    retries = sum(max(0, int(row["attempts"]) - 1) for row in rows)
    finite_errors = [
        float(row["pose_error"])
        for row in rows
        if row.get("pose_error") is not None
    ]
    summary = {
        "config": str(args.config.resolve()),
        "instruction_type": str(args.instruction_type),
        "episodes": len(rows),
        "failures": len(failures),
        "reset_retries": retries,
        "max_pose_error": max(finite_errors) if finite_errors else None,
        "pose_tolerance": float(args.pose_tolerance),
        "passed": not failures,
        "rows": rows,
    }
    run_dir = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "prebuilt_randomized_reset_report.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(report_path)
    print(
        f"episodes={len(rows)} failures={len(failures)} reset_retries={retries} "
        f"max_pose_error={summary['max_pose_error']} passed={summary['passed']}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
