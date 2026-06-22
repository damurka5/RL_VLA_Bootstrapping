#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
    _reverse_shell_counts,
    _validation_env_vars,
)
from rl_vla_bootstrapping.core.config import load_project_config


DEFAULT_CONFIG = ROOT / "configs" / "examples" / "cdpr_openvla_grpo_complex_tasks.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_reset_smoke"


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


def _validator_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        success_distance=0.05,
        directional_displacement_threshold=0.05,
        move_to_object_success_distance=0.10,
        multi_object_scenes=bool(args.multi_object_scenes),
        min_scene_objects=int(args.min_scene_objects),
        max_scene_objects=int(args.max_scene_objects),
        reuse_existing_wrapper_variants=bool(args.reuse_existing_wrapper_variants),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight randomized CDPR resets without loading OpenVLA."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-case", type=int, default=10)
    parser.add_argument("--max-reset-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--hold-steps", type=int, default=None)
    parser.add_argument("--wrapper-dir", type=Path, default=None)
    parser.add_argument("--instruction-types", nargs="+", default=None)
    parser.add_argument(
        "--multi-object-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-scene-objects", type=int, default=3)
    parser.add_argument("--max-scene-objects", type=int, default=4)
    parser.add_argument("--include-reverse-shells", action="store_true")
    parser.add_argument(
        "--reuse-existing-wrapper-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    config = load_project_config(args.config.resolve())
    metadata = dict(getattr(config.task, "metadata", {}) or {})
    dense_instructions = tuple(str(item) for item in metadata.get("dense_stage_instruction_types", ()))
    instruction_types = (
        tuple(str(item) for item in args.instruction_types)
        if args.instruction_types
        else dense_instructions or tuple(str(item) for item in config.task.instruction_types)
    )
    shell_counts = _reverse_shell_counts(tuple(str(item) for item in config.task.instruction_types))
    cases: list[tuple[str, int | None]] = [(name, None) for name in instruction_types]
    if args.include_reverse_shells:
        for instruction_type, shell_count in sorted(shell_counts.items()):
            cases.extend((instruction_type, shell_id) for shell_id in range(int(shell_count)))

    run_dir = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    validator_args = _validator_args(args)
    rows: list[dict[str, object]] = []

    for case_index, (instruction_type, shell_id) in enumerate(cases):
        task_metadata = _instruction_validation_task_metadata(
            config,
            validator_args,
            instruction_type=instruction_type,
        )
        if not args.multi_object_scenes:
            task_metadata["min_scene_objects"] = max(1, int(args.min_scene_objects))
            task_metadata["max_scene_objects"] = max(
                int(task_metadata["min_scene_objects"]),
                int(args.max_scene_objects),
            )
        env_vars = _validation_env_vars(
            config,
            validator_args,
            instruction_type=instruction_type,
            task_metadata_override=task_metadata,
        )
        # Reset-only smoke tests do not render, so avoid requiring the training
        # server's EGL setup on laptops or CPU-only validation hosts.
        env_vars.pop("MUJOCO_GL", None)
        env_vars.pop("PYOPENGL_PLATFORM", None)
        with _temporary_environment(env_vars):
            env = _build_validation_env(
                config=config,
                instruction_type=instruction_type,
                capture_frames=False,
                max_steps=int(args.max_steps),
                hold_steps=args.hold_steps,
                seed=int(args.seed),
                args=validator_args,
                wrapper_dir=args.wrapper_dir.resolve() if args.wrapper_dir else None,
            )
            try:
                for episode_index in range(max(1, int(args.episodes_per_case))):
                    reset_options: dict[str, object] = {"instruction_type": instruction_type}
                    if shell_id is not None:
                        reset_options.update(
                            {
                                "curriculum_mode": "reverse_frontier",
                                "curriculum_shell": int(shell_id),
                            }
                        )
                    seed = int(args.seed) + case_index * 100_000 + episode_index
                    try:
                        _obs, info, attempts = _reset_validation_env_with_retries(
                            env=env,
                            seed=seed,
                            reset_options=reset_options,
                            max_attempts=int(args.max_reset_attempts),
                            quiet=False,
                        )
                        rows.append(
                            {
                                "instruction_type": instruction_type,
                                "curriculum_shell": shell_id,
                                "episode": episode_index,
                                "seed": seed,
                                "success": True,
                                "attempts": attempts,
                                "reason": "",
                                "retry_reasons": " | ".join(
                                    str(item) for item in info.get("reset_retry_errors", ())
                                ),
                                "scene": info.get("scene", ""),
                                "scene_objects": "|".join(
                                    str(item) for item in info.get("scene_objects", ())
                                ),
                            }
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "instruction_type": instruction_type,
                                "curriculum_shell": shell_id,
                                "episode": episode_index,
                                "seed": seed,
                                "success": False,
                                "attempts": int(args.max_reset_attempts),
                                "reason": str(exc),
                                "retry_reasons": "",
                                "scene": "",
                                "scene_objects": "",
                            }
                        )
            finally:
                env.close()

    failures = [row for row in rows if not bool(row["success"])]
    retries = sum(max(0, int(row["attempts"]) - 1) for row in rows)
    summary = {
        "config": str(args.config.resolve()),
        "cases": len(cases),
        "episodes": len(rows),
        "failures": len(failures),
        "reset_retries": retries,
        "passed": not failures,
        "rows": rows,
    }
    json_path = run_dir / "reset_smoke_report.json"
    csv_path = run_dir / "reset_smoke_results.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(json_path)
    print(
        f"episodes={len(rows)} failures={len(failures)} "
        f"reset_retries={retries} passed={not failures}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
