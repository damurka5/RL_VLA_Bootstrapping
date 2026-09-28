#!/usr/bin/env python3
"""Run lightweight simulator comparisons without OpenVLA.

This script benchmarks scripted manipulation and contact predicates. It never
loads OpenVLA checkpoints and never starts policy training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from tools.sim_compare.common import (  # noqa: E402
    DEFAULT_CAMERA_COUNT,
    DEFAULT_CONTACT_OBJECTS,
    DEFAULT_HEIGHT,
    DEFAULT_RENDER_BACKEND,
    DEFAULT_RENDER_STEPS,
    DEFAULT_RESETS,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_TASK_OBJECTS,
    DEFAULT_WIDTH,
    OBJECT_SPECS,
    OUT_DIR,
    SUPPORTED_RENDER_BACKENDS,
    discover_optional_backend_summaries,
    skip_backend_summary,
    write_all_outputs,
)
from tools.sim_compare.mujoco_backend import MujocoCDPRBackend  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resets", "--episodes", dest="resets", type=int, default=DEFAULT_RESETS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--render-steps", type=int, default=DEFAULT_RENDER_STEPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--camera-count", type=int, default=DEFAULT_CAMERA_COUNT)
    parser.add_argument("--render-backend", choices=SUPPORTED_RENDER_BACKENDS, default=DEFAULT_RENDER_BACKEND)
    parser.add_argument("--render", dest="render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--backend", choices=("all", "mujoco_raw_cdpr"), default="all")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--task-objects", nargs="*", default=list(DEFAULT_TASK_OBJECTS), choices=sorted(OBJECT_SPECS))
    parser.add_argument("--contact-objects", nargs="*", default=list(DEFAULT_CONTACT_OBJECTS), choices=sorted(OBJECT_SPECS))
    parser.add_argument("--skip-mujoco", action="store_true", help="Only probe optional candidate backends.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    backend_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    contact_rows: List[Dict[str, Any]] = []
    render_rows: List[Dict[str, Any]] = []

    run_mujoco = not args.skip_mujoco and args.backend in {"all", "mujoco_raw_cdpr"}
    if run_mujoco:
        mujoco_backend = MujocoCDPRBackend(
            seed=args.seed,
            width=args.width,
            height=args.height,
            render_backend=args.render_backend,
            camera_count=args.camera_count,
            render_enabled=args.render,
        )
        if mujoco_backend.is_available():
            try:
                summary, tasks, contacts, renders = mujoco_backend.run(
                    resets=args.resets,
                    steps=args.steps,
                    render_steps=args.render_steps,
                    task_objects=args.task_objects,
                    contact_objects=args.contact_objects,
                )
                backend_rows.append(summary)
                task_rows.extend(tasks)
                contact_rows.extend(contacts)
                render_rows.extend(renders)
            except Exception as exc:
                backend_rows.append(
                    skip_backend_summary(
                        "mujoco_raw_cdpr",
                        MujocoCDPRBackend.robot_embodiment,
                        f"runtime error: {type(exc).__name__}: {exc}",
                        "MuJoCo was importable, but the generated CDPR benchmark scene failed to run.",
                        "No MuJoCo rows produced; inspect tools/sim_compare/out/tmp/mujoco_cdpr_sim_compare_scene.xml.",
                        "low for current pipeline baseline",
                        mujoco_backend.version,
                    )
                )
        else:
            backend_rows.append(
                skip_backend_summary(
                    "mujoco_raw_cdpr",
                    MujocoCDPRBackend.robot_embodiment,
                    "mujoco",
                    "Current raw MuJoCo backend could not be imported.",
                    "Install mujoco to run the baseline.",
                    "low for current pipeline baseline",
                )
            )

    if args.backend == "all":
        backend_rows.extend(discover_optional_backend_summaries())

    settings = {
        "seed": int(args.seed),
        "resets": int(args.resets),
        "steps": int(args.steps),
        "render_steps": int(args.render_steps),
        "width": int(args.width),
        "height": int(args.height),
        "camera_count": int(args.camera_count),
        "render_backend": str(args.render_backend),
        "render": bool(args.render),
        "backend": str(args.backend),
        "task_objects": list(args.task_objects),
        "contact_objects": list(args.contact_objects),
        "out_dir": str(args.out),
    }
    write_all_outputs(backend_rows, task_rows, contact_rows, render_rows, settings, out_dir=args.out)
    print(f"Wrote simulator comparator outputs to {args.out}")
    for row in backend_rows:
        print(
            f"- {row.get('backend_name')}: {row.get('status')} "
            f"physics_fps={row.get('step_fps_no_render', '')} "
            f"rgb_fps={row.get('step_fps_with_rgb', '')} "
            f"reason={row.get('skipped_reason', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
