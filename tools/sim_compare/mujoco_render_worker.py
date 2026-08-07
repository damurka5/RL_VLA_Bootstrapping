#!/usr/bin/env python3
"""Isolated MuJoCo RGB render benchmark worker.

MuJoCo's GL backend is selected when the rendering context is created and may
be difficult to reset in-process after a failure. This worker runs one backend
attempt per subprocess so the main comparator can fail gracefully and try the
next backend in `auto` mode.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from tools.sim_compare.common import gpu_utilization_percent, gpu_vram_mb, platform_label, rss_mb  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--backend", required=True, choices=("egl", "osmesa", "glfw"))
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--camera-count", type=int, required=True)
    return parser.parse_args()


def _free_joint_for_body(mj: Any, model: Any, body_name: str) -> int:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise KeyError(f"missing body {body_name}")
    joint_count = int(model.body_jntnum[body_id])
    joint_adr = int(model.body_jntadr[body_id])
    for offset in range(joint_count):
        joint_id = joint_adr + offset
        if int(model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_FREE):
            return int(joint_id)
    raise KeyError(f"body {body_name} has no free joint")


def _set_free_pose(model: Any, data: Any, joint_id: int, pos: Sequence[float]) -> None:
    qadr = int(model.jnt_qposadr[int(joint_id)])
    dofadr = int(model.jnt_dofadr[int(joint_id)])
    data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    data.qpos[qadr + 3 : qadr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[dofadr : dofadr + 6] = 0.0


def _camera_names(mj: Any, model: Any, requested: int) -> List[str]:
    preferred = ["sim_compare_overview", "ee_camera"]
    names: List[str] = []
    for name in preferred:
        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name) != -1 and name not in names:
            names.append(name)
    for idx in range(int(model.ncam)):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, idx)
        if name and name not in names:
            names.append(name)
    return names[: max(0, int(requested))]


def _gpu_value(value: Any) -> str:
    return "" if value is None else f"{float(value):.2f}"


def main() -> int:
    args = _parse_args()
    os.environ["MUJOCO_GL"] = args.backend
    started = time.perf_counter()
    renderer = None
    try:
        import mujoco as mj

        model = mj.MjModel.from_xml_path(str(Path(args.scene)))
        data = mj.MjData(model)
        renderer = mj.Renderer(model, height=int(args.height), width=int(args.width))

        ee_joint = _free_joint_for_body(mj, model, "ee_base")
        block_joint = _free_joint_for_body(mj, model, "block")
        act_gripper = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        cameras = _camera_names(mj, model, int(args.camera_count))
        if not cameras:
            raise RuntimeError("no MuJoCo cameras available for render benchmark")

        mj.mj_resetData(model, data)
        if act_gripper != -1:
            data.ctrl[act_gripper] = 1.0
        _set_free_pose(model, data, ee_joint, [0.0, 0.0, 0.22])
        _set_free_pose(model, data, block_joint, [0.02, 0.0, 0.052])
        mj.mj_forward(model, data)

        step_time = 0.0
        render_time = 0.0
        frames = 0
        steps = max(1, int(args.steps))
        for idx in range(steps):
            alpha = idx / max(1, steps - 1)
            _set_free_pose(model, data, ee_joint, [-0.18 + 0.30 * alpha, 0.0, 0.18])
            if act_gripper != -1:
                data.ctrl[act_gripper] = 1.0
            s0 = time.perf_counter()
            mj.mj_step(model, data)
            step_time += time.perf_counter() - s0
            r0 = time.perf_counter()
            for camera in cameras:
                renderer.update_scene(data, camera=camera)
                _ = renderer.render()
                frames += 1
            render_time += time.perf_counter() - r0

        total_time = time.perf_counter() - started
        payload: Dict[str, Any] = {
            "ok": True,
            "backend": args.backend,
            "camera_count": len(cameras),
            "rendered_rgb_frames": frames,
            "step_time_s": step_time,
            "render_time_s": render_time,
            "total_time_s": total_time,
            "step_fps_during_rgb": float(steps / total_time) if total_time > 0 else 0.0,
            "rgb_frame_fps": float(frames / render_time) if render_time > 0 else 0.0,
            "cpu_ram_mb": rss_mb(),
            "gpu_vram_mb": gpu_vram_mb(),
            "gpu_utilization_percent": gpu_utilization_percent(),
            "platform": platform_label(),
            "failure_reason": "",
            "engineering_notes": f"Rendered cameras: {', '.join(cameras)}.",
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "backend": args.backend,
            "camera_count": 0,
            "rendered_rgb_frames": 0,
            "step_time_s": 0.0,
            "render_time_s": 0.0,
            "total_time_s": time.perf_counter() - started,
            "step_fps_during_rgb": 0.0,
            "rgb_frame_fps": 0.0,
            "cpu_ram_mb": rss_mb(),
            "gpu_vram_mb": gpu_vram_mb(),
            "gpu_utilization_percent": gpu_utilization_percent(),
            "platform": platform_label(),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "engineering_notes": traceback.format_exc(limit=3),
        }
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        renderer = None
        gc.collect()

    sys.stdout.write(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
