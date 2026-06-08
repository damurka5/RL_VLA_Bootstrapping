#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robots.cdpr.cdpr_mujoco import cdpr_scene_switcher as switcher  # noqa: E402
from robots.cdpr.cdpr_mujoco.cdpr_scene_switcher import STABLE_CONTACT_TIMESTEP  # noqa: E402
from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "contact_debug"
DEFAULT_OBJECTS = ("ycb_apple", "ycb_pear")
TABLE_Z = 0.0
TABLE_GEOM = "contact_debug_table_top"
TABLE_PENETRATION_TOLERANCE = 0.015
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
SCHEMATIC_SIZE = (960, 720)
MUJOCO_VIDEO_SIZE = (640, 480)


def _logical_object_name(object_name: str) -> str:
    name = str(object_name).strip()
    return name.removeprefix("ycb_") or name


def _default_closed_opening(object_name: str) -> float:
    logical = _logical_object_name(object_name).lower()
    if "pear" in logical:
        return 0.0
    if "apple" in logical:
        return 0.0
    return 0.0


def _write_table_scene(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<mujoco model="contact_debug_table_scene">
  <worldbody>
    <light name="contact_debug_key_light" pos="0 -0.6 1.3" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <body name="contact_debug_table" pos="0 0 -0.015">
      <geom name="{TABLE_GEOM}" class="table_collision" type="box" size="0.50 0.36 0.015"
            rgba="0.70 0.70 0.66 1"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


def _build_debug_wrapper(*, object_name: str, run_dir: Path, debug_collision: bool) -> tuple[Path, Path]:
    os.environ["RLVLA_CDPR_USE_STABLE_OBJECTS"] = "1"
    scene_xml = run_dir / "contact_debug_table_scene.xml"
    placed_xml = run_dir / f"{object_name}_placed.xml"
    wrapper_xml = run_dir / f"{object_name}_stable_contact_scene.xml"
    _write_table_scene(scene_xml)

    object_xml = switcher.find_object_xml(object_name)
    switcher.make_placed_object_xml(
        object_xml,
        placed_xml,
        prefix="p0",
        pos=np.array([0.0, 0.0, 0.10], dtype=float),
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        force_dynamic=True,
        logical_name=object_name,
    )
    switcher.build_wrapper_mjcf(
        scene_xml,
        switcher.CDPR_XML,
        [placed_xml],
        wrapper_xml,
        contact_preset="stable_contact",
        table_geom_names=(TABLE_GEOM,),
    )
    if debug_collision:
        os.environ["RLVLA_CDPR_DEBUG_RENDER_COLLISION_GEOMS"] = "1"
    return wrapper_xml, object_xml


def _body_id(model: mj.MjModel, name: str) -> int:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise RuntimeError(f"Body not found: {name}")
    return int(body_id)


def _joint_for_free_body(model: mj.MjModel, body_id: int) -> int:
    for offset in range(int(model.body_jntnum[body_id])):
        joint_id = int(model.body_jntadr[body_id]) + offset
        if int(model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_FREE):
            return joint_id
    raise RuntimeError(f"Body id {body_id} has no free joint.")


def _body_geom_ids(model: mj.MjModel, body_id: int, *, colliding_only: bool = False) -> list[int]:
    ids = [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == int(body_id)]
    if colliding_only:
        ids = [gid for gid in ids if int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0]
    return ids


def _half_extent_along_axis(model: mj.MjModel, data: mj.MjData, geom_id: int, axis: np.ndarray) -> float:
    gid = int(geom_id)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    gtype = int(model.geom_type[gid])
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    if gtype == int(mj.mjtGeom.mjGEOM_BOX):
        half_local = np.array([size[0], size[1], size[2]], dtype=np.float64)
    elif gtype == int(mj.mjtGeom.mjGEOM_CAPSULE):
        half_local = np.array([size[0], size[0], size[1] + size[0]], dtype=np.float64)
    elif gtype == int(mj.mjtGeom.mjGEOM_CYLINDER):
        half_local = np.array([size[0], size[0], size[1]], dtype=np.float64)
    elif gtype == int(mj.mjtGeom.mjGEOM_SPHERE):
        half_local = np.array([size[0], size[0], size[0]], dtype=np.float64)
    else:
        radius = float(model.geom_rbound[gid])
        half_local = np.array([radius, radius, radius], dtype=np.float64)
    xmat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    return float(np.sum(np.abs(xmat.T @ axis) * half_local))


def _geom_interval(model: mj.MjModel, data: mj.MjData, geom_id: int, axis: np.ndarray) -> tuple[float, float]:
    gid = int(geom_id)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    center = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
    projected_center = float(center @ axis)
    half = _half_extent_along_axis(model, data, gid, axis)
    return projected_center - half, projected_center + half


def _body_interval(model: mj.MjModel, data: mj.MjData, body_id: int, axis: np.ndarray) -> tuple[float, float]:
    lo = float("inf")
    hi = float("-inf")
    geom_ids = _body_geom_ids(model, body_id, colliding_only=True) or _body_geom_ids(model, body_id)
    for geom_id in geom_ids:
        geom_lo, geom_hi = _geom_interval(model, data, geom_id, axis)
        lo = min(lo, float(geom_lo))
        hi = max(hi, float(geom_hi))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError(f"Could not compute interval for body id {body_id}.")
    return lo, hi


def _set_free_body_pose(sim: HeadlessCDPRSimulation, body_name: str, pos: np.ndarray, quat_wxyz: np.ndarray | None = None) -> None:
    body_id = _body_id(sim.model, body_name)
    joint_id = _joint_for_free_body(sim.model, body_id)
    qadr = int(sim.model.jnt_qposadr[joint_id])
    dadr = int(sim.model.jnt_dofadr[joint_id])
    sim.data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = IDENTITY_QUAT_WXYZ if quat_wxyz is None else np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    sim.data.qpos[qadr + 3 : qadr + 7] = quat / max(float(np.linalg.norm(quat)), 1e-9)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _set_ee_position(sim: HeadlessCDPRSimulation, pos: np.ndarray) -> None:
    joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, "ee_free")
    if joint_id == -1:
        raise RuntimeError("ee_free joint not found.")
    qadr = int(sim.model.jnt_qposadr[joint_id])
    dadr = int(sim.model.jnt_dofadr[joint_id])
    quat = np.asarray(sim.data.qpos[qadr + 3 : qadr + 7], dtype=np.float64).copy()
    if float(np.linalg.norm(quat)) < 1e-9:
        quat = IDENTITY_QUAT_WXYZ.copy()
    sim.data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    sim.data.qpos[qadr + 3 : qadr + 7] = quat / max(float(np.linalg.norm(quat)), 1e-9)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    sim.target_pos = np.asarray(pos, dtype=float).reshape(3).copy()
    mj.mj_forward(sim.model, sim.data)


def _set_gripper_opening(sim: HeadlessCDPRSimulation, opening_01: float) -> None:
    target = float(np.clip(opening_01, 0.0, 1.0))
    sim.set_gripper(target)
    joint_pos = float(sim.gripper_joint_min + target * (sim.gripper_joint_max - sim.gripper_joint_min))
    for joint_name in ("finger_l", "finger_r"):
        joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            continue
        qadr = int(sim.model.jnt_qposadr[joint_id])
        dadr = int(sim.model.jnt_dofadr[joint_id])
        sim.data.qpos[qadr] = joint_pos
        sim.data.qvel[dadr] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _command_gripper_opening(sim: HeadlessCDPRSimulation, opening_01: float) -> None:
    sim.set_gripper(float(np.clip(opening_01, 0.0, 1.0)))


def _place_object_on_table(sim: HeadlessCDPRSimulation, body_name: str, xy: np.ndarray, *, clearance: float = 0.002) -> np.ndarray:
    body_id = _body_id(sim.model, body_name)
    mj.mj_forward(sim.model, sim.data)
    bottom, _ = _body_interval(sim.model, sim.data, body_id, np.array([0.0, 0.0, 1.0]))
    current = np.asarray(sim.data.xpos[body_id], dtype=np.float64).copy()
    bottom_offset = float(bottom - current[2])
    target = np.array([float(xy[0]), float(xy[1]), TABLE_Z + float(clearance) - bottom_offset], dtype=np.float64)
    _set_free_body_pose(sim, body_name, target)
    return target


def _object_velocity(sim: HeadlessCDPRSimulation, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = _body_id(sim.model, body_name)
    vel = np.zeros(6, dtype=np.float64)
    try:
        mj.mj_objectVelocity(sim.model, sim.data, mj.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    except Exception:
        pass
    return vel[3:].copy(), vel[:3].copy()


def _geom_name(model: mj.MjModel, geom_id: int) -> str:
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(geom_id)) or str(int(geom_id))


def _contact_rows(sim: HeadlessCDPRSimulation) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    min_dist = 0.0
    for idx in range(int(sim.data.ncon)):
        contact = sim.data.contact[idx]
        dist = float(contact.dist)
        min_dist = min(min_dist, dist)
        rows.append(
            {
                "geom1": _geom_name(sim.model, int(contact.geom1)),
                "geom2": _geom_name(sim.model, int(contact.geom2)),
                "dist": dist,
            }
        )
    return rows, float(max(0.0, -min_dist))


def _finger_state(sim: HeadlessCDPRSimulation) -> dict[str, Any]:
    left_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    right_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "right_finger_pad")
    if left_id == -1 or right_id == -1:
        return {"available": False}
    left = np.asarray(sim.data.geom_xpos[left_id], dtype=np.float64).reshape(3)
    right = np.asarray(sim.data.geom_xpos[right_id], dtype=np.float64).reshape(3)
    axis_vec = left - right
    distance = float(np.linalg.norm(axis_vec))
    axis = axis_vec / max(distance, 1e-9)
    center = 0.5 * (left + right)
    left_half = _half_extent_along_axis(sim.model, sim.data, left_id, axis)
    right_half = _half_extent_along_axis(sim.model, sim.data, right_id, axis)
    return {
        "available": True,
        "left": left,
        "right": right,
        "center": center,
        "axis": axis,
        "center_distance_m": distance,
        "inner_gap_m": float(max(0.0, distance - left_half - right_half)),
    }


def _object_between_fingers(sim: HeadlessCDPRSimulation, body_name: str) -> bool:
    finger = _finger_state(sim)
    if not finger.get("available"):
        return False
    object_pos = np.asarray(sim.data.xpos[_body_id(sim.model, body_name)], dtype=np.float64).reshape(3)
    center = np.asarray(finger["center"], dtype=np.float64).reshape(3)
    axis = np.asarray(finger["axis"], dtype=np.float64).reshape(3)
    lateral = abs(float((object_pos - center) @ axis))
    vertical = abs(float(object_pos[2] - center[2]))
    return lateral <= 0.055 and vertical <= 0.080


def _project(point: np.ndarray, bounds: tuple[float, float, float, float], box: tuple[int, int, int, int]) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = bounds
    left, top, right, bottom = box
    point = np.asarray(point, dtype=np.float64).reshape(-1)
    x = (float(point[0]) - x_min) / max(x_max - x_min, 1e-9)
    y = (float(point[1]) - y_min) / max(y_max - y_min, 1e-9)
    px = int(round(left + np.clip(x, 0.0, 1.0) * (right - left)))
    py = int(round(bottom - np.clip(y, 0.0, 1.0) * (bottom - top)))
    return px, py


def _draw_circle(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)


def _write_video(frames: list[np.ndarray], output_path: Path, *, fps: float) -> dict[str, Any]:
    if not frames:
        return {"video": "", "frames": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    try:
        for idx, frame in enumerate(frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(frame_dir / f"{idx:05d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-r",
                f"{float(fps):.6g}",
                "-i",
                str(frame_dir / "%05d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return {"video": output_path.as_posix(), "frames": len(frames)}


class MujocoOverviewRenderer:
    def __init__(self, model: mj.MjModel, *, width: int, height: int, debug_collision: bool):
        self.renderer = mj.Renderer(model, height=int(height), width=int(width))
        self.camera = mj.MjvCamera()
        self.camera.type = mj.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = np.array([0.02, 0.0, 0.18], dtype=float)
        self.camera.distance = 1.05
        self.camera.azimuth = 90
        self.camera.elevation = -24
        self.option = mj.MjvOption()
        mj.mjv_defaultOption(self.option)
        if len(self.option.geomgroup) > 3:
            self.option.geomgroup[3] = 1 if debug_collision else 0

    def capture(self, data: mj.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.option)
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()

    def close(self) -> None:
        self.renderer.close()


def _render_schematic_frame(
    sim: HeadlessCDPRSimulation,
    *,
    object_name: str,
    body_name: str,
    phase: str,
    closeup: bool = False,
) -> np.ndarray:
    width, height = SCHEMATIC_SIZE
    image = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    object_body_id = _body_id(sim.model, body_name)
    object_pos = np.asarray(sim.data.xpos[object_body_id], dtype=np.float64).reshape(3)
    ee_pos = np.asarray(sim.data.xpos[_body_id(sim.model, "ee_base")], dtype=np.float64).reshape(3)
    finger = _finger_state(sim)
    between = _object_between_fingers(sim, body_name)
    bottom_z, _ = _body_interval(sim.model, sim.data, object_body_id, np.array([0.0, 0.0, 1.0]))

    object_color = (206, 45, 35) if "apple" in object_name else (151, 178, 47)
    blue = (32, 100, 180)
    dark = (34, 38, 46)
    muted = (105, 112, 123)

    title = f"{object_name} contact diagnostic"
    draw.text((24, 18), title, fill=dark)
    draw.text((24, 42), f"phase={phase}  t={sim.data.time:.2f}s  between_fingers={between}", fill=muted)

    if closeup:
        panel = (60, 95, width - 60, height - 85)
        center_xy = object_pos[:2]
        if finger.get("available"):
            center_xy = np.asarray(finger["center"], dtype=np.float64)[:2]
        bounds = (center_xy[0] - 0.14, center_xy[0] + 0.14, center_xy[1] - 0.14, center_xy[1] + 0.14)
        draw.rectangle(panel, fill=(255, 255, 255), outline=(190, 195, 202), width=2)
        draw.text((panel[0], panel[1] - 28), "wrist-style closeup, top view", fill=dark)
    else:
        panel = (42, 95, 610, height - 85)
        side = (650, 95, width - 42, height - 85)
        bounds = (-0.36, 0.36, -0.28, 0.28)
        draw.rectangle(panel, fill=(255, 255, 255), outline=(190, 195, 202), width=2)
        draw.rectangle(side, fill=(255, 255, 255), outline=(190, 195, 202), width=2)
        draw.text((panel[0], panel[1] - 28), "overview, top view", fill=dark)
        draw.text((side[0], side[1] - 28), "side view", fill=dark)

        table_left = _project(np.array([-0.50, -0.36]), bounds, panel)
        table_right = _project(np.array([0.50, 0.36]), bounds, panel)
        draw.rectangle((table_left[0], table_right[1], table_right[0], table_left[1]), outline=(175, 160, 128), width=2)

        side_bounds = (-0.36, 0.36, -0.05, 0.55)
        table_a = _project(np.array([-0.36, TABLE_Z]), side_bounds, side)
        table_b = _project(np.array([0.36, TABLE_Z]), side_bounds, side)
        draw.line((table_a[0], table_a[1], table_b[0], table_b[1]), fill=(120, 110, 90), width=3)
        obj_side = _project(np.array([object_pos[0], object_pos[2]]), side_bounds, side)
        ee_side = _project(np.array([ee_pos[0], ee_pos[2]]), side_bounds, side)
        _draw_circle(draw, obj_side, 16, object_color, (110, 70, 40))
        _draw_circle(draw, ee_side, 7, blue, blue)
        bottom_pt = _project(np.array([object_pos[0], bottom_z]), side_bounds, side)
        draw.line((obj_side[0], obj_side[1], bottom_pt[0], bottom_pt[1]), fill=(90, 90, 90), width=1)

    if finger.get("available"):
        left = _project(np.asarray(finger["left"], dtype=np.float64)[:2], bounds, panel)
        right = _project(np.asarray(finger["right"], dtype=np.float64)[:2], bounds, panel)
        draw.line((left[0], left[1], right[0], right[1]), fill=(120, 160, 210), width=3)
        draw.rectangle((left[0] - 9, left[1] - 9, left[0] + 9, left[1] + 9), fill=blue)
        draw.rectangle((right[0] - 9, right[1] - 9, right[0] + 9, right[1] + 9), fill=blue)

    obj_top = _project(object_pos[:2], bounds, panel)
    ee_top = _project(ee_pos[:2], bounds, panel)
    _draw_circle(draw, obj_top, 18, object_color, (110, 70, 40))
    _draw_circle(draw, ee_top, 6, (20, 40, 90), (20, 40, 90))
    draw.text((24, height - 58), f"object=({object_pos[0]:+.3f},{object_pos[1]:+.3f},{object_pos[2]:.3f})  bottom_z={bottom_z:+.3f}", fill=dark)
    if finger.get("available"):
        draw.text((24, height - 34), f"finger_gap={float(finger['inner_gap_m']):.3f} m", fill=dark)
    return np.asarray(image)


def _phase_targets(object_pos: np.ndarray, grasp_offset: np.ndarray | None = None) -> dict[str, np.ndarray]:
    # The CDPR finger-pad centers sit about 20 mm below ee_base.
    # Aim pads near the object's collision-body center instead of pushing
    # the visual finger tips down to the table.
    offset = np.zeros(3, dtype=np.float64) if grasp_offset is None else np.asarray(grasp_offset, dtype=np.float64).reshape(3)
    grasp = np.array([object_pos[0], object_pos[1], object_pos[2] + 0.022], dtype=np.float64) + offset
    return {
        "above": grasp + np.array([0.0, 0.0, 0.24], dtype=np.float64),
        "descend": grasp + np.array([0.0, 0.0, 0.025], dtype=np.float64),
        "grasp": grasp,
        "lift": grasp + np.array([0.0, 0.0, 0.26], dtype=np.float64),
        "carry": grasp + np.array([0.025, 0.0, 0.26], dtype=np.float64),
        "release": grasp + np.array([0.025, 0.0, 0.03], dtype=np.float64),
    }


def _append_step_log(
    *,
    sim: HeadlessCDPRSimulation,
    rows: list[dict[str, Any]],
    object_name: str,
    body_name: str,
    phase: str,
    step_index: int,
) -> dict[str, Any]:
    object_body_id = _body_id(sim.model, body_name)
    ee_body_id = _body_id(sim.model, "ee_base")
    obj_lin, obj_ang = _object_velocity(sim, body_name)
    contacts, max_pen = _contact_rows(sim)
    finger = _finger_state(sim)
    bottom_z, _ = _body_interval(sim.model, sim.data, object_body_id, np.array([0.0, 0.0, 1.0]))
    object_pos = np.asarray(sim.data.xpos[object_body_id], dtype=np.float64).copy()
    ee_pos = np.asarray(sim.data.xpos[ee_body_id], dtype=np.float64).copy()
    row = {
        "step": int(step_index),
        "time": float(sim.data.time),
        "phase": phase,
        "object_name": object_name,
        "object_body": body_name,
        "contact_count": int(sim.data.ncon),
        "contact_geom_names": contacts,
        "object_position": [float(x) for x in object_pos],
        "object_quat_wxyz": [float(x) for x in np.asarray(sim.data.xquat[object_body_id], dtype=np.float64)],
        "gripper_position": [float(x) for x in ee_pos],
        "gripper_quat_wxyz": [float(x) for x in np.asarray(sim.data.xquat[ee_body_id], dtype=np.float64)],
        "object_linear_velocity": [float(x) for x in obj_lin],
        "object_angular_velocity": [float(x) for x in obj_ang],
        "relative_object_to_gripper_position": [float(x) for x in (object_pos - ee_pos)],
        "gripper_opening_normalized": float(sim.get_gripper_opening()),
        "gripper_target_normalized": float(sim.get_gripper_target()),
        "finger_inner_gap_m": float(finger["inner_gap_m"]) if finger.get("available") else None,
        "object_bottom_z": float(bottom_z),
        "object_between_fingers": bool(_object_between_fingers(sim, body_name)),
        "max_contact_penetration": float(max_pen),
    }
    rows.append(row)
    return row


def _run_phase(
    *,
    sim: HeadlessCDPRSimulation,
    body_name: str,
    object_name: str,
    rows: list[dict[str, Any]],
    overview_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    phase: str,
    start_ee: np.ndarray,
    end_ee: np.ndarray,
    start_opening: float,
    end_opening: float,
    seconds: float,
    capture_every: int,
    render_mode: str,
    mujoco_renderer: MujocoOverviewRenderer | None,
    step_index: int,
) -> int:
    steps = max(1, int(round(float(seconds) / float(sim.model.opt.timestep))))
    for local_idx in range(steps):
        alpha = (local_idx + 1) / float(steps)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        _set_ee_position(sim, (1.0 - smooth) * start_ee + smooth * end_ee)
        _set_gripper_opening(sim, (1.0 - smooth) * float(start_opening) + smooth * float(end_opening))
        mj.mj_step(sim.model, sim.data)
        row = _append_step_log(
            sim=sim,
            rows=rows,
            object_name=object_name,
            body_name=body_name,
            phase=phase,
            step_index=step_index,
        )
        if step_index % max(1, int(capture_every)) == 0:
            if render_mode == "mujoco" and mujoco_renderer is not None:
                overview_frames.append(mujoco_renderer.capture(sim.data))
            elif render_mode == "schematic":
                overview_frames.append(_render_schematic_frame(sim, object_name=object_name, body_name=body_name, phase=phase))
                wrist_frames.append(
                    _render_schematic_frame(sim, object_name=object_name, body_name=body_name, phase=phase, closeup=True)
                )
        step_index += 1
        if not np.all(np.isfinite(row["object_position"])):
            break
    return step_index


def _run_phase_controlled(
    *,
    sim: HeadlessCDPRSimulation,
    body_name: str,
    object_name: str,
    rows: list[dict[str, Any]],
    overview_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    phase: str,
    start_ee: np.ndarray,
    end_ee: np.ndarray,
    start_opening: float,
    end_opening: float,
    seconds: float,
    capture_every: int,
    render_mode: str,
    mujoco_renderer: MujocoOverviewRenderer | None,
    step_index: int,
) -> int:
    steps = max(1, int(round(float(seconds) / float(sim.model.opt.timestep))))
    for local_idx in range(steps):
        alpha = (local_idx + 1) / float(steps)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        target_ee = (1.0 - smooth) * start_ee + smooth * end_ee
        sim.set_target_position(target_ee)
        _command_gripper_opening(sim, (1.0 - smooth) * float(start_opening) + smooth * float(end_opening))
        sim.run_simulation_step(capture_frame=False)
        row = _append_step_log(
            sim=sim,
            rows=rows,
            object_name=object_name,
            body_name=body_name,
            phase=phase,
            step_index=step_index,
        )
        if step_index % max(1, int(capture_every)) == 0:
            if render_mode == "mujoco" and mujoco_renderer is not None:
                overview_frames.append(mujoco_renderer.capture(sim.data))
            elif render_mode == "schematic":
                overview_frames.append(_render_schematic_frame(sim, object_name=object_name, body_name=body_name, phase=phase))
                wrist_frames.append(
                    _render_schematic_frame(sim, object_name=object_name, body_name=body_name, phase=phase, closeup=True)
                )
        step_index += 1
        if not np.all(np.isfinite(row["object_position"])):
            break
    return step_index


def _summarize(rows: list[dict[str, Any]], *, lift_threshold: float, hold_duration: float, max_snap_distance: float) -> dict[str, Any]:
    object_positions = np.asarray([row["object_position"] for row in rows], dtype=np.float64)
    bottom_z = np.asarray([row["object_bottom_z"] for row in rows], dtype=np.float64)
    linear_speeds = np.asarray([np.linalg.norm(row["object_linear_velocity"]) for row in rows], dtype=np.float64)
    angular_speeds = np.asarray([np.linalg.norm(row["object_angular_velocity"]) for row in rows], dtype=np.float64)
    penetrations = np.asarray([row["max_contact_penetration"] for row in rows], dtype=np.float64)
    jumps = np.linalg.norm(np.diff(object_positions, axis=0), axis=1) if len(object_positions) > 1 else np.zeros(0)
    lift_rows = [row for row in rows if row["phase"] in {"lift", "hold", "move_laterally"}]
    hold_rows = [row for row in rows if row["phase"] == "hold"]
    hold_between_steps = sum(1 for row in hold_rows if row["object_between_fingers"])
    dt = float(rows[1]["time"] - rows[0]["time"]) if len(rows) > 1 else STABLE_CONTACT_TIMESTEP
    held_between_duration = float(hold_between_steps * max(dt, 0.0))
    min_lift_bottom = min((float(row["object_bottom_z"]) for row in lift_rows), default=float("nan"))
    max_lift_body_z = max((float(row["object_position"][2]) for row in lift_rows), default=float("nan"))
    start_z = float(rows[0]["object_position"][2]) if rows else float("nan")
    criteria = {
        "object_not_through_table": bool(np.all(bottom_z >= TABLE_Z - TABLE_PENETRATION_TOLERANCE)),
        "no_explode_or_jitter": bool(np.all(np.isfinite(object_positions)) and float(np.max(linear_speeds, initial=0.0)) < 5.0 and float(np.max(angular_speeds, initial=0.0)) < 80.0),
        "lifted_above_threshold": bool(np.isfinite(max_lift_body_z) and max_lift_body_z >= start_z + float(lift_threshold)),
        "held_between_fingers": bool(held_between_duration >= min(float(hold_duration), 1.0)),
        "no_teleport_or_snap": bool(float(np.max(jumps, initial=0.0)) <= float(max_snap_distance)),
    }
    return {
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "min_object_height_during_lift": float(min_lift_bottom),
        "max_object_body_z_during_lift": float(max_lift_body_z),
        "object_penetrated_table": bool(not criteria["object_not_through_table"]),
        "slipped_out_after_close": bool(not criteria["held_between_fingers"]),
        "max_contact_penetration": float(np.max(penetrations, initial=0.0)),
        "max_object_linear_speed": float(np.max(linear_speeds, initial=0.0)),
        "max_object_angular_speed": float(np.max(angular_speeds, initial=0.0)),
        "max_object_position_jump": float(np.max(jumps, initial=0.0)),
        "held_between_fingers_duration_s": held_between_duration,
    }


def _run_object(args: argparse.Namespace, object_name: str) -> dict[str, Any]:
    logical = _logical_object_name(object_name)
    closed_opening = (
        _default_closed_opening(object_name)
        if args.closed_opening is None
        else float(args.closed_opening)
    )
    run_dir = Path(args.output_dir).expanduser().resolve() / logical
    run_dir.mkdir(parents=True, exist_ok=True)
    wrapper_xml, object_xml = _build_debug_wrapper(
        object_name=object_name,
        run_dir=run_dir,
        debug_collision=bool(args.debug_render_collision_geoms),
    )

    sim = HeadlessCDPRSimulation(
        str(wrapper_xml),
        output_dir=str(run_dir),
        record_trajectory=False,
        use_model_cache=False,
        timestep=float(args.timestep),
        debug_render_collision_geoms=bool(args.debug_render_collision_geoms),
        render_enabled=False,
    )
    rows: list[dict[str, Any]] = []
    overview_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    mujoco_renderer: MujocoOverviewRenderer | None = None
    try:
        sim.initialize()
        if str(args.render_mode) == "mujoco":
            mujoco_renderer = MujocoOverviewRenderer(
                sim.model,
                width=int(args.video_width),
                height=int(args.video_height),
                debug_collision=bool(args.debug_render_collision_geoms),
            )
        body_name = str(sim.get_object_body_name())
        object_pos = _place_object_on_table(sim, body_name, np.asarray(args.object_xy, dtype=float))
        grasp_offset = np.asarray(args.grasp_offset, dtype=np.float64).reshape(3)
        targets = _phase_targets(object_pos, grasp_offset=grasp_offset)
        _set_ee_position(sim, targets["above"])
        _set_gripper_opening(sim, 1.0)
        sim.hold_current_pose(warm_steps=int(round(0.2 / float(sim.model.opt.timestep))))
        _command_gripper_opening(sim, 1.0)

        phases = [
            ("move_above", targets["above"], targets["above"], 1.0, 1.0, 0.4),
            ("descend", targets["above"], targets["descend"], 1.0, 1.0, 0.7),
            ("open_gripper", targets["descend"], targets["descend"], 1.0, 1.0, 0.2),
            ("center_around_object", targets["descend"], targets["grasp"], 1.0, 1.0, 0.3),
            ("close_gripper", targets["grasp"], targets["grasp"], 1.0, closed_opening, 0.7),
            ("hold_closed_before_lift", targets["grasp"], targets["grasp"], closed_opening, closed_opening, 1.2),
            ("lift", targets["grasp"], targets["lift"], closed_opening, closed_opening, 1.0),
            ("hold", targets["lift"], targets["lift"], closed_opening, closed_opening, float(args.hold_duration)),
            ("move_laterally", targets["lift"], targets["carry"], closed_opening, closed_opening, 1.0),
            ("lower_for_release", targets["carry"], targets["release"], closed_opening, closed_opening, 0.7),
            ("release", targets["release"], targets["release"], closed_opening, 1.0, 0.5),
        ]

        step_index = 0
        for phase, start, end, start_open, end_open, seconds in phases:
            step_index = _run_phase_controlled(
                sim=sim,
                body_name=body_name,
                object_name=object_name,
                rows=rows,
                overview_frames=overview_frames,
                wrist_frames=wrist_frames,
                phase=phase,
                start_ee=np.asarray(start, dtype=np.float64),
                end_ee=np.asarray(end, dtype=np.float64),
                start_opening=float(start_open),
                end_opening=float(end_open),
                seconds=float(seconds),
                capture_every=int(args.capture_every),
                render_mode=str(args.render_mode),
                mujoco_renderer=mujoco_renderer,
                step_index=step_index,
            )

        summary = _summarize(
            rows,
            lift_threshold=float(args.lift_threshold),
            hold_duration=float(args.hold_duration),
            max_snap_distance=float(args.max_snap_distance),
        )
        diagnostics = {
            "object_name": object_name,
            "object_body": body_name,
            "seed": int(args.seed),
            "source_object_xml": object_xml.as_posix(),
            "wrapper_xml": wrapper_xml.as_posix(),
            "timestep": float(sim.model.opt.timestep),
            "solver_settings": {
                "solver": int(sim.model.opt.solver),
                "iterations": int(sim.model.opt.iterations),
                "tolerance": float(sim.model.opt.tolerance),
                "cone": int(sim.model.opt.cone),
                "noslip_iterations": int(sim.model.opt.noslip_iterations),
            },
            "contact_preset": "stable_contact",
            "debug_render_collision_geoms": bool(args.debug_render_collision_geoms),
            "render_mode": str(args.render_mode),
            "video_camera": "overview",
            "video_width": int(args.video_width),
            "video_height": int(args.video_height),
            "closed_opening": float(closed_opening),
            "grasp_offset": [float(x) for x in grasp_offset],
            "control_mode": "cdpr_slider_and_gripper_actuator_commands",
            "summary": summary,
            "steps": rows,
        }
        diagnostics_path = run_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

        videos: dict[str, str] = {}
        if overview_frames:
            overview_path = run_dir / "overview.mp4"
            _write_video(overview_frames, overview_path, fps=float(args.video_fps))
            videos["overview"] = overview_path.as_posix()
        if wrist_frames:
            wrist_path = run_dir / "wrist.mp4"
            _write_video(wrist_frames, wrist_path, fps=float(args.video_fps))
            videos["wrist"] = wrist_path.as_posix()
        return {
            "object_name": object_name,
            "output_dir": run_dir.as_posix(),
            "diagnostics": diagnostics_path.as_posix(),
            "videos": videos,
            **summary,
        }
    finally:
        if mujoco_renderer is not None:
            mujoco_renderer.close()
        sim.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CDPR stable-contact apple/pear grasp diagnostics.")
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestep", type=float, default=STABLE_CONTACT_TIMESTEP)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-width", type=int, default=MUJOCO_VIDEO_SIZE[0])
    parser.add_argument("--video-height", type=int, default=MUJOCO_VIDEO_SIZE[1])
    parser.add_argument("--capture-every", type=int, default=10)
    parser.add_argument("--object-xy", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument("--grasp-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--closed-opening", type=float, default=None)
    parser.add_argument("--hold-duration", type=float, default=1.25)
    parser.add_argument("--lift-threshold", type=float, default=0.08)
    parser.add_argument("--max-snap-distance", type=float, default=0.08)
    parser.add_argument("--render-mode", choices=("schematic", "mujoco", "none"), default="schematic")
    parser.add_argument("--no-video", action="store_true", help="Run physics/contact diagnostics without rendering videos.")
    parser.add_argument("--debug-render-collision-geoms", action="store_true", dest="debug_render_collision_geoms")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    if args.no_video:
        args.render_mode = "none"
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    results = [_run_object(args, str(object_name)) for object_name in args.objects]
    manifest = {
        "started_at": started_at,
        "output_dir": args.output_dir.as_posix(),
        "contact_preset": "stable_contact",
        "objects": list(args.objects),
        "results": results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if all(result.get("pass") for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
