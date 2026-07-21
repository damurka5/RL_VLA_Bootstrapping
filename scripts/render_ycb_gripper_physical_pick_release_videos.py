#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_cdpr_ycb_caught_object_videos import (  # noqa: E402
    DEFAULT_OBJECTS,
    DEFAULT_YCB_ROOT,
    PLATE_CENTER,
    _body_geoms,
    _finger_geometry,
    _force_gripper_opening,
    _geom_interval_along_axis,
    _load_scene_switcher,
    _object_position,
    _set_body_pose,
    _write_video,
)


DEFAULT_OUTPUT = ROOT / "runs" / "ycb_gripper_physical_pick_release_videos"
OBJECT_START_XY = np.array([-0.14, -0.03], dtype=float)
TABLE_TOP_Z = 0.0
PLATE_TOP_Z = float(PLATE_CENTER[2] + 0.006)
PHYSICAL_HELD_OFFSET = np.array([0.0, 0.0, 0.034], dtype=float)


def _patch_object_physics(root: ET.Element, *, mass_scale: float, freejoint_damping: float) -> None:
    scale = float(max(mass_scale, 1e-6))
    for inertial in root.findall(".//inertial"):
        if inertial.get("mass") is not None:
            inertial.set("mass", f"{float(inertial.get('mass')) * scale:.8g}")
        if inertial.get("diaginertia") is not None:
            inertia = np.fromstring(str(inertial.get("diaginertia")), sep=" ", dtype=float)
            if inertia.size == 3:
                inertial.set("diaginertia", " ".join(f"{float(v) * scale:.8g}" for v in inertia))
    for geom_default in root.findall(".//default/geom"):
        if geom_default.get("density") is not None:
            geom_default.set("density", f"{float(geom_default.get('density')) * scale:.8g}")
    damping = float(max(freejoint_damping, 0.0))
    if damping > 0.0:
        for freejoint in root.findall(".//freejoint"):
            freejoint.tag = "joint"
            freejoint.set("type", "free")
            freejoint.set("damping", f"{damping:.8g}")


def _build_physical_gripper_xml(
    *,
    ycb_root: Path,
    object_name: str,
    output_dir: Path,
    object_mass_scale: float,
    object_freejoint_damping: float,
) -> tuple[Path, Path]:
    switcher = _load_scene_switcher(ycb_root)
    object_xml = switcher.find_object_xml(object_name)
    placed_xml = output_dir / f"{object_name}_placed.xml"
    wrapper_xml = output_dir / f"{object_name}_physical_gripper.xml"
    switcher.make_placed_object_xml(
        object_xml,
        placed_xml,
        prefix="p0",
        pos=np.array([0.0, 0.0, 0.08], dtype=float),
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        force_dynamic=True,
        logical_name=object_name,
    )
    tree = ET.parse(placed_xml)
    root = tree.getroot()
    _patch_object_physics(root, mass_scale=object_mass_scale, freejoint_damping=object_freejoint_damping)
    for geom in root.findall(".//geom"):
        if geom.get("contype", "1") != "0":
            geom.set("friction", "12.0 1.0 1.0")
            geom.set("condim", "6")
            geom.set("solref", "0.001 1")
            geom.set("solimp", "0.99 0.999 0.0001")
    try:
        ET.indent(tree)
    except Exception:
        pass
    tree.write(placed_xml, encoding="utf-8", xml_declaration=True)

    wrapper_xml.write_text(
        f"""<mujoco model="physical_ycb_gripper">
  <compiler autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" iterations="100" tolerance="1e-10"/>

  <default>
    <geom friction="12.0 1.0 1.0" condim="6" solref="0.001 1" solimp="0.99 0.999 0.0001"/>
  </default>

  <visual>
    <headlight diffuse=".7 .7 .7"/>
  </visual>

  <worldbody>
    <light name="key_light" pos="0 -0.5 1.2" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="demo_table" pos="0 0 -0.025">
      <geom name="demo_table_top" type="box" size="0.48 0.36 0.025" rgba="0.72 0.72 0.68 1"/>
    </body>
    <body name="demo_plate" pos="{PLATE_CENTER[0]:.4f} {PLATE_CENTER[1]:.4f} {PLATE_CENTER[2]:.4f}">
      <geom name="demo_plate_geom" type="cylinder" size="0.085 0.006" rgba="0.05 0.32 0.85 0.85"/>
    </body>

    <body name="ee_base" pos="0 0 0">
      <joint name="ee_x" type="slide" axis="1 0 0" limited="true" range="-0.8 0.8" damping="80"/>
      <joint name="ee_y" type="slide" axis="0 1 0" limited="true" range="-0.8 0.8" damping="80"/>
      <joint name="ee_z" type="slide" axis="0 0 1" limited="true" range="0.02 0.8" damping="80"/>
      <inertial pos="0 0 0" mass="20" diaginertia="1 1 1"/>

      <body name="ee_platform" pos="0 0 0.08">
        <inertial pos="0 0 0" mass="2" diaginertia="0.02 0.02 0.02"/>
        <geom name="palm" type="box" size="0.03 0.03 0.015" rgba="0.15 0.15 0.15 1"/>

        <body name="finger_left_car" pos="0.02 0 0">
          <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
          <joint name="finger_l" type="slide" axis="1 0 0" limited="true" range="0 0.03" damping="6"/>
          <geom name="finger_l_link" type="box" size="0.012 0.01 0.055" pos="0 0 -0.055"
                rgba="0.2 0.2 0.2 1"/>
          <geom name="finger_l_tip" type="capsule" fromto="0 0 -0.11 0 0 -0.135" size="0.008"
                rgba="0.2 0.2 0.2 1"/>
        </body>

        <body name="finger_right_car" pos="-0.02 0 0">
          <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
          <joint name="finger_r" type="slide" axis="-1 0 0" limited="true" range="0 0.03" damping="6"/>
          <geom name="finger_r_link" type="box" size="0.012 0.01 0.055" pos="0 0 -0.055"
                rgba="0.2 0.2 0.2 1"/>
          <geom name="finger_r_tip" type="capsule" fromto="0 0 -0.11 0 0 -0.135" size="0.008"
                rgba="0.2 0.2 0.2 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <include file="{placed_xml.resolve()}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return wrapper_xml, object_xml


def _sim_from_xml(xml_path: Path):
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return SimpleNamespace(
        model=model,
        data=data,
        gripper_joint_min=0.0,
        gripper_joint_max=0.03,
        set_gripper=lambda _opening: None,
        get_gripper_opening=lambda: _get_gripper_opening(SimpleNamespace(model=model, data=data)),
    )


def _get_gripper_opening(sim) -> float:
    jid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, "finger_l")
    if jid == -1:
        return 0.0
    qadr = int(sim.model.jnt_qposadr[jid])
    return float(np.clip(sim.data.qpos[qadr] / 0.03, 0.0, 1.0))


def _body_name_for_object(sim, object_name: str) -> str:
    preferred = f"p0_{object_name}"
    if mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, preferred) != -1:
        return preferred
    for bid in range(sim.model.nbody):
        name = mj.mj_id2name(sim.model, mj.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith("p0_"):
            return str(name)
    raise RuntimeError(f"Could not find placed object body for {object_name!r}.")


def _set_ee(sim, pos: np.ndarray) -> None:
    for joint_name, value in zip(("ee_x", "ee_y", "ee_z"), np.asarray(pos, dtype=float).reshape(3)):
        jid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        qadr = int(sim.model.jnt_qposadr[jid])
        dadr = int(sim.model.jnt_dofadr[jid])
        sim.data.qpos[qadr] = float(value)
        sim.data.qvel[dadr] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _step(
    sim,
    *,
    steps: int = 1,
    ee_pos: np.ndarray | None = None,
    opening: float | None = None,
) -> None:
    for _ in range(max(1, int(steps))):
        if ee_pos is not None:
            _set_ee(sim, np.asarray(ee_pos, dtype=float))
        if opening is not None:
            _force_gripper_opening(sim, float(opening))
        mj.mj_step(sim.model, sim.data)
    if ee_pos is not None:
        _set_ee(sim, np.asarray(ee_pos, dtype=float))
    if opening is not None:
        _force_gripper_opening(sim, float(opening))


def _body_interval_along_axis(sim, body_name: str, axis: np.ndarray) -> tuple[float, float]:
    lo = float("inf")
    hi = float("-inf")
    for gid in _body_geoms(sim, body_name):
        geom_lo, geom_hi = _geom_interval_along_axis(sim, gid, axis)
        lo = min(lo, float(geom_lo))
        hi = max(hi, float(geom_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError(f"Could not measure body interval for {body_name!r}.")
    return lo, hi


def _body_width_along_axis(sim, body_name: str, axis: np.ndarray) -> float:
    lo, hi = _body_interval_along_axis(sim, body_name, axis)
    return float(hi - lo)


def _body_aabb_center(sim, body_name: str) -> np.ndarray:
    out = []
    for axis in (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ):
        lo, hi = _body_interval_along_axis(sim, body_name, axis)
        out.append(0.5 * (lo + hi))
    return np.asarray(out, dtype=float)


def _set_body_center(sim, body_name: str, center: np.ndarray) -> None:
    body_pos = _object_position(sim, body_name)
    current_center = _body_aabb_center(sim, body_name)
    _set_body_pose(sim, body_name, body_pos + np.asarray(center, dtype=float).reshape(3) - current_center)


def _place_body_bottom_on_z(sim, body_name: str, xy: np.ndarray, surface_z: float, *, clearance: float = 0.001) -> None:
    pos = _object_position(sim, body_name)
    center = _body_aabb_center(sim, body_name)
    lo_z, _ = _body_interval_along_axis(sim, body_name, np.array([0.0, 0.0, 1.0], dtype=float))
    target = pos.copy()
    target[:2] += np.asarray(xy, dtype=float).reshape(2) - center[:2]
    target[2] += float(surface_z) + float(clearance) - float(lo_z)
    _set_body_pose(sim, body_name, target)


def _object_center_for_bottom_on_z(sim, body_name: str, surface_z: float, *, clearance: float = 0.001) -> np.ndarray:
    center = _body_aabb_center(sim, body_name)
    lo_z, _ = _body_interval_along_axis(sim, body_name, np.array([0.0, 0.0, 1.0], dtype=float))
    out = center.copy()
    out[2] += float(surface_z) + float(clearance) - float(lo_z)
    return out


def _measure_fit(sim, body_name: str, *, compression: float) -> dict[str, object]:
    _force_gripper_opening(sim, 0.0)
    closed = _finger_geometry(sim)
    width = _body_width_along_axis(sim, body_name, np.asarray(closed["axis"], dtype=float))
    desired_gap = max(0.0, width - 2.0 * max(0.0, compression))
    raw_opening = (desired_gap - float(closed["inner_gap"])) / 0.06
    opening = float(np.clip(raw_opening, 0.0, 1.0))
    _force_gripper_opening(sim, opening)
    held = _finger_geometry(sim)
    return {
        "object_projected_width_m": float(width),
        "grip_compression_m": float(compression),
        "desired_inner_gap_m": float(desired_gap),
        "closed_inner_gap_m": float(closed["inner_gap"]),
        "held_inner_gap_m": float(held["inner_gap"]),
        "held_gripper_opening_01": float(opening),
        "target_finger_joint_qpos_m": float(opening * 0.03),
        "fits_without_clipping": bool(0.0 <= raw_opening <= 1.0),
    }


def _hold_center_relative_to_ee(sim) -> np.ndarray:
    return np.asarray(_finger_geometry(sim)["center"], dtype=float).reshape(3) - np.asarray(
        sim.data.xpos[mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "ee_base")],
        dtype=float,
    ).reshape(3)


def _ee_for_object_center(sim, object_center: np.ndarray) -> np.ndarray:
    return np.asarray(object_center, dtype=float).reshape(3) - PHYSICAL_HELD_OFFSET - _hold_center_relative_to_ee(sim)


def _held_object_center(sim) -> np.ndarray:
    return np.asarray(_finger_geometry(sim)["center"], dtype=float).reshape(3) + PHYSICAL_HELD_OFFSET


def _make_camera() -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.02, 0.0, 0.20], dtype=float)
    cam.distance = 0.92
    cam.azimuth = 90
    cam.elevation = -24
    return cam


def _render_frame(renderer: mj.Renderer, sim, camera: mj.MjvCamera, lines: list[str]) -> np.ndarray:
    renderer.update_scene(sim.data, camera=camera)
    frame = renderer.render()
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, spacing=4)
    pad = 8
    draw.rectangle((10, 10, 10 + bbox[2] - bbox[0] + 2 * pad, 10 + bbox[3] - bbox[1] + 2 * pad), fill=(0, 0, 0))
    draw.multiline_text((10 + pad, 10 + pad), text, fill=(255, 255, 255), spacing=4)
    return np.asarray(image)


def _capture(renderer, sim, camera, frames, *, object_name: str, object_body: str, label: str, fit: dict[str, object]):
    ee_bid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "ee_base")
    ee = np.asarray(sim.data.xpos[ee_bid], dtype=float)
    obj = _object_position(sim, object_body)
    frames.append(
        _render_frame(
            renderer,
            sim,
            camera,
            [
                f"{object_name}: physical gripper contact",
                label,
                f"width={float(fit['object_projected_width_m']):.3f}m gap={float(fit['held_inner_gap_m']):.3f}m open={_get_gripper_opening(sim):.2f}",
                f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) obj=({obj[0]:+.2f},{obj[1]:+.2f},{obj[2]:.2f})",
            ],
        )
    )


def _move_ee(renderer, sim, camera, frames, *, object_name, object_body, label, fit, target, steps, opening):
    ee_bid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "ee_base")
    start = np.asarray(sim.data.xpos[ee_bid], dtype=float)
    target = np.asarray(target, dtype=float).reshape(3)
    for idx in range(max(1, int(steps))):
        alpha = (idx + 1) / float(max(1, int(steps)))
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        pos = (1.0 - smooth) * start + smooth * target
        _set_ee(sim, pos)
        _force_gripper_opening(sim, opening)
        _step(sim, steps=24, ee_pos=pos, opening=opening)
        _capture(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label=label, fit=fit)


def _render_pick(*, xml_path: Path, object_name: str, run_dir: Path, fps: float, compression: float, keep_frames: bool):
    sim = _sim_from_xml(xml_path)
    object_body = _body_name_for_object(sim, object_name)
    renderer = mj.Renderer(sim.model, height=480, width=640)
    camera = _make_camera()
    frames: list[np.ndarray] = []
    try:
        _place_body_bottom_on_z(sim, object_body, OBJECT_START_XY, TABLE_TOP_Z)
        _set_ee(sim, np.array([OBJECT_START_XY[0], OBJECT_START_XY[1], 0.35], dtype=float))
        fit = _measure_fit(sim, object_body, compression=compression)
        object_center = _body_aabb_center(sim, object_body)
        grasp_ee = _ee_for_object_center(sim, object_center)
        hover = grasp_ee + np.array([0.0, 0.0, 0.22], dtype=float)
        lift = grasp_ee + np.array([0.0, 0.0, 0.29], dtype=float)
        carry = lift + np.array([0.28, 0.12, 0.0], dtype=float)

        _force_gripper_opening(sim, 1.0)
        _set_ee(sim, hover)
        for _ in range(12):
            _capture(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="open above object", fit=fit)
            _step(sim, steps=12, ee_pos=hover, opening=1.0)

        _move_ee(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="descend around object", fit=fit, target=grasp_ee, steps=32, opening=1.0)

        for idx in range(22):
            alpha = (idx + 1) / 22.0
            opening = (1.0 - alpha) * 1.0 + alpha * float(fit["held_gripper_opening_01"])
            _force_gripper_opening(sim, opening)
            _step(sim, steps=24, ee_pos=grasp_ee, opening=opening)
            _capture(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="close fingers, physical contact", fit=fit)

        for _ in range(12):
            _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
            _step(sim, steps=24, ee_pos=grasp_ee, opening=float(fit["held_gripper_opening_01"]))
            _capture(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="grasp stabilized by contact", fit=fit)

        _move_ee(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="lift with contact", fit=fit, target=lift, steps=42, opening=float(fit["held_gripper_opening_01"]))
        _move_ee(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="move while held by friction/contact", fit=fit, target=carry, steps=48, opening=float(fit["held_gripper_opening_01"]))

        out = run_dir / f"physical_pick_lift_move_{object_name}.mp4"
        info = _write_video(frames, out, fps=fps, keep_frames=keep_frames)
        return {"object": object_name, "kind": "physical_pick_lift_move", "object_body": object_body, "final_object_position_m": [float(x) for x in _object_position(sim, object_body)], **fit, **info}
    finally:
        renderer.close()


def _render_caught_lift_move(
    *,
    xml_path: Path,
    object_name: str,
    run_dir: Path,
    fps: float,
    compression: float,
    keep_frames: bool,
):
    sim = _sim_from_xml(xml_path)
    object_body = _body_name_for_object(sim, object_name)
    renderer = mj.Renderer(sim.model, height=480, width=640)
    camera = _make_camera()
    frames: list[np.ndarray] = []
    try:
        _set_ee(sim, np.array([OBJECT_START_XY[0], OBJECT_START_XY[1], 0.35], dtype=float))
        fit = _measure_fit(sim, object_body, compression=compression)

        start_object_center = np.array([OBJECT_START_XY[0], OBJECT_START_XY[1], 0.22], dtype=float)
        start_ee = _ee_for_object_center(sim, start_object_center)
        lift = start_ee + np.array([0.0, 0.0, 0.08], dtype=float)
        carry = lift + np.array([0.18, 0.07, 0.0], dtype=float)

        _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
        _set_ee(sim, start_ee)
        _set_body_center(sim, object_body, _held_object_center(sim))
        for _ in range(36):
            _step(sim, steps=24, ee_pos=start_ee, opening=float(fit["held_gripper_opening_01"]))
            _capture(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="already caught, no pose pinning", fit=fit)

        _move_ee(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="slow lift while caught", fit=fit, target=lift, steps=160, opening=float(fit["held_gripper_opening_01"]))
        _move_ee(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="slow move while caught", fit=fit, target=carry, steps=180, opening=float(fit["held_gripper_opening_01"]))

        out = run_dir / f"physical_caught_lift_move_{object_name}.mp4"
        info = _write_video(frames, out, fps=fps, keep_frames=keep_frames)
        return {"object": object_name, "kind": "physical_caught_lift_move", "object_body": object_body, "final_object_position_m": [float(x) for x in _object_position(sim, object_body)], **fit, **info}
    finally:
        renderer.close()


def _render_release(*, xml_path: Path, object_name: str, run_dir: Path, fps: float, compression: float, keep_frames: bool):
    sim = _sim_from_xml(xml_path)
    object_body = _body_name_for_object(sim, object_name)
    renderer = mj.Renderer(sim.model, height=480, width=640)
    camera = _make_camera()
    frames: list[np.ndarray] = []
    try:
        _set_ee(sim, np.array([PLATE_CENTER[0], PLATE_CENTER[1], 0.35], dtype=float))
        fit = _measure_fit(sim, object_body, compression=compression)
        object_on_plate = _object_center_for_bottom_on_z(sim, object_body, PLATE_TOP_Z, clearance=0.002)
        object_on_plate[:2] = PLATE_CENTER[:2]
        release_ee = _ee_for_object_center(sim, object_on_plate)

        _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
        _set_ee(sim, release_ee)
        _set_body_center(sim, object_body, _held_object_center(sim))
        for _ in range(16):
            _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
            _step(sim, steps=24, ee_pos=release_ee, opening=float(fit["held_gripper_opening_01"]))
            _capture(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="initially held by contact", fit=fit)

        for idx in range(24):
            alpha = (idx + 1) / 24.0
            opening = (1.0 - alpha) * float(fit["held_gripper_opening_01"]) + alpha * 1.0
            _force_gripper_opening(sim, opening)
            _step(sim, steps=24, ee_pos=release_ee, opening=opening)
            _capture(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="open fingers, release contact", fit=fit)

        retreat = release_ee + np.array([0.0, 0.0, 0.18], dtype=float)
        _move_ee(renderer, sim, camera, frames, object_name=object_name, object_body=object_body, label="gripper retreats, object stays free", fit=fit, target=retreat, steps=32, opening=1.0)

        out = run_dir / f"physical_release_{object_name}.mp4"
        info = _write_video(frames, out, fps=fps, keep_frames=keep_frames)
        return {"object": object_name, "kind": "physical_release", "object_body": object_body, "final_object_position_m": [float(x) for x in _object_position(sim, object_body)], **fit, **info}
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render physical-contact YCB gripper pick/release videos.")
    parser.add_argument("--ycb-root", type=Path, default=DEFAULT_YCB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--grip-compression", type=float, default=0.002)
    parser.add_argument("--object-mass-scale", type=float, default=0.10)
    parser.add_argument("--object-freejoint-damping", type=float, default=0.0)
    parser.add_argument("--include-table-pick", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    videos: list[dict[str, object]] = []

    for object_name in args.objects:
        xml_path, object_xml = _build_physical_gripper_xml(
            ycb_root=args.ycb_root,
            object_name=str(object_name),
            output_dir=run_dir,
            object_mass_scale=float(args.object_mass_scale),
            object_freejoint_damping=float(args.object_freejoint_damping),
        )
        renderers = [_render_caught_lift_move, _render_release]
        if args.include_table_pick:
            renderers.insert(0, _render_pick)
        for fn in renderers:
            video = fn(
                xml_path=xml_path,
                object_name=str(object_name),
                run_dir=run_dir,
                fps=float(args.fps),
                compression=float(args.grip_compression),
                keep_frames=bool(args.keep_frames),
            )
            video["wrapper_xml"] = xml_path.as_posix()
            video["source_object_xml"] = object_xml.as_posix()
            videos.append(video)

    manifest = {
        "ycb_root": args.ycb_root.resolve().as_posix(),
        "grip_compression_m": float(args.grip_compression),
        "object_mass_scale": float(args.object_mass_scale),
        "object_freejoint_damping": float(args.object_freejoint_damping),
        "object_pose_pinning": False,
        "gripper_geometry": "original long fingers: palm, finger_l/r_link, finger_l/r_tip only",
        "videos": videos,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
