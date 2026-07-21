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

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CDPR_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_ycb_caught_object_videos"
DEFAULT_YCB_ROOT = (
    ROOT.parent.parent / "CDPR-Dataset" / "external_assets" / "ycb_dataset" / "ycb"
)
DEFAULT_OBJECTS = ("ycb_apple", "ycb_pear", "ycb_peach", "ycb_baseball")

HELD_OFFSET = np.array([0.0, 0.0, -0.035], dtype=float)
PLATE_CENTER = np.array([0.16, 0.08, 0.012], dtype=float)


def _load_scene_switcher(ycb_root: Path):
    os.environ["YCB_ASSETS"] = str(ycb_root.resolve())
    from robots.cdpr.cdpr_mujoco import cdpr_scene_switcher as switcher

    switcher.YCB_ROOT = ycb_root.resolve()
    return switcher


def _import_headless_simulation():
    from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

    return HeadlessCDPRSimulation


def _build_wrapper_xml(
    *,
    cdpr_xml: Path,
    ycb_root: Path,
    object_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    switcher = _load_scene_switcher(ycb_root)
    object_xml = switcher.find_object_xml(object_name)
    placed_xml = output_dir / f"{object_name}_placed.xml"
    wrapper_xml = output_dir / f"{object_name}_caught_start_wrapper.xml"

    switcher.make_placed_object_xml(
        object_xml,
        placed_xml,
        prefix="p0",
        pos=np.array([0.0, 0.0, 0.10], dtype=float),
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        force_dynamic=True,
        logical_name=object_name,
    )

    wrapper_xml.write_text(
        f"""<mujoco>
  <compiler autolimits="true"/>

  <include file="{cdpr_xml.resolve()}"/>

  <worldbody>
    <light name="demo_key_light" pos="0 -0.5 1.2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <body name="demo_table" pos="0 0 -0.025">
      <geom name="demo_table_top" type="box" size="0.48 0.36 0.025"
            rgba="0.72 0.72 0.68 1" contype="1" conaffinity="1"/>
    </body>
    <body name="demo_plate" pos="{PLATE_CENTER[0]:.4f} {PLATE_CENTER[1]:.4f} {PLATE_CENTER[2]:.4f}">
      <geom name="demo_plate_geom" type="cylinder" size="0.085 0.006"
            rgba="0.05 0.32 0.85 0.85" contype="1" conaffinity="1"/>
    </body>
  </worldbody>

  <include file="{placed_xml.resolve()}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return wrapper_xml, object_xml


def _body_geoms(sim, body_name: str) -> list[int]:
    model = sim.model
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise RuntimeError(f"Could not find body {body_name!r}.")

    children = {idx: [] for idx in range(model.nbody)}
    for idx in range(1, model.nbody):
        parent = int(model.body_parentid[idx])
        if parent >= 0:
            children.setdefault(parent, []).append(idx)

    stack = [int(body_id)]
    body_ids: set[int] = set()
    while stack:
        bid = int(stack.pop())
        body_ids.add(bid)
        stack.extend(children.get(bid, ()))
    return [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in body_ids]


def _geom_half_extent_along_axis(sim, geom_id: int, axis: np.ndarray) -> float:
    gid = int(geom_id)
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    gtype = int(sim.model.geom_type[gid])
    size = np.asarray(sim.model.geom_size[gid], dtype=float)
    if gtype == int(mj.mjtGeom.mjGEOM_BOX):
        half_local = np.array([size[0], size[1], size[2]], dtype=float)
    elif gtype == int(mj.mjtGeom.mjGEOM_CAPSULE):
        half_local = np.array([size[0], size[0], size[1] + size[0]], dtype=float)
    elif gtype == int(mj.mjtGeom.mjGEOM_CYLINDER):
        half_local = np.array([size[0], size[0], size[1]], dtype=float)
    elif gtype == int(mj.mjtGeom.mjGEOM_SPHERE):
        half_local = np.array([size[0], size[0], size[0]], dtype=float)
    else:
        radius = float(sim.model.geom_rbound[gid])
        half_local = np.array([radius, radius, radius], dtype=float)
    xmat = np.asarray(sim.data.geom_xmat[gid], dtype=float).reshape(3, 3)
    return float(np.sum(np.abs(xmat.T @ axis) * half_local))


def _geom_interval_along_axis(sim, geom_id: int, axis: np.ndarray) -> tuple[float, float]:
    gid = int(geom_id)
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    center = np.asarray(sim.data.geom_xpos[gid], dtype=float).reshape(3)
    gtype = int(sim.model.geom_type[gid])
    mesh_id = int(sim.model.geom_dataid[gid]) if hasattr(sim.model, "geom_dataid") else -1
    if gtype == int(mj.mjtGeom.mjGEOM_MESH) and mesh_id >= 0:
        start = int(sim.model.mesh_vertadr[mesh_id])
        count = int(sim.model.mesh_vertnum[mesh_id])
        if count > 0:
            verts = np.asarray(sim.model.mesh_vert[start : start + count], dtype=float)
            xmat = np.asarray(sim.data.geom_xmat[gid], dtype=float).reshape(3, 3)
            projected = (center + verts @ xmat.T) @ axis
            return float(np.min(projected)), float(np.max(projected))

    center_projection = float(center @ axis)
    half_extent = _geom_half_extent_along_axis(sim, gid, axis)
    return center_projection - half_extent, center_projection + half_extent


def _body_width_along_axis(sim, body_name: str, axis: np.ndarray) -> float:
    lo = float("inf")
    hi = float("-inf")
    for gid in _body_geoms(sim, body_name):
        geom_lo, geom_hi = _geom_interval_along_axis(sim, gid, axis)
        lo = min(lo, geom_lo)
        hi = max(hi, geom_hi)
    width = float(hi - lo)
    if not np.isfinite(width) or width <= 0.0:
        raise RuntimeError(f"Could not measure projected width for {body_name!r}.")
    return width


def _force_gripper_opening(sim, opening_01: float) -> None:
    target = float(np.clip(opening_01, 0.0, 1.0))
    sim.set_gripper(target)

    joint_min = float(getattr(sim, "gripper_joint_min", 0.0))
    joint_max = float(getattr(sim, "gripper_joint_max", 0.03))
    joint_pos = joint_min + target * max(joint_max - joint_min, 0.0)
    for joint_name in ("finger_l", "finger_r"):
        joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            continue
        qadr = int(sim.model.jnt_qposadr[joint_id])
        dofadr = int(sim.model.jnt_dofadr[joint_id])
        sim.data.qpos[qadr] = float(joint_pos)
        sim.data.qvel[dofadr] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _finger_geometry(sim) -> dict[str, object]:
    left_gid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "finger_l_tip")
    right_gid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "finger_r_tip")
    if left_gid == -1 or right_gid == -1:
        raise RuntimeError("Could not find finger_l_tip/finger_r_tip geoms.")

    left = np.asarray(sim.data.geom_xpos[left_gid], dtype=float).reshape(3)
    right = np.asarray(sim.data.geom_xpos[right_gid], dtype=float).reshape(3)
    separation = left - right
    distance = float(np.linalg.norm(separation))
    axis = separation / max(distance, 1e-9)
    inner_gap = max(
        0.0,
        distance
        - _geom_half_extent_along_axis(sim, left_gid, axis)
        - _geom_half_extent_along_axis(sim, right_gid, axis),
    )
    return {
        "left_gid": int(left_gid),
        "right_gid": int(right_gid),
        "left_tip_center": left,
        "right_tip_center": right,
        "center": 0.5 * (left + right),
        "axis": axis,
        "inner_gap": float(inner_gap),
    }


def _measure_fit(sim, object_body: str, clearance: float) -> dict[str, object]:
    _force_gripper_opening(sim, 0.0)
    closed = _finger_geometry(sim)
    width = _body_width_along_axis(sim, object_body, np.asarray(closed["axis"], dtype=float))
    desired_gap = float(width + 2.0 * max(0.0, clearance))
    joint_span = max(float(sim.gripper_joint_max - sim.gripper_joint_min), 1e-6)
    raw_opening = (desired_gap - float(closed["inner_gap"])) / (2.0 * joint_span)
    opening = float(np.clip(raw_opening, 0.0, 1.0))

    _force_gripper_opening(sim, opening)
    held = _finger_geometry(sim)
    target_joint_qpos = float(sim.gripper_joint_min + opening * joint_span)
    return {
        "object_projected_width_m": float(width),
        "clearance_m": float(clearance),
        "desired_inner_gap_m": float(desired_gap),
        "closed_inner_gap_m": float(closed["inner_gap"]),
        "held_inner_gap_m": float(held["inner_gap"]),
        "raw_gripper_opening_01": float(raw_opening),
        "held_gripper_opening_01": float(opening),
        "target_finger_joint_qpos_m": target_joint_qpos,
        "joint_span_m": float(joint_span),
        "fits_without_clipping": bool(0.0 <= raw_opening <= 1.0),
        "finger_axis": [float(x) for x in np.asarray(held["axis"], dtype=float)],
        "left_tip_center_m": [float(x) for x in np.asarray(held["left_tip_center"], dtype=float)],
        "right_tip_center_m": [float(x) for x in np.asarray(held["right_tip_center"], dtype=float)],
        "hold_center_m": [float(x) for x in np.asarray(held["center"], dtype=float)],
    }


def _set_body_pose(sim, body_name: str, position: np.ndarray) -> None:
    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise RuntimeError(f"Could not find body {body_name!r}.")
    joint_count = int(sim.model.body_jntnum[body_id])
    joint_start = int(sim.model.body_jntadr[body_id])
    free_joint = -1
    for idx in range(joint_count):
        joint_id = joint_start + idx
        if int(sim.model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_FREE):
            free_joint = joint_id
            break
    if free_joint == -1:
        raise RuntimeError(f"Body {body_name!r} has no free joint.")

    qadr = int(sim.model.jnt_qposadr[free_joint])
    dadr = int(sim.model.jnt_dofadr[free_joint])
    current_quat = np.asarray(sim.data.qpos[qadr + 3 : qadr + 7], dtype=float).copy()
    if float(np.linalg.norm(current_quat)) < 1e-9:
        current_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    sim.data.qpos[qadr : qadr + 3] = np.asarray(position, dtype=float).reshape(3)
    sim.data.qpos[qadr + 3 : qadr + 7] = current_quat / max(float(np.linalg.norm(current_quat)), 1e-9)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _set_ee_position(sim, position: np.ndarray) -> None:
    joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, "ee_free")
    if joint_id == -1:
        raise RuntimeError("Could not find ee_free joint.")
    qadr = int(sim.model.jnt_qposadr[joint_id])
    dadr = int(sim.model.jnt_dofadr[joint_id])
    current_quat = np.asarray(sim.data.qpos[qadr + 3 : qadr + 7], dtype=float).copy()
    if float(np.linalg.norm(current_quat)) < 1e-9:
        current_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    sim.data.qpos[qadr : qadr + 3] = np.asarray(position, dtype=float).reshape(3)
    sim.data.qpos[qadr + 3 : qadr + 7] = current_quat / max(float(np.linalg.norm(current_quat)), 1e-9)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _object_position(sim, body_name: str) -> np.ndarray:
    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise RuntimeError(f"Could not find body {body_name!r}.")
    return np.asarray(sim.data.xpos[body_id], dtype=float).copy()


def _hold_position(sim, fit: dict[str, object]) -> np.ndarray:
    center = np.asarray(_finger_geometry(sim)["center"], dtype=float).reshape(3)
    hold_offset = np.asarray(fit["hold_offset_m"], dtype=float).reshape(3)
    return center + hold_offset


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, spacing=4)
    pad = 8
    rect = (10, 10, 10 + bbox[2] - bbox[0] + 2 * pad, 10 + bbox[3] - bbox[1] + 2 * pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.multiline_text((10 + pad, 10 + pad), text, fill=(255, 255, 255), spacing=4)
    return np.asarray(image)


def _write_video(frames: list[np.ndarray], output_path: Path, *, fps: float, keep_frames: bool) -> dict[str, object]:
    if not frames:
        raise RuntimeError("No frames were captured.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
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
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return {"video": output_path.as_posix(), "frames": len(frames)}


def _render_object_video(
    *,
    wrapper_xml: Path,
    object_name: str,
    run_dir: Path,
    fps: float,
    clearance: float,
    keep_frames: bool,
) -> dict[str, object]:
    HeadlessCDPRSimulation = _import_headless_simulation()
    sim = HeadlessCDPRSimulation(str(wrapper_xml), output_dir=str(run_dir), record_trajectory=False)
    frames: list[np.ndarray] = []
    min_hold_error = float("inf")

    def capture(label: str, fit: dict[str, object]) -> None:
        nonlocal min_hold_error
        _set_body_pose(sim, object_body, _hold_position(sim, fit))
        _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
        obj = _object_position(sim, object_body)
        hold = _hold_position(sim, fit)
        ee = np.asarray(sim.get_end_effector_position(), dtype=float)
        min_hold_error = min(min_hold_error, float(np.linalg.norm(obj - hold)))
        frame = sim.capture_frame(sim.overview_cam, "overview")
        frames.append(
            _annotate(
                frame,
                [
                    f"{object_name}: YCB caught-object start",
                    label,
                    f"width={float(fit['object_projected_width_m']):.3f}m gap={float(fit['held_inner_gap_m']):.3f}m open={float(fit['held_gripper_opening_01']):.2f}",
                    f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) obj=({obj[0]:+.2f},{obj[1]:+.2f},{obj[2]:.2f})",
                ],
            )
        )

    try:
        sim.initialize()
        sim.overview_cam.lookat[:] = np.array([0.0, 0.0, 0.28], dtype=float)
        sim.overview_cam.distance = 1.0
        sim.overview_cam.azimuth = 90
        sim.overview_cam.elevation = -22

        object_body = str(sim.get_object_body_name())
        fit = _measure_fit(sim, object_body, clearance=clearance)
        hold_center = np.asarray(fit["hold_center_m"], dtype=float).reshape(3)
        target_pos = hold_center + HELD_OFFSET
        _set_body_pose(sim, object_body, target_pos)
        fit["hold_offset_m"] = [float(x) for x in (target_pos - hold_center)]

        for _ in range(12):
            _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
            capture("already captured: fitted between fingers", fit)

        waypoints = [
            ("carry left", np.array([-0.18, -0.04, 0.42], dtype=float), 30),
            ("carry right", np.array([0.20, 0.05, 0.42], dtype=float), 40),
            ("return center", np.array([0.0, 0.0, 0.44], dtype=float), 30),
        ]
        for label, target, steps in waypoints:
            start = np.asarray(sim.get_end_effector_position(), dtype=float)
            for idx in range(max(1, int(steps))):
                alpha = (idx + 1) / float(max(1, int(steps)))
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                target_pos = (1.0 - smooth) * start + smooth * np.asarray(target, dtype=float)
                _set_ee_position(sim, target_pos)
                _force_gripper_opening(sim, float(fit["held_gripper_opening_01"]))
                capture(label, fit)

        video_path = run_dir / f"caught_start_{object_name}.mp4"
        info = _write_video(frames, video_path, fps=fps, keep_frames=keep_frames)
        fit["final_object_position_m"] = [float(x) for x in _object_position(sim, object_body)]
        return {
            "object": object_name,
            "object_body": object_body,
            "wrapper_xml": wrapper_xml.as_posix(),
            "min_hold_pose_error_m": float(min_hold_error),
            **fit,
            **info,
        }
    finally:
        sim.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render YCB caught-object-start CDPR videos.")
    parser.add_argument("--cdpr-xml", type=Path, default=DEFAULT_CDPR_XML)
    parser.add_argument("--ycb-root", type=Path, default=DEFAULT_YCB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--clearance", type=float, default=0.001)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    videos: list[dict[str, object]] = []
    for object_name in args.objects:
        wrapper_xml, object_xml = _build_wrapper_xml(
            cdpr_xml=args.cdpr_xml,
            ycb_root=args.ycb_root,
            object_name=str(object_name),
            output_dir=run_dir,
        )
        video = _render_object_video(
            wrapper_xml=wrapper_xml,
            object_name=str(object_name),
            run_dir=run_dir,
            fps=float(args.fps),
            clearance=float(args.clearance),
            keep_frames=bool(args.keep_frames),
        )
        video["source_object_xml"] = object_xml.as_posix()
        videos.append(video)

    manifest = {
        "cdpr_xml": args.cdpr_xml.resolve().as_posix(),
        "ycb_root": args.ycb_root.resolve().as_posix(),
        "clearance_m": float(args.clearance),
        "videos": videos,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
