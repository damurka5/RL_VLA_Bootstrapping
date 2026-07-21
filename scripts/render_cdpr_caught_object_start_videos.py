#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_caught_object_start_videos"
CUBE_HALF_SIZE = 0.026
HELD_OFFSET = np.array([0.0, 0.0, -0.035], dtype=float)
PLATE_CENTER = np.array([0.16, 0.08, 0.012], dtype=float)


def _build_demo_xml(xml_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")

    if worldbody.find("./body[@name='demo_table']") is None:
        table = ET.SubElement(worldbody, "body", {"name": "demo_table", "pos": "0 0 -0.025"})
        ET.SubElement(
            table,
            "geom",
            {
                "name": "demo_table_top",
                "type": "box",
                "size": "0.48 0.36 0.025",
                "rgba": "0.72 0.72 0.68 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    if worldbody.find("./body[@name='demo_plate']") is None:
        plate = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "demo_plate",
                "pos": f"{PLATE_CENTER[0]:.4f} {PLATE_CENTER[1]:.4f} {PLATE_CENTER[2]:.4f}",
            },
        )
        ET.SubElement(
            plate,
            "geom",
            {
                "name": "demo_plate_geom",
                "type": "cylinder",
                "size": "0.085 0.006",
                "rgba": "0.05 0.32 0.85 0.85",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    if worldbody.find("./body[@name='target_object']") is None:
        cube = ET.SubElement(
            worldbody,
            "body",
            {"name": "target_object", "pos": f"0 0 {CUBE_HALF_SIZE:.4f}"},
        )
        ET.SubElement(cube, "freejoint", {"name": "target_object_free"})
        ET.SubElement(
            cube,
            "geom",
            {
                "name": "target_box",
                "type": "box",
                "size": f"{CUBE_HALF_SIZE:.4f} {CUBE_HALF_SIZE:.4f} {CUBE_HALF_SIZE:.4f}",
                "rgba": "0.88 0.18 0.08 1",
                "mass": "0.03",
                "friction": "1.6 0.02 0.01",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    out_path = output_dir / "cdpr_caught_object_start_demo.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path


def _import_headless_simulation():
    from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

    return HeadlessCDPRSimulation


def _set_free_body_pose(sim, position: np.ndarray) -> None:
    joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, "target_object_free")
    if joint_id == -1:
        raise RuntimeError("Could not find target_object_free joint.")
    qadr = int(sim.model.jnt_qposadr[joint_id])
    dadr = int(sim.model.jnt_dofadr[joint_id])
    sim.data.qpos[qadr : qadr + 3] = np.asarray(position, dtype=float).reshape(3)
    sim.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    mj.mj_forward(sim.model, sim.data)


def _object_position(sim) -> np.ndarray:
    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "target_object")
    return np.asarray(sim.data.xpos[body_id], dtype=float).copy()


def _force_gripper_opening(sim, opening_01: float) -> None:
    target = float(np.clip(opening_01, 0.0, 1.0))
    sim.set_gripper(target)
    qadr = getattr(sim, "jnt_finger_l_qadr", None)
    if qadr is None:
        mj.mj_forward(sim.model, sim.data)
        return
    joint_min = float(getattr(sim, "gripper_joint_min", 0.0))
    joint_max = float(getattr(sim, "gripper_joint_max", 0.03))
    sim.data.qpos[int(qadr)] = joint_min + target * max(joint_max - joint_min, 0.0)
    joint_id = getattr(sim, "jnt_finger_l", None)
    if joint_id is not None:
        dofadr = int(sim.model.jnt_dofadr[int(joint_id)])
        sim.data.qvel[dofadr] = 0.0
    mj.mj_forward(sim.model, sim.data)


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
    else:
        radius = float(sim.model.geom_rbound[gid])
        half_local = np.array([radius, radius, radius], dtype=float)
    xmat = np.asarray(sim.data.geom_xmat[gid], dtype=float).reshape(3, 3)
    return float(np.sum(np.abs(xmat.T @ axis) * half_local))


def _held_gripper_opening(sim, *, clearance: float = 0.001) -> float:
    _force_gripper_opening(sim, 0.0)
    left_gid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "finger_l_tip")
    right_gid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, "finger_r_tip")
    left = np.asarray(sim.data.geom_xpos[left_gid], dtype=float)
    right = np.asarray(sim.data.geom_xpos[right_gid], dtype=float)
    axis = left - right
    distance = float(np.linalg.norm(axis))
    axis /= max(distance, 1e-9)
    closed_gap = max(
        0.0,
        distance
        - _geom_half_extent_along_axis(sim, left_gid, axis)
        - _geom_half_extent_along_axis(sim, right_gid, axis),
    )
    desired_gap = 2.0 * CUBE_HALF_SIZE + 2.0 * float(clearance)
    joint_span = max(float(sim.gripper_joint_max - sim.gripper_joint_min), 1e-6)
    return float(np.clip((desired_gap - closed_gap) / (2.0 * joint_span), 0.0, 1.0))


def _held_object_position(sim) -> np.ndarray:
    ee_pos = np.asarray(sim.get_end_effector_position(), dtype=float)
    return ee_pos + HELD_OFFSET


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


def _render_sequence(
    *,
    xml_path: Path,
    run_dir: Path,
    filename: str,
    title: str,
    waypoints: list[tuple[str, np.ndarray, int]],
    release_at_end: bool,
    fps: float,
    keep_frames: bool,
) -> dict[str, object]:
    HeadlessCDPRSimulation = _import_headless_simulation()
    sim = HeadlessCDPRSimulation(str(xml_path), output_dir=str(run_dir), record_trajectory=False)
    frames: list[np.ndarray] = []
    min_distance = float("inf")

    def capture(label: str, held_opening: float, attached: bool) -> None:
        nonlocal min_distance
        if attached:
            _set_free_body_pose(sim, _held_object_position(sim))
        obj = _object_position(sim)
        ee = np.asarray(sim.get_end_effector_position(), dtype=float)
        min_distance = min(min_distance, float(np.linalg.norm(obj - _held_object_position(sim))))
        frame = sim.capture_frame(sim.overview_cam, "overview")
        frames.append(
            _annotate(
                frame,
                [
                    title,
                    label,
                    f"held_opening={held_opening:.2f} actual_gripper={sim.get_gripper_opening():.2f}",
                    f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) obj=({obj[0]:+.2f},{obj[1]:+.2f},{obj[2]:.2f})",
                ],
            )
        )

    try:
        sim.initialize()
        held_opening = _held_gripper_opening(sim)
        _force_gripper_opening(sim, held_opening)
        _set_free_body_pose(sim, _held_object_position(sim))
        for _ in range(12):
            _force_gripper_opening(sim, held_opening)
            sim.run_simulation_step(capture_frame=False)
            _set_free_body_pose(sim, _held_object_position(sim))
            capture("already captured: fitted between fingers", held_opening, True)

        attached = True
        for label, target, steps in waypoints:
            start = np.asarray(sim.get_end_effector_position(), dtype=float)
            for idx in range(max(1, int(steps))):
                alpha = (idx + 1) / float(max(1, int(steps)))
                target_pos = (1.0 - alpha) * start + alpha * np.asarray(target, dtype=float)
                _force_gripper_opening(sim, held_opening)
                sim.set_target_position(target_pos)
                sim.run_simulation_step(capture_frame=False)
                capture(label, held_opening, attached)

        if release_at_end:
            attached = False
            drop_pos = np.array([PLATE_CENTER[0], PLATE_CENTER[1], PLATE_CENTER[2] + CUBE_HALF_SIZE], dtype=float)
            for _ in range(16):
                _force_gripper_opening(sim, 1.0)
                _set_free_body_pose(sim, drop_pos)
                sim.run_simulation_step(capture_frame=False)
                capture("release after placement", held_opening, attached)

        output_path = run_dir / filename
        info = _write_video(frames, output_path, fps=fps, keep_frames=keep_frames)
        final_obj = _object_position(sim)
        return {
            "title": title,
            "held_gripper_opening": float(held_opening),
            "final_object_position": [float(x) for x in final_obj],
            "min_hold_pose_error_m": float(min_distance),
            **info,
        }
    finally:
        sim.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render MuJoCo videos for caught-object-start behavior.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _build_demo_xml(args.xml_path, run_dir)

    lateral = _render_sequence(
        xml_path=xml_path,
        run_dir=run_dir,
        filename="caught_object_start_lateral_hold.mp4",
        title="Caught-object start: lateral carry without slip",
        waypoints=[
            ("carry left", np.array([-0.18, -0.04, 0.40], dtype=float), 36),
            ("carry right", np.array([0.20, 0.05, 0.40], dtype=float), 54),
            ("return center", np.array([0.0, 0.0, 0.42], dtype=float), 36),
        ],
        release_at_end=False,
        fps=float(args.fps),
        keep_frames=bool(args.keep_frames),
    )
    plate = _render_sequence(
        xml_path=xml_path,
        run_dir=run_dir,
        filename="caught_object_start_put_into_plate.mp4",
        title="Caught-object start: carry to plate then release",
        waypoints=[
            ("lift held object", np.array([0.0, 0.0, 0.48], dtype=float), 28),
            ("carry to plate", np.array([PLATE_CENTER[0], PLATE_CENTER[1], 0.46], dtype=float), 52),
            ("lower into plate", np.array([PLATE_CENTER[0], PLATE_CENTER[1], 0.20], dtype=float), 34),
        ],
        release_at_end=True,
        fps=float(args.fps),
        keep_frames=bool(args.keep_frames),
    )

    manifest = {"xml": xml_path.as_posix(), "videos": [lateral, plate]}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
