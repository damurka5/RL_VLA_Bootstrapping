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
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_short_horizon_curriculum_videos"
CUBE_HALF_SIZE = 0.024
HELD_OFFSET = np.array([0.0, 0.0, -0.035], dtype=float)
PLATE_CENTER = np.array([0.16, 0.08, 0.016], dtype=float)
TABLE_Z = -0.025
EE_START = np.array([0.0, 0.0, 0.40], dtype=float)


class DirectCDPRDemo:
    """Tiny MuJoCo renderer/controller for near-success curriculum videos."""

    def __init__(self, xml_path: Path, *, width: int, height: int) -> None:
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.width = int(width)
        self.height = int(height)
        self.renderer: mj.Renderer | None = None
        self.overview_cam = mj.MjvCamera()
        self.overview_cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.overview_cam.lookat[:] = np.array([0.0, 0.0, 0.20], dtype=float)
        self.overview_cam.distance = 1.35
        self.overview_cam.azimuth = 90.0
        self.overview_cam.elevation = -32.0

        self.body_ee = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "ee_base")
        if self.body_ee == -1:
            raise RuntimeError("Could not find ee_base body.")
        self.jnt_ee_free = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ee_free")
        if self.jnt_ee_free == -1:
            raise RuntimeError("Could not find ee_free joint.")
        self.jnt_finger_l = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "finger_l")
        self.jnt_finger_r = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "finger_r")
        if self.jnt_finger_l == -1:
            raise RuntimeError("Could not find finger_l joint.")
        self.jnt_finger_l_qadr = int(self.model.jnt_qposadr[self.jnt_finger_l])
        q_lo, q_hi = self.model.jnt_range[self.jnt_finger_l]
        self.gripper_joint_min = float(min(q_lo, q_hi))
        self.gripper_joint_max = float(max(q_lo, q_hi))
        if self.gripper_joint_max <= self.gripper_joint_min:
            self.gripper_joint_min = 0.0
            self.gripper_joint_max = 0.03
        self.act_gripper = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")

    def initialize(self) -> None:
        self.renderer = mj.Renderer(self.model, height=self.height, width=self.width)
        mj.mj_resetData(self.model, self.data)
        self.set_target_position(EE_START)
        self.set_gripper(0.75)
        mj.mj_forward(self.model, self.data)

    def cleanup(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_end_effector_position(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.body_ee], dtype=float).copy()

    def set_target_position(self, target_pos: np.ndarray) -> bool:
        qadr = int(self.model.jnt_qposadr[self.jnt_ee_free])
        dadr = int(self.model.jnt_dofadr[self.jnt_ee_free])
        self.data.qpos[qadr : qadr + 3] = np.asarray(target_pos, dtype=float).reshape(3)
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[dadr : dadr + 6] = 0.0
        mj.mj_forward(self.model, self.data)
        return True

    def set_gripper(self, opening_01: float) -> None:
        opening = float(np.clip(opening_01, 0.0, 1.0))
        joint_pos = self.gripper_joint_min + opening * (self.gripper_joint_max - self.gripper_joint_min)
        for joint_id in (self.jnt_finger_l, self.jnt_finger_r):
            if int(joint_id) == -1:
                continue
            qadr = int(self.model.jnt_qposadr[int(joint_id)])
            dadr = int(self.model.jnt_dofadr[int(joint_id)])
            self.data.qpos[qadr] = joint_pos
            self.data.qvel[dadr] = 0.0
        if self.act_gripper != -1:
            self.data.ctrl[self.act_gripper] = opening
        mj.mj_forward(self.model, self.data)

    def get_gripper_opening(self) -> float:
        span = max(self.gripper_joint_max - self.gripper_joint_min, 1e-9)
        return float(np.clip((self.data.qpos[self.jnt_finger_l_qadr] - self.gripper_joint_min) / span, 0.0, 1.0))

    def run_simulation_step(self, *, capture_frame: bool = False) -> None:
        del capture_frame
        mj.mj_forward(self.model, self.data)

    def capture_frame(self, camera: mj.MjvCamera, camera_name: str) -> np.ndarray:
        del camera_name
        if self.renderer is None:
            raise RuntimeError("Renderer is not initialized.")
        self.renderer.update_scene(self.data, camera=camera)
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()


def _build_demo_xml(xml_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")

    if worldbody.find("./body[@name='curriculum_table']") is None:
        table = ET.SubElement(worldbody, "body", {"name": "curriculum_table", "pos": f"0 0 {TABLE_Z:.4f}"})
        ET.SubElement(
            table,
            "geom",
            {
                "name": "curriculum_table_top",
                "type": "box",
                "size": "0.50 0.38 0.025",
                "rgba": "0.72 0.72 0.68 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    if worldbody.find("./body[@name='curriculum_plate']") is None:
        plate = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "curriculum_plate",
                "pos": f"{PLATE_CENTER[0]:.4f} {PLATE_CENTER[1]:.4f} {PLATE_CENTER[2]:.4f}",
            },
        )
        ET.SubElement(
            plate,
            "geom",
            {
                "name": "curriculum_plate_geom",
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

    out_path = output_dir / "cdpr_short_horizon_curriculum_demo.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path


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


def _geom_half_extent_along_axis(sim, geom_id: int, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    geom_id = int(geom_id)
    geom_type = int(sim.model.geom_type[geom_id])
    size = np.asarray(sim.model.geom_size[geom_id], dtype=float)
    if geom_type == int(mj.mjtGeom.mjGEOM_BOX):
        half_local = np.array([size[0], size[1], size[2]], dtype=float)
    elif geom_type == int(mj.mjtGeom.mjGEOM_CAPSULE):
        half_local = np.array([size[0], size[0], size[1] + size[0]], dtype=float)
    else:
        radius = float(sim.model.geom_rbound[geom_id])
        half_local = np.array([radius, radius, radius], dtype=float)
    xmat = np.asarray(sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
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
    for index, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(frame_dir / f"{index:05d}.png")
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


def _capture(sim, frames: list[np.ndarray], *, title: str, phase: str, success: str) -> None:
    obj = _object_position(sim)
    ee = np.asarray(sim.get_end_effector_position(), dtype=float)
    frame = sim.capture_frame(sim.overview_cam, "overview")
    frames.append(
        _annotate(
            frame,
            [
                title,
                phase,
                success,
                f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) obj=({obj[0]:+.2f},{obj[1]:+.2f},{obj[2]:.2f})",
                f"gripper={sim.get_gripper_opening():.2f}",
            ],
        )
    )


def _drive_ee(sim, target: np.ndarray, *, frames: list[np.ndarray], title: str, phase: str, steps: int, attached: bool, gripper: float, success: str) -> None:
    start = np.asarray(sim.get_end_effector_position(), dtype=float)
    for index in range(max(1, int(steps))):
        alpha = float(index + 1) / float(max(1, int(steps)))
        pos = (1.0 - alpha) * start + alpha * np.asarray(target, dtype=float)
        _force_gripper_opening(sim, gripper)
        sim.set_target_position(pos)
        sim.run_simulation_step(capture_frame=False)
        if attached:
            _set_free_body_pose(sim, _held_object_position(sim))
        _capture(sim, frames, title=title, phase=phase, success=success)


def _render_case(*, xml_path: Path, run_dir: Path, filename: str, title: str, case: str, fps: float, keep_frames: bool) -> dict[str, object]:
    sim = DirectCDPRDemo(xml_path, width=640, height=480)
    frames: list[np.ndarray] = []
    try:
        sim.initialize()
        held_opening = _held_gripper_opening(sim)
        open_near_object = min(1.0, max(held_opening + 0.20, 0.75))
        sim.set_target_position(EE_START)
        for _ in range(10):
            sim.run_simulation_step(capture_frame=False)

        if case == "catch":
            _force_gripper_opening(sim, open_near_object)
            _set_free_body_pose(sim, _held_object_position(sim))
            for _ in range(16):
                sim.run_simulation_step(capture_frame=False)
                _capture(
                    sim,
                    frames,
                    title=title,
                    phase="START: object already centered between open fingers",
                    success="Expected action: close gripper",
                )
            for step in range(24):
                value = open_near_object + (held_opening - open_near_object) * ((step + 1) / 24.0)
                _force_gripper_opening(sim, value)
                sim.run_simulation_step(capture_frame=False)
                _set_free_body_pose(sim, _held_object_position(sim))
                _capture(sim, frames, title=title, phase="closing around object", success="SUCCESS: object gripped")

        elif case == "pick_up":
            _force_gripper_opening(sim, held_opening)
            _set_free_body_pose(sim, _held_object_position(sim))
            _capture(sim, frames, title=title, phase="START: object already caught", success="Expected action: lift upward")
            _drive_ee(
                sim,
                np.array([0.0, 0.0, 0.55], dtype=float),
                frames=frames,
                title=title,
                phase="lifting held object",
                steps=42,
                attached=True,
                gripper=held_opening,
                success="SUCCESS: object lifted above start height",
            )

        elif case == "release":
            _force_gripper_opening(sim, held_opening)
            _set_free_body_pose(sim, _held_object_position(sim))
            for _ in range(12):
                sim.run_simulation_step(capture_frame=False)
                _set_free_body_pose(sim, _held_object_position(sim))
                _capture(sim, frames, title=title, phase="START: object held in gripper", success="Expected action: open fingers")
            drop_pos = _held_object_position(sim).copy()
            drop_pos[2] = max(CUBE_HALF_SIZE, TABLE_Z + 0.025 + CUBE_HALF_SIZE)
            for step in range(30):
                value = held_opening + (1.0 - held_opening) * ((step + 1) / 30.0)
                _force_gripper_opening(sim, value)
                _set_free_body_pose(sim, drop_pos)
                sim.run_simulation_step(capture_frame=False)
                _capture(sim, frames, title=title, phase="opening fingers and dropping object", success="SUCCESS: object released")

        elif case == "put_into_plate":
            _force_gripper_opening(sim, held_opening)
            _set_free_body_pose(sim, _held_object_position(sim))
            _capture(sim, frames, title=title, phase="START: object held near the plate", success="Expected action: carry, lower, release")
            _drive_ee(
                sim,
                np.array([PLATE_CENTER[0], PLATE_CENTER[1], 0.44], dtype=float),
                frames=frames,
                title=title,
                phase="carry above plate",
                steps=42,
                attached=True,
                gripper=held_opening,
                success="approaching container",
            )
            _drive_ee(
                sim,
                np.array([PLATE_CENTER[0], PLATE_CENTER[1], 0.20], dtype=float),
                frames=frames,
                title=title,
                phase="lower into plate",
                steps=28,
                attached=True,
                gripper=held_opening,
                success="ready to release",
            )
            final_pos = np.array([PLATE_CENTER[0], PLATE_CENTER[1], PLATE_CENTER[2] + CUBE_HALF_SIZE], dtype=float)
            for step in range(24):
                value = held_opening + (1.0 - held_opening) * ((step + 1) / 24.0)
                _force_gripper_opening(sim, value)
                _set_free_body_pose(sim, final_pos)
                sim.run_simulation_step(capture_frame=False)
                _capture(sim, frames, title=title, phase="release in plate", success="SUCCESS: object inside plate")
        else:
            raise ValueError(f"Unknown case: {case}")

        info = _write_video(frames, run_dir / filename, fps=fps, keep_frames=keep_frames)
        info.update(
            {
                "case": case,
                "title": title,
                "held_gripper_opening": float(held_opening),
                "final_object_position": [float(x) for x in _object_position(sim)],
            }
        )
        return info
    finally:
        sim.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render CDPR short-horizon curriculum expectation videos.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _build_demo_xml(args.xml_path, run_dir)
    cases = [
        ("near_success_catch_object.mp4", "near-success catch object", "catch"),
        ("near_success_pick_up.mp4", "near-success pick up object", "pick_up"),
        ("near_success_release_object.mp4", "near-success release/drop object", "release"),
        ("near_success_put_into_plate.mp4", "near-success put object into plate", "put_into_plate"),
    ]
    videos = [
        _render_case(
            xml_path=xml_path,
            run_dir=run_dir,
            filename=filename,
            title=title,
            case=case,
            fps=float(args.fps),
            keep_frames=bool(args.keep_frames),
        )
        for filename, title, case in cases
    ]
    manifest = {"xml": xml_path.as_posix(), "videos": videos}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
