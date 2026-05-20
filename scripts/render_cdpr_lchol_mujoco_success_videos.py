#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_lchol_mujoco_success_criteria_videos"

SUPPORT_Z = 0.15
TARGET_HALF_SIZE = 0.026
PLATE_HALF_HEIGHT = 0.006

CURRICULUM_INSTRUCTIONS = (
    "move_to_object",
    "grab_object",
    "pick_up",
    "push_left",
    "push_right",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_between_objects",
)

TEXT = {
    "move_to_object": "move to apple",
    "grab_object": "grab apple",
    "pick_up": "pick up apple",
    "push_left": "push apple left",
    "push_right": "push apple right",
    "put_into_plate": "put apple into plate",
    "move_left_of_object": "move apple to the left of pear",
    "move_right_of_object": "move apple to the right of pear",
    "move_between_objects": "move apple between pear and mug",
}

CRITERIA = {
    "move_to_object": "EE XY distance to target <= 0.025 m",
    "grab_object": "gripper closed and target caught or EE XY distance <= 0.045 m",
    "pick_up": "target grasped and target lift from initial Z >= 0.05 m",
    "push_left": "target X displacement left >= 0.08 m",
    "push_right": "target X displacement right >= 0.08 m",
    "put_into_plate": "target-plate XY error <= 0.08 m and Z error <= 0.10 m",
    "move_left_of_object": "target left of reference >= 0.08 m, Y error <= 0.12 m, target moved >= 0.02 m",
    "move_right_of_object": "target right of reference >= 0.08 m, Y error <= 0.12 m, target moved >= 0.02 m",
    "move_between_objects": "target within 0.07 m of midpoint, projection in segment, target moved >= 0.02 m",
}


@dataclass(frozen=True)
class State:
    ee_pos: np.ndarray
    target_pos: np.ndarray
    gripper_open: float = 1.0
    grasped: bool = False
    reference_pos: np.ndarray | None = None
    second_reference_pos: np.ndarray | None = None
    plate_pos: np.ndarray | None = None
    label: str = ""


def _v(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)


def _lerp(a: np.ndarray | float, b: np.ndarray | float, alpha: float):
    return (1.0 - float(alpha)) * a + float(alpha) * b


def _target_on_table(x: float, y: float) -> np.ndarray:
    return _v(x, y, SUPPORT_Z + TARGET_HALF_SIZE)


def _hidden_pos() -> np.ndarray:
    return _v(2.0, 2.0, -1.0)


def _metric_line(
    instruction: str,
    state: State,
    *,
    initial_target_pos: np.ndarray,
) -> tuple[str, bool]:
    if instruction == "move_to_object":
        distance_xy = float(np.linalg.norm(state.ee_pos[:2] - state.target_pos[:2]))
        return f"ee_target_xy={distance_xy:.3f} m <= 0.025", bool(distance_xy <= 0.025)

    if instruction == "grab_object":
        distance_xy = float(np.linalg.norm(state.ee_pos[:2] - state.target_pos[:2]))
        gripper_closed = bool(state.gripper_open <= 0.35)
        caught = bool(state.grasped or distance_xy <= 0.045)
        success = bool(gripper_closed and caught)
        return f"closed={gripper_closed} caught_or_xy={caught} xy={distance_xy:.3f} m", success

    if instruction == "pick_up":
        lift = max(float(state.target_pos[2] - initial_target_pos[2]), 0.0)
        success = bool(state.grasped and lift >= 0.05)
        return f"grasped={state.grasped} lift={lift:.3f} m >= 0.05", success

    if instruction in {"push_left", "push_right"}:
        sign = -1.0 if instruction == "push_left" else 1.0
        signed_motion = float(sign * (state.target_pos[0] - initial_target_pos[0]))
        return f"signed_x_motion={signed_motion:.3f} m >= 0.08", bool(signed_motion >= 0.08)

    if instruction == "put_into_plate":
        assert state.plate_pos is not None
        xy_error = float(np.linalg.norm(state.target_pos[:2] - state.plate_pos[:2]))
        z_error = float(abs(float(state.target_pos[2]) - float(state.plate_pos[2])))
        success = bool(xy_error <= 0.08 and z_error <= 0.10)
        return f"plate_xy_error={xy_error:.3f} m z_error={z_error:.3f} m", success

    if instruction in {"move_left_of_object", "move_right_of_object"}:
        assert state.reference_pos is not None
        sign = -1.0 if instruction == "move_left_of_object" else 1.0
        offset = float(sign * (state.target_pos[0] - state.reference_pos[0]))
        y_error = float(abs(float(state.target_pos[1]) - float(state.reference_pos[1])))
        motion = float(np.linalg.norm(state.target_pos[:2] - initial_target_pos[:2]))
        success = bool(offset >= 0.08 and y_error <= 0.12 and motion >= 0.02)
        return f"offset={offset:.3f} m y_error={y_error:.3f} m motion={motion:.3f} m", success

    if instruction == "move_between_objects":
        assert state.reference_pos is not None and state.second_reference_pos is not None
        midpoint = 0.5 * (state.reference_pos[:2] + state.second_reference_pos[:2])
        segment = state.second_reference_pos[:2] - state.reference_pos[:2]
        seg_len_sq = float(np.dot(segment, segment))
        projection = 0.5 if seg_len_sq <= 1e-9 else float(np.dot(state.target_pos[:2] - state.reference_pos[:2], segment) / seg_len_sq)
        error = float(np.linalg.norm(state.target_pos[:2] - midpoint))
        motion = float(np.linalg.norm(state.target_pos[:2] - initial_target_pos[:2]))
        success = bool(error <= 0.07 and 0.0 <= projection <= 1.0 and motion >= 0.02)
        return f"midpoint_error={error:.3f} m projection={projection:.2f} motion={motion:.3f} m", success

    raise KeyError(instruction)


def _initial_state(instruction: str) -> State:
    if instruction in {"move_left_of_object", "move_right_of_object"}:
        return State(
            ee_pos=_v(0.0, 0.0, 0.40),
            target_pos=_target_on_table(-0.02, -0.03),
            reference_pos=_target_on_table(0.06, -0.03),
            label="initial state",
        )
    if instruction == "move_between_objects":
        return State(
            ee_pos=_v(0.0, 0.0, 0.40),
            target_pos=_target_on_table(-0.16, -0.03),
            reference_pos=_target_on_table(-0.14, 0.06),
            second_reference_pos=_target_on_table(0.16, 0.06),
            label="initial state",
        )
    if instruction == "put_into_plate":
        return State(
            ee_pos=_v(0.0, 0.0, 0.40),
            target_pos=_target_on_table(-0.16, -0.05),
            plate_pos=_v(0.15, 0.07, SUPPORT_Z + PLATE_HALF_HEIGHT),
            label="initial state",
        )
    return State(
        ee_pos=_v(0.0, 0.0, 0.40),
        target_pos=_target_on_table(-0.12, -0.04),
        label="initial state",
    )


def _interpolated_states(start: State, end: State, *, steps: int, label: str) -> Iterable[State]:
    for idx in range(max(1, int(steps))):
        alpha = (idx + 1) / float(max(1, int(steps)))
        yield State(
            ee_pos=_lerp(start.ee_pos, end.ee_pos, alpha),
            target_pos=_lerp(start.target_pos, end.target_pos, alpha),
            gripper_open=float(_lerp(start.gripper_open, end.gripper_open, alpha)),
            grasped=bool(end.grasped if alpha >= 0.5 else start.grasped),
            reference_pos=start.reference_pos if start.reference_pos is not None else end.reference_pos,
            second_reference_pos=start.second_reference_pos if start.second_reference_pos is not None else end.second_reference_pos,
            plate_pos=start.plate_pos if start.plate_pos is not None else end.plate_pos,
            label=label,
        )


def _build_state_sequence(instruction: str, fps: float) -> tuple[list[State], dict[str, object]]:
    states: list[State] = []
    state = _initial_state(instruction)
    initial_target_pos = state.target_pos.copy()

    def hold(steps: int, label: str) -> None:
        nonlocal state
        state = replace(state, label=label)
        states.extend([state] * max(1, int(steps)))

    def go(label: str, steps: int = 28, **changes) -> None:
        nonlocal state
        target = replace(state, **changes)
        for item in _interpolated_states(state, target, steps=steps, label=label):
            states.append(item)
        state = replace(states[-1], label=label)

    hold(10, "initial state")

    if instruction == "move_to_object":
        go("approach target object", 48, ee_pos=_v(state.target_pos[0], state.target_pos[1], 0.27))

    elif instruction == "grab_object":
        go("approach target", 34, ee_pos=_v(state.target_pos[0], state.target_pos[1], 0.27))
        go("close gripper on target", 20, gripper_open=0.0, grasped=True)

    elif instruction == "pick_up":
        go("approach target", 30, ee_pos=_v(state.target_pos[0], state.target_pos[1], 0.27))
        go("grasp target", 18, gripper_open=0.0, grasped=True)
        lifted_target = state.target_pos + _v(0.0, 0.0, 0.08)
        go("lift target", 38, ee_pos=state.ee_pos + _v(0.0, 0.0, 0.08), target_pos=lifted_target)

    elif instruction in {"push_left", "push_right"}:
        sign = -1.0 if instruction == "push_left" else 1.0
        pre_push = state.target_pos + _v(-0.055 * sign, 0.0, 0.08)
        go("move beside target", 24, ee_pos=pre_push)
        final_target = state.target_pos + _v(0.10 * sign, 0.0, 0.0)
        final_ee = final_target + _v(-0.045 * sign, 0.0, 0.08)
        go("push target", 48, ee_pos=final_ee, target_pos=final_target)

    elif instruction == "put_into_plate":
        assert state.plate_pos is not None
        go("approach target", 28, ee_pos=_v(state.target_pos[0], state.target_pos[1], 0.27))
        go("grasp target", 18, gripper_open=0.0, grasped=True)
        lifted_target = state.target_pos + _v(0.0, 0.0, 0.08)
        go("lift target", 26, ee_pos=state.ee_pos + _v(0.0, 0.0, 0.08), target_pos=lifted_target)
        carried_target = _v(state.plate_pos[0], state.plate_pos[1], state.target_pos[2])
        carried_ee = _v(state.plate_pos[0], state.plate_pos[1], state.ee_pos[2])
        go("carry to plate", 48, ee_pos=carried_ee, target_pos=carried_target)
        placed_target = _v(state.plate_pos[0], state.plate_pos[1], state.plate_pos[2])
        go("place into plate", 24, ee_pos=_v(state.plate_pos[0], state.plate_pos[1], 0.27), target_pos=placed_target)
        go("release", 18, gripper_open=1.0, grasped=False)

    elif instruction in {"move_left_of_object", "move_right_of_object"}:
        assert state.reference_pos is not None
        sign = -1.0 if instruction == "move_left_of_object" else 1.0
        target_xy = state.reference_pos[:2] + np.array([0.12 * sign, 0.0], dtype=float)
        final_target = _v(target_xy[0], target_xy[1], state.target_pos[2])
        go("approach target", 24, ee_pos=_v(state.target_pos[0], state.target_pos[1], 0.27))
        go("grasp target", 16, gripper_open=0.0, grasped=True)
        go("move relative to reference", 48, ee_pos=_v(final_target[0], final_target[1], 0.27), target_pos=final_target)

    elif instruction == "move_between_objects":
        assert state.reference_pos is not None and state.second_reference_pos is not None
        midpoint = 0.5 * (state.reference_pos[:2] + state.second_reference_pos[:2])
        final_target = _v(midpoint[0], midpoint[1], state.target_pos[2])
        go("approach target", 24, ee_pos=_v(state.target_pos[0], state.target_pos[1], 0.27))
        go("grasp target", 16, gripper_open=0.0, grasped=True)
        go("move between references", 54, ee_pos=_v(final_target[0], final_target[1], 0.27), target_pos=final_target)

    else:
        raise KeyError(instruction)

    hold(int(max(1, round(float(fps)))), "final success check")
    metric, success = _metric_line(instruction, state, initial_target_pos=initial_target_pos)
    return states, {
        "instruction_type": instruction,
        "instruction_text": TEXT[instruction],
        "criterion": CRITERIA[instruction],
        "metric": metric,
        "success": bool(success),
        "initial_target_pos": [float(x) for x in initial_target_pos],
        "final_target_pos": [float(x) for x in state.target_pos],
        "final_ee_pos": [float(x) for x in state.ee_pos],
        "reference_pos": None if state.reference_pos is None else [float(x) for x in state.reference_pos],
        "second_reference_pos": None if state.second_reference_pos is None else [float(x) for x in state.second_reference_pos],
        "plate_pos": None if state.plate_pos is None else [float(x) for x in state.plate_pos],
    }


def _add_free_body(
    worldbody: ET.Element,
    *,
    name: str,
    pos: np.ndarray,
    geom_attrs: dict[str, str],
) -> None:
    if worldbody.find(f"./body[@name='{name}']") is not None:
        return
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": name,
            "pos": f"{float(pos[0]):.5f} {float(pos[1]):.5f} {float(pos[2]):.5f}",
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    ET.SubElement(body, "geom", geom_attrs)


def _build_demo_xml(xml_path: Path, run_dir: Path) -> Path:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")

    if worldbody.find("./body[@name='demo_table']") is None:
        table = ET.SubElement(worldbody, "body", {"name": "demo_table", "pos": f"0 0 {SUPPORT_Z - 0.025:.5f}"})
        ET.SubElement(
            table,
            "geom",
            {
                "name": "demo_table_top",
                "type": "box",
                "size": "0.48 0.36 0.025",
                "rgba": "0.72 0.72 0.68 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    _add_free_body(
        worldbody,
        name="target_object",
        pos=_target_on_table(-0.12, -0.04),
        geom_attrs={
            "name": "target_box",
            "type": "box",
            "size": f"{TARGET_HALF_SIZE:.5f} {TARGET_HALF_SIZE:.5f} {TARGET_HALF_SIZE:.5f}",
            "rgba": "0.88 0.18 0.08 1",
            "mass": "0.03",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _add_free_body(
        worldbody,
        name="reference_object",
        pos=_hidden_pos(),
        geom_attrs={
            "name": "reference_box",
            "type": "box",
            "size": "0.027 0.027 0.027",
            "rgba": "0.12 0.55 0.32 1",
            "mass": "0.03",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _add_free_body(
        worldbody,
        name="second_reference_object",
        pos=_hidden_pos(),
        geom_attrs={
            "name": "second_reference_box",
            "type": "box",
            "size": "0.027 0.027 0.027",
            "rgba": "0.40 0.25 0.82 1",
            "mass": "0.03",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _add_free_body(
        worldbody,
        name="demo_plate",
        pos=_hidden_pos(),
        geom_attrs={
            "name": "demo_plate_geom",
            "type": "cylinder",
            "size": f"0.085 {PLATE_HALF_HEIGHT:.5f}",
            "rgba": "0.05 0.32 0.85 0.85",
            "mass": "0.02",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    out_path = run_dir / "cdpr_lchol_mujoco_success_criteria.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path


class MujocoOverviewRenderer:
    def __init__(self, xml_path: Path, *, width: int = 640, height: int = 480) -> None:
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, width=int(width), height=int(height))
        self.camera = mj.MjvCamera()
        self.camera.type = mj.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = np.array([0.0, 0.0, 0.10], dtype=float)
        self.camera.distance = 1.5
        self.camera.azimuth = 90
        self.camera.elevation = -30
        self.ee_qadr = self._joint_qadr("ee_free")
        self.ee_dadr = self._joint_dadr("ee_free")
        self.finger_l_qadr = self._joint_qadr("finger_l", required=False)
        self.finger_r_qadr = self._joint_qadr("finger_r", required=False)
        self.free_joint_qadr = {
            "target_object": self._joint_qadr("target_object_free"),
            "reference_object": self._joint_qadr("reference_object_free"),
            "second_reference_object": self._joint_qadr("second_reference_object_free"),
            "demo_plate": self._joint_qadr("demo_plate_free"),
        }
        self.free_joint_dadr = {
            "target_object": self._joint_dadr("target_object_free"),
            "reference_object": self._joint_dadr("reference_object_free"),
            "second_reference_object": self._joint_dadr("second_reference_object_free"),
            "demo_plate": self._joint_dadr("demo_plate_free"),
        }

    def close(self) -> None:
        self.renderer.close()

    def _joint_qadr(self, name: str, *, required: bool = True) -> int:
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            if required:
                raise RuntimeError(f"Could not find joint {name!r}.")
            return -1
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dadr(self, name: str) -> int:
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise RuntimeError(f"Could not find joint {name!r}.")
        return int(self.model.jnt_dofadr[joint_id])

    def _set_free_joint(self, qadr: int, dadr: int, pos: np.ndarray) -> None:
        self.data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=float).reshape(3)
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[dadr : dadr + 6] = 0.0

    def _set_gripper(self, open_fraction: float) -> None:
        value = 0.03 * float(np.clip(open_fraction, 0.0, 1.0))
        if self.finger_l_qadr >= 0:
            self.data.qpos[self.finger_l_qadr] = value
        if self.finger_r_qadr >= 0:
            self.data.qpos[self.finger_r_qadr] = value
        act_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        if act_id != -1:
            self.data.ctrl[act_id] = float(np.clip(open_fraction, 0.0, 1.0))

    def render(self, state: State, *, instruction: str, metric: str, success: bool, annotate: bool) -> np.ndarray:
        self._set_free_joint(self.ee_qadr, self.ee_dadr, state.ee_pos)
        self._set_gripper(state.gripper_open)
        self._set_free_joint(
            self.free_joint_qadr["target_object"],
            self.free_joint_dadr["target_object"],
            state.target_pos,
        )
        self._set_free_joint(
            self.free_joint_qadr["reference_object"],
            self.free_joint_dadr["reference_object"],
            state.reference_pos if state.reference_pos is not None else _hidden_pos(),
        )
        self._set_free_joint(
            self.free_joint_qadr["second_reference_object"],
            self.free_joint_dadr["second_reference_object"],
            state.second_reference_pos if state.second_reference_pos is not None else _hidden_pos(),
        )
        self._set_free_joint(
            self.free_joint_qadr["demo_plate"],
            self.free_joint_dadr["demo_plate"],
            state.plate_pos if state.plate_pos is not None else _hidden_pos(),
        )
        mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        frame = self.renderer.render()
        if annotate:
            frame = _annotate_frame(
                frame,
                [
                    f"Instruction: {TEXT[instruction]}",
                    f"Phase: {state.label}",
                    f"Criterion: {CRITERIA[instruction]}",
                    metric,
                    f"SUCCESS: {success}",
                ],
            )
        return np.asarray(frame, dtype=np.uint8)


def _annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, spacing=4)
    pad = 8
    rect = (10, 10, 10 + bbox[2] - bbox[0] + 2 * pad, 10 + bbox[3] - bbox[1] + 2 * pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.multiline_text((10 + pad, 10 + pad), text, fill=(255, 255, 255), spacing=4)
    return np.asarray(image)


def _write_video_ffmpeg(frames: list[np.ndarray], output_path: Path, *, fps: float, keep_frames: bool) -> dict[str, object]:
    if not frames:
        raise RuntimeError("No frames to encode.")
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
        return {"video_path": output_path.as_posix(), "frame_count": len(frames), "fps": float(fps)}
    finally:
        if not keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)


def _parse_instruction_types(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip() or raw.strip().lower() == "all":
        return CURRICULUM_INSTRUCTIONS
    requested = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in requested if item not in CURRICULUM_INSTRUCTIONS]
    if unknown:
        raise ValueError(f"Unknown instruction types: {unknown}. Available: {CURRICULUM_INSTRUCTIONS}")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description="Render MuJoCo overview-camera success videos for LC-HOL++ CDPR instructions.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--instruction-types", default="all", help="Comma-separated list, or all.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _build_demo_xml(args.xml_path, run_dir)

    renderer = MujocoOverviewRenderer(xml_path, width=int(args.width), height=int(args.height))
    summaries = []
    try:
        for instruction in _parse_instruction_types(args.instruction_types):
            states, summary = _build_state_sequence(instruction, fps=float(args.fps))
            initial_target = np.asarray(summary["initial_target_pos"], dtype=float)
            frames = []
            for state in states:
                metric, success = _metric_line(instruction, state, initial_target_pos=initial_target)
                frames.append(
                    renderer.render(
                        state,
                        instruction=instruction,
                        metric=metric,
                        success=success,
                        annotate=not bool(args.no_annotations),
                    )
                )
            video_info = _write_video_ffmpeg(
                frames,
                videos_dir / f"{instruction}_success_overview.mp4",
                fps=float(args.fps),
                keep_frames=bool(args.keep_frames),
            )
            summaries.append({**summary, **video_info})
    finally:
        renderer.close()

    manifest = {
        "run_dir": run_dir.as_posix(),
        "generated_at": datetime.now().isoformat(),
        "render_mode": "mujoco_scripted_overview",
        "camera": {
            "type": "free",
            "lookat": [0.0, 0.0, 0.10],
            "distance": 1.5,
            "azimuth": 90,
            "elevation": -30,
            "matches": "robots.cdpr.cdpr_mujoco.headless_cdpr_egl.HeadlessCDPRSimulation.overview_cam",
        },
        "uses_openvla_oft": False,
        "xml": xml_path.as_posix(),
        "instructions": summaries,
        "python": sys.executable,
        "mujoco_version": mj.__version__,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
