#!/usr/bin/env python3
"""Record action-only expert rollouts through the stage-3 CDPR RL environment.

Unlike the older demonstration renderers, this script never rewrites the robot or
object pose after ``env.reset``.  Every post-reset state transition is produced by
``CDPRLanguageRLEnv.step`` with the same normalized five-dimensional action used
by SmolVLA training and validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import mujoco as mj
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CDPR_ROOT = ROOT / "robots" / "cdpr"
if str(CDPR_ROOT) not in sys.path:
    sys.path.insert(0, str(CDPR_ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "examples" / "cdpr_smolvla_stage3_object_dense_complex_resume.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_stage3_training_validation_videos"
DEFAULT_CASES = (
    "move_object_left",
    "move_object_right",
    "move_object_up",
    "move_object_down",
    "push_left",
    "push_right",
    "put_into_plate",
    "put_into_bowl",
)
ALL_MODES = ("training", "validation")
ACTION_NAMES = ("x", "y", "z", "yaw", "grip")


@dataclass
class RolloutState:
    instruction_type: str
    # One four-action cached chunk exposes reset stability before task motion.
    warmup_remaining: int = 1
    phase: str = "settle_reset"
    policy_call: int = 0
    action_in_chunk: int = 0
    cached_action: np.ndarray | None = None
    push_start_object: np.ndarray | None = None
    push_preposition: np.ndarray | None = None
    push_contact: np.ndarray | None = None
    push_axis_sign: float = 0.0


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update({str(key): str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _maybe_silence(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config {path} did not parse to a mapping.")
    return value


def _resolve_path(config_path: Path, raw: Any | None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _dedupe(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    out: list[str] = []
    for value in values:
        name = str(value).strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _training_env(
    *,
    config_path: Path,
    config: dict[str, Any],
    instruction_type: str,
    seed: int,
    quiet: bool,
):
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv

    task = dict(config.get("task") or {})
    metadata = dict(task.get("metadata") or {})
    simulation = dict(config.get("simulation") or {})
    randomization = dict(simulation.get("randomization") or {})
    rl_args = dict(((config.get("training") or {}).get("rl") or {}).get("args") or {})
    catalog = (
        _resolve_path(config_path, simulation.get("catalog_path"))
        or ROOT / "robots" / "cdpr" / "cdpr_dataset" / "datasets" / "cdpr_scene_catalog.yaml"
    )
    desk_textures = _resolve_path(config_path, simulation.get("desk_textures_dir"))
    if desk_textures is None or not desk_textures.exists():
        desk_textures = _resolve_path(config_path, simulation.get("desk_textures_fallback_dir"))
    if desk_textures is not None and not desk_textures.exists():
        desk_textures = None
    allowed = _dedupe(metadata.get("scene_object_pool")) or _dedupe(metadata.get("target_object_pool"))

    env_updates = {"RLVLA_TASK_METADATA_JSON": json.dumps(metadata, sort_keys=True)}
    with _temporary_env(env_updates), _maybe_silence(quiet):
        env = CDPRLanguageRLEnv(
            catalog_path=catalog,
            max_steps=int(rl_args.get("max_env_steps", 96)),
            max_objects=int(rl_args.get("max_objects", randomization.get("max_objects", 4))),
            action_step_xyz=float(rl_args.get("action_step_xyz", 0.015)),
            action_step_yaw=float(rl_args.get("action_step_yaw", 0.08)),
            action_step_gripper=float(rl_args.get("action_step_gripper", 0.05)),
            hold_steps=int(rl_args.get("hold_steps", 6)),
            lock_non_commanded_axes=bool(rl_args.get("lock_non_commanded_axes", True)),
            lock_non_commanded_axes_threshold=float(rl_args.get("lock_non_commanded_axes_threshold", 0.05)),
            randomize_ee_start=bool(rl_args.get("randomize_ee_start", True)),
            ee_start_x_bounds=rl_args.get("ee_start_x_bounds"),
            ee_start_y_bounds=rl_args.get("ee_start_y_bounds"),
            ee_start_z=rl_args.get("ee_start_z"),
            randomize_ee_yaw=bool(rl_args.get("randomize_ee_yaw", True)),
            ee_yaw_bounds=rl_args.get("ee_yaw_bounds"),
            move_distance=float(rl_args.get("move_distance", metadata.get("lateral_goal_offset", 0.40))),
            lift_distance=float(rl_args.get("lift_distance", metadata.get("vertical_goal_offset", 0.10))),
            capture_frames=False,
            record_trajectory=False,
            instruction_types=[instruction_type],
            allowed_objects=allowed or None,
            desk_textures_dir=desk_textures,
            desk_geom_regex=str(simulation.get("desk_geom_regex", ".*desk.*|.*table.*")),
            desk_texrepeat=tuple(simulation.get("desk_texrepeat", (20, 20))),
            wrapper_cleanup=bool(rl_args.get("wrapper_cleanup", False)),
            use_wrapper_cache=bool(rl_args.get("use_wrapper_cache", True)),
            reuse_existing_wrapper_variants=bool(rl_args.get("reuse_existing_wrapper_variants", True)),
            seed=int(seed),
        )
    return env, metadata, env_updates


def _validation_env(
    *,
    config_path: Path,
    instruction_type: str,
    seed: int,
    quiet: bool,
):
    from rl_vla_bootstrapping.cli.validate_cdpr_policy import (
        _build_validation_env,
        _instruction_validation_task_metadata,
        _validation_env_vars,
    )
    from rl_vla_bootstrapping.core.config import load_project_config

    config_obj = load_project_config(config_path)
    metadata_raw = dict(getattr(config_obj.task, "metadata", {}) or {})
    args = SimpleNamespace(
        success_distance=float(metadata_raw.get("success_distance", 0.05)),
        directional_displacement_threshold=float(
            metadata_raw.get("directional_success_displacement_threshold", 0.05)
        ),
        multi_object_scenes=True,
        min_scene_objects=int(metadata_raw.get("min_scene_objects", 3)),
        max_scene_objects=int(metadata_raw.get("max_scene_objects", 4)),
        move_to_object_success_distance=float(
            metadata_raw.get("move_to_object_validation_distance_threshold", 0.15)
        ),
        max_objects=None,
        reuse_existing_wrapper_variants=True,
    )
    metadata = _instruction_validation_task_metadata(
        config_obj,
        args,
        instruction_type=instruction_type,
        target_object="ycb_apple",
    )
    env_updates = _validation_env_vars(
        config_obj,
        args,
        instruction_type=instruction_type,
        task_metadata_override=metadata,
    )
    # MuJoCo is already imported before this point; changing its GL backend now
    # cannot affect physics and can make local macOS rendering fail.
    env_updates.pop("MUJOCO_GL", None)
    env_updates.pop("PYOPENGL_PLATFORM", None)
    rl_args = dict(getattr(getattr(config_obj.training, "rl", None), "args", {}) or {})
    with _temporary_env(env_updates), _maybe_silence(quiet):
        env = _build_validation_env(
            config=config_obj,
            instruction_type=instruction_type,
            capture_frames=False,
            max_steps=int(rl_args.get("max_env_steps", 96)),
            hold_steps=int(rl_args.get("hold_steps", 6)),
            seed=int(seed),
            args=args,
            wrapper_dir=None,
        )
    return env, metadata, env_updates


def _reset_options(instruction_type: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "instruction_type": instruction_type,
        "target_object": "ycb_apple",
        "required_objects": ["ycb_apple"],
    }
    if instruction_type == "put_into_plate":
        options["reference_object"] = "plate"
        options["required_objects"] = ["ycb_apple", "plate"]
    elif instruction_type == "put_into_bowl":
        options["reference_object"] = "bowl"
        options["required_objects"] = ["ycb_apple", "bowl"]
    return options


def _camera(env: Any) -> mj.MjvCamera:
    positions: list[np.ndarray] = [np.asarray(env._get_ee_position(), dtype=float).reshape(3)]
    for body_name in getattr(env, "_object_body_names", ()):
        try:
            positions.append(np.asarray(env._get_body_position(body_name), dtype=float).reshape(3))
        except Exception:
            pass
    arr = np.asarray(positions, dtype=float)
    lookat = np.mean(arr, axis=0)
    support = float(getattr(env, "_support_surface_z", 0.0))
    lookat[2] = max(support + 0.05, min(float(lookat[2]), support + 0.28))
    spread = float(np.max(np.linalg.norm(arr[:, :2] - lookat[:2], axis=1))) if len(arr) > 1 else 0.35
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = float(np.clip(1.05 + 1.6 * spread, 1.15, 1.75))
    camera.azimuth = 90.0
    camera.elevation = -32.0
    return camera


def _format_vec(values: Any, *, count: int = 5) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + " ".join(f"{value:+.3f}" for value in arr[:count]) + "]"


def _target_finger_contact_stats(env: Any) -> tuple[int, float]:
    target_body = str(getattr(env, "_target_body_name", ""))
    target_bid = mj.mj_name2id(env.sim.model, mj.mjtObj.mjOBJ_BODY, target_body)
    pad_gids = {
        gid
        for name in ("left_finger_pad", "right_finger_pad")
        if (gid := mj.mj_name2id(env.sim.model, mj.mjtObj.mjOBJ_GEOM, name)) != -1
    }
    if target_bid == -1 or not pad_gids:
        return 0, 0.0
    count = 0
    max_normal_force = 0.0
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(env.sim.model.geom_bodyid[geom1])
        body2 = int(env.sim.model.geom_bodyid[geom2])
        if not (
            (geom1 in pad_gids and body2 == target_bid)
            or (geom2 in pad_gids and body1 == target_bid)
        ):
            continue
        count += 1
        force = np.zeros(6, dtype=np.float64)
        mj.mj_contactForce(env.sim.model, env.sim.data, index, force)
        max_normal_force = max(max_normal_force, abs(float(force[0])))
    return count, max_normal_force


def _annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_height = 15
    box_height = 10 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 205))
    for index, (line, color) in enumerate(lines):
        draw.text((7, 5 + index * line_height), line, font=font, fill=(*color, 255))
    return np.asarray(image)


def _capture(
    *,
    env: Any,
    renderer: mj.Renderer,
    camera: mj.MjvCamera,
    mode: str,
    instruction: str,
    phase: str,
    step: int,
    policy_call: int,
    chunk_index: int,
    new_output: bool,
    action: np.ndarray,
    reward: float,
    info: dict[str, Any],
) -> np.ndarray:
    mj.mj_forward(env.sim.model, env.sim.data)
    renderer.update_scene(env.sim.data, camera=camera)
    frame = np.asarray(renderer.render(), dtype=np.uint8).copy()
    obj = np.asarray(
        env._current_manipulated_object_position(default=np.zeros(3, dtype=np.float32)), dtype=float
    ).reshape(3)
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    scaled = np.array(
        [
            action[0] * env.action_step_xyz,
            action[1] * env.action_step_xyz,
            action[2] * env.action_step_xyz,
            action[3] * env.action_step_yaw,
            action[4] * env.action_step_gripper,
        ],
        dtype=float,
    )
    caught = bool(info.get("caught_object_is_target", False))
    contact_count, contact_force = _target_finger_contact_stats(env)
    success = bool(info.get("success", False))
    instruction_type = str(info.get("instruction_type", ""))
    if instruction_type.startswith("push_"):
        task_metric = (
            f"push signed_motion={float(info.get('signed_relation_offset', 0.0)):+.3f} m "
            f"required={float(info.get('push_success_displacement', 0.0)):.3f} m "
            f"on_support={int(bool(info.get('push_support_ok', True)))}"
        )
    elif instruction_type.startswith("move_object_"):
        task_metric = (
            f"carried signed_motion={float(info.get('carried_object_signed_motion', 0.0)):+.3f} m "
            f"caught_ok={int(bool(info.get('carried_object_caught_ok', caught)))}"
        )
    elif instruction_type.startswith("put_into_"):
        task_metric = (
            f"put downward_motion={float(info.get('put_downward_motion', 0.0)):+.3f} m "
            f"caught_required={int(bool(info.get('relation_grasp_required', False)))}"
        )
    else:
        task_metric = "task metric unavailable"
    source = "NEW IMITATED VLA OUTPUT" if new_output else "cached VLA chunk"
    if step == 0:
        source = "RESET STATE - no action applied"
    status_color = (100, 255, 120) if success else ((255, 210, 80) if caught else (255, 255, 255))
    lines = [
        (f"{mode.upper()}-LIKE | {source} | call {policy_call} | chunk action {chunk_index + 1}/8", (100, 255, 120) if new_output else (255, 255, 255)),
        (f"step {step:03d} | {instruction} | phase={phase}", (255, 255, 255)),
        (f"normalized action [x y z yaw grip] = {_format_vec(action)}", (255, 255, 255)),
        (f"applied delta    [m m m rad open] = {_format_vec(scaled)}", (255, 255, 255)),
        (f"ee={_format_vec(ee, count=3)} obj={_format_vec(obj, count=3)}", (255, 255, 255)),
        (
            f"grip={float(info.get('gripper_opening', 0.0)):.3f}->{float(info.get('gripper_target', 0.0)):.3f} "
            f"caught_target={int(caught)} reward={reward:+.3f}",
            status_color,
        ),
        (f"target-pad contacts={contact_count} max_normal_force={contact_force:.3f} N", status_color),
        (task_metric, status_color),
        (
            f"SUCCESS={int(success)} sim_valid={int(bool(info.get('simulation_state_valid', True)))} "
            f"scene={info.get('scene', '')}",
            (100, 255, 120) if success else (255, 255, 255),
        ),
    ]
    return _annotate(frame, lines)


def _target_action(env: Any, target: np.ndarray, *, replan_every: int, max_abs: float = 0.85) -> np.ndarray:
    locked = np.asarray(getattr(env, "_locked_target_xyz", env._get_ee_position()), dtype=float).reshape(3)
    delta = np.asarray(target, dtype=float).reshape(3) - locked
    denom = max(float(env.action_step_xyz) * float(replan_every), 1e-6)
    action = delta / denom
    norm = float(np.linalg.norm(action))
    if norm > float(max_abs) > 0.0:
        action *= float(max_abs) / norm
    threshold = float(getattr(env, "lock_non_commanded_axes_threshold", 0.05))
    for axis in range(3):
        if abs(delta[axis]) <= 0.002:
            action[axis] = 0.0
        elif 0.0 < abs(action[axis]) <= threshold:
            action[axis] = math.copysign(threshold + 0.01, action[axis])
    return action.astype(np.float32)


def _yaw_action(env: Any, *, target: float = 0.0, replan_every: int = 4) -> float:
    yaw = float(env._read_current_yaw())
    error = (float(target) - yaw + math.pi) % (2.0 * math.pi) - math.pi
    if abs(error) <= 0.04:
        return 0.0
    return float(np.clip(error / max(env.action_step_yaw * replan_every, 1e-6), -0.25, 0.25))


def _expert_action(env: Any, rollout: RolloutState, *, replan_every: int) -> tuple[np.ndarray, str]:
    instruction_type = rollout.instruction_type
    if rollout.warmup_remaining > 0:
        rollout.warmup_remaining -= 1
        rollout.phase = "settle_reset_without_policy_motion"
        return np.zeros(5, dtype=np.float32), rollout.phase

    if instruction_type.startswith("move_object_"):
        axis_sign = {
            "move_object_left": (0, -1.0),
            "move_object_right": (0, 1.0),
            "move_object_up": (2, 1.0),
            "move_object_down": (2, -1.0),
        }[instruction_type]
        action = np.zeros(5, dtype=np.float32)
        action[axis_sign[0]] = 0.32 * axis_sign[1]
        action[4] = -0.10
        rollout.phase = "carry_held_object_with_directional_vla_actions"
        return action, rollout.phase

    if instruction_type in {"put_into_plate", "put_into_bowl"}:
        action = np.zeros(5, dtype=np.float32)
        action[2] = -0.25
        action[4] = -0.10
        rollout.phase = "lower_held_object_into_prepositioned_container"
        return action, rollout.phase

    if instruction_type not in {"push_left", "push_right"}:
        raise ValueError(f"Unsupported instruction type: {instruction_type}")

    sign = -1.0 if instruction_type == "push_left" else 1.0
    if rollout.push_start_object is None:
        obj = np.asarray(env._current_manipulated_object_position(default=env._goal_position), dtype=float).reshape(3)
        rollout.push_start_object = obj.copy()
        rollout.push_axis_sign = sign
        safe_z = max(float(obj[2] + 0.08), float(getattr(env, "_ee_min_z", 0.08) + 0.06))
        contact_z = max(float(obj[2] + 0.020), float(getattr(env, "_ee_min_z", 0.08)))
        pre_x = float(obj[0] - sign * 0.090)
        rollout.push_preposition = np.array([pre_x, obj[1], safe_z], dtype=float)
        rollout.push_contact = np.array([pre_x, obj[1], contact_z], dtype=float)
        rollout.phase = "preposition_above_object"

    assert rollout.push_preposition is not None
    assert rollout.push_contact is not None
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    grip = float(env._get_gripper_opening() or 0.0)

    if rollout.phase == "preposition_above_object":
        close_enough = bool(np.linalg.norm(ee - rollout.push_preposition) <= 0.025)
        if close_enough and grip <= 0.12:
            rollout.phase = "descend_to_push_contact"

    if rollout.phase == "descend_to_push_contact":
        close_enough = bool(np.linalg.norm(ee - rollout.push_contact) <= 0.018)
        if close_enough:
            rollout.phase = "push_object_toward_surface"

    action = np.zeros(5, dtype=np.float32)
    if rollout.phase == "preposition_above_object":
        action[:3] = _target_action(env, rollout.push_preposition, replan_every=replan_every, max_abs=0.65)
    elif rollout.phase == "descend_to_push_contact":
        action[:3] = _target_action(env, rollout.push_contact, replan_every=replan_every, max_abs=0.62)
    else:
        current_obj = np.asarray(
            env._current_manipulated_object_position(default=env._goal_position), dtype=float
        ).reshape(3)
        action[0] = 0.85 * sign
        action[1] = float(np.clip((rollout.push_start_object[1] - current_obj[1]) / 0.04, -0.25, 0.25))
        action[2] = float(np.clip((rollout.push_contact[2] - ee[2]) / 0.04, -0.20, 0.20))
    action[3] = _yaw_action(env, replan_every=replan_every)
    action[4] = -1.0 if grip > 0.10 else -0.10
    return action, rollout.phase


def _write_video(frames: list[np.ndarray], output: Path, *, fps: float, keep_frames: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output.parent / f"{output.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    for index, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(frames_dir / f"{index:05d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-r",
            f"{float(fps):.8g}",
            "-i",
            str(frames_dir / "%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    if not keep_frames:
        shutil.rmtree(frames_dir)


def _write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _trace_row(
    *,
    mode: str,
    instruction_type: str,
    step: int,
    policy_call: int,
    chunk_index: int,
    new_output: bool,
    phase: str,
    action: np.ndarray,
    reward: float,
    info: dict[str, Any],
    env: Any,
) -> dict[str, Any]:
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    obj = np.asarray(env._current_manipulated_object_position(default=env._goal_position), dtype=float).reshape(3)
    contact_count, contact_force = _target_finger_contact_stats(env)
    row: dict[str, Any] = {
        "mode": mode,
        "instruction_type": instruction_type,
        "step": step,
        "policy_call": policy_call,
        "chunk_action_index": chunk_index,
        "new_policy_output": int(new_output),
        "phase": phase,
    }
    for index, name in enumerate(ACTION_NAMES):
        row[f"action_{name}"] = float(action[index])
    row.update(
        {
            "reward": float(reward),
            "success": int(bool(info.get("success", False))),
            "terminated": int(bool(info.get("terminated", False))),
            "truncated": int(bool(info.get("truncated", False))),
            "simulation_state_valid": int(bool(info.get("simulation_state_valid", True))),
            "caught_object_is_target": int(bool(info.get("caught_object_is_target", False))),
            "caught_object_score": float(info.get("caught_object_score", 0.0)),
            "target_finger_contact_count": int(contact_count),
            "target_finger_max_normal_force": float(contact_force),
            "ee_x": float(ee[0]),
            "ee_y": float(ee[1]),
            "ee_z": float(ee[2]),
            "object_x": float(obj[0]),
            "object_y": float(obj[1]),
            "object_z": float(obj[2]),
            "gripper_opening": float(info.get("gripper_opening", 0.0)),
            "gripper_target": float(info.get("gripper_target", 0.0)),
            "carried_object_signed_motion": float(info.get("carried_object_signed_motion", 0.0)),
            "carried_object_lost": float(info.get("carried_object_lost", 0.0)),
            "push_surface_distance": float(info.get("push_surface_distance", 0.0)),
            "push_signed_motion": float(info.get("signed_relation_offset", 0.0)),
            "push_support_height": float(info.get("push_support_height", -1.0)),
            "push_vertical_drift": float(info.get("push_vertical_drift", 0.0)),
            "push_support_ok": float(info.get("push_support_ok", 1.0)),
            "put_downward_motion": float(info.get("put_downward_motion", 0.0)),
            "manipulation_validation_success": float(info.get("manipulation_validation_success", 0.0)),
        }
    )
    return row


def _rollout(
    *,
    config_path: Path,
    config: dict[str, Any],
    mode: str,
    instruction_type: str,
    seed: int,
    run_dir: Path,
    width: int,
    height: int,
    fps: float,
    keep_frames: bool,
    quiet: bool,
    replan_every: int,
) -> dict[str, Any]:
    builder = _training_env if mode == "training" else _validation_env
    if mode == "training":
        env, metadata, env_updates = builder(
            config_path=config_path,
            config=config,
            instruction_type=instruction_type,
            seed=seed,
            quiet=quiet,
        )
    else:
        env, metadata, env_updates = builder(
            config_path=config_path,
            instruction_type=instruction_type,
            seed=seed,
            quiet=quiet,
        )

    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    renderer: mj.Renderer | None = None
    reset_options = _reset_options(instruction_type)
    try:
        with _temporary_env(env_updates), _maybe_silence(quiet):
            _, reset_info = env.reset(seed=seed, options=reset_options)
        renderer = mj.Renderer(env.sim.model, height=int(height), width=int(width))
        camera = _camera(env)
        instruction = str(reset_info.get("language_instruction", instruction_type))
        initial_obj = np.asarray(
            env._current_manipulated_object_position(default=env._goal_position), dtype=float
        ).reshape(3)
        initial_ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
        initial_contact_count, initial_contact_force = _target_finger_contact_stats(env)
        initial_info = dict(reset_info)
        initial_info.update(
            {
                "caught_object_is_target": bool(reset_info.get("caught_object_start", False)),
                "caught_object_score": 1.0 if reset_info.get("caught_object_start", False) else 0.0,
                "gripper_opening": float(env._get_gripper_opening() or 0.0),
                "gripper_target": float(env._get_gripper_target()),
                "success": False,
                "simulation_state_valid": True,
            }
        )
        frames.append(
            _capture(
                env=env,
                renderer=renderer,
                camera=camera,
                mode=mode,
                instruction=instruction,
                phase="real_env_reset",
                step=0,
                policy_call=0,
                chunk_index=0,
                new_output=False,
                action=np.zeros(5, dtype=np.float32),
                reward=0.0,
                info=initial_info,
            )
        )

        rollout = RolloutState(instruction_type=instruction_type)
        final_info = initial_info
        final_reward = 0.0
        terminated = False
        truncated = False
        max_steps = int(getattr(env, "max_steps", 96))
        cached_remaining = 0
        for step in range(1, max_steps + 1):
            new_output = cached_remaining <= 0 or rollout.cached_action is None
            if new_output:
                action, phase = _expert_action(env, rollout, replan_every=replan_every)
                rollout.cached_action = action.copy()
                rollout.policy_call += 1
                rollout.action_in_chunk = 0
                cached_remaining = int(replan_every)
            else:
                action = rollout.cached_action.copy()
                phase = rollout.phase
            chunk_index = int(rollout.action_in_chunk)
            with _temporary_env(env_updates), _maybe_silence(quiet):
                _, reward, terminated, truncated, info = env.step(action)
            final_info = dict(info)
            final_reward = float(reward)
            rows.append(
                _trace_row(
                    mode=mode,
                    instruction_type=instruction_type,
                    step=step,
                    policy_call=rollout.policy_call,
                    chunk_index=chunk_index,
                    new_output=new_output,
                    phase=phase,
                    action=action,
                    reward=reward,
                    info=info,
                    env=env,
                )
            )
            frames.append(
                _capture(
                    env=env,
                    renderer=renderer,
                    camera=camera,
                    mode=mode,
                    instruction=instruction,
                    phase=phase,
                    step=step,
                    policy_call=rollout.policy_call,
                    chunk_index=chunk_index,
                    new_output=new_output,
                    action=action,
                    reward=reward,
                    info=info,
                )
            )
            rollout.action_in_chunk += 1
            cached_remaining -= 1
            if terminated or truncated:
                break

        final_obj = np.asarray(
            env._current_manipulated_object_position(default=env._goal_position), dtype=float
        ).reshape(3)
        final_ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
        success = bool(final_info.get("success", False))
        support = float(getattr(env, "_support_surface_z", 0.0))
        object_below_support = bool(final_obj[2] < support - 0.01)
        caught_at_finish = bool(final_info.get("caught_object_is_target", False))
        initial_caught = bool(reset_info.get("caught_object_start", False))
        caught_trace = [bool(row["caught_object_is_target"]) for row in rows]
        lost_during_rollout = bool(initial_caught and any(not value for value in caught_trace[4:]))

        # Hold the terminal image long enough to read the final status.  These
        # are duplicate pixels, not extra simulator transitions.
        if frames:
            frames.extend([frames[-1].copy() for _ in range(max(6, int(round(fps))))])

        outcome = "success" if success else "failure"
        stem = f"{mode}_like_{instruction_type}_ycb_apple_{outcome}"
        video_path = run_dir / f"{stem}.mp4"
        trace_path = run_dir / f"{stem}_actions.csv"
        _write_video(frames, video_path, fps=fps, keep_frames=keep_frames)
        _write_trace(trace_path, rows)
        return {
            "mode": mode,
            "instruction_type": instruction_type,
            "instruction": instruction,
            "seed": int(seed),
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": len(rows),
            "video": video_path.as_posix(),
            "action_trace": trace_path.as_posix(),
            "scene": str(reset_info.get("scene", "")),
            "scene_objects": list(reset_info.get("scene_objects", [])),
            "target_object_catalog": str(reset_info.get("target_object_catalog", "")),
            "target_object_body": str(reset_info.get("target_object_body", "")),
            "reference_object_catalog": str(reset_info.get("reference_object_catalog", "")),
            "reference_object_body": str(reset_info.get("reference_object_body", "")),
            "wrapper_xml": str(reset_info.get("wrapper_xml", "")),
            "reset_options": reset_options,
            "caught_object_start": initial_caught,
            "caught_object_start_gripper_opening": float(
                reset_info.get("caught_object_start_gripper_opening", 0.0)
            ),
            "initial_target_finger_contact_count": int(initial_contact_count),
            "initial_target_finger_max_normal_force": float(initial_contact_force),
            "initial_ee_position": [float(value) for value in initial_ee],
            "final_ee_position": [float(value) for value in final_ee],
            "initial_object_position": [float(value) for value in initial_obj],
            "final_object_position": [float(value) for value in final_obj],
            "object_displacement": [float(value) for value in (final_obj - initial_obj)],
            "support_surface_z": support,
            "object_below_support": object_below_support,
            "caught_at_finish": caught_at_finish,
            "caught_lost_during_rollout": lost_during_rollout,
            "final_reward": final_reward,
            "final_sparse_success": float(final_info.get("sparse_success", 0.0)),
            "final_manipulation_validation_success": float(
                final_info.get("manipulation_validation_success", 0.0)
            ),
            "final_carried_object_lost": float(final_info.get("carried_object_lost", 0.0)),
            "final_put_downward_motion": float(final_info.get("put_downward_motion", 0.0)),
            "final_push_surface_distance": float(final_info.get("push_surface_distance", 0.0)),
            "final_push_signed_motion": float(final_info.get("signed_relation_offset", 0.0)),
            "final_push_support_height": float(final_info.get("push_support_height", -1.0)),
            "final_push_vertical_drift": float(final_info.get("push_vertical_drift", 0.0)),
            "final_push_support_ok": float(final_info.get("push_support_ok", 1.0)),
            "hold_steps": int(getattr(env, "hold_steps", 0)),
            "action_step_xyz": float(env.action_step_xyz),
            "action_step_yaw": float(env.action_step_yaw),
            "action_step_gripper": float(env.action_step_gripper),
            "task_metadata": metadata,
        }
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record real stage-3 training-like and validation-like action-only CDPR rollouts."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(ALL_MODES))
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=20260713)
    # The checked-in MuJoCo scene defines a 640x480 offscreen framebuffer.
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--verbose-env", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    run_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(args.modes):
        for case_index, instruction_type in enumerate(args.cases):
            seed = int(args.seed) + mode_index * 100_000 + case_index * 997
            print(f"[{mode}] {instruction_type} seed={seed}", flush=True)
            result = _rollout(
                config_path=config_path,
                config=config,
                mode=str(mode),
                instruction_type=str(instruction_type),
                seed=seed,
                run_dir=run_dir,
                width=int(args.width),
                height=int(args.height),
                fps=float(args.fps),
                keep_frames=bool(args.keep_frames),
                quiet=not bool(args.verbose_env),
                replan_every=max(1, int(args.replan_every)),
            )
            results.append(result)
            print(
                f"  success={result['success']} steps={result['steps']} "
                f"caught_finish={result['caught_at_finish']} dropped={result['object_below_support']}",
                flush=True,
            )

    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    renderer_path = Path(__file__).resolve()
    manifest = {
        "created_at": datetime.now().isoformat(),
        "config": config_path.as_posix(),
        "config_sha256": config_sha256,
        "renderer": renderer_path.as_posix(),
        "renderer_sha256": hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
        "recorder_post_reset_pose_writes": False,
        "caught_start_constraint_is_environment_behavior": bool(
            any(result["task_metadata"].get("caught_object_start_pin_object", False) for result in results)
        ),
        "state_transition_api": "CDPRLanguageRLEnv.step(normalized_5d_action)",
        "action_order": list(ACTION_NAMES),
        "target_object": "ycb_apple",
        "vla_chunk_length": 8,
        "vla_replan_every": max(1, int(args.replan_every)),
        "results": results,
        "all_successful": bool(results and all(result["success"] for result in results)),
        "any_object_below_support": any(result["object_below_support"] for result in results),
        "any_caught_object_loss": any(result["caught_lost_during_rollout"] for result in results),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path, flush=True)
    return 0 if manifest["all_successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
