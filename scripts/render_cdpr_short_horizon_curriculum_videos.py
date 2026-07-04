#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import mujoco as mj
import numpy as np
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CDPR_ROOT = ROOT / "robots" / "cdpr"
if str(CDPR_ROOT) not in sys.path:
    sys.path.insert(0, str(CDPR_ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "examples" / "cdpr_octo_small_dense_simple.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_short_horizon_curriculum_videos"
DEFAULT_CASES = (
    "move_to_object",
    "catch_object",
    "grip_object",
    "pick_up",
    "release_object",
    "put_into_plate",
)


@dataclass(frozen=True)
class CaseSpec:
    instruction_type: str
    filename: str
    start_options: dict[str, Any]
    start_label: str
    finish_label: str


CASE_SPECS: dict[str, CaseSpec] = {
    "move_to_object": CaseSpec(
        instruction_type="move_to_object",
        filename="training_scene_move_to_object.mp4",
        start_options={},
        start_label="START: sampled training scene",
        finish_label="SUCCESS: end effector above target object",
    ),
    "catch_object": CaseSpec(
        instruction_type="catch_object",
        filename="training_scene_catch_object.mp4",
        start_options={"start_with_target_at_gripper": True},
        start_label="START: target placed between open fingers",
        finish_label="SUCCESS: fingers closed around target",
    ),
    "grip_object": CaseSpec(
        instruction_type="grip_object",
        filename="training_scene_grip_object.mp4",
        start_options={"start_with_target_at_gripper": True},
        start_label="START: target placed between open fingers",
        finish_label="SUCCESS: target gripped",
    ),
    "pick_up": CaseSpec(
        instruction_type="pick_up",
        filename="training_scene_pick_up.mp4",
        start_options={"start_with_caught_object": True},
        start_label="START: target already caught by gripper",
        finish_label="SUCCESS: caught target lifted",
    ),
    "release_object": CaseSpec(
        instruction_type="release_object",
        filename="training_scene_release_object.mp4",
        start_options={"start_with_caught_object": True},
        start_label="START: target already held",
        finish_label="SUCCESS: gripper opened and target released",
    ),
    "put_into_plate": CaseSpec(
        instruction_type="put_into_plate",
        filename="training_scene_put_into_plate.mp4",
        start_options={"start_with_caught_object": True},
        start_label="START: target held near sampled container",
        finish_label="SUCCESS: target released into container",
    ),
}


@contextmanager
def _task_metadata_env(metadata: dict[str, Any]) -> Iterator[None]:
    old_value = os.environ.get("RLVLA_TASK_METADATA_JSON")
    os.environ["RLVLA_TASK_METADATA_JSON"] = json.dumps(metadata)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("RLVLA_TASK_METADATA_JSON", None)
        else:
            os.environ["RLVLA_TASK_METADATA_JSON"] = old_value


@contextmanager
def _maybe_silence(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def _resolve_path(config_path: Path, raw: Any | None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} did not parse to a mapping.")
    return data


def _dedupe(values: Any) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    if values is None:
        return ()
    for item in values:
        name = str(item).strip()
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return tuple(out)


def _build_env(
    *,
    config_path: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    instruction_types: tuple[str, ...],
    seed: int,
    quiet: bool,
):
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv

    task = dict(config.get("task") or {})
    simulation = dict(config.get("simulation") or {})
    randomization = dict(simulation.get("randomization") or {})
    training_args = dict(((config.get("training") or {}).get("rl") or {}).get("args") or {})

    catalog_path = (
        _resolve_path(config_path, simulation.get("catalog_path"))
        or _resolve_path(config_path, task.get("catalog_path"))
        or ROOT / "robots" / "cdpr" / "cdpr_dataset" / "datasets" / "cdpr_scene_catalog.yaml"
    )
    desk_textures_dir = _resolve_path(config_path, simulation.get("desk_textures_dir"))
    if desk_textures_dir is not None and not desk_textures_dir.exists():
        desk_textures_dir = _resolve_path(config_path, simulation.get("desk_textures_fallback_dir"))
    if desk_textures_dir is not None and not desk_textures_dir.exists():
        desk_textures_dir = None
    scene_objects = _dedupe(metadata.get("scene_object_pool"))
    target_objects = _dedupe(metadata.get("target_object_pool"))
    allowed_objects = scene_objects or target_objects or None

    with _task_metadata_env(metadata), _maybe_silence(quiet):
        return CDPRLanguageRLEnv(
            catalog_path=catalog_path,
            max_steps=int(training_args.get("max_env_steps", 64)),
            max_objects=int(training_args.get("max_objects", randomization.get("max_objects", 3))),
            action_step_xyz=float(training_args.get("action_step_xyz", 0.015)),
            action_step_yaw=float(training_args.get("action_step_yaw", 0.08)),
            action_step_gripper=float(training_args.get("action_step_gripper", 0.05)),
            hold_steps=int(training_args.get("hold_steps", 0)),
            lock_non_commanded_axes=bool(training_args.get("lock_non_commanded_axes", False)),
            lock_non_commanded_axes_threshold=float(
                training_args.get("lock_non_commanded_axes_threshold", 0.05)
            ),
            randomize_ee_start=bool(training_args.get("randomize_ee_start", True)),
            ee_start_x_bounds=training_args.get("ee_start_x_bounds"),
            ee_start_y_bounds=training_args.get("ee_start_y_bounds"),
            ee_start_z=training_args.get("ee_start_z"),
            randomize_ee_yaw=bool(training_args.get("randomize_ee_yaw", True)),
            ee_yaw_bounds=training_args.get("ee_yaw_bounds"),
            move_distance=float(training_args.get("move_distance", 0.40)),
            lift_distance=float(training_args.get("lift_distance", 0.10)),
            capture_frames=False,
            record_trajectory=False,
            instruction_types=instruction_types,
            allowed_objects=allowed_objects,
            desk_textures_dir=desk_textures_dir,
            desk_geom_regex=str(simulation.get("desk_geom_regex", ".*desk.*|.*table.*")),
            desk_texrepeat=tuple(simulation.get("desk_texrepeat", (20, 20))),
            wrapper_cleanup=bool(training_args.get("wrapper_cleanup", False)),
            use_wrapper_cache=bool(training_args.get("use_wrapper_cache", True)),
            reuse_existing_wrapper_variants=bool(training_args.get("reuse_existing_wrapper_variants", True)),
            seed=int(seed),
        )


def _camera_for_env(env: Any) -> mj.MjvCamera:
    positions: list[np.ndarray] = []
    try:
        positions.append(np.asarray(env._get_ee_position(), dtype=float).reshape(3))
    except Exception:
        pass
    for body_name in getattr(env, "_object_body_names", ()):
        try:
            positions.append(np.asarray(env._get_body_position(str(body_name)), dtype=float).reshape(3))
        except Exception:
            continue

    if positions:
        arr = np.asarray(positions, dtype=float)
        lookat = np.mean(arr, axis=0)
        spread = np.max(np.linalg.norm(arr[:, :2] - lookat[:2], axis=1)) if arr.shape[0] > 1 else 0.4
    else:
        lookat = np.array([0.0, 0.0, 0.20], dtype=float)
        spread = 0.4
    support_z = float(getattr(env, "_support_surface_z", lookat[2]))
    lookat[2] = max(support_z + 0.05, min(float(lookat[2]), support_z + 0.35))

    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = float(np.clip(1.10 + 1.8 * spread, 1.15, 1.85))
    camera.azimuth = 90.0
    camera.elevation = -34.0
    return camera


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


def _capture(
    *,
    env: Any,
    renderer: mj.Renderer,
    camera: mj.MjvCamera,
    frames: list[np.ndarray],
    title: str,
    phase: str,
    outcome: str,
) -> None:
    mj.mj_forward(env.sim.model, env.sim.data)
    renderer.update_scene(env.sim.data, camera=camera)
    frame = np.asarray(renderer.render(), dtype=np.uint8).copy()
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    obj = np.asarray(env._current_manipulated_object_position(default=env._goal_position), dtype=float).reshape(3)
    gripper = float(env._get_gripper_opening())
    frames.append(
        _annotate(
            frame,
            [
                title,
                phase,
                outcome,
                f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) obj=({obj[0]:+.2f},{obj[1]:+.2f},{obj[2]:.2f})",
                f"gripper={gripper:.2f}",
            ],
        )
    )


def _write_video(frames: list[np.ndarray], output_path: Path, *, fps: float, keep_frames: bool) -> dict[str, Any]:
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


def _set_ee_position(env: Any, position: np.ndarray) -> None:
    joint_id = mj.mj_name2id(env.sim.model, mj.mjtObj.mjOBJ_JOINT, "ee_free")
    if joint_id == -1:
        raise RuntimeError("Could not find ee_free joint.")
    qadr = int(env.sim.model.jnt_qposadr[joint_id])
    dadr = int(env.sim.model.jnt_dofadr[joint_id])
    quat = np.asarray(env.sim.data.qpos[qadr + 3 : qadr + 7], dtype=float).copy()
    if float(np.linalg.norm(quat)) < 1e-9:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    env.sim.data.qpos[qadr : qadr + 3] = np.asarray(position, dtype=float).reshape(3)
    env.sim.data.qpos[qadr + 3 : qadr + 7] = quat / max(float(np.linalg.norm(quat)), 1e-9)
    env.sim.data.qvel[dadr : dadr + 6] = 0.0
    mj.mj_forward(env.sim.model, env.sim.data)
    env._locked_target_xyz = np.asarray(position, dtype=np.float32).reshape(3).copy()


def _set_body_position(env: Any, body_name: str, position: np.ndarray) -> None:
    if not env._set_body_position(str(body_name), np.asarray(position, dtype=np.float32).reshape(3)):
        raise RuntimeError(f"Could not set body position for {body_name!r}.")


def _set_gripper(env: Any, opening_01: float) -> None:
    env._force_gripper_opening(float(np.clip(opening_01, 0.0, 1.0)))
    mj.mj_forward(env.sim.model, env.sim.data)


def _target_position(env: Any) -> np.ndarray:
    return np.asarray(env._current_manipulated_object_position(default=env._goal_position), dtype=np.float32).reshape(3)


def _reference_position(env: Any) -> np.ndarray:
    return np.asarray(env._reference_object_position(default=env._goal_position), dtype=np.float32).reshape(3)


def _hold_offset(env: Any) -> np.ndarray:
    try:
        return (_target_position(env) - np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3)).astype(np.float32)
    except Exception:
        return np.array([0.0, 0.0, -0.035], dtype=np.float32)


def _held_opening(env: Any) -> float:
    opening = float(getattr(env, "_caught_object_start_gripper_opening", 0.0))
    if not np.isfinite(opening) or opening <= 0.0:
        opening = float(env._get_gripper_opening())
    return float(np.clip(opening, 0.0, 1.0))


def _target_fit_opening(env: Any) -> float:
    getter = getattr(env, "_caught_object_start_gripper_opening_for_body", None)
    if callable(getter) and getattr(env, "_target_body_name", ""):
        try:
            return float(np.clip(getter(env._target_body_name), 0.0, 1.0))
        except Exception:
            pass
    return _held_opening(env)


def _target_at_hold_center(env: Any) -> np.ndarray:
    target = _target_position(env)
    hold_getter = getattr(env, "_caught_object_start_hold_center", None)
    if not callable(hold_getter):
        return target
    try:
        hold_center = hold_getter()
    except Exception:
        hold_center = None
    if hold_center is None:
        return target
    current_hold = np.asarray(hold_center, dtype=np.float32).reshape(3)
    return current_hold + (target - current_hold)


def _released_object_position(env: Any) -> np.ndarray:
    target = _target_position(env)
    geometry_getter = getattr(env, "_finger_pair_geometry", None)
    width_getter = getattr(env, "_body_width_along_axis", None)
    if callable(geometry_getter) and callable(width_getter) and getattr(env, "_target_body_name", ""):
        try:
            geometry = geometry_getter()
            axis = np.asarray(geometry["axis"], dtype=np.float32).reshape(3)
            center = np.asarray(geometry["center"], dtype=np.float32).reshape(3)
            inner_gap = float(geometry["inner_gap"])
            width = float(width_getter(env._target_body_name, axis))
            if np.all(np.isfinite(axis)) and float(np.linalg.norm(axis)) > 1e-9 and np.isfinite(width):
                axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
                target = center + axis * (0.5 * max(inner_gap, 0.0) + 0.5 * max(width, 0.0) + 0.06)
        except Exception:
            pass
    support_z = float(getattr(env, "_support_surface_z", target[2]))
    target = np.asarray(target, dtype=np.float32).reshape(3)
    target[2] = support_z + 0.025
    return target


def _drive(
    *,
    env: Any,
    renderer: mj.Renderer,
    camera: mj.MjvCamera,
    frames: list[np.ndarray],
    target_ee: np.ndarray,
    held_offset: np.ndarray | None,
    gripper: float,
    title: str,
    phase: str,
    outcome: str,
    steps: int,
) -> None:
    start_ee = np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3)
    for index in range(max(1, int(steps))):
        alpha = float(index + 1) / float(max(1, int(steps)))
        ee_pos = (1.0 - alpha) * start_ee + alpha * np.asarray(target_ee, dtype=np.float32).reshape(3)
        _set_ee_position(env, ee_pos)
        _set_gripper(env, gripper)
        if held_offset is not None and getattr(env, "_target_body_name", ""):
            _set_body_position(env, env._target_body_name, ee_pos + held_offset)
        _capture(env=env, renderer=renderer, camera=camera, frames=frames, title=title, phase=phase, outcome=outcome)


def _success_move_to_object(env: Any, metadata: dict[str, Any]) -> None:
    obj = _target_position(env)
    z_low = float(metadata.get("move_to_object_z_window_low", 0.10))
    z_high = float(metadata.get("move_to_object_z_window_high", 0.20))
    if z_low > z_high:
        z_low, z_high = z_high, z_low
    final_ee = np.array([obj[0], obj[1], 0.5 * (z_low + z_high)], dtype=np.float32)
    _set_ee_position(env, final_ee)


def _success_catch_or_grip(env: Any) -> None:
    _set_gripper(env, _target_fit_opening(env))
    if getattr(env, "_target_body_name", ""):
        _set_body_position(env, env._target_body_name, _target_at_hold_center(env))


def _success_pick_up(env: Any, metadata: dict[str, Any], held_offset: np.ndarray) -> None:
    lift = max(float(metadata.get("pick_lift_success_height", 0.05)) + 0.03, 0.07)
    final_ee = np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3) + np.array([0.0, 0.0, lift], dtype=np.float32)
    _set_ee_position(env, final_ee)
    _set_gripper(env, _held_opening(env))
    _set_body_position(env, env._target_body_name, final_ee + held_offset)


def _success_release(env: Any) -> None:
    target = _released_object_position(env)
    if getattr(env, "_target_body_name", ""):
        _set_body_position(env, env._target_body_name, target)
    _set_gripper(env, 1.0)
    env._caught_object_start_active = False


def _success_put_into_plate(env: Any) -> np.ndarray:
    reference = _reference_position(env)
    final_obj = reference.copy()
    support_z = float(getattr(env, "_support_surface_z", final_obj[2]))
    final_obj[2] = max(float(final_obj[2]), support_z + 0.025)
    if getattr(env, "_target_body_name", ""):
        _set_body_position(env, env._target_body_name, final_obj)
    _set_gripper(env, 1.0)
    env._caught_object_start_active = False
    final_ee = final_obj + np.array([0.0, 0.0, 0.16], dtype=np.float32)
    _set_ee_position(env, final_ee)
    return final_obj


def _evaluate_final_state(env: Any) -> dict[str, Any]:
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import (
        _call_with_supported_kwargs,
        _normalize_reward_result,
        _normalize_success_result,
    )

    state = copy.deepcopy(env._reward_state)
    ee = np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3)
    goal_pos = env._current_target_reference_position()
    gripper_opening = env._get_gripper_opening()
    caught_body, caught_catalog, caught_score, caught_is_target = env._detect_caught_object(ee)
    reward_kwargs = {
        "spec": env._instruction_spec,
        "ee_pos": ee,
        "obj_pos": goal_pos,
        "goal_pos": goal_pos,
        "reward_state": state,
        "action": np.zeros((5,), dtype=np.float32),
        "camera_alignment": env._get_ee_camera_alignment(
            target_pos=goal_pos,
            direction=env._current_goal_motion_direction(ee_pos=ee, goal_pos=goal_pos),
        ),
        "goal_direction": env._current_goal_motion_direction(ee_pos=ee, goal_pos=goal_pos),
        "goal_region": env._goal_region,
        "goal_relation": env._goal_relation,
        "dense_reward_terms": env._dense_reward_terms,
        "task_metadata": env._task_metadata,
        "env": env,
        "sim": env.sim,
        "scene_name": env._scene_name,
        "target_catalog_name": env._target_catalog_name,
        "target_body_name": env._target_body_name,
        "reference_catalog_name": env._reference_catalog_name,
        "reference_body_name": env._reference_body_name,
        "second_reference_catalog_name": env._second_reference_catalog_name,
        "second_reference_body_name": env._second_reference_body_name,
        "gripper_opening": gripper_opening,
        "support_surface_z": env._support_surface_z,
        "caught_object_body": caught_body,
        "caught_object_catalog": caught_catalog,
        "caught_object_score": float(caught_score),
        "caught_object_is_target": bool(caught_is_target),
    }
    reward, success, reward_info = _normalize_reward_result(
        _call_with_supported_kwargs(env._reward_fn, **reward_kwargs)
    )
    if env._success_fn is not None:
        success, success_info = _normalize_success_result(
            _call_with_supported_kwargs(
                env._success_fn,
                **reward_kwargs,
                reward=float(reward),
                reward_info=reward_info,
                current_success=bool(success),
            ),
            bool(success),
        )
        reward_info.update(success_info)
    return {
        "reward": float(reward),
        "success": bool(success),
        "caught_object_body": str(caught_body),
        "caught_object_catalog": str(caught_catalog),
        "caught_object_score": float(caught_score),
        "caught_object_is_target": bool(caught_is_target),
        "reward_info_subset": {
            str(key): float(value)
            for key, value in reward_info.items()
            if isinstance(value, (int, float, np.floating, np.integer))
            and key
            in {
                "dense_gripper_success",
                "sparse_success",
                "pick_up_validation_success",
                "manipulation_validation_success",
                "move_to_object_validation_success",
                "reward_raw_before_output_normalization",
                "reward_output_value",
            }
        },
    }


def _render_case(
    *,
    config_path: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    run_dir: Path,
    case: CaseSpec,
    seed: int,
    fps: float,
    width: int,
    height: int,
    keep_frames: bool,
    quiet_env: bool,
) -> dict[str, Any]:
    instruction_types = _dedupe(config.get("task", {}).get("instruction_types")) or DEFAULT_CASES
    env = _build_env(
        config_path=config_path,
        config=config,
        metadata=metadata,
        instruction_types=instruction_types,
        seed=seed,
        quiet=quiet_env,
    )
    frames: list[np.ndarray] = []
    renderer: mj.Renderer | None = None
    try:
        reset_options = {"instruction_type": case.instruction_type, **case.start_options}
        with _task_metadata_env(metadata), _maybe_silence(quiet_env):
            _, info = env.reset(seed=seed, options=reset_options)

        renderer = mj.Renderer(env.sim.model, height=int(height), width=int(width))
        camera = _camera_for_env(env)
        title = f"{case.instruction_type}: {info.get('language_instruction', '')}"
        target_name = str(info.get("target_object_catalog") or "")
        reference_name = str(info.get("reference_object_catalog") or "")
        held_offset = _hold_offset(env)
        held_opening = _held_opening(env)
        open_opening = 1.0

        for _ in range(14):
            _capture(
                env=env,
                renderer=renderer,
                camera=camera,
                frames=frames,
                title=title,
                phase=case.start_label,
                outcome=f"scene={info.get('scene', '')} target={target_name} ref={reference_name}",
            )

        if case.instruction_type == "move_to_object":
            obj = _target_position(env)
            z_low = float(metadata.get("move_to_object_z_window_low", 0.10))
            z_high = float(metadata.get("move_to_object_z_window_high", 0.20))
            if z_low > z_high:
                z_low, z_high = z_high, z_low
            final_ee = np.array([obj[0], obj[1], 0.5 * (z_low + z_high)], dtype=np.float32)
            _drive(
                env=env,
                renderer=renderer,
                camera=camera,
                frames=frames,
                target_ee=final_ee,
                held_offset=None,
                gripper=float(env._get_gripper_opening()),
                title=title,
                phase="moving end effector to object",
                outcome=case.finish_label,
                steps=36,
            )
            _success_move_to_object(env, metadata)

        elif case.instruction_type in {"catch_object", "grip_object"}:
            fit_opening = _target_fit_opening(env)
            for step in range(30):
                alpha = float(step + 1) / 30.0
                _set_gripper(env, (1.0 - alpha) * open_opening + alpha * fit_opening)
                _capture(
                    env=env,
                    renderer=renderer,
                    camera=camera,
                    frames=frames,
                    title=title,
                    phase="closing gripper on target",
                    outcome=case.finish_label,
                )
            _success_catch_or_grip(env)

        elif case.instruction_type == "pick_up":
            lift = max(float(metadata.get("pick_lift_success_height", 0.05)) + 0.03, 0.07)
            final_ee = np.asarray(env._get_ee_position(), dtype=np.float32).reshape(3) + np.array(
                [0.0, 0.0, lift],
                dtype=np.float32,
            )
            _drive(
                env=env,
                renderer=renderer,
                camera=camera,
                frames=frames,
                target_ee=final_ee,
                held_offset=held_offset,
                gripper=held_opening,
                title=title,
                phase="lifting already-held target",
                outcome=case.finish_label,
                steps=40,
            )
            _success_pick_up(env, metadata, held_offset)

        elif case.instruction_type == "release_object":
            target = _released_object_position(env)
            for step in range(30):
                alpha = float(step + 1) / 30.0
                _set_gripper(env, (1.0 - alpha) * held_opening + alpha)
                _set_body_position(env, env._target_body_name, target)
                _capture(
                    env=env,
                    renderer=renderer,
                    camera=camera,
                    frames=frames,
                    title=title,
                    phase="opening gripper and leaving target on table",
                    outcome=case.finish_label,
                )
            _success_release(env)

        elif case.instruction_type == "put_into_plate":
            reference = _reference_position(env)
            above = reference + np.array([0.0, 0.0, 0.22], dtype=np.float32)
            lower = reference + np.array([0.0, 0.0, 0.12], dtype=np.float32)
            _drive(
                env=env,
                renderer=renderer,
                camera=camera,
                frames=frames,
                target_ee=above,
                held_offset=held_offset,
                gripper=held_opening,
                title=title,
                phase="carrying target over sampled container",
                outcome="approaching container",
                steps=34,
            )
            _drive(
                env=env,
                renderer=renderer,
                camera=camera,
                frames=frames,
                target_ee=lower,
                held_offset=held_offset,
                gripper=held_opening,
                title=title,
                phase="lowering target into container",
                outcome="ready to release",
                steps=24,
            )
            final_obj = _success_put_into_plate(env)
            for _ in range(18):
                _set_body_position(env, env._target_body_name, final_obj)
                _capture(
                    env=env,
                    renderer=renderer,
                    camera=camera,
                    frames=frames,
                    title=title,
                    phase="target released in sampled container",
                    outcome=case.finish_label,
                )
        else:
            raise ValueError(f"Unsupported case: {case.instruction_type}")

        for _ in range(10):
            _capture(
                env=env,
                renderer=renderer,
                camera=camera,
                frames=frames,
                title=title,
                phase="FINAL CHECK",
                outcome=case.finish_label,
            )

        evaluation = _evaluate_final_state(env)
        video_info = _write_video(frames, run_dir / case.filename, fps=fps, keep_frames=keep_frames)
        video_info.update(
            {
                "case": case.instruction_type,
                "seed": int(seed),
                "scene": str(info.get("scene", "")),
                "scene_objects": list(info.get("scene_objects", [])),
                "target_object_catalog": target_name,
                "target_object_body": str(info.get("target_object_body", "")),
                "reference_object_catalog": reference_name,
                "reference_object_body": str(info.get("reference_object_body", "")),
                "language_instruction": str(info.get("language_instruction", "")),
                "reset_options": reset_options,
                "caught_object_start": bool(info.get("caught_object_start", False)),
                "final_evaluation": evaluation,
            }
        )
        return video_info
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render short-horizon CDPR videos from real training-config MuJoCo scenes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), choices=tuple(CASE_SPECS))
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--verbose-env", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    metadata = dict((config.get("task") or {}).get("metadata") or {})
    if not metadata:
        raise ValueError(f"Config {config_path} has no task.metadata block.")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = [CASE_SPECS[str(name)] for name in args.cases]
    videos = [
        _render_case(
            config_path=config_path,
            config=config,
            metadata=metadata,
            run_dir=run_dir,
            case=case,
            seed=int(args.seed) + 997 * index,
            fps=float(args.fps),
            width=int(args.width),
            height=int(args.height),
            keep_frames=bool(args.keep_frames),
            quiet_env=not bool(args.verbose_env),
        )
        for index, case in enumerate(cases)
    ]
    manifest = {
        "config": config_path.as_posix(),
        "task_metadata_source": "task.metadata",
        "training_scene_renderer": True,
        "videos": videos,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
