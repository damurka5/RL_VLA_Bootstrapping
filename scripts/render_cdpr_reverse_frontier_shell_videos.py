#!/usr/bin/env python3
"""Record successful Reverse Frontier episodes through the real CDPR RL API.

The recorder deliberately shares the active GRPO configuration, scene sampler,
Reverse Frontier reset code, normalized five-dimensional action interface, hold
steps, and sparse success predicate.  After ``env.reset`` it never teleports the
robot or an object: every transition is produced by ``CDPRLanguageRLEnv.step``.

The policy is a deterministic task-state oracle that imitates SmolVLA chunk
outputs.  It is useful for inspecting what a successful episode *can* look like;
it is not a learned checkpoint evaluation.
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
from typing import Any, Iterator, Sequence

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

from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import (
    SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS,
    SMOLVLA_COMPLEX_PROFILE,
    get_cdpr_reverse_shell_specs,
)

DEFAULT_CONFIG = ROOT / "configs" / "examples" / "cdpr_smolvla_complex_reverse_frontier_grpo.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_reverse_frontier_shell_videos"
DEFAULT_TARGET = "ycb_apple"
MODES = ("training", "validation")
ACTION_NAMES = ("x", "y", "z", "yaw", "gripper")
SHELL_POLICY_DECISION_BOUNDS = SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS
PLACEMENT_TYPES = ("put_into_plate", "put_into_bowl")
RELATION_TYPES = ("move_left_of_object", "move_right_of_object", "move_between_objects")


@dataclass
class OracleState:
    instruction_type: str
    shell_id: int
    phase: str = "inspect_reverse_shell_reset"
    policy_call: int = 0
    chunk_index: int = 0
    grasp_steps: int = 0
    caught_once: bool = False
    grasp_lifted: bool = False
    grasp_lift_goal_object_z: float | None = None
    desired_object_position: np.ndarray | None = None
    move_to_start_z: float | None = None
    push_start_y: float | None = None
    shell3_push_preposition: np.ndarray | None = None
    shell3_push_contact: np.ndarray | None = None


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
    out: list[str] = []
    for value in values or ():
        name = str(value).strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _config_parts(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task = dict(config.get("task") or {})
    simulation = dict(config.get("simulation") or {})
    rl_args = dict(((config.get("training") or {}).get("rl") or {}).get("args") or {})
    return task, simulation, rl_args


def _build_env(
    *,
    config_path: Path,
    config: dict[str, Any],
    instruction_type: str,
    seed: int,
    quiet: bool,
):
    """Mirror the environment arguments used by SmolVLA GRPO `_build_env`."""
    from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv

    task, simulation, rl_args = _config_parts(config)
    metadata = dict(task.get("metadata") or {})
    randomization = dict(simulation.get("randomization") or {})
    catalog = (
        _resolve_path(config_path, simulation.get("catalog_path"))
        or ROOT / "robots" / "cdpr" / "cdpr_dataset" / "datasets" / "cdpr_scene_catalog.yaml"
    )
    desk_textures = _resolve_path(config_path, simulation.get("desk_textures_dir"))
    if desk_textures is None or not desk_textures.exists():
        desk_textures = _resolve_path(config_path, simulation.get("desk_textures_fallback_dir"))
    if desk_textures is not None and not desk_textures.exists():
        desk_textures = None
    allowed = _dedupe(task.get("target_objects"))
    env_updates = {
        "RLVLA_TASK_METADATA_JSON": json.dumps(metadata, sort_keys=True),
        "RLVLA_TASK_SUCCESS_ATTRIBUTE": "compute_instruction_validation_success",
        "RLVLA_TASK_SUCCESS_FILE": (
            ROOT / "robots" / "cdpr" / "cdpr_dataset" / "rl_instruction_tasks.py"
        ).as_posix(),
    }
    # MuJoCo is already imported.  Switching its backend at this point is unsafe
    # on local macOS and does not alter physics.
    # Remote-only EGL and compiled-model-cache process settings are intentionally
    # not injected into an already-running local macOS Python process.  They do
    # not alter scene sampling, physics, actions, or success semantics, and the
    # compiled cache can retain a renderer/model lock across local resets.

    with _temporary_env(env_updates), _maybe_silence(quiet):
        env = CDPRLanguageRLEnv(
            catalog_path=catalog,
            max_steps=int(rl_args.get("max_env_steps", 96)),
            max_objects=int(rl_args.get("max_objects", randomization.get("max_objects", 4))),
            action_step_xyz=float(rl_args.get("action_step_xyz", 0.015)),
            action_step_yaw=float(rl_args.get("action_step_yaw", 0.08)),
            action_step_gripper=float(rl_args.get("action_step_gripper", 0.05)),
            hold_steps=int(rl_args.get("hold_steps", 6)),
            lock_non_commanded_axes=bool(rl_args.get("lock_non_commanded_axes", False)),
            lock_non_commanded_axes_threshold=float(
                rl_args.get("lock_non_commanded_axes_threshold", 0.05)
            ),
            randomize_ee_start=bool(rl_args.get("randomize_ee_start", True)),
            ee_start_x_bounds=rl_args.get("ee_start_x_bounds", (-0.20, 0.20)),
            ee_start_y_bounds=rl_args.get("ee_start_y_bounds", (-0.20, 0.20)),
            ee_start_z=rl_args.get("ee_start_z"),
            randomize_ee_yaw=bool(rl_args.get("randomize_ee_yaw", True)),
            ee_yaw_bounds=rl_args.get("ee_yaw_bounds", (-math.pi, math.pi)),
            move_distance=float(rl_args.get("move_distance", 0.40)),
            lift_distance=float(rl_args.get("lift_distance", 0.10)),
            capture_frames=False,
            record_trajectory=False,
            instruction_types=[instruction_type],
            allowed_objects=allowed or None,
            desk_textures_dir=desk_textures,
            desk_geom_regex=str(simulation.get("desk_geom_regex", r"(table|desk|workbench|counter|surface)")),
            desk_texrepeat=tuple(simulation.get("desk_texrepeat", (20, 20))),
            wrapper_cleanup=bool(rl_args.get("wrapper_cleanup", False)),
            use_wrapper_cache=bool(rl_args.get("use_wrapper_cache", True)),
            reuse_existing_wrapper_variants=bool(
                rl_args.get("reuse_existing_wrapper_variants", True)
            ),
            seed=int(seed),
        )
    return env, metadata, env_updates


def _reset_options(instruction_type: str, shell_id: int, target_object: str) -> dict[str, Any]:
    # The first five keys are exactly ComplexSmolVLARuntime.reset_options().
    # The explicit target makes the 64-video matrix comparable across cases.
    options: dict[str, Any] = {
        "instruction_type": str(instruction_type),
        "curriculum_mode": "reverse_frontier",
        "curriculum_shell": int(shell_id),
        "curriculum_sample_source": "recorder_frontier",
        "start_with_caught_object": False,
        "start_with_target_at_gripper": False,
        "target_object": str(target_object),
        "required_objects": [str(target_object)],
    }
    if instruction_type == "put_into_plate":
        options.update(reference_object="plate", required_objects=[str(target_object), "plate"])
    elif instruction_type == "put_into_bowl":
        options.update(reference_object="bowl", required_objects=[str(target_object), "bowl"])
    return options


def _case_seed(
    *,
    mode: str,
    base_seed: int,
    validation_seed: int,
    global_step: int,
    instruction_index: int,
    shell_id: int,
    attempt: int,
) -> int:
    if mode == "validation":
        # Same formula as `_run_smolvla_distinct_validation`, with shell and
        # retry offsets occupying the episode-index part of the seed.
        return (
            int(validation_seed)
            + int(global_step) * 17
            + int(instruction_index) * 1009
            + int(shell_id) * 101
            + int(attempt)
        )
    return int(base_seed) + int(instruction_index) * 1009 + int(shell_id) * 101 + int(attempt)


def _policy_step_bounds(hold_steps: int, shell_id: int) -> tuple[int, int]:
    del hold_steps
    return SHELL_POLICY_DECISION_BOUNDS[
        int(np.clip(shell_id, 0, len(SHELL_POLICY_DECISION_BOUNDS) - 1))
    ]


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
    lookat[2] = max(support + 0.05, min(float(lookat[2]), support + 0.30))
    spread = float(np.max(np.linalg.norm(arr[:, :2] - lookat[:2], axis=1))) if len(arr) > 1 else 0.35
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = float(np.clip(1.05 + 1.7 * spread, 1.15, 1.80))
    camera.azimuth = 90.0
    camera.elevation = -32.0
    return camera


def _format_vec(values: Any, *, count: int = 5) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + " ".join(f"{value:+.3f}" for value in arr[:count]) + "]"


def _annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_height = 15
    box_height = 10 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 210))
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
    shell_id: int,
    phase: str,
    policy_step: int,
    policy_call: int,
    action: np.ndarray,
    reward: float,
    info: dict[str, Any],
) -> np.ndarray:
    mj.mj_forward(env.sim.model, env.sim.data)
    renderer.update_scene(env.sim.data, camera=camera)
    frame = np.asarray(renderer.render(), dtype=np.uint8).copy()
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    obj = np.asarray(
        env._current_manipulated_object_position(default=np.zeros(3, dtype=np.float32)), dtype=float
    ).reshape(3)
    ref = np.asarray(env._reference_object_position(default=obj), dtype=float).reshape(3)
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
    policy_low, policy_high = _policy_step_bounds(int(env.hold_steps), shell_id)
    policy_low = int(info.get("curriculum_shell_policy_steps_low", policy_low))
    policy_high = int(info.get("curriculum_shell_policy_steps_high", policy_high))
    target_policy_steps = int(info.get("curriculum_shell_target_policy_steps", -1))
    sim_low = int(info.get("curriculum_shell_sim_steps_low", policy_low * (1 + env.hold_steps)))
    sim_high = int(info.get("curriculum_shell_sim_steps_high", policy_high * (1 + env.hold_steps)))
    success = bool(info.get("success", False))
    caught = bool(info.get("caught_object_is_target", False))
    pinned = bool(getattr(env, "_caught_object_start_active", False))
    dynamically_latched = bool(
        getattr(env, "_reverse_frontier_dynamic_grasp_latched", False)
    )
    status = (100, 255, 120) if success else ((255, 210, 80) if caught else (255, 255, 255))
    source = "RESET - no policy action" if policy_step == 0 else "NEW IMITATED VLA CHUNK"
    lines = [
        (f"{mode.upper()}-LIKE | Reverse Frontier shell {shell_id} | {source}", (100, 220, 255)),
        (f"{instruction} | phase={phase}", (255, 255, 255)),
        (
            f"policy step={policy_step:03d} call={policy_call:03d} applied chunk action=1/8 "
            f"(replan_every=1)",
            (255, 255, 255),
        ),
        (f"normalized action [x y z yaw grip] = {_format_vec(action)}", (255, 255, 255)),
        (f"applied delta    [m m m rad open] = {_format_vec(scaled)}", (255, 255, 255)),
        (
            f"shell requires {policy_low}-{policy_high} policy decisions; sampled target="
            f"{target_policy_steps}; hold_steps={env.hold_steps} ({sim_low}-{sim_high} sim steps)",
            (255, 210, 80),
        ),
        (f"ee={_format_vec(ee, count=3)} target={_format_vec(obj, count=3)} ref={_format_vec(ref, count=3)}", status),
        (
            f"gripper={float(info.get('gripper_opening', env._get_gripper_opening() or 0.0)):.3f} "
            f"caught={int(caught)} ever-grasped={int(float(info.get('ever_grasped', 0.0)) >= 0.5)} "
            f"carry-latch={int(pinned)} caught-after-policy-close={int(dynamically_latched)} "
            f"reward={float(reward):+.3f}",
            status,
        ),
        (
            f"catch-gate={getattr(env, '_reverse_frontier_grasp_latch_status', 'inactive')} "
            f"align-xy={float(getattr(env, '_reverse_frontier_grasp_latch_xy_distance', float('nan'))):.3f}m "
            f"align-z={float(getattr(env, '_reverse_frontier_grasp_latch_z_distance', float('nan'))):.3f}m "
            f"finger-contacts={int(getattr(env, '_reverse_frontier_grasp_latch_contact_count', 0))}",
            status,
        ),
        (
            f"SUCCESS={int(success)} raw={int(float(info.get('reverse_frontier_raw_success', success)) >= 0.5)} "
            f"decision-gate={int(float(info.get('reverse_frontier_policy_gate_satisfied', 0.0)) >= 0.5)} "
            f"remaining={int(info.get('reverse_frontier_policy_steps_remaining', max(0, target_policy_steps - policy_step)))} "
            f"sim_valid={int(bool(info.get('simulation_state_valid', True)))} scene={info.get('scene', '')}",
            (100, 255, 120) if success else (255, 255, 255),
        ),
    ]
    return _annotate(frame, lines)


def _target_action(env: Any, target: np.ndarray, *, max_norm: float = 0.70) -> np.ndarray:
    if bool(getattr(env, "lock_non_commanded_axes", False)):
        reference = np.asarray(
            getattr(env, "_locked_target_xyz", env._get_ee_position()), dtype=float
        ).reshape(3)
    else:
        reference = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    delta = np.asarray(target, dtype=float).reshape(3) - reference
    action = delta / max(float(env.action_step_xyz), 1e-6)
    norm = float(np.linalg.norm(action))
    if norm > float(max_norm) > 0.0:
        action *= float(max_norm) / norm
    threshold = float(getattr(env, "lock_non_commanded_axes_threshold", 0.05))
    for axis in range(3):
        if abs(delta[axis]) <= 0.0015:
            action[axis] = 0.0
        elif 0.0 < abs(action[axis]) <= threshold:
            action[axis] = math.copysign(threshold + 0.01, action[axis])
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _gripper_action_to_target(env: Any, target: float) -> float:
    current = float(env._get_gripper_target())
    error = float(np.clip(target, 0.0, 1.0)) - current
    if abs(error) <= 0.008:
        return 0.0
    return float(np.clip(error / max(float(env.action_step_gripper), 1e-6), -1.0, 1.0))


def _held_opening(env: Any) -> float:
    getter = getattr(env, "_caught_object_start_gripper_opening_for_body", None)
    if callable(getter):
        try:
            return float(np.clip(getter(env._target_body_name), 0.0, 1.0))
        except Exception:
            pass
    return 0.75


def _physical_grasp_target(env: Any) -> float:
    # Reverse-held shells pin the body and use the fitted opening exactly.  A
    # free-object grasp shell uses extra preload before its aligned close latch.
    return float(np.clip(_held_opening(env) - 0.03, 0.0, 1.0))


def _held_offset(env: Any) -> np.ndarray:
    raw = dict(getattr(env, "_task_metadata", {}) or {}).get(
        "caught_object_start_object_offset", (0.0, 0.0, 0.005)
    )
    return np.asarray(raw, dtype=float).reshape(-1)[:3]


def _grasp_ee_position(env: Any, object_position: np.ndarray) -> np.ndarray:
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    center_getter = getattr(env, "_caught_object_start_hold_center", None)
    center = center_getter() if callable(center_getter) else None
    if center is None:
        desired = np.asarray(object_position, dtype=float).reshape(3).copy()
        desired[2] += 0.09
        return desired
    center_arr = np.asarray(center, dtype=float).reshape(3)
    return ee + (np.asarray(object_position, dtype=float).reshape(3) - _held_offset(env) - center_arr)


def _desired_object_position(env: Any, instruction_type: str, current: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float).reshape(3)
    if instruction_type in PLACEMENT_TYPES:
        ref = np.asarray(env._reference_object_position(default=current), dtype=float).reshape(3)
        desired = current.copy()
        desired[:2] = ref[:2]
        # Keep the held object safely above the receptacle until release.
        clearance = 0.10 if instruction_type == "put_into_bowl" else 0.09
        desired[2] = max(float(current[2]), float(ref[2] + clearance))
        return desired
    if instruction_type in {"move_left_of_object", "move_right_of_object"}:
        ref = np.asarray(env._reference_object_position(default=current), dtype=float).reshape(3)
        offset = float(dict(env._task_metadata).get("relation_left_right_offset", 0.10))
        sign = -1.0 if instruction_type == "move_left_of_object" else 1.0
        desired = current.copy()
        desired[0] = float(ref[0] + sign * offset)
        desired[1] = float(ref[1])
        desired[2] = max(float(current[2]), float(ref[2] + 0.04))
        return desired
    if instruction_type == "move_between_objects":
        first = np.asarray(env._reference_object_position(default=current), dtype=float).reshape(3)
        second = np.asarray(
            env._reference_object_position(second=True, default=current), dtype=float
        ).reshape(3)
        desired = current.copy()
        desired[:2] = 0.5 * (first[:2] + second[:2])
        desired[2] = max(float(current[2]), float(0.5 * (first[2] + second[2]) + 0.04))
        return desired
    return current.copy()


def _shell3_position_only_action(
    env: Any,
    oracle: OracleState,
    obj: np.ndarray,
    desired: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Exploit the configured no-grasp sparse predicate using only RL actions."""
    action = np.zeros(5, dtype=np.float32)
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    delta_xy = np.asarray(desired[:2] - obj[:2], dtype=float)
    distance = float(np.linalg.norm(delta_xy))
    if distance <= 1e-8:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = delta_xy / distance

    # Enter the success zone with margin before opening, because the fingers can
    # nudge a free object while the release command is applied.
    instruction_type = oracle.instruction_type
    if instruction_type in PLACEMENT_TYPES:
        tolerance = float(dict(env._task_metadata).get("put_container_xy_tolerance", 0.08))
    elif instruction_type == "move_between_objects":
        tolerance = float(dict(env._task_metadata).get("between_xy_tolerance", 0.06))
    else:
        zone = float(dict(env._task_metadata).get("move_relation_success_zone_size", 0.06))
        tolerance = 0.5 * zone
    release_ready = bool(distance <= max(0.008, tolerance * 0.85))
    if release_ready:
        action[4] = 1.0
        oracle.phase = "shell3_position_only_goal_reached_open_for_release_gate"
        return action, oracle.phase

    if oracle.phase == "shell3_position_only_push_into_sparse_success_region":
        object_ahead = float(np.dot(obj[:2] - ee[:2], direction))
        if object_ahead < -0.010 or float(np.linalg.norm(obj[:2] - ee[:2])) > 0.130:
            # The end effector has passed the freely moving target.  Reposition
            # behind its live pose for another genuine contact push.
            oracle.shell3_push_preposition = None
            oracle.shell3_push_contact = None
            oracle.phase = "shell3_position_only_retreat_above_live_target"

    if oracle.shell3_push_preposition is None or oracle.shell3_push_contact is None:
        behind = np.asarray(obj[:2] - direction * 0.085, dtype=float)
        ee_min = float(getattr(env, "_ee_min_z", obj[2] + 0.04))
        safe_z = max(float(obj[2] + 0.10), ee_min + 0.055)
        oracle.shell3_push_preposition = np.array([behind[0], behind[1], safe_z], dtype=float)
        oracle.shell3_push_contact = np.array([behind[0], behind[1], ee_min], dtype=float)
        if oracle.phase != "shell3_position_only_retreat_above_live_target":
            oracle.phase = "shell3_position_only_preposition_behind_target"

    assert oracle.shell3_push_preposition is not None
    assert oracle.shell3_push_contact is not None
    gripper = float(env._get_gripper_opening() or 0.0)
    if oracle.phase == "shell3_position_only_retreat_above_live_target":
        retreat = ee.copy()
        retreat[2] = oracle.shell3_push_preposition[2]
        if abs(float(ee[2] - retreat[2])) <= 0.018:
            oracle.phase = "shell3_position_only_preposition_behind_target"
        else:
            action[:3] = _target_action(env, retreat, max_norm=0.65)
            action[4] = _gripper_action_to_target(env, 0.0)
            return action, oracle.phase
    if oracle.phase == "shell3_position_only_preposition_behind_target":
        if float(np.linalg.norm(ee - oracle.shell3_push_preposition)) <= 0.025 and gripper <= 0.12:
            oracle.phase = "shell3_position_only_descend_to_push_contact"
    if oracle.phase == "shell3_position_only_descend_to_push_contact":
        if float(np.linalg.norm(ee - oracle.shell3_push_contact)) <= 0.020:
            oracle.phase = "shell3_position_only_push_into_sparse_success_region"

    if oracle.phase == "shell3_position_only_preposition_behind_target":
        action[:3] = _target_action(env, oracle.shell3_push_preposition, max_norm=0.65)
        action[4] = _gripper_action_to_target(env, 0.0)
    elif oracle.phase == "shell3_position_only_descend_to_push_contact":
        action[:3] = _target_action(env, oracle.shell3_push_contact, max_norm=0.60)
        action[4] = _gripper_action_to_target(env, 0.0)
    else:
        # Re-aim at the live goal each step while preserving the original contact
        # height.  The current task metadata evaluates position, not grasp.
        action[0] = float(np.clip(direction[0] * 0.85, -0.85, 0.85))
        action[1] = float(np.clip(direction[1] * 0.85, -0.85, 0.85))
        action[2] = float(
            np.clip((oracle.shell3_push_contact[2] - ee[2]) / 0.04, -0.20, 0.20)
        )
        action[4] = _gripper_action_to_target(env, 0.0)
    return action, oracle.phase


def _oracle_action(
    env: Any,
    oracle: OracleState,
    info: dict[str, Any],
) -> tuple[np.ndarray, str]:
    instruction_type = oracle.instruction_type
    action = np.zeros(5, dtype=np.float32)
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    obj = np.asarray(env._current_manipulated_object_position(default=env._goal_position), dtype=float).reshape(3)

    if (
        float(info.get("reverse_frontier_raw_success", 0.0)) >= 0.5
        and float(info.get("reverse_frontier_policy_gate_satisfied", 0.0)) < 0.5
    ):
        oracle.phase = "hold_raw_success_until_required_policy_decision"
        return action, oracle.phase

    if instruction_type == "move_to_object":
        if oracle.move_to_start_z is None:
            oracle.move_to_start_z = float(ee[2])
        target = ee.copy()
        target[:2] = obj[:2]
        target[2] = float(oracle.move_to_start_z)
        action[:3] = _target_action(env, target, max_norm=0.60)
        oracle.phase = "move_end_effector_in_strict_xy_plane_to_target"
        return action, oracle.phase

    if instruction_type in {"push_left", "push_right"}:
        sign = -1.0 if instruction_type == "push_left" else 1.0
        if oracle.push_start_y is None:
            oracle.push_start_y = float(obj[1])
        action[0] = float(0.88 * sign)
        action[1] = float(np.clip((oracle.push_start_y - obj[1]) / 0.04, -0.25, 0.25))
        action[2] = float(np.clip((obj[2] + 0.045 - ee[2]) / 0.04, -0.20, 0.20))
        action[4] = _gripper_action_to_target(env, 0.0)
        oracle.phase = "push_target_along_commanded_axis"
        return action, oracle.phase

    if instruction_type not in PLACEMENT_TYPES + RELATION_TYPES:
        raise ValueError(f"No imitation oracle for {instruction_type!r}.")

    # Replan from the live reference pose just like the training policy replans
    # from a fresh observation. Receptacles and reference objects are free
    # MuJoCo bodies and can drift after contact; caching their reset pose makes
    # a visually correct controller chase a stale goal.
    oracle.desired_object_position = _desired_object_position(env, instruction_type, obj)
    desired = np.asarray(oracle.desired_object_position, dtype=float).reshape(3)
    env._recorder_oracle_desired_object_position = desired.copy()
    pinned = bool(getattr(env, "_caught_object_start_active", False))
    caught = bool(info.get("caught_object_is_target", False)) or pinned
    if oracle.shell_id == 3 and not pinned and instruction_type != "put_into_bowl":
        return _shell3_position_only_action(env, oracle, obj, desired)
    if (
        oracle.shell_id >= 3
        and oracle.caught_once
        and not caught
        and (
            float(np.linalg.norm(ee[:2] - obj[:2])) > 0.050
            or float(ee[2] - obj[2]) > 0.105
        )
    ):
        # The wide-object catch boolean can remain false, so use live geometry
        # to notice a genuinely lost contact and re-center for another grasp.
        oracle.caught_once = False
        oracle.grasp_lifted = False
        oracle.grasp_lift_goal_object_z = None
        oracle.grasp_steps = 0

    # Shells 0-3 are progressively farther held/pinned starts. Shells 4-6
    # deliberately enter the unpinned branch below to learn close/catch,
    # approach/catch, and the full randomized grasp-to-placement task.
    if oracle.shell_id < 3 or caught or oracle.caught_once:
        if caught:
            oracle.caught_once = True
        if instruction_type in PLACEMENT_TYPES + RELATION_TYPES and oracle.shell_id >= 4 and pinned:
            if oracle.grasp_lift_goal_object_z is None:
                ref = np.asarray(
                    env._reference_object_position(default=obj), dtype=float
                ).reshape(3)
                if instruction_type == "move_between_objects":
                    second_ref = np.asarray(
                        env._reference_object_position(second=True, default=ref),
                        dtype=float,
                    ).reshape(3)
                    ref[2] = max(float(ref[2]), float(second_ref[2]))
                oracle.grasp_lift_goal_object_z = max(
                    float(obj[2] + 0.070), float(ref[2] + 0.090)
                )
            if not oracle.grasp_lifted:
                if float(obj[2]) >= float(oracle.grasp_lift_goal_object_z) - 0.010:
                    oracle.grasp_lifted = True
                else:
                    lift_target = ee.copy()
                    lift_target[2] += 0.015
                    action[:3] = _target_action(env, lift_target, max_norm=0.65)
                    action[4] = _gripper_action_to_target(env, _held_opening(env))
                    oracle.phase = "post_catch_lift_target_clear_of_reference"
                    return action, oracle.phase
        if instruction_type == "put_into_bowl" and oracle.shell_id >= 3 and not pinned:
            if oracle.grasp_lift_goal_object_z is None:
                ref = np.asarray(
                    env._reference_object_position(default=obj), dtype=float
                ).reshape(3)
                oracle.grasp_lift_goal_object_z = max(
                    float(obj[2] + 0.080), float(ref[2] + 0.130)
                )
            if not oracle.grasp_lifted:
                if float(obj[2]) >= float(oracle.grasp_lift_goal_object_z) - 0.012:
                    oracle.grasp_lifted = True
                else:
                    # Verify a genuine contact grasp by lifting vertically before
                    # translating.  The next policy call drops back to re-grasp
                    # if the live object does not follow the end effector.
                    lift_target = ee.copy()
                    lift_target[2] += 0.012
                    action[:3] = _target_action(env, lift_target, max_norm=0.22)
                    action[4] = _gripper_action_to_target(env, _physical_grasp_target(env))
                    oracle.phase = "shell3_lift_grasped_target_clear_of_bowl_rim"
                    return action, oracle.phase
        xy_error = float(np.linalg.norm(obj[:2] - desired[:2]))
        z_error = float(abs(obj[2] - desired[2]))
        xy_ready = 0.045 if instruction_type == "put_into_bowl" else 0.010
        z_ready = 0.070 if instruction_type == "put_into_bowl" else 0.018
        if xy_error > xy_ready or z_error > z_ready:
            ee_target = ee + (desired - obj)
            action[:3] = _target_action(
                env,
                ee_target,
                max_norm=0.60 if pinned else 0.20,
            )
            grip_target = _held_opening(env) if pinned else _physical_grasp_target(env)
            action[4] = _gripper_action_to_target(env, grip_target)
            oracle.phase = "carry_held_target_to_success_region"
            return action, oracle.phase
        action[4] = 1.0
        oracle.phase = "open_gripper_to_release_in_success_region"
        return action, oracle.phase

    grasp_ee = _grasp_ee_position(env, obj)
    grasp_error = float(np.linalg.norm(ee - grasp_ee))
    grasp_xy_error = float(np.linalg.norm(ee[:2] - grasp_ee[:2]))
    ee_min_z = float(getattr(env, "_ee_min_z", float("-inf")))
    at_workspace_floor = bool(
        np.isfinite(ee_min_z)
        and ee[2] <= ee_min_z + 0.004
        and grasp_ee[2] <= ee_min_z + 0.004
    )
    gripper = float(env._get_gripper_opening() or 0.0)
    grasp_target = _physical_grasp_target(env)
    if 0 < oracle.grasp_steps < 12:
        # Once closing has begun, do not reopen merely because contact nudges the
        # live object a centimetre and shifts the ideal geometric center.
        action[4] = _gripper_action_to_target(env, grasp_target)
        oracle.grasp_steps += 1
        oracle.phase = "shell3_close_fingers_and_stabilize_physical_grasp"
        return action, oracle.phase
    if grasp_error > 0.010 and not (at_workspace_floor and grasp_xy_error <= 0.025):
        action[:3] = _target_action(env, grasp_ee, max_norm=0.65)
        action[4] = _gripper_action_to_target(env, 1.0)
        oracle.phase = "shell3_descend_open_gripper_around_target"
        return action, oracle.phase

    # Advance when the *commanded* actuator target has converged.  Measured
    # opening is contact-dependent and can legitimately remain a few percent
    # wider while the object is squeezed between the fingers.
    commanded_gripper = float(env._get_gripper_target())
    if commanded_gripper > grasp_target + 0.008 or oracle.grasp_steps < 12:
        action[4] = _gripper_action_to_target(env, grasp_target)
        oracle.grasp_steps += 1
        oracle.phase = "shell3_close_fingers_and_stabilize_physical_grasp"
        return action, oracle.phase

    if bool((getattr(env, "_curriculum_reset_info", {}) or {}).get("curriculum_grasp_required", False)):
        # A close that did not engage the environment's contact/alignment latch
        # is not a catch. Reopen and recenter on the live object instead of
        # pretending it was grasped and pushing it toward the goal.
        oracle.grasp_steps = 0
        oracle.caught_once = False
        oracle.grasp_lifted = False
        oracle.grasp_lift_goal_object_z = None
        action[4] = _gripper_action_to_target(env, 1.0)
        oracle.phase = "grasp_retry_reopen_and_recenter_on_live_target"
        return action, oracle.phase

    # The generic detector calls a gripper "closed" only below 35% opening, while
    # wide YCB objects are physically held around 60-90% opening.  After centered
    # contact stabilization, continue with a gentle probe even if that detector
    # remains false; subsequent control still depends on the live object pose.
    oracle.caught_once = True
    lift_target = ee.copy()
    lift_target[2] += 0.012
    action[:3] = _target_action(env, lift_target, max_norm=0.22)
    action[4] = _gripper_action_to_target(env, grasp_target)
    oracle.phase = "shell3_probe_lift_for_caught_target"
    return action, oracle.phase


def _trace_row(
    *,
    mode: str,
    instruction_type: str,
    shell_id: int,
    seed: int,
    attempt: int,
    policy_step: int,
    policy_call: int,
    phase: str,
    action: np.ndarray,
    reward: float,
    info: dict[str, Any],
    env: Any,
) -> dict[str, Any]:
    ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
    obj = np.asarray(env._current_manipulated_object_position(default=env._goal_position), dtype=float).reshape(3)
    ref = np.asarray(env._reference_object_position(default=obj), dtype=float).reshape(3)
    desired = np.asarray(
        getattr(env, "_recorder_oracle_desired_object_position", np.full(3, np.nan)),
        dtype=float,
    ).reshape(3)
    row: dict[str, Any] = {
        "mode": mode,
        "instruction_type": instruction_type,
        "shell_id": int(shell_id),
        "seed": int(seed),
        "attempt": int(attempt),
        "policy_step": int(policy_step),
        "policy_call": int(policy_call),
        "chunk_action_index": 0,
        "new_imitation_vla_output": 1,
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
            "reverse_pin_active": int(bool(getattr(env, "_caught_object_start_active", False))),
            "dynamic_grasp_latched": int(
                bool(getattr(env, "_reverse_frontier_dynamic_grasp_latched", False))
            ),
            "grasp_latch_status": str(
                getattr(env, "_reverse_frontier_grasp_latch_status", "inactive")
            ),
            "grasp_latch_distance": float(
                getattr(env, "_reverse_frontier_grasp_latch_distance", float("nan"))
            ),
            "grasp_latch_xy_distance": float(
                getattr(env, "_reverse_frontier_grasp_latch_xy_distance", float("nan"))
            ),
            "grasp_latch_z_distance": float(
                getattr(env, "_reverse_frontier_grasp_latch_z_distance", float("nan"))
            ),
            "grasp_latch_contact_count": int(
                getattr(env, "_reverse_frontier_grasp_latch_contact_count", 0)
            ),
            "ee_x": float(ee[0]),
            "ee_y": float(ee[1]),
            "ee_z": float(ee[2]),
            "object_x": float(obj[0]),
            "object_y": float(obj[1]),
            "object_z": float(obj[2]),
            "reference_x": float(ref[0]),
            "reference_y": float(ref[1]),
            "reference_z": float(ref[2]),
            "oracle_desired_x": float(desired[0]),
            "oracle_desired_y": float(desired[1]),
            "oracle_desired_z": float(desired[2]),
            "gripper_opening": float(info.get("gripper_opening", 0.0)),
            "gripper_target": float(info.get("gripper_target", 0.0)),
            "sparse_success": float(info.get("sparse_success", 0.0)),
            "ever_grasped": float(info.get("ever_grasped", 0.0)),
            "relation_grasp_history_required": float(
                info.get("relation_grasp_history_required", 0.0)
            ),
            "relation_grasp_history_ok": float(
                info.get("relation_grasp_history_ok", 0.0)
            ),
            "reverse_frontier_raw_success": float(
                info.get("reverse_frontier_raw_success", info.get("success", False))
            ),
            "reverse_frontier_policy_gate_satisfied": float(
                info.get("reverse_frontier_policy_gate_satisfied", 0.0)
            ),
            "reverse_frontier_policy_steps_remaining": int(
                info.get("reverse_frontier_policy_steps_remaining", -1)
            ),
            "relation_error": float(info.get("relation_error", 0.0)),
            "relation_motion_ok": float(info.get("relation_motion_ok", 0.0)),
            "relation_grasp_ok": float(info.get("relation_grasp_ok", 0.0)),
            "put_release_ok": float(info.get("put_release_ok", 0.0)),
            "put_container_z_error": float(info.get("put_container_z_error", 0.0)),
            "target_motion_xy": float(info.get("target_motion_xy", 0.0)),
            "signed_relation_offset": float(info.get("signed_relation_offset", 0.0)),
            "push_support_ok": float(info.get("push_support_ok", 1.0)),
            "shell_policy_steps_low": int(info.get("curriculum_shell_policy_steps_low", -1)),
            "shell_policy_steps_high": int(info.get("curriculum_shell_policy_steps_high", -1)),
            "shell_target_policy_steps": int(
                info.get("curriculum_shell_target_policy_steps", -1)
            ),
        }
    )
    return row


def _write_video(frames: list[np.ndarray], output: Path, *, fps: float, keep_frames: bool) -> None:
    if not frames:
        raise RuntimeError("No frames captured.")
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
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _attempt_rollout(
    *,
    config_path: Path,
    config: dict[str, Any],
    mode: str,
    instruction_type: str,
    instruction_index: int,
    shell_id: int,
    target_object: str,
    seed: int,
    attempt: int,
    run_dir: Path,
    width: int,
    height: int,
    fps: float,
    keep_frames: bool,
    quiet: bool,
    save_failure: bool,
    shell3_max_initial_goal_distance: float,
) -> dict[str, Any]:
    env, metadata, env_updates = _build_env(
        config_path=config_path,
        config=config,
        instruction_type=instruction_type,
        seed=seed,
        quiet=quiet,
    )
    renderer: mj.Renderer | None = None
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    options = _reset_options(instruction_type, shell_id, target_object)
    try:
        with _temporary_env(env_updates), _maybe_silence(quiet):
            _, reset_info = env.reset(seed=seed, options=options)
        renderer = mj.Renderer(env.sim.model, height=int(height), width=int(width))
        camera = _camera(env)
        instruction = str(reset_info.get("language_instruction", instruction_type))
        initial_obj = np.asarray(
            env._current_manipulated_object_position(default=env._goal_position), dtype=float
        ).reshape(3)
        initial_ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
        desired_at_reset = _desired_object_position(env, instruction_type, initial_obj)
        initial_goal_distance = float(np.linalg.norm(desired_at_reset[:2] - initial_obj[:2]))
        prefiltered = bool(
            shell_id >= 3
            and instruction_type in PLACEMENT_TYPES + RELATION_TYPES
            and not bool(reset_info.get("curriculum_target_grasped", False))
            and not bool(reset_info.get("curriculum_grasp_required", False))
            and initial_goal_distance > float(shell3_max_initial_goal_distance)
        )
        initial_info = dict(reset_info)
        initial_info.update(
            success=False,
            simulation_state_valid=True,
            gripper_opening=float(env._get_gripper_opening() or 0.0),
            caught_object_is_target=bool(reset_info.get("curriculum_target_grasped", False)),
        )
        frames.append(
            _capture(
                env=env,
                renderer=renderer,
                camera=camera,
                mode=mode,
                instruction=instruction,
                shell_id=shell_id,
                phase="real_reverse_frontier_reset",
                policy_step=0,
                policy_call=0,
                action=np.zeros(5, dtype=np.float32),
                reward=0.0,
                info=initial_info,
            )
        )

        oracle = OracleState(instruction_type=instruction_type, shell_id=shell_id)
        final_info = initial_info
        final_reward = 0.0
        terminated = False
        truncated = False
        for policy_step in range(1, (0 if prefiltered else int(env.max_steps)) + 1):
            oracle.policy_call += 1
            action, phase = _oracle_action(env, oracle, final_info)
            with _temporary_env(env_updates), _maybe_silence(quiet):
                _, reward, terminated, truncated, step_info = env.step(action)
            final_info = dict(step_info)
            final_reward = float(reward)
            rows.append(
                _trace_row(
                    mode=mode,
                    instruction_type=instruction_type,
                    shell_id=shell_id,
                    seed=seed,
                    attempt=attempt,
                    policy_step=policy_step,
                    policy_call=oracle.policy_call,
                    phase=phase,
                    action=action,
                    reward=reward,
                    info=final_info,
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
                    shell_id=shell_id,
                    phase=phase,
                    policy_step=policy_step,
                    policy_call=oracle.policy_call,
                    action=action,
                    reward=reward,
                    info=final_info,
                )
            )
            if terminated or truncated:
                break

        success = bool(final_info.get("success", False))
        policy_low, policy_high = _policy_step_bounds(int(env.hold_steps), shell_id)
        policy_low = int(
            reset_info.get("curriculum_shell_policy_steps_low", policy_low)
        )
        policy_high = int(
            reset_info.get("curriculum_shell_policy_steps_high", policy_high)
        )
        policy_steps_within_shell_range = bool(
            policy_low <= len(rows) <= policy_high
        )
        # Training samples a minimum decision gate inside the shell range.  The
        # upper bound is not an episode timeout: a rollout may legitimately
        # finish later (for example, while waiting for a released object to
        # settle).  Select videos with the same rule used by training instead
        # of imposing an artificial recorder-only maximum.
        minimum_decision_gate_satisfied = bool(
            final_info.get(
                "reverse_frontier_policy_gate_satisfied",
                len(rows) >= policy_low,
            )
        )
        selected_success = bool(success and minimum_decision_gate_satisfied)
        final_obj = np.asarray(
            env._current_manipulated_object_position(default=env._goal_position), dtype=float
        ).reshape(3)
        final_ee = np.asarray(env._get_ee_position(), dtype=float).reshape(3)
        if frames:
            frames.extend([frames[-1].copy() for _ in range(max(6, int(round(fps))))])

        mode_dir = run_dir / mode / instruction_type
        outcome = "success" if selected_success else "failure"
        stem = f"{mode}_like_{instruction_type}_shell_{shell_id:02d}_{outcome}"
        video_path = mode_dir / f"{stem}.mp4"
        trace_path = mode_dir / f"{stem}_actions.csv"
        save_artifacts = bool(selected_success or (save_failure and not prefiltered))
        if save_artifacts:
            _write_video(frames, video_path, fps=fps, keep_frames=keep_frames)
            _write_trace(trace_path, rows)

        return {
            "mode": mode,
            "instruction_type": instruction_type,
            "instruction_index": int(instruction_index),
            "instruction": instruction,
            "shell_id": int(shell_id),
            "seed": int(seed),
            "attempt": int(attempt),
            "success": success,
            "selected_success": selected_success,
            "minimum_decision_gate_satisfied": minimum_decision_gate_satisfied,
            "policy_steps_within_shell_range": policy_steps_within_shell_range,
            "shell_policy_steps_low": policy_low,
            "shell_policy_steps_high": policy_high,
            "prefiltered": prefiltered,
            "prefilter_reason": (
                "shell3_initial_goal_distance_above_success-video_limit" if prefiltered else ""
            ),
            "initial_goal_distance_m": initial_goal_distance,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "policy_steps": len(rows),
            "simulator_steps_per_policy_action": int(1 + env.hold_steps),
            "video": video_path.as_posix() if save_artifacts else "",
            "action_trace": trace_path.as_posix() if save_artifacts else "",
            "scene": str(reset_info.get("scene", "")),
            "scene_objects": list(reset_info.get("scene_objects", [])),
            "wrapper_xml": str(reset_info.get("wrapper_xml", "")),
            "target_object_catalog": str(reset_info.get("target_object_catalog", "")),
            "reference_object_catalog": str(reset_info.get("reference_object_catalog", "")),
            "second_reference_object_catalog": str(
                reset_info.get("second_reference_object_catalog", "")
            ),
            "curriculum_shell_profile": str(reset_info.get("curriculum_shell_profile", "")),
            "curriculum_shell_relation": str(reset_info.get("curriculum_shell_relation", "")),
            "curriculum_shell_target_sim_steps": int(
                reset_info.get("curriculum_shell_target_sim_steps", -1)
            ),
            "curriculum_shell_target_policy_steps": int(
                reset_info.get("curriculum_shell_target_policy_steps", -1)
            ),
            "curriculum_target_grasped": bool(
                reset_info.get("curriculum_target_grasped", False)
            ),
            "initial_ee_position": [float(value) for value in initial_ee],
            "final_ee_position": [float(value) for value in final_ee],
            "initial_object_position": [float(value) for value in initial_obj],
            "final_object_position": [float(value) for value in final_obj],
            "object_displacement": [float(value) for value in final_obj - initial_obj],
            "final_reward": final_reward,
            "final_sparse_success": float(final_info.get("sparse_success", 0.0)),
            "final_reverse_frontier_raw_success": float(
                final_info.get("reverse_frontier_raw_success", final_info.get("success", False))
            ),
            "final_reverse_frontier_policy_gate_satisfied": float(
                final_info.get("reverse_frontier_policy_gate_satisfied", 0.0)
            ),
            "final_reverse_frontier_policy_steps_remaining": int(
                final_info.get("reverse_frontier_policy_steps_remaining", -1)
            ),
            "final_relation_error": float(final_info.get("relation_error", 0.0)),
            "final_simulation_state_valid": bool(
                final_info.get("simulation_state_valid", True)
            ),
            "final_simulation_state_reason": str(
                final_info.get("simulation_state_reason", "")
            ),
            "hold_steps": int(env.hold_steps),
            "action_step_xyz": float(env.action_step_xyz),
            "action_step_yaw": float(env.action_step_yaw),
            "action_step_gripper": float(env.action_step_gripper),
            "reset_options": options,
            "task_metadata": metadata,
        }
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _selected_instructions(config: dict[str, Any], requested: Sequence[str] | None) -> tuple[str, ...]:
    configured = _dedupe(dict(config.get("task") or {}).get("instruction_types"))
    if requested:
        unknown = [item for item in requested if item not in configured]
        if unknown:
            raise ValueError(f"Instructions not in active config: {unknown}; configured={list(configured)}")
        return tuple(str(item) for item in requested)
    return configured


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record successful training-like and validation-like Reverse Frontier shell episodes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--instruction-types", nargs="+", default=None)
    parser.add_argument(
        "--shells",
        nargs="+",
        type=int,
        choices=range(len(SHELL_POLICY_DECISION_BOUNDS)),
        default=None,
        help="Optional shell subset; by default each instruction records all configured shells.",
    )
    parser.add_argument("--target-object", default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-step", type=int, default=6_700_000)
    parser.add_argument("--attempts-per-case", type=int, default=20)
    parser.add_argument(
        "--shell3-max-initial-goal-distance",
        type=float,
        default=0.20,
        help=(
            "For successful-example recording, retry shell-3 placement/relation seeds whose "
            "initial target-to-goal XY distance exceeds this value."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--verbose-env", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    _, _, rl_args = _config_parts(config)
    instructions = _selected_instructions(config, args.instruction_types)
    shell_counts = {
        spec.instruction_id: int(spec.shell_count)
        for spec in get_cdpr_reverse_shell_specs(
            instructions,
            profile=SMOLVLA_COMPLEX_PROFILE,
        )
    }
    shells_by_instruction = {
        instruction: tuple(
            shell_id
            for shell_id in (
                args.shells
                if args.shells is not None
                else range(int(shell_counts[instruction]))
            )
            if int(shell_id) < int(shell_counts[instruction])
        )
        for instruction in instructions
    }
    validation_seed = int(rl_args.get("validation_seed", 1_000_000))
    run_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    successful: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for mode in args.modes:
        for instruction_index, instruction_type in enumerate(instructions):
            for shell_id in shells_by_instruction[instruction_type]:
                case_result: dict[str, Any] | None = None
                for attempt in range(max(1, int(args.attempts_per_case))):
                    seed = _case_seed(
                        mode=mode,
                        base_seed=int(args.seed),
                        validation_seed=validation_seed,
                        global_step=int(args.global_step),
                        instruction_index=instruction_index,
                        shell_id=int(shell_id),
                        attempt=attempt,
                    )
                    print(
                        f"[{mode}] {instruction_type} shell={shell_id} attempt={attempt + 1} "
                        f"seed={seed}",
                        flush=True,
                    )
                    result = _attempt_rollout(
                        config_path=config_path,
                        config=config,
                        mode=str(mode),
                        instruction_type=str(instruction_type),
                        instruction_index=instruction_index,
                        shell_id=int(shell_id),
                        target_object=str(args.target_object),
                        seed=seed,
                        attempt=attempt,
                        run_dir=run_dir,
                        width=int(args.width),
                        height=int(args.height),
                        fps=float(args.fps),
                        keep_frames=bool(args.keep_frames),
                        quiet=not bool(args.verbose_env),
                        save_failure=bool(args.keep_failures),
                        shell3_max_initial_goal_distance=float(
                            args.shell3_max_initial_goal_distance
                        ),
                    )
                    attempts.append(
                        {
                            key: value
                            for key, value in result.items()
                            if key not in {"task_metadata"}
                        }
                    )
                    print(
                        f"  success={result['success']} selected={result['selected_success']} "
                        f"policy_steps={result['policy_steps']} "
                        f"scene={result['scene']}",
                        flush=True,
                    )
                    if result["selected_success"]:
                        case_result = result
                        successful.append(result)
                        break
                if case_result is None:
                    print(
                        f"  FAILED all attempts: {mode}/{instruction_type}/shell_{shell_id}",
                        file=sys.stderr,
                        flush=True,
                    )

    expected_count = len(args.modes) * sum(
        len(shells) for shells in shells_by_instruction.values()
    )
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    renderer_path = Path(__file__).resolve()
    hold_steps = int(rl_args.get("hold_steps", 6))
    manifest = {
        "created_at": datetime.now().isoformat(),
        "config": config_path.as_posix(),
        "config_sha256": config_sha256,
        "renderer": renderer_path.as_posix(),
        "renderer_sha256": hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
        "source_training_entrypoint": (
            ROOT / "scripts" / "train_cdpr_smolvla_complex_grpo_dual_remote.sh"
        ).as_posix(),
        "reverse_shell_implementation": (
            ROOT / "robots" / "cdpr" / "cdpr_dataset" / "cdpr_reverse_shells.py"
        ).as_posix(),
        "training_runtime": (
            ROOT / "rl_vla_bootstrapping" / "policy" / "smolvla_grpo_finetune_cdpr.py"
        ).as_posix(),
        "state_transition_api": "CDPRLanguageRLEnv.step(normalized_5d_action)",
        "recorder_post_reset_pose_writes": False,
        "policy": "deterministic state oracle imitating 8-action SmolVLA chunk outputs",
        "policy_is_learned_checkpoint": False,
        "action_order": list(ACTION_NAMES),
        "chunk_size": int(rl_args.get("chunk_size", 8)),
        "replan_every": int(rl_args.get("replan_every", 1)),
        "hold_steps": hold_steps,
        "simulator_steps_per_policy_action": 1 + hold_steps,
        "video_selection_rule": (
            "Training-equivalent: successful after the shell's sampled minimum policy-"
            "decision gate. The shell range upper bound is not an episode timeout."
        ),
        "nominal_shell_sim_step_bounds": [
            [low * (1 + hold_steps), high * (1 + hold_steps)]
            for low, high in SHELL_POLICY_DECISION_BOUNDS
        ],
        "nominal_shell_policy_step_bounds": [
            list(_policy_step_bounds(hold_steps, shell_id))
            for shell_id in range(len(SHELL_POLICY_DECISION_BOUNDS))
        ],
        "shell3_discontinuity": False,
        "shell_progression": (
            "Shells 0-3 use progressively farther held starts. Plate, left/right relation, "
            "and between-object tasks add aligned catch, near-object approach/catch, and "
            "full randomized approach/catch shells 4-6."
        ),
        "modes": list(args.modes),
        "instruction_types": list(instructions),
        "shell_counts": shell_counts,
        "shells_by_instruction": {
            instruction: [int(value) for value in shells]
            for instruction, shells in shells_by_instruction.items()
        },
        "target_object": str(args.target_object),
        "base_training_seed": int(args.seed),
        "validation_seed": validation_seed,
        "validation_global_step": int(args.global_step),
        "expected_video_count": expected_count,
        "successful_video_count": len(successful),
        "all_successful": len(successful) == expected_count,
        "results": successful,
        "attempts": attempts,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path, flush=True)
    return 0 if manifest["all_successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
