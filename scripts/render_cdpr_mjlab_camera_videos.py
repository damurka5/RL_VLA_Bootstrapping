#!/usr/bin/env python3
"""Render representative CDPR training/validation videos from both policy cameras.

The default ``auto`` backend uses the production MJ-Lab/MJWarp backend when its
pinned CUDA runtime is available. On a laptop without CUDA it uses MuJoCo's
local renderer with the same fixed MJWarp MJCF, curated RoboCasa visual
meshes, CDPR primitive colliders, camera names, and visual variants. The
manifest records which backend produced the videos so a local reference render
cannot be mistaken for a CUDA/MJWarp result.

These are deterministic scripted-policy previews.  They show the observations
and robot motion used by training/validation, but do not claim to evaluate a
learned checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CDPR_ROOT = ROOT / "robots" / "cdpr"
if str(CDPR_ROOT) not in sys.path:
    sys.path.insert(0, str(CDPR_ROOT))

from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    CATALOG_TO_ID,
    GEOM_SLOT_NAMES,
    OBJECT_VARIANTS,
    compile_catalog_variant_models,
    slot_geom_name,
)


DEFAULT_XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr_mjwarp_smoke.xml"
DEFAULT_OUTPUT = ROOT / "runs" / "cdpr_mjlab_camera_videos"
CAMERA_NAMES = ("overview", "ee_camera")
ACTION_NAMES = ("x", "y", "z", "yaw", "gripper")
WORKSPACE_MIN = np.array((-0.28, -0.28, 0.20), dtype=np.float64)
WORKSPACE_MAX = np.array((0.28, 0.28, 1.20), dtype=np.float64)
ACTION_STEP = np.array((0.015, 0.015, 0.015, 0.08, 0.05), dtype=np.float64)
PHYSICS_SUBSTEPS = 7
SUPPORT_SURFACE_Z = 0.15
CAUGHT_OBJECT_OFFSET_Z = -0.0025

# Normalized finger positions calibrated for caught objects used by these
# scripted scenarios. These are physical grip targets, not visual-mesh
# bounding-box estimates.
CAUGHT_GRIPPER_OPENINGS = {
    "robocasa_apple": 0.46,
}


@dataclass(frozen=True)
class Phase:
    name: str
    target_position: tuple[float, float, float]
    target_yaw: float
    target_gripper: float
    max_steps: int
    position_tolerance: float = 0.018
    minimum_steps: int = 3
    translation_action_limit: float = 0.72


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    instruction: str
    catalogs: tuple[str, str, str, str]
    object_positions: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    object_yaws: tuple[float, float, float, float]
    ee_start: tuple[float, float, float]
    ee_yaw: float
    gripper_opening: float
    texture_variant: int
    background_rgba: tuple[float, float, float, float]
    gripper_shade: float
    phases: tuple[Phase, ...]


def _rest_z(catalog: str) -> float:
    return SUPPORT_SURFACE_Z + float(OBJECT_VARIANTS[catalog].rest_height)


def _caught_opening(catalog: str) -> float:
    try:
        return float(CAUGHT_GRIPPER_OPENINGS[catalog])
    except KeyError as exc:
        raise KeyError(f"No caught-object finger target for {catalog!r}.") from exc


def _caught_position(
    ee_position: Sequence[float],
) -> tuple[float, float, float]:
    return (
        float(ee_position[0]),
        float(ee_position[1]),
        float(ee_position[2]) + CAUGHT_OBJECT_OFFSET_Z,
    )


SCENARIOS: dict[str, Scenario] = {
    "training_put_into_bowl": Scenario(
        name="training_put_into_bowl",
        mode="training",
        instruction="put apple into bowl",
        catalogs=(
            "robocasa_apple",
            "robocasa_bowl",
            "robocasa_carrot",
            "robocasa_mug",
        ),
        object_positions=(
            _caught_position((-0.16, -0.12, 0.38)),
            (0.12, 0.09, _rest_z("robocasa_bowl")),
            (-0.13, 0.12, _rest_z("robocasa_carrot")),
            (0.16, -0.12, _rest_z("robocasa_mug")),
        ),
        object_yaws=(0.0, 0.25, -0.5, 0.0),
        ee_start=(-0.16, -0.12, 0.38),
        ee_yaw=-0.35,
        gripper_opening=_caught_opening("robocasa_apple"),
        texture_variant=2,
        background_rgba=(0.035, 0.050, 0.075, 1.0),
        gripper_shade=0.70,
        phases=(
            Phase(
                "carry_at_safe_height",
                (0.12, 0.09, 0.38),
                0.35,
                _caught_opening("robocasa_apple"),
                64,
            ),
            Phase(
                "position_above_bowl",
                (0.12, 0.09, 0.36),
                0.35,
                _caught_opening("robocasa_apple"),
                40,
                position_tolerance=0.009,
                minimum_steps=6,
                translation_action_limit=0.25,
            ),
            Phase(
                "lower_into_bowl",
                (0.12, 0.09, 0.30),
                0.35,
                _caught_opening("robocasa_apple"),
                40,
                position_tolerance=0.009,
                minimum_steps=6,
                translation_action_limit=0.25,
            ),
            Phase(
                "release",
                (0.12, 0.09, 0.30),
                0.35,
                1.0,
                28,
                position_tolerance=0.012,
                minimum_steps=18,
                translation_action_limit=0.10,
            ),
            Phase("lift_clear", (0.12, 0.09, 0.43), 0.0, 1.0, 40),
            Phase("retreat", (0.02, -0.02, 0.43), 0.0, 1.0, 40),
        ),
    ),
    "training_put_on_plate": Scenario(
        name="training_put_on_plate",
        mode="training",
        instruction="put apple on plate",
        catalogs=(
            "robocasa_apple",
            "robocasa_plate",
            "robocasa_bell_pepper",
            "robocasa_orange",
        ),
        object_positions=(
            _caught_position((0.16, -0.12, 0.38)),
            (-0.12, 0.09, _rest_z("robocasa_plate")),
            (0.13, 0.12, _rest_z("robocasa_bell_pepper")),
            (-0.16, -0.13, _rest_z("robocasa_orange")),
        ),
        object_yaws=(0.0, 0.2, 0.0, -0.3),
        ee_start=(0.16, -0.12, 0.38),
        ee_yaw=0.25,
        gripper_opening=_caught_opening("robocasa_apple"),
        texture_variant=6,
        background_rgba=(0.035, 0.050, 0.075, 1.0),
        gripper_shade=0.86,
        phases=(
            Phase(
                "carry_at_safe_height",
                (-0.12, 0.09, 0.38),
                -0.25,
                _caught_opening("robocasa_apple"),
                64,
            ),
            Phase(
                "position_over_plate",
                (-0.12, 0.09, 0.34),
                -0.25,
                _caught_opening("robocasa_apple"),
                40,
                position_tolerance=0.009,
                minimum_steps=6,
                translation_action_limit=0.25,
            ),
            Phase(
                "lower_onto_plate",
                (-0.12, 0.09, 0.30),
                -0.25,
                _caught_opening("robocasa_apple"),
                40,
                position_tolerance=0.009,
                minimum_steps=6,
                translation_action_limit=0.25,
            ),
            Phase(
                "release",
                (-0.12, 0.09, 0.30),
                -0.25,
                1.0,
                28,
                position_tolerance=0.012,
                minimum_steps=18,
                translation_action_limit=0.10,
            ),
            Phase("lift_clear", (-0.12, 0.09, 0.43), 0.0, 1.0, 40),
            Phase("retreat", (0.0, -0.08, 0.43), 0.0, 1.0, 40),
        ),
    ),
    "validation_move_to_apple": Scenario(
        name="validation_move_to_apple",
        mode="validation",
        instruction="move to apple",
        catalogs=(
            "robocasa_apple",
            "robocasa_banana",
            "robocasa_tomato",
            "robocasa_potato",
        ),
        object_positions=(
            (-0.12, 0.08, _rest_z("robocasa_apple")),
            (0.13, 0.11, _rest_z("robocasa_banana")),
            (-0.14, -0.13, _rest_z("robocasa_tomato")),
            (0.14, -0.10, _rest_z("robocasa_potato")),
        ),
        object_yaws=(0.0, 0.2, -0.4, 0.0),
        ee_start=(0.18, -0.16, 0.40),
        ee_yaw=0.0,
        gripper_opening=1.0,
        texture_variant=0,
        background_rgba=(0.035, 0.050, 0.075, 1.0),
        gripper_shade=0.94,
        phases=(
            Phase("move_above_target", (-0.12, 0.08, 0.40), 0.35, 1.0, 46),
            Phase("approach_target", (-0.12, 0.08, 0.31), 0.35, 1.0, 18),
            Phase("inspect_target", (-0.12, 0.08, 0.265), -0.35, 1.0, 18),
            Phase("finish", (-0.12, 0.08, 0.29), 0.0, 1.0, 10, minimum_steps=8),
        ),
    ),
    "validation_move_to_carrot": Scenario(
        name="validation_move_to_carrot",
        mode="validation",
        instruction="move to carrot",
        catalogs=(
            "robocasa_carrot",
            "robocasa_mug",
            "robocasa_plate",
            "robocasa_bowl",
        ),
        object_positions=(
            (0.10, -0.08, _rest_z("robocasa_carrot")),
            (-0.13, 0.11, _rest_z("robocasa_mug")),
            (-0.15, -0.12, _rest_z("robocasa_plate")),
            (0.14, -0.13, _rest_z("robocasa_bowl")),
        ),
        object_yaws=(0.0, 0.0, -0.4, 0.0),
        ee_start=(-0.18, 0.16, 0.40),
        ee_yaw=-0.5,
        gripper_opening=1.0,
        texture_variant=0,
        background_rgba=(0.035, 0.050, 0.075, 1.0),
        gripper_shade=0.94,
        phases=(
            Phase("move_above_target", (0.10, -0.08, 0.40), 0.45, 1.0, 44),
            Phase("approach_target", (0.10, -0.08, 0.32), 0.45, 1.0, 18),
            Phase("inspect_target", (0.10, -0.08, 0.285), -0.35, 1.0, 16),
            Phase("finish", (0.10, -0.08, 0.31), 0.0, 1.0, 10, minimum_steps=8),
        ),
    ),
}


def _name_id(mujoco: Any, model: Any, objtype: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, objtype, name))
    if value < 0:
        raise RuntimeError(f"Missing MJCF element {name!r}.")
    return value


def _yaw_quaternions(yaws: Sequence[float]) -> np.ndarray:
    values = np.asarray(yaws, dtype=np.float64)
    output = np.zeros((len(values), 4), dtype=np.float64)
    output[:, 0] = np.cos(0.5 * values)
    output[:, 3] = np.sin(0.5 * values)
    return output


def _angle_delta(target: float, current: float) -> float:
    return float((target - current + math.pi) % (2.0 * math.pi) - math.pi)


def _mjwarp_available() -> bool:
    required = ("torch", "warp", "mujoco_warp", "mjlab")
    if not all(importlib.util.find_spec(name) is not None for name in required):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class PreviewRunner:
    backend_name: str
    physics_dtype: str
    exact_production_backend: bool

    def reset(self, scenario: Scenario) -> None:
        raise NotImplementedError

    def observe(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def render(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MuJoCoReferenceRunner(PreviewRunner):
    backend_name = "mujoco_reference_of_mjlab_scene"
    physics_dtype = "float64"
    exact_production_backend = False

    def __init__(self, *, xml_path: Path, width: int, height: int) -> None:
        import mujoco as mj
        from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import (
            HeadlessCDPRSimulation,
        )

        self.mj = mj
        self.width = int(width)
        self.height = int(height)
        self.sim = HeadlessCDPRSimulation(
            str(xml_path),
            record_trajectory=False,
            use_model_cache=False,
            timestep=0.002,
            render_enabled=False,
        )
        self.sim.initialize()
        self._catalog_models = compile_catalog_variant_models(
            self.mj, xml_path
        )
        self.renderer: Any | None = None
        self.scene_option = self.mj.MjvOption()
        self.mj.mjv_defaultOption(self.scene_option)
        self.scene_option.geomgroup[:] = 0
        # Group 3 is collision-only. Keeping it out of the preview matches the
        # production MJWarp policy-camera contract and prevents black proxies.
        self.scene_option.geomgroup[:3] = 1
        self._scenario: Scenario | None = None
        self._yaw_target = 0.0
        self._gripper_target = 1.0

    def _apply_catalogs(self, catalogs: Sequence[str]) -> None:
        for slot, catalog in enumerate(catalogs):
            reference = self._catalog_models[catalog]
            for geom_slot in GEOM_SLOT_NAMES:
                name = slot_geom_name(slot, geom_slot)
                geom = _name_id(
                    self.mj,
                    self.sim.model,
                    self.mj.mjtObj.mjOBJ_GEOM,
                    name,
                )
                reference_geom = _name_id(
                    self.mj,
                    reference,
                    self.mj.mjtObj.mjOBJ_GEOM,
                    name,
                )
                self.sim.model.geom_dataid[geom] = reference.geom_dataid[
                    reference_geom
                ]
                self.sim.model.geom_size[geom] = reference.geom_size[
                    reference_geom
                ]
                self.sim.model.geom_pos[geom] = reference.geom_pos[
                    reference_geom
                ]
                self.sim.model.geom_quat[geom] = reference.geom_quat[
                    reference_geom
                ]
                self.sim.model.geom_rgba[geom] = reference.geom_rgba[
                    reference_geom
                ]
                self.sim.model.geom_matid[geom] = reference.geom_matid[
                    reference_geom
                ]
                self.sim.model.geom_aabb[geom] = reference.geom_aabb[
                    reference_geom
                ]
                self.sim.model.geom_rbound[geom] = reference.geom_rbound[
                    reference_geom
                ]
            body = _name_id(
                self.mj,
                self.sim.model,
                self.mj.mjtObj.mjOBJ_BODY,
                f"mjwarp_object_slot_{slot}",
            )
            reference_body = _name_id(
                self.mj,
                reference,
                self.mj.mjtObj.mjOBJ_BODY,
                f"mjwarp_object_slot_{slot}",
            )
            self.sim.model.body_mass[body] = reference.body_mass[
                reference_body
            ]
            self.sim.model.body_inertia[body] = reference.body_inertia[
                reference_body
            ]
        self.mj.mj_setConst(self.sim.model, self.sim.data)
        self.mj.mj_forward(self.sim.model, self.sim.data)

    def _set_object_poses(
        self,
        positions: Sequence[Sequence[float]],
        yaws: Sequence[float],
    ) -> None:
        quaternions = _yaw_quaternions(yaws)
        for slot in range(4):
            joint = _name_id(
                self.mj,
                self.sim.model,
                self.mj.mjtObj.mjOBJ_JOINT,
                f"mjwarp_object_slot_{slot}_free",
            )
            qadr = int(self.sim.model.jnt_qposadr[joint])
            dofadr = int(self.sim.model.jnt_dofadr[joint])
            self.sim.data.qpos[qadr : qadr + 3] = positions[slot]
            self.sim.data.qpos[qadr + 3 : qadr + 7] = quaternions[slot]
            self.sim.data.qvel[dofadr : dofadr + 6] = 0.0

    def reset(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self.sim.reset_data_state()
        self._apply_catalogs(scenario.catalogs)
        self._set_object_poses(scenario.object_positions, scenario.object_yaws)

        ee_joint = _name_id(
            self.mj, self.sim.model, self.mj.mjtObj.mjOBJ_JOINT, "ee_free"
        )
        ee_qadr = int(self.sim.model.jnt_qposadr[ee_joint])
        ee_dofadr = int(self.sim.model.jnt_dofadr[ee_joint])
        self.sim.data.qpos[ee_qadr : ee_qadr + 3] = scenario.ee_start
        self.sim.data.qvel[ee_dofadr : ee_dofadr + 6] = 0.0
        self.sim.data.qpos[self.sim.jnt_yaw_qadr] = float(scenario.ee_yaw)
        yaw_dofadr = int(
            self.sim.model.jnt_dofadr[
                _name_id(
                    self.mj,
                    self.sim.model,
                    self.mj.mjtObj.mjOBJ_JOINT,
                    "ee_yaw",
                )
            ]
        )
        self.sim.data.qvel[yaw_dofadr] = 0.0

        normalized = float(np.clip(scenario.gripper_opening, 0.0, 1.0))
        if self.sim.jnt_finger_l_qadr is not None:
            self.sim.data.qpos[self.sim.jnt_finger_l_qadr] = (
                self.sim.gripper_joint_min
                + normalized
                * (self.sim.gripper_joint_max - self.sim.gripper_joint_min)
            )
        self.mj.mj_forward(self.sim.model, self.sim.data)
        self.sim._sync_controller_geometry_from_state()
        self.sim._match_sliders_to_ee_lengths(max_iter=12, tol=1.0e-6)
        self.sim.target_pos = self.sim.get_end_effector_position().copy()
        self.sim.set_yaw(scenario.ee_yaw)
        self.sim.set_gripper(normalized)
        self._set_object_poses(scenario.object_positions, scenario.object_yaws)

        desk_visual = _name_id(
            self.mj,
            self.sim.model,
            self.mj.mjtObj.mjOBJ_GEOM,
            "mjwarp_desk_surface_visual",
        )
        material = _name_id(
            self.mj,
            self.sim.model,
            self.mj.mjtObj.mjOBJ_MATERIAL,
            f"mjwarp_desk_mat_{scenario.texture_variant}",
        )
        self.sim.model.geom_matid[desk_visual] = material
        for name in (
            "palm",
            "finger_l_shoulder",
            "finger_l_link",
            "finger_l_tip",
            "left_finger_pad",
            "finger_r_shoulder",
            "finger_r_link",
            "finger_r_tip",
            "right_finger_pad",
        ):
            geom = _name_id(
                self.mj, self.sim.model, self.mj.mjtObj.mjOBJ_GEOM, name
            )
            self.sim.model.geom_rgba[geom, :3] = float(scenario.gripper_shade)

        self._yaw_target = float(scenario.ee_yaw)
        self._gripper_target = normalized
        self.mj.mj_forward(self.sim.model, self.sim.data)

        if self.renderer is not None:
            self.renderer.close()
        self.renderer = self.mj.Renderer(
            self.sim.model, height=self.height, width=self.width
        )

    def observe(self) -> dict[str, Any]:
        object_bodies = [
            _name_id(
                self.mj,
                self.sim.model,
                self.mj.mjtObj.mjOBJ_BODY,
                f"mjwarp_object_slot_{slot}",
            )
            for slot in range(4)
        ]
        object_positions = np.asarray(
            self.sim.data.xpos[object_bodies], dtype=np.float64
        ).copy()
        object_quaternions = np.asarray(
            self.sim.data.xquat[object_bodies], dtype=np.float64
        ).copy()
        return {
            "ee_position": self.sim.get_end_effector_position().copy(),
            "ee_yaw": float(self.sim.get_yaw()),
            "gripper_opening": float(self.sim.get_gripper_opening()),
            "gripper_target": float(self.sim.get_gripper_target()),
            "target_position": self.sim.get_target_position().copy(),
            "tendon_lengths": self.sim.get_cable_lengths().copy(),
            "object_position": object_positions[0],
            "reference_position": object_positions[1],
            "object_positions": object_positions,
            "object_quaternions": object_quaternions,
            "pinned": False,
        }

    def step(self, action: np.ndarray) -> dict[str, Any]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ee = self.sim.get_end_effector_position()
        target = np.clip(
            ee + action[:3] * ACTION_STEP[:3], WORKSPACE_MIN, WORKSPACE_MAX
        )
        self._yaw_target = float(
            np.clip(
                self._yaw_target + action[3] * ACTION_STEP[3],
                -math.pi,
                math.pi,
            )
        )
        self._gripper_target = float(
            np.clip(
                self._gripper_target + action[4] * ACTION_STEP[4], 0.0, 1.0
            )
        )
        self.sim.target_pos = target
        self.sim.set_yaw(self._yaw_target)
        self.sim.set_gripper(self._gripper_target)
        for _ in range(PHYSICS_SUBSTEPS):
            self.sim.run_simulation_step(capture_frame=False)
        self.mj.mj_forward(self.sim.model, self.sim.data)
        return self.observe()

    def render(self) -> dict[str, np.ndarray]:
        if self.renderer is None:
            raise RuntimeError("Runner must be reset before rendering.")
        output: dict[str, np.ndarray] = {}
        for camera in CAMERA_NAMES:
            self.renderer.update_scene(
                self.sim.data,
                camera=camera,
                scene_option=self.scene_option,
            )
            self.renderer.scene.flags[
                self.mj.mjtRndFlag.mjRND_SKYBOX
            ] = 1
            output[camera] = np.asarray(
                self.renderer.render(), dtype=np.uint8
            ).copy()
        return output

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.sim.cleanup()


class MJLabMJWarpRunner(PreviewRunner):
    backend_name = "mjlab_mjwarp"
    physics_dtype = "float32"
    exact_production_backend = True

    def __init__(
        self,
        *,
        xml_path: Path,
        width: int,
        height: int,
        device: str,
    ) -> None:
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_backend import CDPRBackendConfig
        from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import (
            MJLabMJWarpCDPRBackend,
        )

        self.torch = torch
        self.worlds = 8
        self.backend = MJLabMJWarpCDPRBackend(
            config=CDPRBackendConfig(
                backend="mjlab_mjwarp",
                worlds_per_rank=self.worlds,
                groups_per_rank=1,
                grpo_group_size=8,
                hold_steps=6,
                render_width=int(width),
                render_height=int(height),
                device=str(device),
                xml_path=xml_path,
            ),
            create_renderer=True,
            require_mjlab=True,
        )
        self._yaw_target = 0.0
        self._gripper_target = 1.0

    def _repeat(self, value: Any, *, dtype: Any) -> Any:
        tensor = self.torch.as_tensor(
            value, dtype=dtype, device=self.backend.device
        )
        return tensor.unsqueeze(0).repeat(self.worlds, *([1] * tensor.ndim))

    def reset(self, scenario: Scenario) -> None:
        torch = self.torch
        worlds = torch.arange(
            self.worlds, dtype=torch.int64, device=self.backend.device
        )
        self.backend.reset_worlds(worlds)
        catalogs = self._repeat(
            [CATALOG_TO_ID[name] for name in scenario.catalogs],
            dtype=torch.int64,
        )
        self.backend.set_object_catalogs(catalogs)
        positions = self._repeat(
            scenario.object_positions, dtype=torch.float32
        )
        quaternions = self._repeat(
            _yaw_quaternions(scenario.object_yaws), dtype=torch.float32
        )
        self.backend.set_free_body_poses(
            self.backend.object_body_ids, positions, quaternions
        )
        ee_positions = self._repeat(scenario.ee_start, dtype=torch.float32)
        yaws = torch.full(
            (self.worlds,),
            float(scenario.ee_yaw),
            dtype=torch.float32,
            device=self.backend.device,
        )
        self.backend.set_end_effector_poses(ee_positions, yaws)
        openings = torch.full(
            (self.worlds,),
            float(scenario.gripper_opening),
            dtype=torch.float32,
            device=self.backend.device,
        )
        self.backend.set_gripper_openings(openings)
        self.backend.set_free_body_poses(
            self.backend.object_body_ids, positions, quaternions
        )
        self.backend.set_visual_variants(
            torch.full(
                (self.worlds,),
                int(scenario.texture_variant),
                dtype=torch.int64,
                device=self.backend.device,
            ),
            self._repeat(scenario.background_rgba, dtype=torch.float32),
            torch.full(
                (self.worlds,),
                float(scenario.gripper_shade),
                dtype=torch.float32,
                device=self.backend.device,
            ),
        )
        self._yaw_target = float(scenario.ee_yaw)
        self._gripper_target = float(scenario.gripper_opening)

    def observe(self) -> dict[str, Any]:
        state = self.backend.low_dim_observations()
        object_positions = (
            state.object_positions[0].detach().cpu().numpy().copy()
        )
        object_quaternions = (
            state.object_quaternions[0].detach().cpu().numpy().copy()
        )
        return {
            "ee_position": state.ee_position[0].detach().cpu().numpy().copy(),
            "ee_yaw": float(state.ee_yaw[0].item()),
            "gripper_opening": float(state.gripper_opening[0].item()),
            "gripper_target": float(self._gripper_target),
            "target_position": (
                state.target_position[0].detach().cpu().numpy().copy()
            ),
            "tendon_lengths": (
                state.tendon_lengths[0].detach().cpu().numpy().copy()
            ),
            "object_position": object_positions[0],
            "reference_position": object_positions[1],
            "object_positions": object_positions,
            "object_quaternions": object_quaternions,
            "pinned": False,
        }

    def step(self, action: np.ndarray) -> dict[str, Any]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        actions = self._repeat(action, dtype=self.torch.float32)
        active = self.torch.ones(
            (self.worlds,), dtype=self.torch.bool, device=self.backend.device
        )
        self.backend.step(actions, active)
        self._yaw_target = float(
            np.clip(
                self._yaw_target + action[3] * ACTION_STEP[3],
                -math.pi,
                math.pi,
            )
        )
        self._gripper_target = float(
            np.clip(
                self._gripper_target + action[4] * ACTION_STEP[4], 0.0, 1.0
            )
        )
        return self.observe()

    def render(self) -> dict[str, np.ndarray]:
        cameras = self.backend.render_policy_cameras()

        def frame(value: Any) -> np.ndarray:
            array = (
                value[0].permute(1, 2, 0).detach().cpu().numpy()
            )
            return np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)

        return {
            "overview": frame(cameras.overview),
            "ee_camera": frame(cameras.wrist),
        }

    def close(self) -> None:
        self.backend.close()


def _make_runner(
    backend: str,
    *,
    xml_path: Path,
    width: int,
    height: int,
    device: str,
) -> PreviewRunner:
    selected = backend
    if selected == "auto":
        selected = "mjlab-mjwarp" if _mjwarp_available() else "mujoco-reference"
    if selected == "mjlab-mjwarp":
        return MJLabMJWarpRunner(
            xml_path=xml_path,
            width=width,
            height=height,
            device=device,
        )
    return MuJoCoReferenceRunner(
        xml_path=xml_path,
        width=width,
        height=height,
    )


def _policy_action(state: dict[str, Any], phase: Phase) -> np.ndarray:
    action = np.zeros(5, dtype=np.float32)
    ee = np.asarray(state["ee_position"], dtype=np.float64)
    delta = np.asarray(phase.target_position, dtype=np.float64) - ee
    translation_limit = float(phase.translation_action_limit)
    action[:3] = np.clip(
        delta / ACTION_STEP[:3], -translation_limit, translation_limit
    )
    action[3] = float(
        np.clip(
            _angle_delta(phase.target_yaw, float(state["ee_yaw"]))
            / ACTION_STEP[3],
            -0.65,
            0.65,
        )
    )
    action[4] = float(
        np.clip(
            (phase.target_gripper - float(state["gripper_opening"]))
            / ACTION_STEP[4],
            -1.0,
            1.0,
        )
    )
    return action


def _phase_complete(state: dict[str, Any], phase: Phase, steps: int) -> bool:
    if steps < int(phase.minimum_steps):
        return False
    position_error = float(
        np.linalg.norm(
            np.asarray(phase.target_position, dtype=np.float64)
            - np.asarray(state["ee_position"], dtype=np.float64)
        )
    )
    gripper_error = abs(
        float(phase.target_gripper) - float(state["gripper_opening"])
    )
    return position_error <= float(phase.position_tolerance) and gripper_error <= 0.06


def _scenario_metrics(
    scenario: Scenario,
    state: dict[str, Any],
) -> dict[str, Any]:
    target = np.asarray(state["object_position"], dtype=np.float64)
    reference = np.asarray(state["reference_position"], dtype=np.float64)
    xy_error = float(np.linalg.norm(target[:2] - reference[:2]))
    z_offset = float(target[2] - reference[2])
    is_put = scenario.name in {
        "training_put_into_bowl",
        "training_put_on_plate",
    }
    released = float(state["gripper_opening"]) >= 0.55
    success = bool(
        is_put
        and released
        and xy_error <= 0.03
        and -0.01 <= z_offset <= 0.12
    )
    return {
        "xy_error": xy_error,
        "z_offset": z_offset,
        "released": released,
        "success": success,
    }


def _telemetry_lines(
    *,
    camera_label: str,
    scenario: Scenario,
    phase: str,
    step: int,
    action: np.ndarray,
    state: dict[str, Any],
) -> list[str]:
    action = np.asarray(action, dtype=np.float64).reshape(5)
    applied = action * ACTION_STEP
    ee = np.asarray(state["ee_position"], dtype=np.float64)
    target = np.asarray(state["target_position"], dtype=np.float64)
    obj = np.asarray(state["object_position"], dtype=np.float64)
    reference = np.asarray(state["reference_position"], dtype=np.float64)
    cables = np.asarray(state["tendon_lengths"], dtype=np.float64).reshape(4)
    metrics = _scenario_metrics(scenario, state)

    def vector(values: np.ndarray, digits: int = 2) -> str:
        return " ".join(f"{float(value):+.{digits}f}" for value in values)

    return [
        f"{camera_label} | {scenario.mode.upper()} | {scenario.instruction}",
        f"phase={phase}  policy_step={step}",
        f"VLA-like normalized action [{vector(action, 2)}]",
        f"executed delta [{vector(applied[:3], 3)} m | "
        f"yaw={applied[3]:+.3f} rad grip={applied[4]:+.3f}]",
        f"ee_xyz=[{vector(ee, 3)}] controller_target=[{vector(target, 3)}]",
        f"controller_error={np.linalg.norm(target - ee):.4f} m  "
        f"gripper={float(state['gripper_opening']):.3f}->"
        f"{float(state['gripper_target']):.3f}",
        f"object_xyz=[{vector(obj, 3)}] receptacle_xyz=[{vector(reference, 3)}]",
        f"receptacle_xy_error={metrics['xy_error']:.4f} m  "
        f"z_offset={metrics['z_offset']:+.4f} m  success={metrics['success']}",
        f"cable_lengths_m=[{vector(cables, 3)}]",
    ]


def _annotated_frame(
    frame: np.ndarray,
    *,
    camera_label: str,
    scenario: Scenario,
    phase: str,
    step: int,
    action: np.ndarray,
    state: dict[str, Any],
) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    panel_height = 210 if width < 480 else 134
    image = Image.new("RGB", (width, height + panel_height), (14, 17, 23))
    image.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_chars = max(24, int((width - 14) // 6))
    wrapped: list[str] = []
    for line in _telemetry_lines(
        camera_label=camera_label,
        scenario=scenario,
        phase=phase,
        step=step,
        action=action,
        state=state,
    ):
        wrapped.extend(
            textwrap.wrap(
                line,
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    for index, line in enumerate(wrapped):
        y = height + 6 + 11 * index
        if y + 10 > height + panel_height:
            break
        color = (174, 210, 255) if "VLA-like" in line else (238, 241, 245)
        draw.text((7, y), line, fill=color, font=font)
    return np.asarray(image)


def _composite_frame(
    overview: np.ndarray,
    wrist: np.ndarray,
    *,
    scenario: Scenario,
    phase: str,
    step: int,
    action: np.ndarray,
    state: dict[str, Any],
) -> np.ndarray:
    combined = np.concatenate((overview, wrist), axis=1)
    return _annotated_frame(
        combined,
        camera_label="overview | ee_camera (wrist)",
        scenario=scenario,
        phase=phase,
        step=step,
        action=action,
        state=state,
    )


def _write_video(
    frames: Sequence[np.ndarray],
    output: Path,
    *,
    fps: float,
    keep_frames: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output.parent / f".{output.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    try:
        for index, frame in enumerate(frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                frames_dir / f"{index:05d}.png"
            )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
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
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    finally:
        if not keep_frames and frames_dir.exists():
            shutil.rmtree(frames_dir)


def _write_trace(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trace_row(
    *,
    scenario: Scenario,
    phase: str,
    step: int,
    action: np.ndarray,
    state: dict[str, Any],
) -> dict[str, Any]:
    ee = np.asarray(state["ee_position"], dtype=np.float64)
    controller_target = np.asarray(state["target_position"], dtype=np.float64)
    obj = np.asarray(state["object_position"], dtype=np.float64)
    reference = np.asarray(state["reference_position"], dtype=np.float64)
    tendon_lengths = np.asarray(
        state["tendon_lengths"], dtype=np.float64
    ).reshape(4)
    metrics = _scenario_metrics(scenario, state)
    row: dict[str, Any] = {
        "mode": scenario.mode,
        "scenario": scenario.name,
        "instruction": scenario.instruction,
        "phase": phase,
        "action_step": int(step),
    }
    for index, name in enumerate(ACTION_NAMES):
        row[f"action_{name}"] = float(action[index])
        row[f"executed_delta_{name}"] = float(action[index] * ACTION_STEP[index])
    row.update(
        {
            "ee_x": float(ee[0]),
            "ee_y": float(ee[1]),
            "ee_z": float(ee[2]),
            "ee_yaw": float(state["ee_yaw"]),
            "controller_target_x": float(controller_target[0]),
            "controller_target_y": float(controller_target[1]),
            "controller_target_z": float(controller_target[2]),
            "controller_error": float(
                np.linalg.norm(controller_target - ee)
            ),
            "gripper_opening": float(state["gripper_opening"]),
            "gripper_target": float(state["gripper_target"]),
            "object_x": float(obj[0]),
            "object_y": float(obj[1]),
            "object_z": float(obj[2]),
            "reference_x": float(reference[0]),
            "reference_y": float(reference[1]),
            "reference_z": float(reference[2]),
            "receptacle_xy_error": float(metrics["xy_error"]),
            "receptacle_z_offset": float(metrics["z_offset"]),
            "scenario_success": int(bool(metrics["success"])),
            "cable_1_length": float(tendon_lengths[0]),
            "cable_2_length": float(tendon_lengths[1]),
            "cable_3_length": float(tendon_lengths[2]),
            "cable_4_length": float(tendon_lengths[3]),
            "object_pinned": int(bool(state["pinned"])),
        }
    )
    return row


def _render_scenario(
    *,
    runner: PreviewRunner,
    scenario: Scenario,
    output_dir: Path,
    fps: float,
    terminal_hold_seconds: float,
    keep_frames: bool,
    write_composite: bool,
) -> dict[str, Any]:
    runner.reset(scenario)
    frames: dict[str, list[np.ndarray]] = {name: [] for name in CAMERA_NAMES}
    composite: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    state = runner.observe()
    initial_grasp_offset = (
        np.asarray(state["object_position"], dtype=np.float64)
        - np.asarray(state["ee_position"], dtype=np.float64)
    )
    initial = runner.render()
    zero_action = np.zeros(5, dtype=np.float32)
    initial_hold = max(2, int(round(0.6 * fps)))
    for _ in range(initial_hold):
        for camera in CAMERA_NAMES:
            frames[camera].append(
                _annotated_frame(
                    initial[camera],
                    camera_label=camera,
                    scenario=scenario,
                    phase="reset",
                    step=0,
                    action=zero_action,
                    state=state,
                )
            )
        if write_composite:
            composite.append(
                _composite_frame(
                    initial["overview"],
                    initial["ee_camera"],
                    scenario=scenario,
                    phase="reset",
                    step=0,
                    action=zero_action,
                    state=state,
                )
            )

    total_steps = 0
    phase_counts: dict[str, int] = {}
    phase_completed: dict[str, bool] = {}
    held_gripper_openings: list[float] = []
    held_object_slips: list[float] = []
    for phase in scenario.phases:
        used = 0
        completed = False
        for _ in range(int(phase.max_steps)):
            action = _policy_action(state, phase)
            state = runner.step(action)
            total_steps += 1
            used += 1
            rendered = runner.render()
            for camera in CAMERA_NAMES:
                frames[camera].append(
                    _annotated_frame(
                        rendered[camera],
                        camera_label=camera,
                        scenario=scenario,
                        phase=phase.name,
                        step=total_steps,
                        action=action,
                        state=state,
                    )
                )
            if write_composite:
                composite.append(
                    _composite_frame(
                        rendered["overview"],
                        rendered["ee_camera"],
                        scenario=scenario,
                        phase=phase.name,
                        step=total_steps,
                        action=action,
                        state=state,
                    )
                )
            trace.append(
                _trace_row(
                    scenario=scenario,
                    phase=phase.name,
                    step=total_steps,
                    action=action,
                    state=state,
                )
            )
            if phase.target_gripper < 0.55:
                held_gripper_openings.append(
                    float(state["gripper_opening"])
                )
                held_object_slips.append(
                    float(
                        np.linalg.norm(
                            np.asarray(
                                state["object_position"], dtype=np.float64
                            )
                            - np.asarray(
                                state["ee_position"], dtype=np.float64
                            )
                            - initial_grasp_offset
                        )
                    )
                )
            if _phase_complete(state, phase, used):
                completed = True
                break
        phase_counts[phase.name] = used
        phase_completed[phase.name] = completed

    hold = max(2, int(round(float(terminal_hold_seconds) * fps)))
    for _ in range(hold):
        for camera in CAMERA_NAMES:
            frames[camera].append(frames[camera][-1].copy())
        if write_composite:
            composite.append(composite[-1].copy())

    video_paths: dict[str, str] = {}
    for camera in CAMERA_NAMES:
        path = output_dir / f"{scenario.name}_{camera}.mp4"
        _write_video(frames[camera], path, fps=fps, keep_frames=keep_frames)
        video_paths[camera] = path.as_posix()
    if write_composite:
        path = output_dir / f"{scenario.name}_both_cameras.mp4"
        _write_video(composite, path, fps=fps, keep_frames=keep_frames)
        video_paths["both_cameras"] = path.as_posix()

    trace_path = output_dir / f"{scenario.name}_actions.csv"
    _write_trace(trace_path, trace)
    final_object = np.asarray(state["object_position"], dtype=np.float64)
    initial_object = np.asarray(
        scenario.object_positions[0], dtype=np.float64
    )
    metrics = _scenario_metrics(scenario, state)
    settled_openings = held_gripper_openings[5:]
    held_error = (
        max(
            abs(value - float(scenario.gripper_opening))
            for value in settled_openings
        )
        if settled_openings
        else 0.0
    )
    held_max_step = (
        float(np.max(np.abs(np.diff(settled_openings))))
        if len(settled_openings) >= 2
        else 0.0
    )
    held_object_max_slip = max(held_object_slips, default=0.0)
    return {
        "scenario": scenario.name,
        "mode": scenario.mode,
        "instruction": scenario.instruction,
        "frames": len(frames["overview"]),
        "action_steps": total_steps,
        "phase_action_steps": phase_counts,
        "phase_completed": phase_completed,
        "duration_seconds": len(frames["overview"]) / float(fps),
        "videos": video_paths,
        "action_trace": trace_path.as_posix(),
        "final_ee_position": [
            float(value) for value in np.asarray(state["ee_position"]).reshape(3)
        ],
        "final_object_position": [float(value) for value in final_object],
        "object_displacement": [
            float(value) for value in (final_object - initial_object)
        ],
        "final_gripper_opening": float(state["gripper_opening"]),
        "held_gripper_steady_max_error": float(held_error),
        "held_gripper_max_step_after_settle": held_max_step,
        "held_object_max_slip": float(held_object_max_slip),
        "final_object_pinned": bool(state["pinned"]),
        "receptacle_xy_error": float(metrics["xy_error"]),
        "receptacle_z_offset": float(metrics["z_offset"]),
        "scenario_success": bool(metrics["success"]),
    }


def _write_contact_sheet(
    runner: PreviewRunner,
    selected: Iterable[Scenario],
    output: Path,
) -> None:
    rows: list[Image.Image] = []
    font = ImageFont.load_default()
    for scenario in selected:
        runner.reset(scenario)
        cameras = runner.render()
        height, width = cameras["overview"].shape[:2]
        row = Image.new("RGB", (2 * width, height + 28), (16, 19, 25))
        row.paste(Image.fromarray(cameras["overview"]), (0, 28))
        row.paste(Image.fromarray(cameras["ee_camera"]), (width, 28))
        draw = ImageDraw.Draw(row)
        draw.text((7, 5), f"{scenario.name} | overview", fill=(245, 245, 245), font=font)
        draw.text((width + 7, 5), "ee_camera", fill=(245, 245, 245), font=font)
        draw.text((7, 16), scenario.instruction, fill=(174, 210, 255), font=font)
        rows.append(row)
    sheet = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height for row in rows)),
        (16, 19, 25),
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render representative MJ-Lab CDPR videos from both policy cameras."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend",
        choices=("auto", "mjlab-mjwarp", "mujoco-reference"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=[
            "training_put_into_bowl",
            "training_put_on_plate",
            "validation_move_to_apple",
            "validation_move_to_carrot",
        ],
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=60.0 / PHYSICS_SUBSTEPS)
    parser.add_argument("--terminal-hold-seconds", type=float, default=1.0)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--composite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--timestamped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a timestamped child directory under --output-dir.",
    )
    parser.add_argument(
        "--verify-scenarios",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exit nonzero when a selected training put scenario misses a "
            "phase, drops the caught object, or fails the 3 cm placement test."
        ),
    )
    args = parser.parse_args()
    if args.width < 32 or args.height < 32:
        parser.error("Video dimensions must be at least 32x32.")
    if args.fps <= 0.0:
        parser.error("--fps must be positive.")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required to encode MP4 videos.")

    xml_path = args.xml.expanduser().resolve()
    if not xml_path.exists():
        parser.error(f"MJCF does not exist: {xml_path}")
    base_output = args.output_dir.expanduser().resolve()
    output_dir = (
        base_output / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.timestamped
        else base_output
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [SCENARIOS[name] for name in args.scenarios]

    os.environ.setdefault("RLVLA_CDPR_QUIET", "1")
    runner = _make_runner(
        args.backend,
        xml_path=xml_path,
        width=int(args.width),
        height=int(args.height),
        device=str(args.device),
    )
    print(
        f"backend={runner.backend_name} exact_production_backend="
        f"{runner.exact_production_backend}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    try:
        for scenario in selected:
            print(
                f"[{scenario.mode}] {scenario.name}: {scenario.instruction}",
                flush=True,
            )
            result = _render_scenario(
                runner=runner,
                scenario=scenario,
                output_dir=output_dir,
                fps=float(args.fps),
                terminal_hold_seconds=float(args.terminal_hold_seconds),
                keep_frames=bool(args.keep_frames),
                write_composite=bool(args.composite),
            )
            results.append(result)
            print(
                f"  {result['action_steps']} actions, "
                f"{result['duration_seconds']:.1f}s, "
                f"object_delta={result['object_displacement']}, "
                f"success={result['scenario_success']}",
                flush=True,
            )
        contact_sheet = output_dir / "camera_contact_sheet.png"
        _write_contact_sheet(runner, selected, contact_sheet)
    finally:
        runner.close()

    failed_training_scenarios = [
        result["scenario"]
        for result in results
        if result["scenario"].startswith("training_put_")
        and (
            not bool(result["scenario_success"])
            or not all(result["phase_completed"].values())
            or float(result["held_gripper_steady_max_error"]) > 0.05
            or float(result["held_gripper_max_step_after_settle"]) > 0.02
            or float(result["held_object_max_slip"]) > 0.03
        )
    ]
    manifest = {
        "created_at": datetime.now().isoformat(),
        "renderer_backend": runner.backend_name,
        "exact_production_backend": runner.exact_production_backend,
        "physics_dtype": runner.physics_dtype,
        "production_backend": "mjlab_mjwarp",
        "exact_robocasa_visual_assets": True,
        "rendered_geom_groups": [0, 1, 2, 4],
        "collision_geom_group": 3,
        "cable_visual_geom_group": 4,
        "preview_type": "deterministic_scripted_policy",
        "videos_include_vla_action_telemetry": True,
        "scenario_verification_enabled": bool(args.verify_scenarios),
        "scenario_verification_passed": not failed_training_scenarios,
        "failed_training_scenarios": failed_training_scenarios,
        "learned_checkpoint_evaluation": False,
        "xml": xml_path.as_posix(),
        "camera_order": ["overview", "ee_camera", "ee_camera"],
        "physical_camera_videos": ["overview", "ee_camera"],
        "third_smolvla_slot": "exact duplicate of ee_camera in production",
        "render_resolution": [int(args.width), int(args.height)],
        "rgb_contract": (
            "MP4 files are uint8 RGB-derived H.264; production MJWarp camera "
            "tensors are normalized float32 BCHW RGB on CUDA."
        ),
        "action_order": list(ACTION_NAMES),
        "action_step": [float(value) for value in ACTION_STEP],
        "hold_steps": PHYSICS_SUBSTEPS - 1,
        "physics_substeps_per_action": PHYSICS_SUBSTEPS,
        "video_fps": float(args.fps),
        "camera_contact_sheet": contact_sheet.as_posix(),
        "local_reference_limitations": (
            []
            if runner.exact_production_backend
            else [
                "Physics is MuJoCo float64 rather than MJWarp float32 CUDA.",
                "Pixels are from MuJoCo's renderer; MJWarp pixels may differ.",
                "The MJCF, curated RoboCasa visuals, CDPR native primitive "
                "colliders, named "
                "cameras, controller action scales, visual variants, and "
                "substep timing are shared.",
            ]
        ),
        "scenarios": [asdict(scenario) for scenario in selected],
        "results": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(manifest_path, flush=True)
    if args.verify_scenarios and failed_training_scenarios:
        print(
            "Training scenario verification failed: "
            + ", ".join(failed_training_scenarios),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
