"""Can the WidowX-200 reach, and can its cameras see, every scene the pipeline scores?

CPU only. No MJWarp, no policy, no physics step.

Scenes. The pipeline evaluates and harvests through ``tools/audit/sil_record.py``,
which builds the VALIDATION resetter (``BatchedReverseFrontierResetter``,
validation seed, frontier 1.0, balanced catalogs, the skill's ``eval_cap``) and
calls ``reset(update_index=0, round_index=r, allow_prelifted=False)``. This
script builds exactly that resetter from the pipeline's own ``eval_config`` and
feeds it a recording stub (``start_distance_probe.build_recording_backend``), so
the poses below are the poses the simulator is handed, not a re-derivation of
the sampling. A GRPO round is 64 scenes x 8 identical copies; one world per
group is audited.

Reach. Every point a scene requires -- the gripper start, the grasp point, the
lift point, the carry segment, the release/hover goal the curriculum publishes
-- is pushed through the controller's own ``joint_targets_from_task_targets``
with the spec ``WidowX200MJWarpBackend._control_spec`` builds (layout XY box,
config Z bounds intersected with the layout's, waist keep-out, 0.02 m reach
margin). A point counts as reachable when the commanded target survives every
clamp to within 2 mm in XY. Separately: raw top-down IK including joint limits,
and the IK swept over the controller's whole +-pi/2 yaw range at the grasp
point (a parallel gripper's needed yaw is defined modulo pi, so full-range
reach means every object orientation is graspable).

Visibility. The scene is rebuilt on CPU MuJoCo with the backend's per-slot
catalog switching (the compiled variant models' geom fields), the arm posed by
the same IK, and both policy cameras rendered at the configured
``render_width x render_height`` with geom groups 0,1,2,4 (the MJWarp render
context's). Segmentation renders give, per object and camera: pixels visible in
the full scene, pixels with the arm removed, and pixels of the object alone --
hence frustum coverage, pixel area, and occlusion split into arm and
other-object parts.

    MUJOCO_GL=cgl RLVLA_EMBODIMENT=widowx200 python tools/audit/widowx200_scene_coverage.py \
        --output tools/audit/out/widowx200_scene_coverage
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/examples/widowx200_smolvla_pipeline.yaml"
# Evaluation rounds (eval_rounds=2) and the SIL harvest rounds
# (harvest_first_round=100, harvest_rounds=4).
DEFAULT_ROUNDS = (0, 1, 100, 101, 102, 103)
# (label, skill, composed)
AUDITS: tuple[tuple[str, str, bool], ...] = (
    ("move_to", "move_to", False),
    ("pick_up", "pick_up", False),
    ("put_into", "put_into", False),
    ("put_into_composed", "put_into", True),
)
XY_TOLERANCE = 0.002
CARRY_SAMPLES = 9
YAW_SWEEP = 13


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------


def pipeline_view(config_path: Path) -> Any:
    from tools.train.run_widowx200_pipeline import PIPELINE_KEY, load_yaml

    raw = load_yaml(config_path)
    spec = raw.pop(PIPELINE_KEY)
    view = types.SimpleNamespace(base=raw, spec=spec, smoke=False)
    view.skill = lambda name: dict(spec["skills"][name])
    return view


def eval_config(view: Any, skill: str, composed: bool) -> dict[str, Any]:
    from tools.train.run_widowx200_pipeline import Pipeline

    return Pipeline.eval_config(view, skill, composed=composed)


def sample_scenes(
    config: Mapping[str, Any], *, cap: float, rounds: Sequence[int]
) -> list[dict[str, Any]]:
    import torch

    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        BatchedReverseFrontierResetter,
        RankLocalCurriculum,
        goal_slots_for_reset,
    )
    from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        ACTIVE_INSTRUCTION_TYPES,
        INSTRUCTION_TO_ID,
    )
    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
        ACTIVE_CDPR_CATALOGS,
    )
    from tools.audit.start_distance_probe import build_recording_backend

    simulator = config["simulator"]
    rl_args = config["training"]["rl"]["args"]
    task = config["task"]
    instructions = tuple(task["instruction_types"])
    layout = RankLocalGroupLayout(
        worlds_per_rank=int(simulator["worlds_per_rank"]),
        groups_per_rank=int(simulator["groups_per_rank"]),
        group_size=int(rl_args["grpo_group_size"]),
    )
    backend = build_recording_backend(torch)
    recorded: dict[str, Any] = {}

    def set_catalogs(ids: Any) -> None:
        recorded["catalogs"] = torch.as_tensor(ids).clone()

    backend.set_object_catalogs = set_catalogs
    resetter = BatchedReverseFrontierResetter(
        backend=backend,
        layout=layout,
        curriculum=RankLocalCurriculum(device=backend.device),
        rank=0,
        base_seed=int(rl_args["validation_seed"]),
        instruction_types=instructions,
        allowed_objects=tuple(task["target_objects"]),
        frontier_probability=1.0,
        rehearsal_probability=0.0,
        balanced_target_catalogs=True,
        task_metadata=dict(task["metadata"]),
    )
    resetter.set_random_start_max_goal_distance(
        {int(INSTRUCTION_TO_ID[name]): float(cap) for name in instructions}
    )
    scenes: list[dict[str, Any]] = []
    group = int(layout.group_size)
    for round_index in rounds:
        reset = resetter.reset(
            update_index=0, round_index=int(round_index), allow_prelifted=False
        )
        state = reset.task_state
        goal_slots = goal_slots_for_reset(torch, reset)
        goal = reset.curriculum_goal_xyz
        for scene, world in enumerate(range(0, int(layout.worlds_per_rank), group)):
            catalogs = [int(v) for v in recorded["catalogs"][world].tolist()]
            positions = backend.object_positions[world].tolist()
            quats = backend.object_quaternions[world].tolist()
            target_slot = int(state.target_slots[world])
            reference_slot = int(state.reference_slots[world])
            scenes.append(
                {
                    "round": int(round_index),
                    "scene": scene,
                    "world": world,
                    "instruction": ACTIVE_INSTRUCTION_TYPES[int(state.instruction_ids[world])],
                    "catalogs": [
                        ACTIVE_CDPR_CATALOGS[c] if c >= 0 else None for c in catalogs
                    ],
                    "positions": positions,
                    "quaternions": quats,
                    "target_slot": target_slot,
                    "reference_slot": reference_slot,
                    "goal_slot": int(goal_slots[world]),
                    "goal_xyz": None if goal is None else goal[world].tolist(),
                    "ee_xyz": backend.ee_positions[world].tolist(),
                    "ee_yaw": float(backend.ee_yaw[world]),
                    "opening": float(backend.gripper_openings[world]),
                }
            )
    return scenes


# --------------------------------------------------------------------------
# Reach
# --------------------------------------------------------------------------


def control_spec(config: Mapping[str, Any]) -> Any:
    """``WidowX200MJWarpBackend._control_spec`` without a backend."""

    from robots.widowx200.widowx200_mujoco import workspace as widowx_workspace
    from robots.widowx200.widowx200_mujoco.controller import (
        WidowX200ControlSpec,
        WidowX200MountPose,
    )

    layout = widowx_workspace.DEFAULT_LAYOUT
    rl_args = config["training"]["rl"]["args"]
    bounds = rl_args.get("controller_workspace_z_bounds") or layout.workspace_z
    z_low = max(float(bounds[0]), float(layout.workspace_z[0]))
    z_high = min(float(bounds[1]), float(layout.workspace_z[1]))
    return WidowX200ControlSpec(
        mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
        hold_steps=int(rl_args.get("hold_steps", 24)),
        workspace_x=tuple(layout.workspace_x),
        workspace_y=tuple(layout.workspace_y),
        workspace_z=(z_low, z_high),
        min_reach_radius=float(layout.min_reach_radius),
    )


def reach_check(spec: Any, points: Sequence[Sequence[float]], yaws: Sequence[float]) -> list[dict[str, Any]]:
    """Per point: does the controller deliver it, and if not, which clamp bit."""

    import numpy as np

    from robots.widowx200.widowx200_mujoco.controller import (
        joint_targets_from_task_targets,
    )
    from robots.widowx200.widowx200_mujoco.kinematics import top_down_ik

    world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    yaw = np.asarray(yaws, dtype=np.float64).reshape(-1)
    solved = joint_targets_from_task_targets(world, yaw, np.ones_like(yaw), spec)
    achieved = np.asarray(solved["achieved_target"])
    base = spec.mount.world_to_base(_numpy_ops(), world)
    radius = np.hypot(base[:, 0], base[:, 1])
    raw_ik = top_down_ik(
        base, yaw - spec.mount.yaw, geometry=spec.geometry, gamma=spec.pitch,
        clamp_to_reach=False,
    )
    rows = []
    for index in range(world.shape[0]):
        x, y, z = world[index]
        xy_error = float(np.hypot(*(achieved[index, :2] - world[index, :2])))
        rows.append(
            {
                "base_radius": float(radius[index]),
                "in_box_xy": bool(
                    spec.workspace_x[0] <= x <= spec.workspace_x[1]
                    and spec.workspace_y[0] <= y <= spec.workspace_y[1]
                ),
                "z_below_floor": float(max(0.0, spec.workspace_z[0] - z)),
                "z_above_ceiling": float(max(0.0, z - spec.workspace_z[1])),
                "outside_keepout": bool(radius[index] >= spec.min_reach_radius),
                "ik_reachable": bool(np.asarray(raw_ik.reachable)[index]),
                "xy_error": xy_error,
                "z_error": float(abs(achieved[index, 2] - z)),
                "delivered": xy_error <= XY_TOLERANCE,
            }
        )
    return rows


def _numpy_ops() -> Any:
    from robots.widowx200.widowx200_mujoco.kinematics import _NumpyOps

    return _NumpyOps()


def yaw_coverage(spec: Any, point: Sequence[float]) -> float:
    """Fraction of the controller's yaw range the IK solves at ``point``."""

    yaws = [
        -0.5 * math.pi + math.pi * index / (YAW_SWEEP - 1) for index in range(YAW_SWEEP)
    ]
    rows = reach_check(spec, [point] * len(yaws), yaws)
    return sum(1 for row in rows if row["ik_reachable"] and row["delivered"]) / len(rows)


def scene_waypoints(
    scene: Mapping[str, Any], metadata: Mapping[str, Any]
) -> list[tuple[str, list[float]]]:
    """The points this instruction requires the tool to occupy."""

    instruction = scene["instruction"]
    target = scene["positions"][scene["target_slot"]]
    offset = float(metadata.get("pick_grasp_height_offset", 0.0))
    points: list[tuple[str, list[float]]] = [("start", list(scene["ee_xyz"]))]
    if scene["goal_xyz"] is not None:
        points.append(("goal", list(scene["goal_xyz"])))
    if instruction == "move_to_object":
        points.append(
            ("hover", [target[0], target[1], float(metadata.get("move_to_object_approach_z", 0.23))])
        )
        return points
    grasp = [target[0], target[1], target[2] + offset]
    lift = [target[0], target[1], target[2] + offset + float(metadata.get("pick_lift_success_height", 0.05))]
    composed_grasp = instruction == "pick_up" or not scene_is_caught(scene, metadata)
    if composed_grasp:
        points += [("grasp", grasp), ("lift", lift)]
    if instruction in ("put_into_plate", "put_into_bowl"):
        container = scene["positions"][scene["reference_slot"]]
        key = "put_plate_release_height" if instruction == "put_into_plate" else "put_bowl_release_height"
        release = [container[0], container[1], container[2] + float(metadata.get(key, 0.07))]
        points.append(("release", release))
        carry_from = lift if composed_grasp else list(scene["ee_xyz"])
        points += [("carry", point) for point in arc_carry(carry_from, release)]
    return points


def arc_carry(start: Sequence[float], end: Sequence[float]) -> list[list[float]]:
    """Carry path interpolated in polar coordinates about the arm's base.

    A straight chord between the two inner lattice cells passes within 0.10 m
    of the waist axis, inside the 0.15 m keep-out, so a chord would score as
    unreachable a carry the arm makes by swinging its waist. The arc is the
    path a top-down arm actually takes; whether the chord crosses the keep-out
    is reported separately (``straight_carry_min_radius``).
    """

    from robots.widowx200.widowx200_mujoco import workspace as widowx_workspace

    cx, cy, _ = widowx_workspace.DEFAULT_LAYOUT.base_position
    r0, r1 = math.hypot(start[0] - cx, start[1] - cy), math.hypot(end[0] - cx, end[1] - cy)
    a0, a1 = math.atan2(start[1] - cy, start[0] - cx), math.atan2(end[1] - cy, end[0] - cx)
    out = []
    for step in range(1, CARRY_SAMPLES + 1):
        t = step / (CARRY_SAMPLES + 1)
        radius, angle = r0 + t * (r1 - r0), a0 + t * (a1 - a0)
        out.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle), start[2] + t * (end[2] - start[2])])
    return out


def straight_carry_min_radius(scene: Mapping[str, Any]) -> float | str:
    """Closest approach of the target->container chord to the waist axis."""

    if scene["reference_slot"] < 0 or not scene["instruction"].startswith("put_into"):
        return ""
    from robots.widowx200.widowx200_mujoco import workspace as widowx_workspace

    cx, cy, _ = widowx_workspace.DEFAULT_LAYOUT.base_position
    a = scene["positions"][scene["target_slot"]]
    b = scene["positions"][scene["reference_slot"]]
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((cx - a[0]) * dx + (cy - a[1]) * dy) / max(dx * dx + dy * dy, 1e-12)))
    return round(math.hypot(a[0] + t * dx - cx, a[1] + t * dy - cy), 4)


def scene_is_caught(scene: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    """A placement start holding its object: the EE sits on the target."""

    target = scene["positions"][scene["target_slot"]]
    ee = scene["ee_xyz"]
    return math.dist(target[:2], ee[:2]) < 0.005 and abs(ee[2] - target[2]) < 0.005


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------


class SceneRenderer:
    """CPU MuJoCo copy of the MJWarp scene with per-slot catalog switching."""

    GROUPS = (0, 1, 2, 4)

    def __init__(
        self,
        xml_path: Path,
        *,
        width: int,
        height: int,
        overview: Mapping[str, Any] | None = None,
    ) -> None:
        import mujoco

        from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
            ACTIVE_CDPR_CATALOGS,
            compile_catalog_variant_models,
        )
        from robots.widowx200.widowx200_mujoco import workspace as widowx_workspace

        self.mujoco = mujoco
        self.variants = compile_catalog_variant_models(mujoco, xml_path)
        self.catalogs = ACTIVE_CDPR_CATALOGS
        # A fresh compile, so editing it never touches the variant tables.
        self.model = compile_catalog_variant_models(mujoco, xml_path)[ACTIVE_CDPR_CATALOGS[0]]
        model = self.model
        for variant in self.variants.values():
            widowx_workspace.mount_widowx200(variant, mujoco=mujoco)
        widowx_workspace.mount_widowx200(model, mujoco=mujoco)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.overview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        self.wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
        if overview:
            if "pos" in overview:
                model.cam_pos[self.overview_id] = overview["pos"]
            if "xyaxes" in overview:
                model.cam_quat[self.overview_id] = _xyaxes_to_quat(overview["xyaxes"])
            if "fovy" in overview:
                model.cam_fovy[self.overview_id] = float(overview["fovy"])
        self.data = mujoco.MjData(model)
        self.width, self.height = int(width), int(height)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.option = mujoco.MjvOption()
        for group in range(len(self.option.geomgroup)):
            self.option.geomgroup[group] = 1 if group in self.GROUPS else 0
        # The MJWarp ray renderer draws geoms only; MuJoCo's default would also
        # draw the grasp-point and wrist-camera marker sites into the images.
        for group in range(len(self.option.sitegroup)):
            self.option.sitegroup[group] = 0

        def body_of(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        self.slot_bodies = [body_of(f"mjwarp_object_slot_{slot}") for slot in range(4)]
        self.slot_geoms = [
            [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
            for body in self.slot_bodies
        ]
        self.slot_visual = [
            next(g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
            for geoms in self.slot_geoms
        ]
        self.slot_qadr = [
            int(model.jnt_qposadr[model.body_jntadr[body]]) for body in self.slot_bodies
        ]
        mount = body_of("wx200_mount")
        arm_bodies = {mount}
        for body in range(model.nbody):
            parent = body
            while parent > 0:
                if parent == mount:
                    arm_bodies.add(body)
                    break
                parent = int(model.body_parentid[parent])
        self.arm_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies}
        from robots.widowx200.widowx200_mujoco.kinematics import JOINT_NAMES

        def qadr(name: str) -> int:
            return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])

        self.joint_qadr = [qadr(name) for name in JOINT_NAMES]
        self.finger_qadr = [qadr("left_finger"), qadr("right_finger")]

    # -- posing ---------------------------------------------------------------

    def pose(self, scene: Mapping[str, Any], spec: Any) -> None:
        import numpy as np

        from robots.widowx200.widowx200_mujoco.controller import (
            joint_targets_from_task_targets,
        )

        model, data = self.model, self.data
        fields = (
            "geom_dataid", "geom_size", "geom_pos", "geom_quat", "geom_rgba",
            "geom_matid", "geom_aabb", "geom_rbound",
        )
        for slot, catalog in enumerate(scene["catalogs"]):
            source = self.variants[catalog or self.catalogs[0]]
            for field in fields:
                getattr(model, field)[self.slot_geoms[slot]] = getattr(source, field)[self.slot_geoms[slot]]
        self.mujoco.mj_resetData(model, data)
        for slot in range(4):
            adr = self.slot_qadr[slot]
            data.qpos[adr : adr + 3] = scene["positions"][slot]
            data.qpos[adr + 3 : adr + 7] = scene["quaternions"][slot]
        solved = joint_targets_from_task_targets(
            np.asarray([scene["ee_xyz"]]), np.asarray([scene["ee_yaw"]]),
            np.asarray([scene["opening"]]), spec,
        )
        q = np.asarray(solved["q"]).reshape(-1)
        for column, adr in enumerate(self.joint_qadr):
            data.qpos[adr] = q[column]
        for adr in self.finger_qadr:
            data.qpos[adr] = float(np.asarray(solved["gripper_ctrl"]).reshape(-1)[0])
        self.mujoco.mj_forward(model, data)

    # -- rendering ------------------------------------------------------------

    def _scene(self, camera: int) -> None:
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)

    def _hide(self, keep: Any) -> None:
        geom_type = int(self.mujoco.mjtObj.mjOBJ_GEOM)
        scn = self.renderer.scene
        for index in range(scn.ngeom):
            g = scn.geoms[index]
            if not keep(int(g.objtype) == geom_type, int(g.objid)):
                g.pos[:] = (100.0, 100.0, 100.0)

    def segmentation(self, camera: int, keep: Any = None) -> Any:
        self.renderer.enable_segmentation_rendering()
        self._scene(camera)
        if keep is not None:
            self._hide(keep)
        seg = self.renderer.render()[..., 0].copy()
        self.renderer.disable_segmentation_rendering()
        return seg

    def rgb(self, camera: int) -> Any:
        self._scene(camera)
        return self.renderer.render().copy()

    def project(self, camera: int, points: Any) -> Any:
        """Pixel coordinates (u, v) and depth for world points."""

        import numpy as np

        position = self.data.cam_xpos[camera]
        rotation = self.data.cam_xmat[camera].reshape(3, 3)
        local = (np.asarray(points) - position) @ rotation
        depth = -local[:, 2]
        focal = 0.5 * self.height / math.tan(math.radians(float(self.model.cam_fovy[camera])) / 2.0)
        u = 0.5 * self.width + focal * local[:, 0] / np.maximum(depth, 1e-9)
        v = 0.5 * self.height - focal * local[:, 1] / np.maximum(depth, 1e-9)
        return u, v, depth

    def object_corners(self, slot: int) -> Any:
        import numpy as np

        geom = self.slot_visual[slot]
        aabb = self.model.geom_aabb[geom]
        centre, half = aabb[:3], aabb[3:]
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        local = centre + signs * half
        rotation = self.data.geom_xmat[geom].reshape(3, 3)
        return self.data.geom_xpos[geom] + local @ rotation.T

    def measure(self, scene: Mapping[str, Any], camera: int) -> dict[int, dict[str, Any]]:
        import numpy as np

        slot_of = {}
        for slot, geoms in enumerate(self.slot_geoms):
            for g in geoms:
                slot_of[g] = slot
        full = self.segmentation(camera)
        no_arm = self.segmentation(camera, keep=lambda is_geom, objid: not (is_geom and objid in self.arm_geoms))
        pixels = float(self.width * self.height)
        out: dict[int, dict[str, Any]] = {}
        for slot, catalog in enumerate(scene["catalogs"]):
            if catalog is None:
                continue
            ids = np.asarray(self.slot_geoms[slot])
            alone = self.segmentation(
                camera, keep=lambda is_geom, objid, s=slot: is_geom and slot_of.get(objid) == s
            )
            area_alone = float(np.isin(alone, ids).sum())
            area_full = float(np.isin(full, ids).sum())
            area_no_arm = float(np.isin(no_arm, ids).sum())
            u, v, depth = self.project(camera, self.object_corners(slot))
            centre_u, centre_v, centre_depth = self.project(camera, np.asarray([scene["positions"][slot]]))
            inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) & (depth > 0)
            out[slot] = {
                "centre_in_frame": bool(
                    0 <= centre_u[0] < self.width and 0 <= centre_v[0] < self.height and centre_depth[0] > 0
                ),
                "bbox_in_frame_fraction": float(inside.mean()),
                "pixels_alone": area_alone,
                "pixels_visible": area_full,
                "area_fraction": area_full / pixels,
                "visible_fraction": area_full / area_alone if area_alone > 0 else 0.0,
                "occluded_by_arm": (area_no_arm - area_full) / area_alone if area_alone > 0 else 0.0,
                "occluded_by_objects": (area_alone - area_no_arm) / area_alone if area_alone > 0 else 0.0,
            }
        return out


def lookat_xyaxes(position: Sequence[float], target: Sequence[float]) -> list[float]:
    """MJCF ``xyaxes`` for a camera at ``position`` looking at ``target``, +z up."""

    import numpy as np

    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [float(v) for v in (*right, *up)]


def _xyaxes_to_quat(xyaxes: Sequence[float]) -> Any:
    import numpy as np

    x = np.asarray(xyaxes[:3], dtype=np.float64)
    x /= np.linalg.norm(x)
    y = np.asarray(xyaxes[3:6], dtype=np.float64)
    y -= x * float(x @ y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    quat = np.zeros(4)
    import mujoco

    mujoco.mju_mat2Quat(quat, np.stack([x, y, z], axis=1).reshape(-1))
    return quat


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def audit(
    *,
    config_path: Path,
    rounds: Sequence[int],
    output: Path,
    render: bool,
    overview: Mapping[str, Any] | None,
    labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    view = pipeline_view(config_path)
    selected = set(labels or [label for label, *_ in AUDITS])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    renderer: SceneRenderer | None = None
    scenes_by_label: dict[str, list[dict[str, Any]]] = {}
    for label, skill, composed in AUDITS:
        if label not in selected:
            continue
        config = eval_config(view, skill, composed)
        metadata = config["task"]["metadata"]
        cap = float(view.skill(skill).get("eval_cap", 0.2))
        spec = control_spec(config)
        scenes = sample_scenes(config, cap=cap, rounds=rounds)
        scenes_by_label[label] = scenes
        if render and renderer is None:
            xml = (config_path.parent / config["simulator"]["fixed_scene_xml"]).resolve()
            renderer = SceneRenderer(
                xml,
                width=int(config["simulator"]["render_width"]),
                height=int(config["simulator"]["render_height"]),
                overview=overview,
            )
        print(f"[coverage] {label}: {len(scenes)} scenes at cap {cap}", flush=True)
        for scene in scenes:
            waypoints = scene_waypoints(scene, metadata)
            checks = reach_check(spec, [p for _, p in waypoints], [scene["ee_yaw"]] * len(waypoints))
            reach: dict[str, dict[str, Any]] = {}
            for (name, _), check in zip(waypoints, checks):
                merged = reach.setdefault(name, dict(check))
                if name == "carry":
                    merged["delivered"] = merged["delivered"] and check["delivered"]
                    merged["ik_reachable"] = merged["ik_reachable"] and check["ik_reachable"]
                    merged["xy_error"] = max(merged["xy_error"], check["xy_error"])
            target = scene["positions"][scene["target_slot"]]
            grasp_point = [target[0], target[1], target[2] + float(metadata.get("pick_grasp_height_offset", 0.0))]
            vis: dict[str, dict[int, dict[str, Any]]] = {}
            if renderer is not None:
                renderer.pose(scene, spec)
                vis["overview"] = renderer.measure(scene, renderer.overview_id)
                vis["wrist"] = renderer.measure(scene, renderer.wrist_id)
            row: dict[str, Any] = {
                "audit": label,
                "round": scene["round"],
                "scene": scene["scene"],
                "instruction": scene["instruction"],
                "target": scene["catalogs"][scene["target_slot"]],
                "target_x": round(target[0], 4),
                "target_y": round(target[1], 4),
                "target_z": round(target[2], 4),
                "target_base_radius": round(math.hypot(target[0] - spec.mount.position[0], target[1] - spec.mount.position[1]), 4),
                "container": scene["catalogs"][scene["reference_slot"]] if scene["reference_slot"] >= 0 else "",
                "container_x": round(scene["positions"][scene["reference_slot"]][0], 4) if scene["reference_slot"] >= 0 else "",
                "container_y": round(scene["positions"][scene["reference_slot"]][1], 4) if scene["reference_slot"] >= 0 else "",
                "ee_x": round(scene["ee_xyz"][0], 4),
                "ee_y": round(scene["ee_xyz"][1], 4),
                "ee_z": round(scene["ee_xyz"][2], 4),
                "caught_start": scene_is_caught(scene, metadata) if scene["instruction"].startswith("put_into") else "",
                "grasp_yaw_coverage": round(yaw_coverage(spec, grasp_point), 3)
                if scene["instruction"] != "move_to_object" else "",
            }
            row["reach_all"] = all(entry["delivered"] for entry in reach.values())
            for name in ("start", "goal", "hover", "grasp", "lift", "release", "carry"):
                entry = reach.get(name)
                row[f"reach_{name}"] = "" if entry is None else entry["delivered"]
                row[f"ik_{name}"] = "" if entry is None else entry["ik_reachable"]
                row[f"xy_err_{name}"] = "" if entry is None else round(entry["xy_error"], 4)
            row["straight_carry_min_radius"] = straight_carry_min_radius(scene)
            row["grasp_z_below_floor"] = (
                round(reach["grasp"]["z_below_floor"], 4) if "grasp" in reach else ""
            )
            for camera, per_slot in vis.items():
                for role, slot in (("target", scene["target_slot"]), ("container", scene["reference_slot"])):
                    entry = per_slot.get(slot) if slot >= 0 else None
                    for key in ("centre_in_frame", "bbox_in_frame_fraction", "pixels_visible", "area_fraction",
                                "visible_fraction", "occluded_by_arm", "occluded_by_objects"):
                        value = "" if entry is None else entry[key]
                        row[f"{camera}_{role}_{key}"] = round(value, 4) if isinstance(value, float) else value
                others = [
                    v for s, v in per_slot.items() if s not in (scene["target_slot"], scene["reference_slot"])
                ]
                row[f"{camera}_min_other_visible_fraction"] = (
                    round(min(v["visible_fraction"] for v in others), 4) if others else ""
                )
            rows.append(row)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (output / "scenes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if renderer is not None:
        contact_sheet(renderer, scenes_by_label, rows, view, output)
    return summary


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by[row["audit"]].append(row)
        by[f"{row['audit']}/{row['instruction']}"].append(row)

    def frac(items: Sequence[Mapping[str, Any]], key: str, test: Any = bool) -> float | None:
        values = [row[key] for row in items if row.get(key, "") != ""]
        return None if not values else round(sum(1 for v in values if test(v)) / len(values), 4)

    def stat(items: Sequence[Mapping[str, Any]], key: str, fn: Any) -> float | None:
        values = [float(row[key]) for row in items if row.get(key, "") != ""]
        return None if not values else round(float(fn(values)), 4)

    out: dict[str, Any] = {}
    for name, items in sorted(by.items()):
        entry: dict[str, Any] = {"scenes": len(items), "reach_all": frac(items, "reach_all")}
        for point in ("start", "goal", "hover", "grasp", "lift", "release", "carry"):
            entry[f"reach_{point}"] = frac(items, f"reach_{point}")
            entry[f"ik_{point}"] = frac(items, f"ik_{point}")
        entry["grasp_yaw_full"] = frac(items, "grasp_yaw_coverage", lambda v: float(v) >= 0.999)
        for camera in ("overview", "wrist"):
            for role in ("target", "container"):
                prefix = f"{camera}_{role}"
                entry[f"{prefix}_centre_in_frame"] = frac(items, f"{prefix}_centre_in_frame")
                entry[f"{prefix}_fully_in_frame"] = frac(items, f"{prefix}_bbox_in_frame_fraction", lambda v: float(v) >= 0.999)
                entry[f"{prefix}_visible_ge_50pct"] = frac(items, f"{prefix}_visible_fraction", lambda v: float(v) >= 0.5)
                entry[f"{prefix}_pixels_min"] = stat(items, f"{prefix}_pixels_visible", min)
                entry[f"{prefix}_pixels_median"] = stat(items, f"{prefix}_pixels_visible", lambda v: sorted(v)[len(v) // 2])
                entry[f"{prefix}_occluded_by_arm_max"] = stat(items, f"{prefix}_occluded_by_arm", max)
                entry[f"{prefix}_occluded_by_objects_max"] = stat(items, f"{prefix}_occluded_by_objects", max)
        out[name] = {k: v for k, v in entry.items() if v is not None}
    return out


def contact_sheet(
    renderer: SceneRenderer,
    scenes_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
    view: Any,
    output: Path,
) -> None:
    """Best and worst overview-visibility scenes per audit, with wrist views."""

    import numpy as np
    from PIL import Image, ImageDraw

    tiles = []
    for label, scenes in scenes_by_label.items():
        audit_rows = [r for r in rows if r["audit"] == label]
        if not audit_rows:
            continue
        index = {(r["round"], r["scene"]): r for r in audit_rows}

        def score(row: Mapping[str, Any]) -> float:
            target = float(row.get("overview_target_visible_fraction") or 0.0)
            container = row.get("overview_container_visible_fraction")
            container = 1.0 if container in ("", None) else float(container)
            return min(target, container) + 0.001 * float(row.get("overview_target_pixels_visible") or 0.0)

        ordered = sorted(scenes, key=lambda s: score(index[(s["round"], s["scene"])]))
        picks = [("worst", ordered[0]), ("worst", ordered[1]), ("median", ordered[len(ordered) // 2]), ("best", ordered[-1])]
        config_label = next(a for a in AUDITS if a[0] == label)
        spec = control_spec(eval_config(view, config_label[1], config_label[2]))
        for tag, scene in picks:
            row = index[(scene["round"], scene["scene"])]
            renderer.pose(scene, spec)
            overview = renderer.rgb(renderer.overview_id)
            wrist = renderer.rgb(renderer.wrist_id)
            tile = Image.fromarray(np.concatenate([overview, wrist], axis=1))
            draw = ImageDraw.Draw(tile)
            text = (
                f"{label} {tag} r{scene['round']}s{scene['scene']} {scene['instruction']} "
                f"tgt={str(row['target']).replace('robocasa_', '')} "
                f"vis={row.get('overview_target_visible_fraction')} px={row.get('overview_target_pixels_visible')} "
                f"reach={row['reach_all']}"
            )
            draw.rectangle((0, 0, tile.width, 14), fill=(0, 0, 0))
            draw.text((3, 1), text, fill=(255, 255, 255))
            tiles.append(tile)
    if not tiles:
        return
    width = tiles[0].width
    sheet = Image.new("RGB", (width * 2, tiles[0].height * math.ceil(len(tiles) / 2)))
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % 2) * width, (position // 2) * tile.height))
    sheet.save(output / "contact_sheet.png")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rounds", type=int, nargs="+", default=list(DEFAULT_ROUNDS))
    parser.add_argument("--output", type=Path, default=ROOT / "tools/audit/out/widowx200_scene_coverage")
    parser.add_argument("--audits", nargs="+", default=None, choices=[label for label, *_ in AUDITS])
    parser.add_argument("--no-render", action="store_true", help="Reach only; skip the camera renders.")
    parser.add_argument("--overview-pos", type=float, nargs=3, default=None)
    parser.add_argument("--overview-xyaxes", type=float, nargs=6, default=None)
    parser.add_argument("--overview-fovy", type=float, default=None)
    parser.add_argument(
        "--overview-lookat", type=float, nargs=3, default=None,
        help="With --overview-pos: aim the overview at this world point, world +z up.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.environ.get("RLVLA_EMBODIMENT", "").strip().lower() != "widowx200":
        raise SystemExit("Set RLVLA_EMBODIMENT=widowx200: it sizes the object catalog at import.")
    overview: dict[str, Any] = {}
    if args.overview_pos:
        overview["pos"] = args.overview_pos
    if args.overview_xyaxes:
        overview["xyaxes"] = args.overview_xyaxes
    if args.overview_lookat:
        if not args.overview_pos:
            raise SystemExit("--overview-lookat needs --overview-pos.")
        overview["xyaxes"] = lookat_xyaxes(args.overview_pos, args.overview_lookat)
        print(f"[coverage] overview xyaxes {' '.join(f'{v:.5f}' for v in overview['xyaxes'])}", flush=True)
    if args.overview_fovy:
        overview["fovy"] = args.overview_fovy
    summary = audit(
        config_path=args.config.resolve(),
        rounds=args.rounds,
        output=args.output,
        render=not args.no_render,
        overview=overview or None,
        labels=args.audits,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
