"""MuJoCo/CDPR backend for the simulator comparator."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.sim_compare.common import (
    CONTACT_RESULT_FIELDS,
    DEFAULT_CAMERA_COUNT,
    DEFAULT_HEIGHT,
    DEFAULT_RENDER_BACKEND,
    DEFAULT_WIDTH,
    MAX_CONTACT_FORCE_N,
    MAX_SETTLED_ANGULAR_VELOCITY_RADPS,
    MAX_SETTLED_LINEAR_VELOCITY_MPS,
    MAX_TRANSIENT_ANGULAR_VELOCITY_RADPS,
    MAX_TRANSIENT_LINEAR_VELOCITY_MPS,
    MOVE_XY_THRESHOLD_M,
    OBJECT_SPECS,
    SUPPORTED_RENDER_BACKENDS,
    PUSH_DISPLACEMENT_THRESHOLD_M,
    RELATION_OFFSET_M,
    RELATION_TOLERANCE_M,
    TABLE_PENETRATION_TOLERANCE_M,
    REPO_ROOT,
    TABLE_TOP_Z,
    TMP_DIR,
    format_vec,
    gpu_utilization_percent,
    gpu_vram_mb,
    mean,
    module_version,
    platform_label,
    move_to_object_success,
    predicate_self_check,
    push_success,
    relation_success,
    rss_mb,
    timer,
)


class MujocoCDPRBackend:
    backend_name = "mujoco_raw_cdpr"
    robot_embodiment = "current CDPR gripper MJCF with kinematic waypoint EE controller"
    num_envs = 1

    gripper_max_travel_m = 0.03
    finger_base_half_gap_m = 0.02
    finger_half_width_m = 0.012
    finger_center_z_from_ee_base_m = 0.025
    finger_lowest_z_from_ee_base_m = -0.030
    finger_tip_lowest_z_from_ee_base_m = -0.063
    gripper_min_inner_gap_m = 2.0 * (finger_base_half_gap_m - finger_half_width_m)
    gripper_max_inner_gap_m = 2.0 * (finger_base_half_gap_m + gripper_max_travel_m - finger_half_width_m)

    def __init__(
        self,
        seed: int,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        render_backend: str = DEFAULT_RENDER_BACKEND,
        camera_count: int = DEFAULT_CAMERA_COUNT,
        render_enabled: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.width = int(width)
        self.height = int(height)
        self.render_backend = str(render_backend)
        if self.render_backend not in SUPPORTED_RENDER_BACKENDS:
            raise ValueError(f"unsupported render backend: {render_backend}")
        self.camera_count = int(camera_count)
        self.render_enabled = bool(render_enabled)
        self.version = module_version("mujoco")
        self.scene_path = TMP_DIR / "mujoco_cdpr_sim_compare_scene.xml"
        self.model = None
        self.data = None
        self.mj = None
        self._body_ids: Dict[str, int] = {}
        self._joint_ids: Dict[str, int] = {}
        self._ee_joint_id = -1
        self._ee_body_id = -1
        self._act_gripper_id = -1
        self._ee_yaw_joint_id = -1
        self._ee_ball_joint_id = -1
        self._finger_l_joint_id = -1
        self._finger_r_joint_id = -1
        self._overview_camera_name = "sim_compare_overview"

    def is_available(self) -> bool:
        try:
            import mujoco  # noqa: F401

            return True
        except Exception:
            return False

    def setup(self) -> None:
        import mujoco as mj

        self.mj = mj
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.scene_path.write_text(self._scene_xml())
        self.model = mj.MjModel.from_xml_path(str(self.scene_path))
        self.data = mj.MjData(self.model)
        self._ee_body_id = self._body_id("ee_base")
        self._ee_joint_id = self._joint_id("ee_free")
        self._ee_yaw_joint_id = self._joint_id("ee_yaw")
        self._ee_ball_joint_id = self._joint_id("ee_ball")
        self._finger_l_joint_id = self._joint_id("finger_l")
        self._finger_r_joint_id = self._joint_id("finger_r")
        self._act_gripper_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        for key in OBJECT_SPECS:
            self._body_ids[key] = self._body_id(key)
            self._joint_ids[key] = self._free_joint_for_body(key)
        self.reset_world()

    def run(
        self,
        resets: int,
        steps: int,
        render_steps: int,
        task_objects: Sequence[str],
        contact_objects: Sequence[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.is_available():
            raise RuntimeError("mujoco is not importable")
        self.setup()

        task_rows: List[Dict[str, Any]] = []
        contact_rows: List[Dict[str, Any]] = []
        render_rows: List[Dict[str, Any]] = []

        predicate_text, _predicate_score = predicate_self_check()
        for object_name in task_objects:
            task_rows.extend(self.run_move_to_object(object_name, resets, steps, predicate_text))
            task_rows.extend(self.run_push_object(object_name, resets, steps, predicate_text))
        task_rows.extend(self.run_place_relation("block", "plate", resets, steps, predicate_text))

        for object_name in contact_objects:
            for test_name in ("drop", "rest_on_table", "push", "gripper_squeeze", "lift"):
                contact_rows.append(self.run_contact_test(object_name, test_name, steps))

        render_rows.append(self.run_render_profile(max(1, int(render_steps))))

        reset_times = [float(row["reset_time_s"]) for row in task_rows if row.get("reset_time_s") != ""]
        step_times = [float(row["step_time_s"]) for row in task_rows if row.get("step_time_s") != ""]
        step_counts = [int(row["steps"]) for row in task_rows if row.get("steps") != ""]
        total_step_time = sum(step_times)
        total_steps = sum(step_counts)
        step_fps = float(total_steps / total_step_time) if total_step_time > 0 else 0.0

        stability_rows = [row for row in contact_rows if row["test_name"] in {"drop", "rest_on_table", "push"}]
        stable_good = sum(1 for row in stability_rows if row.get("pass_fail") in {"pass", "warn"})
        stability_rate = float(stable_good / len(stability_rows)) if stability_rows else 0.0
        anomalies = [
            str(row.get("contact_anomalies", ""))
            for row in contact_rows
            if row.get("contact_anomalies") and row.get("pass_fail") not in {"pass", "warn"}
        ]
        render_fps = 0.0
        render_backend_text = self.render_backend
        if render_rows:
            render_fps = float(render_rows[0].get("step_fps_during_rgb") or 0.0)
            render_backend_text = str(render_rows[0].get("render_backend") or render_backend_text)

        gpu_vram = gpu_vram_mb()
        gpu_util = gpu_utilization_percent()

        summary = {
            "backend_name": self.backend_name,
            "status": "ran",
            "simulator_version": self.version,
            "robot_embodiment": self.robot_embodiment,
            "num_environments": self.num_envs,
            "reset_time_mean_s": f"{mean(reset_times):.6f}",
            "step_fps_no_render": f"{step_fps:.2f}",
            "step_fps_with_rgb": f"{render_fps:.2f}",
            "render_resolution": f"{self.width}x{self.height}",
            "render_backend": render_backend_text,
            "platform": platform_label(),
            "cpu_ram_mb": f"{rss_mb():.2f}",
            "gpu_vram_mb": "" if gpu_vram is None else f"{gpu_vram:.2f}",
            "gpu_utilization_percent": "" if gpu_util is None else f"{gpu_util:.1f}",
            "success_predicate_correctness": predicate_text,
            "object_stability_pass_rate": f"{stability_rate:.3f}",
            "contact_anomalies": f"{len(anomalies)} anomaly rows",
            "engineering_notes": (
                "Uses current CDPR MJCF; scripted waypoint controller directly sets ee_free qpos. "
                "Manipulation predicates are geometric body-position checks."
            ),
            "missing_features": "No OpenVLA, no learned policy, no robust grasp planner; place_relation uses direct object placement.",
            "migration_difficulty": "low for current pipeline baseline",
            "skipped_reason": "",
        }
        return summary, task_rows, contact_rows, render_rows

    def _scene_xml(self) -> str:
        cdpr_xml = REPO_ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr.xml"
        object_xml = "\n".join(self._object_body_xml(spec) for spec in OBJECT_SPECS.values())
        return f"""<mujoco model="sim_compare_cdpr">
  <compiler autolimits="true" angle="degree"/>
  <option timestep="0.002" solver="Newton" iterations="50"/>
  <include file="{cdpr_xml.as_posix()}"/>
  <worldbody>
    <camera name="{self._overview_camera_name}" pos="0.72 -0.72 0.72" xyaxes="0.707 0.707 0 -0.408 0.408 0.816" fovy="45"/>
    <body name="benchmark_table" pos="0 0 0">
      <geom name="benchmark_table_top" type="box" size="0.55 0.36 0.015"
            friction="1.1 0.04 0.01" condim="6" solref="0.006 1" solimp="0.92 0.98 0.001"
            margin="0.0005" gap="0" rgba="0.70 0.70 0.66 1"/>
    </body>
{object_xml}
  </worldbody>
</mujoco>
"""

    def _object_body_xml(self, spec: Any) -> str:
        rgba = " ".join(str(x) for x in spec.rgba)
        friction = " ".join(str(x) for x in spec.friction)
        solref = " ".join(str(x) for x in spec.solref)
        solimp = " ".join(str(x) for x in spec.solimp)
        contact_attrs = (
            f'friction="{friction}" condim="{int(spec.condim)}" solref="{solref}" solimp="{solimp}" '
            f'margin="{float(spec.margin):.6f}" gap="{float(spec.gap):.6f}"'
        )
        inertia = self._object_inertia(spec)
        if spec.geom_type == "box":
            geom = (
                f'<geom name="{spec.key}_collision" type="box" '
                f'size="{spec.size[0]} {spec.size[1]} {spec.size[2]}" '
                f'{contact_attrs} rgba="{rgba}"/>'
            )
        elif spec.geom_type == "cylinder":
            geom = (
                f'<geom name="{spec.key}_collision" type="cylinder" '
                f'size="{spec.size[0]} {spec.size[1]}" '
                f'{contact_attrs} rgba="{rgba}"/>'
            )
        elif spec.geom_type == "sphere":
            geom = (
                f'<geom name="{spec.key}_collision" type="sphere" size="{spec.size[0]}" '
                f'{contact_attrs} rgba="{rgba}"/>'
            )
        elif spec.geom_type == "compound_bowl":
            geom = f"""
      <geom name="{spec.key}_base_collision" type="cylinder" size="0.070 0.014"
            {contact_attrs} rgba="{rgba}"/>
      <geom name="{spec.key}_wall_x_pos" type="box" size="0.007 0.062 0.024" pos="0.069 0 0.014"
            {contact_attrs} rgba="{rgba}"/>
      <geom name="{spec.key}_wall_x_neg" type="box" size="0.007 0.062 0.024" pos="-0.069 0 0.014"
            {contact_attrs} rgba="{rgba}"/>
      <geom name="{spec.key}_wall_y_pos" type="box" size="0.062 0.007 0.024" pos="0 0.069 0.014"
            {contact_attrs} rgba="{rgba}"/>
      <geom name="{spec.key}_wall_y_neg" type="box" size="0.062 0.007 0.024" pos="0 -0.069 0.014"
            {contact_attrs} rgba="{rgba}"/>"""
        else:
            raise ValueError(spec.geom_type)
        return f"""    <body name="{spec.key}" pos="0 0 {spec.center_z:.4f}">
      <freejoint name="{spec.key}_free"/>
      <inertial pos="0 0 0" mass="{spec.mass}" diaginertia="{inertia}"/>
      {geom}
    </body>"""

    def _object_inertia(self, spec: Any) -> str:
        mass = float(spec.mass)
        if spec.geom_type == "box":
            x, y, z = (2.0 * float(value) for value in spec.size[:3])
            ixx = mass * (y * y + z * z) / 12.0
            iyy = mass * (x * x + z * z) / 12.0
            izz = mass * (x * x + y * y) / 12.0
        elif spec.geom_type == "cylinder":
            radius = float(spec.size[0])
            half_height = float(spec.size[1])
            height = 2.0 * half_height
            ixx = mass * (3.0 * radius * radius + height * height) / 12.0
            iyy = ixx
            izz = 0.5 * mass * radius * radius
        elif spec.geom_type == "sphere":
            radius = float(spec.size[0])
            ixx = iyy = izz = 0.4 * mass * radius * radius
        elif spec.geom_type == "compound_bowl":
            radius = float(spec.size[0])
            height = 2.0 * float(spec.half_height)
            ixx = iyy = mass * (3.0 * radius * radius + height * height) / 12.0
            izz = 0.5 * mass * radius * radius
        else:
            raise ValueError(spec.geom_type)
        floor = 1e-8
        return f"{max(ixx, floor):.9f} {max(iyy, floor):.9f} {max(izz, floor):.9f}"

    def _body_id(self, name: str) -> int:
        body_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, str(name))
        if body_id == -1:
            raise KeyError(f"Missing MuJoCo body: {name}")
        return int(body_id)

    def _joint_id(self, name: str) -> int:
        joint_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id == -1:
            raise KeyError(f"Missing MuJoCo joint: {name}")
        return int(joint_id)

    def _free_joint_for_body(self, body_name: str) -> int:
        body_id = self._body_id(body_name)
        joint_count = int(self.model.body_jntnum[body_id])
        joint_adr = int(self.model.body_jntadr[body_id])
        for offset in range(joint_count):
            joint_id = joint_adr + offset
            if int(self.model.jnt_type[joint_id]) == int(self.mj.mjtJoint.mjJNT_FREE):
                return int(joint_id)
        raise RuntimeError(f"Body {body_name!r} has no free joint")

    def _set_free_pose(self, joint_id: int, pos: Sequence[float], quat: Optional[Sequence[float]] = None) -> None:
        qadr = int(self.model.jnt_qposadr[int(joint_id)])
        dofadr = int(self.model.jnt_dofadr[int(joint_id)])
        self.data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        self.data.qpos[qadr + 3 : qadr + 7] = np.asarray(quat or [1.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4)
        self.data.qvel[dofadr : dofadr + 6] = 0.0

    def _set_object_pose(self, object_name: str, pos: Sequence[float]) -> None:
        self._set_free_pose(self._joint_ids[object_name], pos)

    def _set_ee_pose(self, pos: Sequence[float]) -> None:
        self._set_free_pose(self._ee_joint_id, pos)

    def _set_gripper_open_fraction(self, fraction: float) -> None:
        value = float(np.clip(fraction, 0.0, 1.0))
        qpos = value * self.gripper_max_travel_m
        for joint_id in (self._finger_l_joint_id, self._finger_r_joint_id):
            if joint_id == -1:
                continue
            qadr = int(self.model.jnt_qposadr[int(joint_id)])
            dofadr = int(self.model.jnt_dofadr[int(joint_id)])
            self.data.qpos[qadr] = qpos
            self.data.qvel[dofadr] = 0.0
        if self._act_gripper_id != -1:
            self.data.ctrl[self._act_gripper_id] = value

    def _set_tool_orientation_neutral(self) -> None:
        if self._ee_yaw_joint_id != -1:
            qadr = int(self.model.jnt_qposadr[int(self._ee_yaw_joint_id)])
            dofadr = int(self.model.jnt_dofadr[int(self._ee_yaw_joint_id)])
            self.data.qpos[qadr] = 0.0
            self.data.qvel[dofadr] = 0.0
        if self._ee_ball_joint_id != -1:
            qadr = int(self.model.jnt_qposadr[int(self._ee_ball_joint_id)])
            dofadr = int(self.model.jnt_dofadr[int(self._ee_ball_joint_id)])
            self.data.qpos[qadr : qadr + 4] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.data.qvel[dofadr : dofadr + 3] = 0.0

    def _gripper_test_ee_z(self, spec: Any) -> float:
        table_clearance_z = TABLE_TOP_Z + 0.004 - self.finger_tip_lowest_z_from_ee_base_m
        center_aligned_z = float(spec.center_z) - self.finger_center_z_from_ee_base_m
        return float(max(table_clearance_z, center_aligned_z))

    def _gripper_geometry_note(self, spec: Any) -> str:
        if float(spec.grasp_width) > self.gripper_max_inner_gap_m + 0.001:
            return (
                f"object wider than gripper maximum opening "
                f"({spec.grasp_width:.3f} m > {self.gripper_max_inner_gap_m:.3f} m)"
            )
        return ""

    def _gripper_open_fraction_for_gap(self, gap_m: float) -> float:
        span = self.gripper_max_inner_gap_m - self.gripper_min_inner_gap_m
        if span <= 0:
            return 0.0
        return float(np.clip((float(gap_m) - self.gripper_min_inner_gap_m) / span, 0.0, 1.0))

    def _gripper_target_open_fraction(self, spec: Any, test_name: str) -> float:
        compression = 0.006 if test_name == "gripper_squeeze" else 0.010
        target_gap = max(self.gripper_min_inner_gap_m, float(spec.grasp_width) - compression)
        return self._gripper_open_fraction_for_gap(target_gap)

    def _object_pos(self, object_name: str) -> np.ndarray:
        self.mj.mj_forward(self.model, self.data)
        return np.asarray(self.data.xpos[self._body_ids[object_name]], dtype=np.float64).copy()

    def _ee_pos(self) -> np.ndarray:
        self.mj.mj_forward(self.model, self.data)
        return np.asarray(self.data.xpos[self._ee_body_id], dtype=np.float64).copy()

    def reset_world(self) -> None:
        self.mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._set_ee_pose([0.0, 0.0, 0.22])
        self._set_tool_orientation_neutral()
        self._set_gripper_open_fraction(1.0)
        for idx, key in enumerate(OBJECT_SPECS):
            spec = OBJECT_SPECS[key]
            self._set_object_pose(key, [-0.30 + 0.10 * idx, 0.26, spec.center_z])
        self.data.xfrc_applied[:, :] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def _reset_task_layout(self, active_positions: Dict[str, Sequence[float]]) -> float:
        start = timer()
        self.reset_world()
        for key, pos in active_positions.items():
            self._set_object_pose(key, pos)
        self.mj.mj_forward(self.model, self.data)
        return timer() - start

    def _step_waypoint(self, ee_pos: Sequence[float], gripper: float = 1.0, stabilize_tool: bool = False) -> None:
        self._set_ee_pose(ee_pos)
        if stabilize_tool:
            self._set_tool_orientation_neutral()
        self.data.ctrl[:] = 0.0
        if self._act_gripper_id != -1:
            self.data.ctrl[self._act_gripper_id] = float(gripper)
        self.mj.mj_step(self.model, self.data)

    def _run_ee_line(
        self,
        start_xyz: Sequence[float],
        end_xyz: Sequence[float],
        steps: int,
        gripper: float = 1.0,
    ) -> float:
        start = np.asarray(start_xyz, dtype=np.float64)
        end = np.asarray(end_xyz, dtype=np.float64)
        t0 = timer()
        for idx in range(max(1, int(steps))):
            alpha = (idx + 1) / max(1, int(steps))
            pos = start * (1.0 - alpha) + end * alpha
            self._step_waypoint(pos, gripper=gripper)
        return timer() - t0

    def run_move_to_object(self, object_name: str, resets: int, steps: int, predicate_text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        spec = OBJECT_SPECS[object_name]
        rng = np.random.default_rng(self.seed + 1000 + self._stable_object_index(object_name) * 97)
        for episode in range(int(resets)):
            target_xy = np.array(
                [rng.uniform(-0.06, 0.08), rng.uniform(-0.06, 0.06)],
                dtype=np.float64,
            )
            target = [float(target_xy[0]), float(target_xy[1]), spec.center_z]
            reset_s = self._reset_task_layout({object_name: target})
            initial_ee = self._ee_pos()
            initial_obj = self._object_pos(object_name)
            final_ee_target = [target[0] + 0.006, target[1] - 0.006, 0.18]
            step_s = self._run_ee_line([-0.22, 0.0, 0.20], final_ee_target, int(steps))
            final_ee = self._ee_pos()
            final_obj = self._object_pos(object_name)
            success = move_to_object_success(final_ee, final_obj, MOVE_XY_THRESHOLD_M, "xy")
            rows.append(
                self._task_row(
                    "move_to_object",
                    object_name,
                    episode,
                    reset_s,
                    steps,
                    step_s,
                    success,
                    predicate_text,
                    initial_ee,
                    final_ee,
                    initial_obj,
                    final_obj,
                    "",
                    MOVE_XY_THRESHOLD_M,
                    "XY predicate on ee_base to target body; waypoint controller only.",
                    "scripted_waypoint_predicate",
                )
            )
        return rows

    def run_push_object(self, object_name: str, resets: int, steps: int, predicate_text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        spec = OBJECT_SPECS[object_name]
        rng = np.random.default_rng(self.seed + 2000 + self._stable_object_index(object_name) * 97)
        for episode in range(int(resets)):
            initial = np.array([rng.uniform(-0.04, 0.02), rng.uniform(-0.025, 0.025), spec.center_z], dtype=np.float64)
            reset_s = self._reset_task_layout({object_name: initial})
            initial_ee = self._ee_pos()
            initial_obj = self._object_pos(object_name)
            lateral = initial[1]
            ee_z = TABLE_TOP_Z + spec.half_height + 0.052
            start = [initial[0] - 0.14, lateral, ee_z]
            end = [initial[0] + 0.08, lateral, ee_z]
            step_s = self._run_ee_line(start, end, int(steps), gripper=0.0)
            final_ee = self._ee_pos()
            final_obj = self._object_pos(object_name)
            success = push_success(initial_obj, final_obj, [1.0, 0.0, 0.0], PUSH_DISPLACEMENT_THRESHOLD_M)
            rows.append(
                self._task_row(
                    "push_object",
                    object_name,
                    episode,
                    reset_s,
                    steps,
                    step_s,
                    success,
                    predicate_text,
                    initial_ee,
                    final_ee,
                    initial_obj,
                    final_obj,
                    "",
                    PUSH_DISPLACEMENT_THRESHOLD_M,
                    "Kinematic CDPR gripper push along +X; success is signed object displacement.",
                    "scripted_waypoint_predicate",
                )
            )
        return rows

    def run_place_relation(
        self,
        object_name: str,
        reference_name: str,
        resets: int,
        steps: int,
        predicate_text: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        target_spec = OBJECT_SPECS[object_name]
        ref_spec = OBJECT_SPECS[reference_name]
        relations = ("left", "right", "front", "behind")
        ref = np.array([0.08, 0.0, ref_spec.center_z], dtype=np.float64)
        for episode in range(int(resets)):
            relation = relations[episode % len(relations)]
            start_target = np.array([-0.10, -0.08, target_spec.center_z], dtype=np.float64)
            reset_s = self._reset_task_layout({object_name: start_target, reference_name: ref})
            initial_ee = self._ee_pos()
            initial_obj = self._object_pos(object_name)
            reference_obj = self._object_pos(reference_name)
            target = ref.copy()
            if relation == "left":
                target[0] -= RELATION_OFFSET_M + 0.025
            elif relation == "right":
                target[0] += RELATION_OFFSET_M + 0.025
            elif relation == "front":
                target[1] += RELATION_OFFSET_M + 0.025
            else:
                target[1] -= RELATION_OFFSET_M + 0.025
            target[2] = target_spec.center_z
            start = timer()
            for idx in range(max(1, int(steps))):
                if idx == max(1, int(steps)) // 3:
                    self._set_object_pose(object_name, target)
                self._step_waypoint([float(target[0]), float(target[1]), 0.20])
            step_s = timer() - start
            final_ee = self._ee_pos()
            final_obj = self._object_pos(object_name)
            success = relation_success(final_obj, reference_obj, relation, RELATION_OFFSET_M, RELATION_TOLERANCE_M)
            rows.append(
                self._task_row(
                    f"place_relation_{relation}",
                    object_name,
                    episode,
                    reset_s,
                    steps,
                    step_s,
                    success,
                    predicate_text,
                    initial_ee,
                    final_ee,
                    initial_obj,
                    final_obj,
                    reference_obj,
                    RELATION_OFFSET_M,
                    "Direct object pose set used because scripted robust grasping is outside this scaffold.",
                    "predicate_validation_only",
                )
            )
        return rows

    def _task_row(
        self,
        task_name: str,
        object_name: str,
        episode: int,
        reset_s: float,
        steps: int,
        step_s: float,
        success: bool,
        predicate_text: str,
        initial_ee: Sequence[float],
        final_ee: Sequence[float],
        initial_obj: Sequence[float],
        final_obj: Sequence[float],
        reference_obj: Any,
        threshold: float,
        notes: str,
        validation_scope: str,
    ) -> Dict[str, Any]:
        spec = OBJECT_SPECS[object_name]
        fps = float(int(steps) / step_s) if step_s > 0 else 0.0
        return {
            "backend_name": self.backend_name,
            "simulator_version": self.version,
            "task_name": task_name,
            "object_category": spec.category,
            "object_name": object_name,
            "robot_embodiment": self.robot_embodiment,
            "num_environments": self.num_envs,
            "validation_scope": validation_scope,
            "episode": int(episode),
            "reset_time_s": f"{float(reset_s):.6f}",
            "steps": int(steps),
            "step_time_s": f"{float(step_s):.6f}",
            "step_fps_no_render": f"{fps:.2f}",
            "success": int(bool(success)),
            "success_predicate_correctness": predicate_text,
            "initial_ee_xyz": format_vec(initial_ee),
            "final_ee_xyz": format_vec(final_ee),
            "initial_object_xyz": format_vec(initial_obj),
            "final_object_xyz": format_vec(final_obj),
            "reference_object_xyz": "" if isinstance(reference_obj, str) and reference_obj == "" else format_vec(reference_obj),
            "threshold_m": f"{float(threshold):.4f}",
            "engineering_notes": notes,
        }

    def _stable_object_index(self, object_name: str) -> int:
        keys = list(OBJECT_SPECS)
        return keys.index(object_name) if object_name in OBJECT_SPECS else 0

    def run_contact_test(self, object_name: str, test_name: str, steps: int) -> Dict[str, Any]:
        spec = OBJECT_SPECS[object_name]
        start = timer()
        failure_reason = ""
        anomaly = ""
        finger_count: Any = 0
        try:
            self.reset_world()
            side_table_pos = np.array([0.25, 0.0, spec.center_z], dtype=np.float64)
            pos = side_table_pos.copy()
            if test_name == "drop":
                pos[2] = TABLE_TOP_Z + spec.half_height + 0.15
                self._set_object_pose(object_name, pos)
                self.mj.mj_forward(self.model, self.data)
                metrics = self._step_collect(object_name, int(steps), ignore_initial_steps=int(0.65 * steps))
            elif test_name == "rest_on_table":
                self._set_object_pose(object_name, pos)
                self.mj.mj_forward(self.model, self.data)
                metrics = self._step_collect(object_name, int(steps), ignore_initial_steps=int(0.70 * steps))
            elif test_name == "push":
                self._set_object_pose(object_name, pos)
                self.mj.mj_forward(self.model, self.data)
                for _ in range(60):
                    self.mj.mj_step(self.model, self.data)
                self.data.xfrc_applied[self._body_ids[object_name], 0] = min(0.30, max(0.12, 2.2 * float(spec.mass)))
                push_metrics = self._step_collect(object_name, max(1, int(steps) // 3))
                self.data.xfrc_applied[:, :] = 0.0
                settle_steps = max(1, int(steps) - max(1, int(steps) // 3))
                settle_metrics = self._step_collect(
                    object_name,
                    settle_steps,
                    ignore_initial_steps=max(1, int(0.55 * settle_steps)),
                )
                metrics = self._merge_metrics(push_metrics, settle_metrics)
            elif test_name in {"gripper_squeeze", "lift"}:
                geometry_note = self._gripper_geometry_note(spec)
                if geometry_note:
                    self._set_object_pose(object_name, side_table_pos)
                    self.mj.mj_forward(self.model, self.data)
                    metrics = self._empty_metrics()
                    self._sample_object_metrics(object_name, metrics)
                    finger_count = 0
                    status = "fail"
                    failure_reason = geometry_note
                    anomaly = geometry_note
                    raise StopIteration
                metrics, finger_count, extra_failures = self._run_gripper_contact_sequence(object_name, test_name, int(steps))
            else:
                raise ValueError(test_name)
            status, failure_reason = self._contact_verdict(
                metrics,
                spec,
                require_contact=test_name in {"gripper_squeeze", "lift"},
                finger_contact_count=int(finger_count),
                extra_failures=extra_failures if test_name in {"gripper_squeeze", "lift"} else None,
            )
            anomaly = failure_reason
        except StopIteration:
            pass
        except Exception as exc:
            metrics = {
                "max_linear_velocity": "",
                "max_angular_velocity": "",
                "settled_linear_velocity": "",
                "settled_angular_velocity": "",
                "max_normal_contact_force": "",
                "min_body_z": "",
                "min_bottom_z": "",
            }
            finger_count = ""
            status = "error"
            failure_reason = f"{type(exc).__name__}: {exc}"
            anomaly = failure_reason
        row = {
            "backend_name": self.backend_name,
            "simulator_version": self.version,
            "object_category": spec.category,
            "object_name": object_name,
            "test_name": test_name,
            "pass_fail": status,
            "failure_reason": failure_reason,
            "steps": int(steps),
            "duration_s": f"{timer() - start:.6f}",
            "max_linear_velocity": metrics.get("max_linear_velocity", ""),
            "max_angular_velocity": metrics.get("max_angular_velocity", ""),
            "settled_linear_velocity": metrics.get("settled_linear_velocity", ""),
            "settled_angular_velocity": metrics.get("settled_angular_velocity", ""),
            "max_normal_contact_force": metrics.get("max_normal_contact_force", ""),
            "min_body_z": metrics.get("min_body_z", ""),
            "min_bottom_z": metrics.get("min_bottom_z", ""),
            "finger_contact_count": finger_count,
            "contact_anomalies": anomaly,
            "engineering_notes": (
                "Deterministic contact check with table-penetration, settled-velocity, force, explosion/spin, "
                "and expected gripper-contact criteria."
            ),
        }
        return {field: row.get(field, "") for field in CONTACT_RESULT_FIELDS}

    def _step_collect(self, object_name: str, steps: int, ignore_initial_steps: int = 0) -> Dict[str, float]:
        metrics = self._empty_metrics()
        sampled = False
        for idx in range(max(1, int(steps))):
            self.mj.mj_step(self.model, self.data)
            if idx < int(ignore_initial_steps):
                continue
            self._sample_object_metrics(object_name, metrics)
            sampled = True
        if not sampled:
            self.mj.mj_forward(self.model, self.data)
            self._sample_object_metrics(object_name, metrics)
        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            "max_linear_velocity": 0.0,
            "max_angular_velocity": 0.0,
            "settled_linear_velocity": 0.0,
            "settled_angular_velocity": 0.0,
            "max_normal_contact_force": 0.0,
            "min_body_z": float("inf"),
            "min_bottom_z": float("inf"),
        }

    def _sample_object_metrics(self, object_name: str, metrics: Dict[str, float]) -> None:
        spec = OBJECT_SPECS[object_name]
        joint_id = self._joint_ids[object_name]
        body_id = self._body_ids[object_name]
        dofadr = int(self.model.jnt_dofadr[joint_id])
        qvel = np.asarray(self.data.qvel[dofadr : dofadr + 6], dtype=np.float64)
        linear = float(np.linalg.norm(qvel[:3]))
        angular = float(np.linalg.norm(qvel[3:]))
        body_z = float(self.data.xpos[body_id, 2])
        metrics["max_linear_velocity"] = max(float(metrics["max_linear_velocity"]), linear)
        metrics["max_angular_velocity"] = max(float(metrics["max_angular_velocity"]), angular)
        metrics["settled_linear_velocity"] = linear
        metrics["settled_angular_velocity"] = angular
        metrics["max_normal_contact_force"] = max(float(metrics["max_normal_contact_force"]), self._max_contact_force(body_id))
        metrics["min_body_z"] = min(float(metrics["min_body_z"]), body_z)
        metrics["min_bottom_z"] = min(float(metrics["min_bottom_z"]), spec.bottom_z(body_z))

    def _merge_metrics(self, *metrics_list: Dict[str, float]) -> Dict[str, float]:
        merged = self._empty_metrics()
        last = metrics_list[-1] if metrics_list else merged
        for metrics in metrics_list:
            merged["max_linear_velocity"] = max(merged["max_linear_velocity"], float(metrics["max_linear_velocity"]))
            merged["max_angular_velocity"] = max(merged["max_angular_velocity"], float(metrics["max_angular_velocity"]))
            merged["max_normal_contact_force"] = max(merged["max_normal_contact_force"], float(metrics["max_normal_contact_force"]))
            merged["min_body_z"] = min(merged["min_body_z"], float(metrics["min_body_z"]))
            merged["min_bottom_z"] = min(merged["min_bottom_z"], float(metrics["min_bottom_z"]))
        merged["settled_linear_velocity"] = float(last["settled_linear_velocity"])
        merged["settled_angular_velocity"] = float(last["settled_angular_velocity"])
        return merged

    def _run_gripper_contact_sequence(self, object_name: str, test_name: str, steps: int) -> Tuple[Dict[str, float], int, List[str]]:
        spec = OBJECT_SPECS[object_name]
        pos = np.array([0.0, 0.0, spec.center_z], dtype=np.float64)
        ee_z = self._gripper_test_ee_z(spec)
        self._set_object_pose(object_name, pos)
        self._set_ee_pose([pos[0], pos[1], ee_z])
        self._set_tool_orientation_neutral()
        self._set_gripper_open_fraction(1.0)
        self.mj.mj_forward(self.model, self.data)

        for _ in range(60):
            self._step_waypoint([pos[0], pos[1], ee_z], gripper=1.0, stabilize_tool=True)

        close_steps = max(100, int(0.40 * max(1, steps)))
        hold_steps = max(120, int(0.35 * max(1, steps)))
        target_open_fraction = self._gripper_target_open_fraction(spec, test_name)
        metrics = self._empty_metrics()
        finger_contact_samples = 0
        squeeze_contact_samples = 0

        for value in np.linspace(1.0, target_open_fraction, num=close_steps):
            self._step_waypoint([pos[0], pos[1], ee_z], gripper=float(value), stabilize_tool=True)
            self._sample_object_metrics(object_name, metrics)
            current_contacts = self._finger_contact_count(object_name)
            finger_contact_samples += current_contacts
            squeeze_contact_samples += current_contacts

        for _ in range(hold_steps):
            self._step_waypoint([pos[0], pos[1], ee_z], gripper=target_open_fraction, stabilize_tool=True)
            self._sample_object_metrics(object_name, metrics)
            current_contacts = self._finger_contact_count(object_name)
            finger_contact_samples += current_contacts
            squeeze_contact_samples += current_contacts

        extra_failures: List[str] = []
        if squeeze_contact_samples <= 0:
            extra_failures.append("no gripper/object contact detected during squeeze")
        squeeze_final_z = float(self._object_pos(object_name)[2])
        if squeeze_contact_samples > 0 and squeeze_final_z > float(spec.center_z) + 0.05:
            extra_failures.append("gripper geometry/actuation limitation: squeeze contact ejects object")

        if test_name == "lift":
            squeeze_unstable = (
                squeeze_contact_samples <= 0
                or squeeze_final_z > float(spec.center_z) + 0.05
                or float(metrics["settled_linear_velocity"]) > MAX_SETTLED_LINEAR_VELOCITY_MPS
                or float(metrics["settled_angular_velocity"]) > MAX_SETTLED_ANGULAR_VELOCITY_RADPS
                or float(metrics["max_linear_velocity"]) > MAX_TRANSIENT_LINEAR_VELOCITY_MPS
                or float(metrics["max_angular_velocity"]) > MAX_TRANSIENT_ANGULAR_VELOCITY_RADPS
                or float(metrics["max_normal_contact_force"]) > MAX_CONTACT_FORCE_N
            )
            if squeeze_unstable:
                extra_failures.append("lift not attempted because squeeze contact was not stable")
                return metrics, int(finger_contact_samples), extra_failures
            object_z_before_lift = float(self._object_pos(object_name)[2])
            lift_metrics = self._empty_metrics()
            lift_steps = max(100, int(0.35 * max(1, steps)))
            post_lift_hold_steps = max(80, int(0.20 * max(1, steps)))
            lift_distance = 0.10
            for idx in range(lift_steps):
                alpha = (idx + 1) / lift_steps
                self._step_waypoint(
                    [pos[0], pos[1], ee_z + lift_distance * alpha],
                    gripper=target_open_fraction,
                    stabilize_tool=True,
                )
                self._sample_object_metrics(object_name, lift_metrics)
                finger_contact_samples += self._finger_contact_count(object_name)
            for _ in range(post_lift_hold_steps):
                self._step_waypoint(
                    [pos[0], pos[1], ee_z + lift_distance],
                    gripper=target_open_fraction,
                    stabilize_tool=True,
                )
                self._sample_object_metrics(object_name, lift_metrics)
                finger_contact_samples += self._finger_contact_count(object_name)
            final_z = float(self._object_pos(object_name)[2])
            metrics = self._merge_metrics(metrics, lift_metrics)
            if squeeze_contact_samples > 0 and final_z - object_z_before_lift < 0.04:
                extra_failures.append("gripper contacted object but object did not follow during lift")

        return metrics, int(finger_contact_samples), extra_failures

    def _max_contact_force(self, body_id: int) -> float:
        max_force = 0.0
        for idx in range(int(self.data.ncon)):
            contact = self.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            bodies = {int(self.model.geom_bodyid[geom1]), int(self.model.geom_bodyid[geom2])}
            if int(body_id) not in bodies:
                continue
            force = np.zeros(6, dtype=np.float64)
            try:
                self.mj.mj_contactForce(self.model, self.data, idx, force)
            except Exception:
                continue
            max_force = max(max_force, abs(float(force[0])))
        return float(max_force)

    def _finger_contact_count(self, object_name: str) -> int:
        object_body = self._body_ids[object_name]
        finger_bodies = {
            self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "finger_left_car"),
            self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "finger_right_car"),
            self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "ee_platform"),
        }
        count = 0
        for idx in range(int(self.data.ncon)):
            contact = self.data.contact[idx]
            bodies = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            if int(object_body) in bodies and bodies.intersection(finger_bodies):
                count += 1
        return int(count)

    def _contact_verdict(
        self,
        metrics: Dict[str, float],
        spec: Any,
        require_contact: bool = False,
        finger_contact_count: int = 0,
        extra_failures: Optional[Sequence[str]] = None,
    ) -> Tuple[str, str]:
        reasons: List[str] = []
        values = [float(value) for value in metrics.values()]
        if not np.isfinite(values).all():
            reasons.append("non-finite state")
        if require_contact and int(finger_contact_count) <= 0:
            reasons.append("missing expected gripper/object contact")
        if float(metrics.get("min_bottom_z", float("inf"))) < TABLE_TOP_Z - TABLE_PENETRATION_TOLERANCE_M:
            reasons.append("body z below table tolerance")
        if (
            float(metrics.get("max_linear_velocity", 0.0)) > MAX_TRANSIENT_LINEAR_VELOCITY_MPS
            or float(metrics.get("max_angular_velocity", 0.0)) > MAX_TRANSIENT_ANGULAR_VELOCITY_RADPS
        ):
            reasons.append("object explosion/spin")
        if float(metrics.get("settled_linear_velocity", 0.0)) > MAX_SETTLED_LINEAR_VELOCITY_MPS:
            reasons.append("excessive linear velocity after settling")
        if float(metrics.get("settled_angular_velocity", 0.0)) > MAX_SETTLED_ANGULAR_VELOCITY_RADPS:
            reasons.append("excessive angular velocity after settling")
        if float(metrics.get("max_normal_contact_force", 0.0)) > MAX_CONTACT_FORCE_N:
            reasons.append("excessive normal/contact force")
        for reason in extra_failures or []:
            if reason and reason not in reasons:
                reasons.append(str(reason))
        return ("fail", "; ".join(reasons)) if reasons else ("pass", "")

    def run_render_profile(self, render_steps: int) -> Dict[str, Any]:
        if not self.render_enabled:
            gpu_vram = gpu_vram_mb()
            gpu_util = gpu_utilization_percent()
            return {
                "backend_name": self.backend_name,
                "simulator_version": self.version,
                "robot_embodiment": self.robot_embodiment,
                "num_environments": self.num_envs,
                "camera_count": 0,
                "render_resolution": f"{self.width}x{self.height}",
                "render_backend": self.render_backend,
                "platform": platform_label(),
                "rendered_rgb_frames": 0,
                "sim_steps": int(render_steps),
                "step_time_s": "0.000000",
                "render_time_s": "0.000000",
                "total_time_s": "0.000000",
                "step_fps_during_rgb": "0.00",
                "rgb_frame_fps": "0.00",
                "cpu_ram_mb": f"{rss_mb():.2f}",
                "gpu_vram_mb": "" if gpu_vram is None else f"{gpu_vram:.2f}",
                "gpu_utilization_percent": "" if gpu_util is None else f"{gpu_util:.1f}",
                "failure_reason": "render skipped by --no-render",
                "engineering_notes": "RGB rendering was not requested for this run.",
            }

        attempts = self._render_backend_attempts()
        failures: List[str] = []
        last_payload: Dict[str, Any] = {}
        for backend in attempts:
            payload = self._run_render_worker(backend, int(render_steps))
            last_payload = payload
            if bool(payload.get("ok")):
                return self._render_row_from_payload(payload, int(render_steps), failures)
            failures.append(
                f"{backend}: {payload.get('failure_reason') or 'render worker failed'}"
            )
        if not last_payload:
            last_payload = {
                "backend": ",".join(attempts) if attempts else self.render_backend,
                "camera_count": 0,
                "rendered_rgb_frames": 0,
                "step_time_s": 0.0,
                "render_time_s": 0.0,
                "total_time_s": 0.0,
                "step_fps_during_rgb": 0.0,
                "rgb_frame_fps": 0.0,
                "cpu_ram_mb": rss_mb(),
                "gpu_vram_mb": gpu_vram_mb(),
                "gpu_utilization_percent": gpu_utilization_percent(),
                "platform": platform_label(),
                "failure_reason": "no render backend attempts were selected",
                "engineering_notes": "",
            }
        return self._render_row_from_payload(last_payload, int(render_steps), failures)

    def _render_backend_attempts(self) -> List[str]:
        requested = self.render_backend
        if requested != "auto":
            return [requested]
        env_backend = os.environ.get("MUJOCO_GL", "").strip().lower()
        attempts: List[str] = []
        if env_backend in {"egl", "osmesa", "glfw"}:
            attempts.append(env_backend)
        system = platform.system().lower()
        if system == "linux":
            if shutil.which("nvidia-smi"):
                attempts.extend(["egl", "osmesa", "glfw"])
            else:
                attempts.extend(["osmesa", "egl", "glfw"])
        elif system == "darwin":
            attempts.append("glfw")
        else:
            attempts.append("glfw")
        deduped: List[str] = []
        for backend in attempts:
            if backend not in deduped:
                deduped.append(backend)
        return deduped

    def _run_render_worker(self, backend: str, render_steps: int) -> Dict[str, Any]:
        worker = REPO_ROOT / "tools" / "sim_compare" / "mujoco_render_worker.py"
        env = dict(os.environ)
        env["MUJOCO_GL"] = backend
        if backend == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")
        elif backend == "osmesa":
            env.setdefault("PYOPENGL_PLATFORM", "osmesa")
        cmd = [
            sys.executable,
            str(worker),
            "--scene",
            str(self.scene_path),
            "--backend",
            backend,
            "--width",
            str(int(self.width)),
            "--height",
            str(int(self.height)),
            "--steps",
            str(max(1, int(render_steps))),
            "--camera-count",
            str(max(1, int(self.camera_count))),
        ]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=max(20, int(render_steps) * 2))
        except Exception as exc:
            return {
                "ok": False,
                "backend": backend,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "engineering_notes": self._render_actionable_note(backend, f"{type(exc).__name__}: {exc}"),
            }
        payload: Dict[str, Any]
        try:
            payload = json.loads(proc.stdout.strip() or "{}")
        except Exception:
            payload = {}
        if proc.returncode != 0 or not payload:
            stderr = proc.stderr.strip()
            payload = {
                "ok": False,
                "backend": backend,
                "camera_count": 0,
                "rendered_rgb_frames": 0,
                "step_time_s": 0.0,
                "render_time_s": 0.0,
                "total_time_s": 0.0,
                "step_fps_during_rgb": 0.0,
                "rgb_frame_fps": 0.0,
                "cpu_ram_mb": rss_mb(),
                "gpu_vram_mb": gpu_vram_mb(),
                "gpu_utilization_percent": gpu_utilization_percent(),
                "platform": platform_label(),
                "failure_reason": stderr or f"render worker exited {proc.returncode}",
                "engineering_notes": "",
            }
        if proc.stderr.strip() and not payload.get("ok"):
            payload["failure_reason"] = f"{payload.get('failure_reason', '')}; stderr: {proc.stderr.strip()}"
        payload.setdefault("backend", backend)
        payload["engineering_notes"] = self._render_actionable_note(backend, str(payload.get("failure_reason", "")), str(payload.get("engineering_notes", "")))
        return payload

    def _render_row_from_payload(self, payload: Dict[str, Any], render_steps: int, failures: Sequence[str]) -> Dict[str, Any]:
        ok = bool(payload.get("ok"))
        failure_reason = "" if ok else str(payload.get("failure_reason") or "; ".join(failures))
        note = str(payload.get("engineering_notes", ""))
        if failures and ok:
            note = f"{note} Earlier failed attempts: {' | '.join(failures)}"
        return {
            "backend_name": self.backend_name,
            "simulator_version": self.version,
            "robot_embodiment": self.robot_embodiment,
            "num_environments": self.num_envs,
            "camera_count": int(payload.get("camera_count") or 0),
            "render_resolution": f"{self.width}x{self.height}",
            "render_backend": str(payload.get("backend") or self.render_backend),
            "platform": str(payload.get("platform") or platform_label()),
            "rendered_rgb_frames": int(payload.get("rendered_rgb_frames") or 0),
            "sim_steps": int(render_steps),
            "step_time_s": f"{float(payload.get('step_time_s') or 0.0):.6f}",
            "render_time_s": f"{float(payload.get('render_time_s') or 0.0):.6f}",
            "total_time_s": f"{float(payload.get('total_time_s') or 0.0):.6f}",
            "step_fps_during_rgb": f"{float(payload.get('step_fps_during_rgb') or 0.0):.2f}",
            "rgb_frame_fps": f"{float(payload.get('rgb_frame_fps') or 0.0):.2f}",
            "cpu_ram_mb": f"{float(payload.get('cpu_ram_mb') or rss_mb()):.2f}",
            "gpu_vram_mb": "" if payload.get("gpu_vram_mb") is None else f"{float(payload.get('gpu_vram_mb') or 0.0):.2f}",
            "gpu_utilization_percent": "" if payload.get("gpu_utilization_percent") is None else f"{float(payload.get('gpu_utilization_percent') or 0.0):.1f}",
            "failure_reason": failure_reason,
            "engineering_notes": note,
        }

    def _render_actionable_note(self, backend: str, failure: str, worker_note: str = "") -> str:
        system = platform.system().lower()
        prefix = f"Render backend attempted: {backend}; platform: {platform_label()}."
        if not failure:
            return f"{prefix} {worker_note or 'RGB rendering succeeded.'}".strip()
        if backend == "egl":
            advice = (
                "For Linux/NVIDIA remote servers run with `MUJOCO_GL=egl`; verify the NVIDIA driver, "
                "`nvidia-smi`, and EGL libraries are visible inside the environment."
            )
        elif backend == "osmesa":
            advice = "For CPU headless rendering install/enable OSMesa and run with `MUJOCO_GL=osmesa`."
        elif system == "darwin":
            advice = (
                "Local macOS uses MuJoCo GLFW/CGL; `CGLError: invalid CoreGraphics connection` means this "
                "process lacks a usable GUI/CoreGraphics session. Run from an interactive macOS login session, "
                "or profile RGB on the Linux server with EGL."
            )
        else:
            advice = "For local GUI rendering run with `MUJOCO_GL=glfw` and a valid display."
        brief_failure = failure.split("; stderr:", 1)[0].strip()
        return f"{prefix} {advice} Failure: {brief_failure}. {worker_note}".strip()
