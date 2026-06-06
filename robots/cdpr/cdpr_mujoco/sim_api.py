"""Small simulator-agnostic API adapter for the current CDPR MuJoCo stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import mujoco as mj
import numpy as np


@dataclass
class MujocoCDPRStateAPI:
    """Adapter exposing state, contacts, and rendering through stable methods."""

    owner: Any

    @property
    def sim(self) -> Any:
        return getattr(self.owner, "sim", self.owner)

    @property
    def model(self) -> Any:
        return self.sim.model

    @property
    def data(self) -> Any:
        return self.sim.data

    def get_body_pose(self, name: str) -> dict[str, np.ndarray]:
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, str(name))
        if body_id == -1:
            raise KeyError(f"MuJoCo body not found: {name}")
        pos = np.asarray(self.data.xpos[body_id], dtype=np.float64).copy()
        quat = np.asarray(self.data.xquat[body_id], dtype=np.float64).copy()
        return {"position": pos, "quat_wxyz": quat}

    def get_body_velocity(self, name: str) -> dict[str, np.ndarray]:
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, str(name))
        if body_id == -1:
            raise KeyError(f"MuJoCo body not found: {name}")
        vel = np.zeros(6, dtype=np.float64)
        try:
            mj.mj_objectVelocity(
                self.model,
                self.data,
                mj.mjtObj.mjOBJ_BODY,
                int(body_id),
                vel,
                0,
            )
        except Exception:
            pass
        return {"angular": vel[:3].copy(), "linear": vel[3:].copy()}

    def get_ee_pose(self) -> dict[str, np.ndarray]:
        return self.get_body_pose("ee_base")

    def get_gripper_state(self) -> dict[str, float]:
        sim = self.sim
        opening = None
        target = None
        if hasattr(sim, "get_gripper_opening"):
            opening = sim.get_gripper_opening()
        if hasattr(sim, "get_gripper_target"):
            target = sim.get_gripper_target()
        return {
            "opening": float("nan") if opening is None else float(opening),
            "target": float("nan") if target is None else float(target),
        }

    def get_contact_summary(self, body_a: str, body_b: str) -> dict[str, Any]:
        body_a_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, str(body_a))
        body_b_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, str(body_b))
        if body_a_id == -1 or body_b_id == -1:
            return {"count": 0, "max_normal_force": 0.0, "pairs": []}

        pairs: list[dict[str, Any]] = []
        max_normal_force = 0.0
        for idx in range(int(self.data.ncon)):
            contact = self.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            c_body1 = int(self.model.geom_bodyid[geom1])
            c_body2 = int(self.model.geom_bodyid[geom2])
            if {c_body1, c_body2} != {int(body_a_id), int(body_b_id)}:
                continue
            force = np.zeros(6, dtype=np.float64)
            try:
                mj.mj_contactForce(self.model, self.data, idx, force)
            except Exception:
                pass
            max_normal_force = max(max_normal_force, abs(float(force[0])))
            pairs.append(
                {
                    "geom1": mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom1) or str(geom1),
                    "geom2": mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom2) or str(geom2),
                    "normal_force": float(force[0]),
                }
            )
        return {
            "count": len(pairs),
            "max_normal_force": float(max_normal_force),
            "pairs": pairs,
        }

    def capture_state(self) -> dict[str, Any]:
        if hasattr(self.owner, "capture_state"):
            return self.owner.capture_state()
        return self.sim.capture_state()

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if hasattr(self.owner, "restore_state"):
            self.owner.restore_state(dict(state))
            return
        self.sim.restore_state(dict(state))

    def set_object_pose(self, name: str, pose: Mapping[str, Sequence[float]]) -> None:
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, str(name))
        if body_id == -1:
            raise KeyError(f"MuJoCo body not found: {name}")
        joint_count = int(self.model.body_jntnum[body_id])
        joint_adr = int(self.model.body_jntadr[body_id])
        free_joint = -1
        for offset in range(joint_count):
            jid = joint_adr + offset
            if int(self.model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE):
                free_joint = int(jid)
                break
        if free_joint == -1:
            raise ValueError(f"Body {name!r} does not have a freejoint.")

        qadr = int(self.model.jnt_qposadr[free_joint])
        pos = np.asarray(pose.get("position", pose.get("pos", self.data.qpos[qadr : qadr + 3])), dtype=np.float64)
        quat = np.asarray(
            pose.get("quat_wxyz", pose.get("quat", self.data.qpos[qadr + 3 : qadr + 7])),
            dtype=np.float64,
        )
        self.data.qpos[qadr : qadr + 3] = pos.reshape(-1)[:3]
        self.data.qpos[qadr + 3 : qadr + 7] = quat.reshape(-1)[:4]
        dofadr = int(self.model.jnt_dofadr[free_joint])
        self.data.qvel[dofadr : dofadr + 6] = 0.0
        mj.mj_forward(self.model, self.data)

    def render(self, camera_names: Sequence[str]) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for name in camera_names:
            camera_name = str(name)
            if camera_name in {"overview", "front", "main"}:
                camera = getattr(self.sim, "overview_cam")
            elif camera_name in {"ee_camera", "wrist"}:
                camera = getattr(self.sim, "ee_cam")
            else:
                raise KeyError(f"Unsupported CDPR camera: {camera_name}")
            frames[camera_name] = self.sim.capture_frame(camera, camera_name)
        return frames
