from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CDPRPolicyControlSpec:
    xyz_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    action_step_xyz: float
    action_step_yaw: float
    open_gripper_threshold: float = -0.2
    close_gripper_threshold: float = 0.2
    hold_steps: int = 4

    @property
    def sim_steps_per_policy_action(self) -> int:
        return max(1, int(self.hold_steps) + 1)

    @property
    def policy_period_s(self) -> float | None:
        return None


def clamp_xyz_to_limits(
    xyz: np.ndarray,
    xyz_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float32).reshape(3).copy()
    for idx, (lo, hi) in enumerate(xyz_limits):
        out[idx] = np.float32(np.clip(out[idx], float(lo), float(hi)))
    return out


def policy_action_period_s(sim_dt: float, hold_steps: int) -> float:
    return float(sim_dt) * float(max(1, int(hold_steps) + 1))


def policy_action_frequency_hz(sim_dt: float, hold_steps: int) -> float:
    period = policy_action_period_s(sim_dt, hold_steps)
    if period <= 0:
        raise ValueError("Policy action period must be positive.")
    return 1.0 / period


def _set_ee_target(sim: Any, xyz: np.ndarray) -> None:
    if hasattr(sim, "set_end_effector_target"):
        sim.set_end_effector_target(xyz)
        return
    if hasattr(sim, "set_ee_target"):
        sim.set_ee_target(xyz)
        return
    if hasattr(sim, "set_target_position"):
        sim.set_target_position(xyz)
        return
    raise RuntimeError("Simulator has no supported end-effector target setter.")


def apply_normalized_cdpr_action(
    sim: Any,
    normalized_action: np.ndarray,
    spec: CDPRPolicyControlSpec,
    *,
    ee_min_z: float | None = None,
    capture_last_frame: bool = True,
) -> dict[str, Any]:
    action = np.asarray(normalized_action, dtype=np.float32).reshape(-1)
    if action.size != 5:
        raise ValueError(f"Expected normalized action shape (5,), got {action.shape}")
    action = np.clip(action, -1.0, 1.0)

    current_xyz = np.asarray(sim.get_end_effector_position(), dtype=np.float32).reshape(-1)[:3]
    target_xyz = current_xyz + np.asarray(action[:3] * float(spec.action_step_xyz), dtype=np.float32)
    target_xyz = clamp_xyz_to_limits(target_xyz, spec.xyz_limits)
    if ee_min_z is not None:
        target_xyz[2] = np.float32(max(float(target_xyz[2]), float(ee_min_z)))
    _set_ee_target(sim, target_xyz)

    current_yaw = None
    target_yaw = None
    if hasattr(sim, "get_yaw"):
        current_yaw = float(sim.get_yaw())
    if current_yaw is not None:
        target_yaw = float(current_yaw + float(action[3]) * float(spec.action_step_yaw))
        if hasattr(sim, "set_yaw"):
            sim.set_yaw(target_yaw)

    gripper_command = float(action[4])
    if gripper_command >= float(spec.close_gripper_threshold) and hasattr(sim, "close_gripper"):
        sim.close_gripper()
    elif gripper_command <= float(spec.open_gripper_threshold) and hasattr(sim, "open_gripper"):
        sim.open_gripper()

    steps = spec.sim_steps_per_policy_action
    for step_idx in range(steps):
        capture = bool(capture_last_frame and step_idx == (steps - 1))
        sim.run_simulation_step(capture_frame=capture)

    result: dict[str, Any] = {
        "commanded_action": action.copy(),
        "target_xyz": target_xyz.copy(),
        "gripper_command": gripper_command,
        "sim_steps": steps,
    }
    if target_yaw is not None:
        result["target_yaw"] = float(target_yaw)
    if hasattr(sim, "get_end_effector_position"):
        result["ee_position"] = np.asarray(sim.get_end_effector_position(), dtype=np.float32).reshape(-1)[:3].copy()
    if hasattr(sim, "get_yaw"):
        result["ee_yaw"] = float(sim.get_yaw())
    if hasattr(sim, "get_gripper_opening"):
        result["gripper_opening"] = float(sim.get_gripper_opening())
    return result
