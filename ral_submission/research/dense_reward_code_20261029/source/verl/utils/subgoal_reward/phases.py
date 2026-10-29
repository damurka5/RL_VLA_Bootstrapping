from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .libero_state import LiberoState


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class Thresholds:
    reach_distance: float = 0.05
    target_distance: float = 0.06
    lift_height: float = 0.08


class Phase:
    name = "phase"

    def __init__(self, thresholds: Thresholds, initial_object_z: float | None = None):
        self.thresholds = thresholds
        self.initial_object_z = initial_object_z

    def compute_progress(self, state: LiberoState) -> float:
        raise NotImplementedError

    def is_done(self, state: LiberoState) -> bool:
        raise NotImplementedError


class ReachObjectPhase(Phase):
    name = "reach_object"

    def compute_progress(self, state: LiberoState) -> float:
        if state.gripper_position is None or state.object_position is None:
            return 0.0
        dist = float(np.linalg.norm(state.gripper_position - state.object_position))
        return _clip01(1.0 - dist / max(self.thresholds.reach_distance * 4.0, 1e-6))

    def is_done(self, state: LiberoState) -> bool:
        if state.gripper_position is None or state.object_position is None:
            return False
        return float(np.linalg.norm(state.gripper_position - state.object_position)) <= self.thresholds.reach_distance


class GraspObjectPhase(ReachObjectPhase):
    name = "grasp_object"

    def compute_progress(self, state: LiberoState) -> float:
        if state.object_in_gripper:
            return 1.0
        reach = super().compute_progress(state)
        closed_bonus = 0.2 if state.gripper_open is not None and state.gripper_open < 0.5 else 0.0
        return _clip01(0.8 * reach + closed_bonus)

    def is_done(self, state: LiberoState) -> bool:
        if state.object_in_gripper:
            return True
        return super().is_done(state) and state.gripper_open is not None and state.gripper_open < 0.5


class LiftObjectPhase(Phase):
    name = "lift_object"

    def compute_progress(self, state: LiberoState) -> float:
        if state.object_position is None or self.initial_object_z is None:
            return 1.0 if state.object_in_gripper else 0.0
        lift = float(state.object_position[2] - self.initial_object_z)
        grasp_bonus = 0.2 if state.object_in_gripper else 0.0
        return _clip01(lift / max(self.thresholds.lift_height, 1e-6) + grasp_bonus)

    def is_done(self, state: LiberoState) -> bool:
        if state.object_position is None or self.initial_object_z is None:
            return bool(state.object_in_gripper)
        return float(state.object_position[2] - self.initial_object_z) >= self.thresholds.lift_height


class MoveToTargetPhase(Phase):
    name = "move_to_target"

    def _target_distance(self, state: LiberoState) -> float | None:
        if state.object_position is None or state.target_position is None:
            return None
        return float(np.linalg.norm(state.object_position[:2] - state.target_position[:2]))

    def compute_progress(self, state: LiberoState) -> float:
        dist = self._target_distance(state)
        if dist is None:
            return 0.0
        return _clip01(1.0 - dist / max(self.thresholds.target_distance * 5.0, 1e-6))

    def is_done(self, state: LiberoState) -> bool:
        dist = self._target_distance(state)
        if dist is None:
            return False
        return dist <= self.thresholds.target_distance


class PlaceOrSuccessPhase(MoveToTargetPhase):
    name = "place_or_success"

    def compute_progress(self, state: LiberoState) -> float:
        if state.success:
            return 1.0
        near_target = super().compute_progress(state)
        open_bonus = 0.1 if state.gripper_open is not None and state.gripper_open > 0.5 else 0.0
        return _clip01(0.9 * near_target + open_bonus)

    def is_done(self, state: LiberoState) -> bool:
        return bool(state.success) or (
            super().is_done(state) and state.gripper_open is not None and state.gripper_open > 0.5
        )
