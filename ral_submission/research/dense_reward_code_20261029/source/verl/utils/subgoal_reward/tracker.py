from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .libero_state import LiberoState
from .task_specs import TaskSpec


@dataclass
class SubgoalStepInfo:
    supported: bool
    phase_id: int
    phase_name: str
    progress: float
    best_progress: float
    positive_delta: float
    phase_completed: float
    success: bool
    action_delta_l2: float

    def as_numeric_dict(self) -> dict[str, float]:
        return {
            "subgoal_supported": float(self.supported),
            "subgoal_phase_id": float(self.phase_id),
            "subgoal_progress": self.progress,
            "subgoal_best_progress": self.best_progress,
            "subgoal_positive_delta": self.positive_delta,
            "subgoal_phase_completed": self.phase_completed,
            "success": float(self.success),
            "action_delta_l2": self.action_delta_l2,
        }


class OnlineSubgoalTracker:
    def __init__(
        self,
        task_spec: TaskSpec | None = None,
        use_best_progress: bool = True,
        auto_complete_on_success: bool = True,
    ):
        self.task_spec = task_spec
        self.use_best_progress = use_best_progress
        self.auto_complete_on_success = auto_complete_on_success
        self.reset(task_spec=task_spec)

    def reset(self, task_spec: TaskSpec | None = None):
        if task_spec is not None:
            self.task_spec = task_spec
        self.phase_id = 0
        self.best_progress_so_far = 0.0
        self.last_progress = 0.0
        self.prev_action = None

    def update(self, state: LiberoState, action: Any = None) -> SubgoalStepInfo:
        action_delta_l2 = self._action_delta(action)
        if self.task_spec is None or not self.task_spec.supported or not self.task_spec.phases:
            return SubgoalStepInfo(
                supported=False,
                phase_id=-1,
                phase_name="terminal_only",
                progress=1.0 if state.success else 0.0,
                best_progress=1.0 if state.success else 0.0,
                positive_delta=0.0,
                phase_completed=0.0,
                success=bool(state.success),
                action_delta_l2=action_delta_l2,
            )

        self.phase_id = max(0, min(self.phase_id, len(self.task_spec.phases) - 1))
        phase = self.task_spec.phases[self.phase_id]
        progress = float(np.clip(phase.compute_progress(state), 0.0, 1.0))
        if self.use_best_progress:
            positive_delta = max(progress - self.best_progress_so_far, 0.0)
            self.best_progress_so_far = max(self.best_progress_so_far, progress)
        else:
            positive_delta = progress - self.last_progress
            self.best_progress_so_far = progress
        self.last_progress = progress

        completed_count = 0.0
        while phase.is_done(state):
            completed_count += 1.0
            if self.phase_id >= len(self.task_spec.phases) - 1:
                break
            self.phase_id += 1
            self.best_progress_so_far = 0.0
            self.last_progress = 0.0
            phase = self.task_spec.phases[self.phase_id]

        if self.auto_complete_on_success and state.success and self.phase_id < len(self.task_spec.phases) - 1:
            completed_count += float(len(self.task_spec.phases) - self.phase_id - 1)
            self.phase_id = len(self.task_spec.phases) - 1

        return SubgoalStepInfo(
            supported=True,
            phase_id=self.phase_id,
            phase_name=self.task_spec.phases[self.phase_id].name,
            progress=progress,
            best_progress=self.best_progress_so_far,
            positive_delta=positive_delta,
            phase_completed=completed_count,
            success=bool(state.success),
            action_delta_l2=action_delta_l2,
        )

    def _action_delta(self, action: Any) -> float:
        if action is None:
            return 0.0
        try:
            current = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            return 0.0
        if self.prev_action is None or self.prev_action.shape != current.shape:
            self.prev_action = current.copy()
            return 0.0
        delta = float(np.linalg.norm(current - self.prev_action))
        self.prev_action = current.copy()
        return delta
