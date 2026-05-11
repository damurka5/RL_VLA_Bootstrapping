from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    allowed_options: tuple[str, ...]
    gate_options: tuple[str, ...]
    success_threshold: float


DEFAULT_CDPR_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        name="approach",
        allowed_options=("move_to_object", "grab_object"),
        gate_options=("move_to_object", "grab_object"),
        success_threshold=0.60,
    ),
    CurriculumStage(
        name="grasp",
        allowed_options=("move_to_object", "grab_object", "pick_up"),
        gate_options=("grab_object", "pick_up"),
        success_threshold=0.60,
    ),
    CurriculumStage(
        name="push",
        allowed_options=("move_to_object", "grab_object", "pick_up", "push_left", "push_right"),
        gate_options=("push_left", "push_right"),
        success_threshold=0.50,
    ),
    CurriculumStage(
        name="place",
        allowed_options=("move_to_object", "grab_object", "pick_up", "push_left", "push_right", "put_into_plate"),
        gate_options=("put_into_plate",),
        success_threshold=0.40,
    ),
    CurriculumStage(
        name="binary_relation",
        allowed_options=(
            "move_to_object",
            "grab_object",
            "pick_up",
            "push_left",
            "push_right",
            "put_into_plate",
            "move_left_of_object",
            "move_right_of_object",
        ),
        gate_options=("move_left_of_object", "move_right_of_object"),
        success_threshold=0.40,
    ),
    CurriculumStage(
        name="full_mix",
        allowed_options=(
            "move_to_object",
            "grab_object",
            "pick_up",
            "push_left",
            "push_right",
            "put_into_plate",
            "move_left_of_object",
            "move_right_of_object",
            "move_between_objects",
        ),
        gate_options=("move_between_objects",),
        success_threshold=0.35,
    ),
)


class StrictSuccessCurriculum:
    def __init__(
        self,
        *,
        stages: Sequence[CurriculumStage] = DEFAULT_CDPR_STAGES,
        min_success_samples: int = 24,
        window_size: int = 96,
        weakest_mode_oversample_strength: float = 2.5,
        newest_stage_weight: float = 1.4,
    ):
        if not stages:
            raise ValueError("StrictSuccessCurriculum needs at least one stage.")
        self.stages = tuple(stages)
        self.min_success_samples = max(1, int(min_success_samples))
        self.weakest_mode_oversample_strength = max(1.0, float(weakest_mode_oversample_strength))
        self.newest_stage_weight = max(1.0, float(newest_stage_weight))
        self.stage_index = 0
        self._success: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(1, int(window_size))))

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    def record(self, info: Mapping[str, Any]) -> None:
        option = str(info.get("instruction_type") or info.get("option_name") or "").strip()
        if not option:
            return
        success = _success_value(info)
        self._success[option].append(float(success))
        self._maybe_promote()

    def allowed_options(self, available_options: Sequence[str] | None = None) -> tuple[str, ...]:
        allowed = tuple(self.stage.allowed_options)
        if available_options is None:
            return allowed
        available = {str(option) for option in available_options}
        return tuple(option for option in allowed if option in available)

    def success_rate(self, option_name: str) -> float:
        values = self._success.get(str(option_name))
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def sample_option(
        self,
        *,
        rng: np.random.Generator,
        available_options: Sequence[str] | None = None,
    ) -> str:
        options = list(self.allowed_options(available_options))
        if not options:
            raise ValueError("No options are available in the active LC-HOL curriculum stage.")

        rates = np.asarray([self.success_rate(option) for option in options], dtype=np.float64)
        weakness = 1.0 - np.clip(rates, 0.0, 1.0)
        weights = 1.0 + float(self.weakest_mode_oversample_strength) * weakness
        newest = set(self.stage.gate_options)
        for idx, option in enumerate(options):
            if option in newest:
                weights[idx] *= float(self.newest_stage_weight)
        weights = weights / float(weights.sum())
        return str(rng.choice(options, p=weights))

    def metrics(self) -> dict[str, float]:
        out = {
            "stage_index": float(self.stage_index),
            "stage_success_threshold": float(self.stage.success_threshold),
        }
        for option in self.stage.allowed_options:
            values = self._success.get(option)
            out[f"success_rate/{option}"] = float(sum(values) / len(values)) if values else 0.0
            out[f"samples/{option}"] = float(len(values) if values else 0)
        return out

    def _maybe_promote(self) -> None:
        while self.stage_index < len(self.stages) - 1 and self._stage_gate_passed(self.stage):
            self.stage_index += 1

    def _stage_gate_passed(self, stage: CurriculumStage) -> bool:
        gate_values: list[float] = []
        for option in stage.gate_options:
            values = self._success.get(option)
            if not values or len(values) < self.min_success_samples:
                return False
            gate_values.append(float(sum(values) / len(values)))
        return bool(gate_values and min(gate_values) >= float(stage.success_threshold))


def _success_value(info: Mapping[str, Any]) -> float:
    for key in ("success", "sparse_success", "caught_object_is_target", "relation_motion_ok"):
        if key not in info:
            continue
        raw = info.get(key)
        try:
            return 1.0 if float(raw) >= 0.5 else 0.0
        except (TypeError, ValueError):
            return 1.0 if bool(raw) else 0.0
    return 0.0
