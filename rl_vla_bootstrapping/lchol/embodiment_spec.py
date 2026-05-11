from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class AchievedOption:
    option_name: str
    first_timestep: int
    instruction: str
    target_object: str = ""
    reference_object: str = ""
    second_reference_object: str = ""
    predicate_value: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HindsightBCRecord:
    option_name: str
    instruction: str
    action: Any
    source_instruction: str
    first_timestep: int
    image_primary: Any = None
    image_wrist: Any = None
    prefix_actions: tuple[Any, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EmbodimentLCHOLSpec(Protocol):
    option_names: Sequence[str]

    def phase_score(self, trajectory: Sequence[Mapping[str, Any]], option_name: str | None = None) -> float:
        ...

    def achieved_options(self, trajectory: Sequence[Mapping[str, Any]]) -> list[AchievedOption]:
        ...

    def relabel_instruction(self, achieved_option: AchievedOption) -> str:
        ...

    def synthetic_completion_action(self, achieved_option: AchievedOption, state: Mapping[str, Any]) -> Any:
        ...

    def build_hindsight_records(
        self,
        trajectory: Sequence[Mapping[str, Any]],
        *,
        prefix_max_steps: int,
    ) -> list[HindsightBCRecord]:
        ...
