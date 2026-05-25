from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rl_vla_bootstrapping.lchol.reverse_shells import ReverseShellSpec, clamp_shell_id


@dataclass(frozen=True)
class FrontierSchedulerConfig:
    promotion_success: float = 0.50
    demotion_success: float = 0.20
    validation_rollouts_per_shell: int = 50
    min_train_updates_before_validation: int = 5
    max_shell_jump: int = 1
    saturation_abort_threshold: float = 0.30
    sample_frontier_probability: float = 0.80
    sample_rehearsal_probability: float = 0.20


@dataclass
class FrontierInstructionState:
    active_shell: int = 0
    validation_success: float = 0.0
    train_updates: int = 0
    last_promoted_update: int = 0
    validation_rollouts: int = 0
    action_saturation: float = 0.0


@dataclass(frozen=True)
class FrontierSample:
    instruction_id: str
    shell_id: int
    source: str = "frontier"


@dataclass(frozen=True)
class ShellValidationResult:
    instruction_id: str
    shell_id: int
    success_rate: float
    rollouts: int
    action_saturation_rate: float = 0.0


class FrontierScheduler:
    def __init__(
        self,
        *,
        specs: Sequence[ReverseShellSpec],
        config: FrontierSchedulerConfig | None = None,
        state: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.specs = {str(spec.instruction_id): spec for spec in specs}
        if not self.specs:
            raise ValueError("FrontierScheduler needs at least one reverse shell spec.")
        self.config = config or FrontierSchedulerConfig()
        self.state: dict[str, FrontierInstructionState] = {}
        for instruction_id in self.specs:
            raw = dict((state or {}).get(instruction_id, {}) or {})
            shell_count = int(self.specs[instruction_id].shell_count)
            active_shell = clamp_shell_id(int(raw.get("active_shell", 0)), shell_count)
            self.state[instruction_id] = FrontierInstructionState(
                active_shell=active_shell,
                validation_success=float(raw.get("validation_success", 0.0)),
                train_updates=int(raw.get("train_updates", 0)),
                last_promoted_update=int(raw.get("last_promoted_update", 0)),
                validation_rollouts=int(raw.get("validation_rollouts", 0)),
                action_saturation=float(raw.get("action_saturation", 0.0)),
            )

    @property
    def active_shells(self) -> dict[str, int]:
        return {key: int(value.active_shell) for key, value in self.state.items()}

    def sample(self, *, rng: np.random.Generator) -> FrontierSample:
        frontier_ids = [
            instruction_id
            for instruction_id, state in self.state.items()
            if int(state.active_shell) < int(self.specs[instruction_id].shell_count) - 1
        ]
        instruction_ids = frontier_ids or list(self.state)
        use_rehearsal = bool(
            frontier_ids
            and float(self.config.sample_rehearsal_probability) > 0.0
            and float(rng.random()) >= float(self.config.sample_frontier_probability)
        )
        instruction_id = str(instruction_ids[int(rng.integers(0, len(instruction_ids)))])
        active_shell = int(self.state[instruction_id].active_shell)
        if use_rehearsal and active_shell > 0:
            shell_id = int(rng.integers(0, active_shell + 1))
            source = "rehearsal"
        else:
            shell_id = active_shell
            source = "frontier"
        return FrontierSample(instruction_id=instruction_id, shell_id=shell_id, source=source)

    def record_train_update(self, instruction_id: str | None = None) -> None:
        if instruction_id:
            keys = [str(instruction_id)] if str(instruction_id) in self.state else []
        else:
            keys = list(self.state)
        for key in keys:
            self.state[key].train_updates += 1

    def update(
        self,
        results: Mapping[str, ShellValidationResult | Mapping[str, Any]] | Sequence[ShellValidationResult | Mapping[str, Any]],
    ) -> dict[str, FrontierInstructionState]:
        if isinstance(results, Mapping):
            iterable = results.values()
        else:
            iterable = results
        for raw_result in iterable:
            result = self._coerce_result(raw_result)
            if result.instruction_id not in self.state:
                continue
            spec = self.specs[result.instruction_id]
            state = self.state[result.instruction_id]
            state.validation_success = float(np.clip(result.success_rate, 0.0, 1.0))
            state.validation_rollouts = max(0, int(result.rollouts))
            state.action_saturation = float(np.clip(result.action_saturation_rate, 0.0, 1.0))
            if int(result.shell_id) != int(state.active_shell):
                continue
            if (
                state.validation_success <= float(self.config.demotion_success)
                and int(state.active_shell) > 0
            ):
                state.active_shell = max(0, int(state.active_shell) - 1)
                continue
            can_validate = (
                int(state.train_updates) - int(state.last_promoted_update)
                >= int(self.config.min_train_updates_before_validation)
            )
            saturation_ok = state.action_saturation < float(self.config.saturation_abort_threshold)
            if (
                can_validate
                and saturation_ok
                and state.validation_success >= float(self.config.promotion_success)
                and int(state.active_shell) < int(spec.shell_count) - 1
            ):
                jump = max(1, int(self.config.max_shell_jump))
                state.active_shell = min(int(spec.shell_count) - 1, int(state.active_shell) + jump)
                state.last_promoted_update = int(state.train_updates)
        return self.state

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for instruction_id, state in sorted(self.state.items()):
            prefix = f"reverse_frontier/{instruction_id}"
            out[f"{prefix}/active_shell"] = float(state.active_shell)
            out[f"{prefix}/validation_success"] = float(state.validation_success)
            out[f"{prefix}/validation_rollouts"] = float(state.validation_rollouts)
            out[f"{prefix}/train_updates"] = float(state.train_updates)
            out[f"{prefix}/action_saturation"] = float(state.action_saturation)
        return out

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "state": {key: asdict(value) for key, value in sorted(self.state.items())},
        }

    def save(self, path: Path | str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        specs: Sequence[ReverseShellSpec],
        config: FrontierSchedulerConfig | None = None,
    ) -> "FrontierScheduler":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded_config = config
        if loaded_config is None and isinstance(raw.get("config"), Mapping):
            loaded_config = FrontierSchedulerConfig(**dict(raw["config"]))
        return cls(specs=specs, config=loaded_config, state=dict(raw.get("state") or {}))

    @staticmethod
    def _coerce_result(raw: ShellValidationResult | Mapping[str, Any]) -> ShellValidationResult:
        if isinstance(raw, ShellValidationResult):
            return raw
        data = dict(raw)
        instruction_id = str(data.get("instruction_id") or data.get("instruction_type") or "")
        if not instruction_id:
            raise ValueError(f"Validation result is missing instruction_id: {data!r}")
        return ShellValidationResult(
            instruction_id=instruction_id,
            shell_id=int(data.get("shell_id", data.get("curriculum_shell", 0))),
            success_rate=float(data.get("success_rate", data.get("validation_success", 0.0))),
            rollouts=int(data.get("rollouts", data.get("validation_rollouts", 0))),
            action_saturation_rate=float(
                data.get("action_saturation_rate", data.get("action_saturation", 0.0))
            ),
        )
