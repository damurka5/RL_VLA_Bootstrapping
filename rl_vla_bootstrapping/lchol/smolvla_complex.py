from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Mapping, Sequence

import numpy as np

from rl_vla_bootstrapping.lchol.frontier_scheduler import (
    FrontierScheduler,
    FrontierSchedulerConfig,
    ShellValidationResult,
)
from rl_vla_bootstrapping.lchol.replay_buffers import PerOptionReplayBuffer
from robots.cdpr.cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec
from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import (
    SMOLVLA_COMPLEX_PROFILE,
    get_cdpr_reverse_shell_specs,
)


COMPLEX_INSTRUCTION_TYPES: tuple[str, ...] = (
    "move_to_object",
    "push_left",
    "push_right",
    "put_into_bowl",
    "put_into_plate",
    "move_left_of_object",
    "move_right_of_object",
    "move_between_objects",
)

PLACEMENT_INSTRUCTION_TYPES: frozenset[str] = frozenset(
    {
        "put_into_bowl",
        "put_into_plate",
        "move_left_of_object",
        "move_right_of_object",
        "move_between_objects",
    }
)


class SmolVLAComplexRuntime:
    """Curriculum and hindsight state shared by the two SmolVLA experiments."""

    def __init__(self, *, args: Any, seed: int) -> None:
        self.args = args
        self.mode = str(getattr(args, "complex_training_approach", "none") or "none")
        configured = tuple(str(item) for item in (getattr(args, "instruction_types", None) or ()))
        self.instruction_types = tuple(item for item in configured if item in COMPLEX_INSTRUCTION_TYPES)
        if not self.instruction_types:
            self.instruction_types = COMPLEX_INSTRUCTION_TYPES
        self.rng = np.random.default_rng(int(seed))
        self.spec = CDPRLCHOLSpec()
        capacity = max(1, int(getattr(args, "lchol_hindsight_replay_capacity", 20_000)))
        self.replay = PerOptionReplayBuffer(max(1, capacity // max(1, len(self.spec.option_names))))
        self.trajectories: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.seen_relabels: dict[int, set[tuple[str, str, int]]] = defaultdict(set)
        self.relabel_counts: Counter[str] = Counter()
        self.relabels_added_since_log = 0
        self.episode_success: dict[str, deque[float]] = {
            name: deque(maxlen=max(1, int(getattr(args, "metrics_window_episodes", 100))))
            for name in self.instruction_types
        }

        self.frontier: FrontierScheduler | None = None
        if self.mode == "reverse_frontier":
            specs = get_cdpr_reverse_shell_specs(
                self.instruction_types,
                profile=SMOLVLA_COMPLEX_PROFILE,
            )
            self.frontier = FrontierScheduler(
                specs=specs,
                config=FrontierSchedulerConfig(
                    promotion_success=float(getattr(args, "reverse_frontier_promotion_success", 0.80)),
                    demotion_success=float(getattr(args, "reverse_frontier_demotion_success", -1.0)),
                    validation_rollouts_per_shell=int(
                        getattr(args, "reverse_frontier_validation_episodes", 50)
                    ),
                    min_train_updates_before_validation=int(
                        getattr(args, "reverse_frontier_min_train_updates", 1)
                    ),
                    max_shell_jump=1,
                    saturation_abort_threshold=float(
                        getattr(args, "reverse_frontier_saturation_abort_threshold", 1.01)
                    ),
                    sample_frontier_probability=float(
                        getattr(args, "reverse_frontier_sample_probability", 0.80)
                    ),
                    sample_rehearsal_probability=float(
                        getattr(args, "reverse_frontier_rehearsal_probability", 0.20)
                    ),
                ),
            )

        self.put_stage: dict[str, int] = {
            name: 0 for name in self.instruction_types if name in PLACEMENT_INSTRUCTION_TYPES
        }
        history_size = max(1, int(getattr(args, "put_stage_history_episodes", 50)))
        self.put_stage_history: dict[str, deque[float]] = {
            name: deque(maxlen=history_size) for name in self.put_stage
        }
        self.put_stage_promotions: Counter[str] = Counter()

    @property
    def hindsight_enabled(self) -> bool:
        return self.mode == "lchol_hindsight"

    @property
    def reverse_frontier_enabled(self) -> bool:
        return self.frontier is not None

    def reset_options(self) -> dict[str, Any]:
        if self.frontier is not None:
            sample = self.frontier.sample(rng=self.rng)
            return {
                "instruction_type": str(sample.instruction_id),
                "curriculum_mode": "reverse_frontier",
                "curriculum_shell": int(sample.shell_id),
                "curriculum_sample_source": str(sample.source),
                "start_with_caught_object": False,
                "start_with_target_at_gripper": False,
            }

        instruction = str(self.instruction_types[int(self.rng.integers(0, len(self.instruction_types)))])
        options: dict[str, Any] = {"instruction_type": instruction}
        if self.hindsight_enabled and instruction in self.put_stage:
            stage = int(self.put_stage[instruction])
            options.update(
                {
                    "curriculum_mode": "lchol_put_stage",
                    "curriculum_shell": stage,
                    "put_start_stage": stage,
                    "start_with_caught_object": bool(stage == 0),
                    "start_with_target_at_gripper": False,
                }
            )
        return options

    def reset_episode(self, slot_index: int) -> None:
        self.trajectories[int(slot_index)] = []
        self.seen_relabels[int(slot_index)] = set()

    def append_trajectory_step(
        self,
        slot_index: int,
        step: Mapping[str, Any],
    ) -> list[Any]:
        if not self.hindsight_enabled:
            return []
        slot_index = int(slot_index)
        self.trajectories[slot_index].append(dict(step))
        records = self.spec.build_hindsight_records(
            self.trajectories[slot_index],
            prefix_max_steps=max(1, int(getattr(self.args, "lchol_hindsight_prefix_max_steps", 16))),
        )
        new_records = []
        seen = self.seen_relabels[slot_index]
        for record in records:
            key = (str(record.option_name), str(record.instruction), int(record.first_timestep))
            if key in seen:
                continue
            seen.add(key)
            new_records.append(record)
        return new_records

    def add_hindsight_record(self, record: Mapping[str, Any]) -> None:
        option = str(record.get("option_name") or "unknown")
        self.replay.add(option, dict(record))
        self.relabel_counts[option] += 1
        self.relabels_added_since_log += 1

    def sample_hindsight(self, rollout_record_count: int) -> list[dict[str, Any]]:
        ratio = max(0.0, float(getattr(self.args, "lchol_hindsight_replay_ratio", 0.25)))
        batch_size = int(round(max(0, int(rollout_record_count)) * ratio))
        if batch_size <= 0 or len(self.replay) == 0:
            return []
        return [
            dict(record)
            for record in self.replay.sample_balanced(batch_size=batch_size, rng=self.rng)
        ]

    def record_episode(
        self,
        *,
        instruction_type: str,
        success: bool,
        episode_put_stage: int = -1,
    ) -> bool:
        instruction = str(instruction_type)
        if instruction in self.episode_success:
            self.episode_success[instruction].append(1.0 if success else 0.0)
        if not self.hindsight_enabled or instruction not in self.put_stage:
            return False
        if int(episode_put_stage) != int(self.put_stage[instruction]):
            return False
        history = self.put_stage_history[instruction]
        history.append(1.0 if success else 0.0)
        min_episodes = max(1, int(getattr(self.args, "put_stage_min_episodes", 30)))
        threshold = float(getattr(self.args, "put_stage_promotion_success", 0.80))
        if (
            int(self.put_stage[instruction]) == 0
            and len(history) >= min_episodes
            and float(np.mean(history)) >= threshold
        ):
            self.put_stage[instruction] = 1
            self.put_stage_promotions[instruction] += 1
            history.clear()
            return True
        return False

    def record_train_update(self, instruction_types: Sequence[str]) -> None:
        if self.frontier is None:
            return
        for instruction in sorted({str(item) for item in instruction_types}):
            self.frontier.record_train_update(instruction)

    def reverse_validation_plan(self) -> list[tuple[str, int]]:
        if self.frontier is None:
            return []
        return [
            (instruction, int(shell))
            for instruction, shell in sorted(self.frontier.active_shells.items())
        ]

    def record_reverse_validation(self, results: Sequence[Mapping[str, Any]]) -> None:
        if self.frontier is None:
            return
        self.frontier.update(
            [
                ShellValidationResult(
                    instruction_id=str(item["instruction_id"]),
                    shell_id=int(item["shell_id"]),
                    success_rate=float(item["success_rate"]),
                    rollouts=int(item["rollouts"]),
                    action_saturation_rate=float(item.get("action_saturation_rate", 0.0)),
                )
                for item in results
            ]
        )

    def metrics(self, *, consume_interval_counts: bool = False) -> dict[str, float]:
        out: dict[str, float] = {
            "complex/approach_reverse_frontier": float(self.reverse_frontier_enabled),
            "complex/approach_lchol_hindsight": float(self.hindsight_enabled),
            "lchol/replay_size": float(len(self.replay)),
            "lchol/relabels_added_interval": float(self.relabels_added_since_log),
        }
        if self.frontier is not None:
            out.update(self.frontier.metrics())
        for instruction, history in sorted(self.episode_success.items()):
            token = instruction.replace(" ", "_")
            out[f"complex/success_rate/{token}"] = float(np.mean(history)) if history else 0.0
            out[f"complex/success_episodes/{token}"] = float(len(history))
        for instruction, stage in sorted(self.put_stage.items()):
            token = instruction.replace(" ", "_")
            history = self.put_stage_history[instruction]
            out[f"lchol/put_stage/{token}/active_stage"] = float(stage)
            out[f"lchol/put_stage/{token}/success_rate"] = float(np.mean(history)) if history else 0.0
            out[f"lchol/put_stage/{token}/episodes"] = float(len(history))
            out[f"lchol/put_stage/{token}/promotions"] = float(self.put_stage_promotions[instruction])
        for option, size in self.replay.sizes().items():
            out[f"lchol/replay_size_by_option/{option}"] = float(size)
        for option, count in sorted(self.relabel_counts.items()):
            out[f"lchol/relabel_count/{option}"] = float(count)
        if consume_interval_counts:
            self.relabels_added_since_log = 0
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rng_state": self.rng.bit_generator.state,
            "frontier": None if self.frontier is None else self.frontier.to_json_dict(),
            "put_stage": dict(self.put_stage),
            "put_stage_history": {
                key: list(values) for key, values in self.put_stage_history.items()
            },
            "put_stage_promotions": dict(self.put_stage_promotions),
            "episode_success": {key: list(values) for key, values in self.episode_success.items()},
            "replay": self.replay.state_dict(),
            "relabel_counts": dict(self.relabel_counts),
        }

    def json_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frontier": None if self.frontier is None else self.frontier.to_json_dict(),
            "put_stage": dict(self.put_stage),
            "put_stage_success_history": {
                key: list(values) for key, values in self.put_stage_history.items()
            },
            "put_stage_promotions": dict(self.put_stage_promotions),
            "replay_sizes": self.replay.sizes(),
            "relabel_counts": dict(self.relabel_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        state = dict(state or {})
        if state.get("rng_state"):
            self.rng.bit_generator.state = dict(state["rng_state"])
        frontier_state = state.get("frontier")
        if self.frontier is not None and isinstance(frontier_state, Mapping):
            raw_state = dict(frontier_state.get("state") or {})
            for instruction, raw in raw_state.items():
                if instruction not in self.frontier.state:
                    continue
                target = self.frontier.state[instruction]
                for key, value in dict(raw).items():
                    if hasattr(target, key):
                        setattr(target, key, value)
        for instruction, value in dict(state.get("put_stage") or {}).items():
            if instruction in self.put_stage:
                self.put_stage[instruction] = int(np.clip(int(value), 0, 1))
        for instruction, values in dict(state.get("put_stage_history") or {}).items():
            if instruction in self.put_stage_history:
                self.put_stage_history[instruction].extend(float(item) for item in values)
        self.put_stage_promotions.update(dict(state.get("put_stage_promotions") or {}))
        for instruction, values in dict(state.get("episode_success") or {}).items():
            if instruction in self.episode_success:
                self.episode_success[instruction].extend(float(item) for item in values)
        if isinstance(state.get("replay"), Mapping):
            self.replay.load_state_dict(dict(state["replay"]))
        self.relabel_counts.update(dict(state.get("relabel_counts") or {}))
