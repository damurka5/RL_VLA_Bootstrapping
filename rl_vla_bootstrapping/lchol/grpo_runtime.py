from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rl_vla_bootstrapping.lchol.curriculum import StrictSuccessCurriculum
from rl_vla_bootstrapping.lchol.embodiment_spec import HindsightBCRecord
from rl_vla_bootstrapping.lchol.frontier_scheduler import (
    FrontierScheduler,
    FrontierSchedulerConfig,
    ShellValidationResult,
)
from rl_vla_bootstrapping.lchol.replay_buffers import PerOptionReplayBuffer


@dataclass(frozen=True)
class LCHOLGRPOConfig:
    enabled: bool = False
    group_score: str = "phase_shaped"
    hindsight_bc_coef: float = 0.20
    hindsight_done_weight: float = 0.20
    hindsight_replay_capacity: int = 20_000
    hindsight_replay_ratio: float = 0.50
    hindsight_prefix_max_steps: int = 16
    option_prior_bc_coef: float = 0.20
    option_prior_min_coef: float = 0.035
    option_prior_decay_updates: int = 80
    curriculum: str = "strict_staged"
    strict_min_success_samples: int = 24
    weakest_mode_oversample_strength: float = 2.5
    newest_stage_weight: float = 1.4
    reverse_promotion_success: float = 0.50
    reverse_demotion_success: float = 0.20
    reverse_validation_rollouts_per_shell: int = 50
    reverse_min_train_updates_before_validation: int = 5
    reverse_max_shell_jump: int = 1
    reverse_saturation_abort_threshold: float = 0.30
    reverse_sample_frontier_probability: float = 0.80
    reverse_sample_rehearsal_probability: float = 0.20


class LCHOLGRPORuntime:
    def __init__(
        self,
        *,
        config: LCHOLGRPOConfig,
        spec: Any,
        available_options: Sequence[str],
        seed: int,
    ):
        self.config = config
        self.spec = spec
        self.available_options = tuple(str(option) for option in available_options if str(option))
        self.rng = np.random.default_rng(int(seed))
        per_option_capacity = max(1, int(config.hindsight_replay_capacity) // max(1, len(self.available_options)))
        self.replay = PerOptionReplayBuffer(per_option_capacity)
        self.curriculum = StrictSuccessCurriculum(
            min_success_samples=int(config.strict_min_success_samples),
            weakest_mode_oversample_strength=float(config.weakest_mode_oversample_strength),
            newest_stage_weight=float(config.newest_stage_weight),
        )
        self.reverse_scheduler: FrontierScheduler | None = None
        if self._curriculum_name() == "reverse_frontier":
            self.reverse_scheduler = FrontierScheduler(
                specs=self._build_reverse_shell_specs(),
                config=FrontierSchedulerConfig(
                    promotion_success=float(config.reverse_promotion_success),
                    demotion_success=float(config.reverse_demotion_success),
                    validation_rollouts_per_shell=int(config.reverse_validation_rollouts_per_shell),
                    min_train_updates_before_validation=int(config.reverse_min_train_updates_before_validation),
                    max_shell_jump=int(config.reverse_max_shell_jump),
                    saturation_abort_threshold=float(config.reverse_saturation_abort_threshold),
                    sample_frontier_probability=float(config.reverse_sample_frontier_probability),
                    sample_rehearsal_probability=float(config.reverse_sample_rehearsal_probability),
                ),
            )
        self.source_counts = {
            "pg": 0,
            "hindsight_new": 0,
            "hindsight_replay": 0,
            "option_prior_bc": 0,
        }
        self.replay_episode_keys: set[tuple[Any, ...]] = set()
        self.replay_episode_keys_by_option: dict[str, set[tuple[Any, ...]]] = {}
        self.phase_scores: list[float] = []
        self.grpo_non_pg_count = 0
        self.grpo_batch_count = 0
        self._last_log_update = 0
        self.reverse_shell_validation: dict[tuple[str, int], ShellValidationResult] = {}

    def sample_instruction_type(self) -> str | None:
        if self.reverse_scheduler is not None:
            return self.sample_reset_options().get("instruction_type")
        if self._curriculum_name() != "strict_staged":
            return None
        return self.curriculum.sample_option(rng=self.rng, available_options=self.available_options)

    def sample_reset_options(self) -> dict[str, Any]:
        if self.reverse_scheduler is not None:
            sample = self.reverse_scheduler.sample(rng=self.rng)
            return {
                "instruction_type": sample.instruction_id,
                "curriculum_mode": "reverse_frontier",
                "curriculum_shell": int(sample.shell_id),
                "curriculum_sample_source": sample.source,
            }
        sampled = self.sample_instruction_type()
        return {"instruction_type": sampled} if sampled else {}

    def phase_score(self, info: Mapping[str, Any], *, fallback: float) -> float:
        if str(self.config.group_score).strip().lower() != "phase_shaped":
            return float(fallback)
        try:
            score = float(self.spec.phase_score([info]))
        except Exception:
            score = float(fallback)
        if not np.isfinite(score):
            score = float(fallback)
        self.phase_scores.append(float(score))
        return score

    def capture_candidate(
        self,
        *,
        obs: Mapping[str, Any],
        step_info: Mapping[str, Any],
        sampled_action: Any,
        group_score: float,
        update: int,
        global_step: int,
    ) -> None:
        info = dict(step_info)
        info.setdefault("action", sampled_action)
        info.setdefault("image_primary", obs.get("image_primary"))
        info.setdefault("image_wrist", obs.get("image_wrist"))
        info.setdefault("source_instruction", obs.get("instruction", ""))
        info.setdefault("lchol_group_score", float(group_score))
        info.setdefault("update", int(update))
        info.setdefault("global_step", int(global_step))

        self.source_counts["pg"] += 1
        self.curriculum.record(info)

        try:
            records = self.spec.build_hindsight_records(
                [info],
                prefix_max_steps=int(self.config.hindsight_prefix_max_steps),
            )
        except Exception:
            records = []
        for record in records:
            self.replay.add(record.option_name, record)
            episode_key = self._episode_key(info)
            self.replay_episode_keys.add(episode_key)
            self.replay_episode_keys_by_option.setdefault(str(record.option_name), set()).add(episode_key)
            self.source_counts["hindsight_new"] += 1

    def sample_bc_records(self, batch_size: int) -> list[HindsightBCRecord]:
        if self._curriculum_name() == "strict_staged":
            allowed = self.curriculum.allowed_options(self.available_options)
        else:
            allowed = self.available_options
        weights = self._option_sample_weights(allowed)
        records = self.replay.sample_balanced(
            batch_size=batch_size,
            rng=self.rng,
            allowed_options=allowed,
            option_weights=weights,
        )
        self.source_counts["hindsight_replay"] += len(records)
        return records

    def record_grpo_batch_audit(self, *, total: int, non_pg: int) -> None:
        self.grpo_batch_count += max(0, int(total))
        self.grpo_non_pg_count += max(0, int(non_pg))

    def bc_loss(
        self,
        *,
        policy: Any,
        ppo_module: Any,
        device: Any,
        args: Any,
        num_actions_chunk: int,
    ) -> Any:
        import torch

        coef = float(self.config.hindsight_bc_coef)
        if coef <= 0.0 or len(self.replay) <= 0:
            return torch.zeros((), dtype=torch.float32, device=device)

        batch_size = max(1, int(round(float(args.minibatch_size) * float(self.config.hindsight_replay_ratio))))
        records = self.sample_bc_records(batch_size)
        if not records:
            return torch.zeros((), dtype=torch.float32, device=device)

        images_primary = [record.image_primary for record in records]
        images_wrist = [record.image_wrist for record in records] if int(args.num_images_in_input) > 1 else None
        instructions = [record.instruction for record in records]
        actions_np = np.asarray([record.action for record in records], dtype=np.float32)

        mean_action, std_action, _, mean_pre_action = policy(
            images_primary=images_primary,
            images_wrist=images_wrist,
            instructions=instructions,
        )
        del mean_action
        actions = torch.tensor(actions_np, dtype=torch.float32, device=device)
        logprob = ppo_module.squashed_gaussian_log_prob(
            actions,
            mean_pre_action,
            std_action,
        ).sum(dim=-1)
        nll = -logprob.mean() / float(max(1, int(num_actions_chunk)))
        return coef * nll

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {
            f"source/{key}": float(value) for key, value in self.source_counts.items()
        }
        out["grpo/batch_count"] = float(self.grpo_batch_count)
        out["grpo/non_pg_count"] = float(self.grpo_non_pg_count)
        out["replay/total_records"] = float(len(self.replay))
        out["replay/episodes_total"] = float(len(self.replay_episode_keys))
        out.update({f"replay/{key}": float(value) for key, value in self.replay.sizes().items()})
        out.update(
            {
                f"replay/episodes/{key}": float(len(value))
                for key, value in sorted(self.replay_episode_keys_by_option.items())
            }
        )
        out.update({f"curriculum/{key}": float(value) for key, value in self.curriculum.metrics().items()})
        if self.reverse_scheduler is not None:
            out.update({f"curriculum/{key}": float(value) for key, value in self.reverse_scheduler.metrics().items()})
            for (instruction_id, shell_id), result in sorted(self.reverse_shell_validation.items()):
                instruction_tag = self._metric_token(instruction_id)
                out[
                    f"reverse_frontier/shell_success_rate/{instruction_tag}/shell_{int(shell_id):02d}"
                ] = float(result.success_rate)
        if self.phase_scores:
            out["phase_score/mean"] = float(np.mean(np.asarray(self.phase_scores, dtype=np.float32)))
            out["phase_score/std"] = float(np.std(np.asarray(self.phase_scores, dtype=np.float32)))
        else:
            out["phase_score/mean"] = 0.0
            out["phase_score/std"] = 0.0
        return out

    def log_update(self, *, update: int, global_step: int, tb_writer: Any, is_main: bool) -> None:
        if not is_main or int(update) == self._last_log_update:
            return
        self._last_log_update = int(update)
        metrics = self.metrics()
        print(
            "[lchol] "
            f"stage={self.curriculum.stage_index}:{self.curriculum.stage.name} "
            f"phase_mean={metrics.get('phase_score/mean', 0.0):.4f} "
            f"hindsight_new={int(self.source_counts['hindsight_new'])} "
            f"hindsight_replay={int(self.source_counts['hindsight_replay'])} "
            f"grpo_non_pg={int(self.grpo_non_pg_count)} "
            f"replay_total={len(self.replay)} "
            f"replay_episodes={len(self.replay_episode_keys)}",
            flush=True,
        )
        if tb_writer is not None:
            for key, value in metrics.items():
                tb_writer.add_scalar(f"lchol/{key}", float(value), int(global_step))
            tb_writer.flush()
        self.phase_scores.clear()
        self.source_counts["hindsight_replay"] = 0
        self.grpo_non_pg_count = 0
        self.grpo_batch_count = 0

    def after_rollout(self, *, update: int) -> None:
        del update
        if self.reverse_scheduler is not None:
            self.reverse_scheduler.record_train_update()

    def reverse_validation_options(self, instruction_id: str, shell_id: int) -> dict[str, Any]:
        return {
            "instruction_type": str(instruction_id),
            "curriculum_mode": "reverse_frontier",
            "curriculum_shell": int(shell_id),
        }

    def reverse_validation_plan(self) -> list[tuple[str, int]]:
        if self.reverse_scheduler is None:
            return []
        return [
            (instruction_id, int(shell_id))
            for instruction_id, shell_id in sorted(self.reverse_scheduler.active_shells.items())
        ]

    def record_reverse_validation(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        run_dir: Any | None = None,
        update: int | None = None,
    ) -> None:
        if self.reverse_scheduler is None:
            return
        coerced = [
            ShellValidationResult(
                instruction_id=str(item["instruction_id"]),
                shell_id=int(item["shell_id"]),
                success_rate=float(item["success_rate"]),
                rollouts=int(item["rollouts"]),
                action_saturation_rate=float(item.get("action_saturation_rate", 0.0)),
            )
            for item in results
        ]
        for result in coerced:
            self.reverse_shell_validation[(str(result.instruction_id), int(result.shell_id))] = result
        self.reverse_scheduler.update(coerced)
        if run_dir is not None:
            from pathlib import Path

            state_dir = Path(run_dir) / "lchol_reverse_frontier"
            state_dir.mkdir(parents=True, exist_ok=True)
            if update is not None:
                self.reverse_scheduler.save(state_dir / f"state_update_{int(update):05d}.json")
            self.reverse_scheduler.save(state_dir / "state_latest.json")

    def _option_sample_weights(self, options: Sequence[str]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for option in options:
            rate = self.curriculum.success_rate(option)
            weights[str(option)] = 1.0 + float(self.config.weakest_mode_oversample_strength) * (1.0 - rate)
        return weights

    def _curriculum_name(self) -> str:
        return str(self.config.curriculum).strip().lower()

    def _build_reverse_shell_specs(self):
        try:
            from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import get_cdpr_reverse_shell_specs
        except ModuleNotFoundError:
            from cdpr_dataset.cdpr_reverse_shells import get_cdpr_reverse_shell_specs

        specs = get_cdpr_reverse_shell_specs(self.available_options)
        if not specs:
            raise ValueError("No CDPR reverse shell specs match the available LC-HOL options.")
        return specs

    @staticmethod
    def _metric_token(value: Any) -> str:
        token = str(value).strip().lower().replace(" ", "_")
        token = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in token)
        return token.strip("_") or "unknown"

    @staticmethod
    def _episode_key(info: Mapping[str, Any]) -> tuple[Any, ...]:
        explicit = info.get("source_rollout_id") or info.get("rollout_id")
        if explicit:
            return ("rollout", str(explicit))
        return (
            info.get("env_instance_id", ""),
            info.get("episode_index", ""),
            info.get("instruction_type", ""),
            info.get("target_object_catalog", info.get("target_object_name", "")),
            info.get("reference_object_catalog", ""),
            info.get("second_reference_object_catalog", ""),
            info.get("scene", ""),
        )
