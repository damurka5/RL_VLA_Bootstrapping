from __future__ import annotations

import json
import os
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
        self.base_task_metadata = _load_task_metadata_from_env()
        self.dense_instruction_types = _metadata_name_tuple(
            self.base_task_metadata,
            "dense_stage_instruction_types",
            "dense_warmup_instruction_types",
        )
        self.sparse_instruction_types = (
            _metadata_name_tuple(
                self.base_task_metadata,
                "sparse_stage_instruction_types",
                "lchol_sparse_instruction_types",
            )
            or self.available_options
        )
        self.dense_success_threshold = float(
            self.base_task_metadata.get(
                "dense_to_sparse_success_threshold",
                self.base_task_metadata.get("dense_stage_success_threshold", 0.0),
            )
            or 0.0
        )
        self.dense_min_success_samples = max(
            1,
            int(
                self.base_task_metadata.get(
                    "dense_to_sparse_min_success_samples",
                    self.base_task_metadata.get("dense_stage_min_success_samples", 1),
                )
                or 1
            ),
        )
        self.dense_validation_episodes = max(
            0,
            int(
                self.base_task_metadata.get(
                    "dense_stage_validation_episodes",
                    self.base_task_metadata.get("dense_validation_episodes", 0),
                )
                or 0
            ),
        )
        self.dense_stage_max_updates = max(
            0,
            int(
                self.base_task_metadata.get(
                    "dense_stage_max_updates",
                    self.base_task_metadata.get(
                        "stage1_total_updates",
                        self.base_task_metadata.get("stage1_updates", 0),
                    ),
                )
                or 0
            ),
        )
        self.sparse_stage_max_updates = max(
            0,
            int(
                self.base_task_metadata.get(
                    "sparse_stage_max_updates",
                    self.base_task_metadata.get(
                        "stage2_total_updates",
                        self.base_task_metadata.get("stage2_updates", 0),
                    ),
                )
                or 0
            ),
        )
        self.dense_stage_open_on_max_updates = _metadata_bool(
            self.base_task_metadata,
            "dense_stage_open_on_max_updates",
            True,
        )
        self.dense_gate_armed = not (
            self.dense_instruction_types
            and np.isfinite(self.dense_success_threshold)
            and self.dense_success_threshold > 0.0
        )
        self.dense_gate_success: dict[str, float] = {}
        self.dense_gate_rollouts: dict[str, int] = {}
        self.dense_gate_rewards: dict[str, float] = {}
        self.dense_updates_completed = 0
        self.sparse_updates_completed = 0
        self._stage_at_update_start = "sparse" if self.dense_gate_armed else "dense"
        self._dense_gate_open_reason = "initially_armed" if self.dense_gate_armed else ""
        self._grpo_stats_reset_pending = False
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

    def dense_gate_active(self) -> bool:
        return bool(not self.dense_gate_armed)

    def dense_validation_plan(self) -> tuple[str, ...]:
        return tuple(self.dense_instruction_types) if self.dense_gate_active() else ()

    def dense_validation_episodes_per_instruction(self, fallback: int) -> int:
        configured = int(self.dense_validation_episodes)
        return max(1, configured if configured > 0 else int(fallback))

    def current_task_metadata(self) -> dict[str, Any]:
        metadata = dict(self.base_task_metadata)
        if self.dense_gate_active():
            metadata.update(dict(metadata.get("dense_stage_metadata") or {}))
            metadata.setdefault("reward_mode", "dense")
            return metadata
        metadata.update(dict(metadata.get("sparse_stage_metadata") or {}))
        return metadata

    def configure_env_for_current_stage(self, env_wrapper: Any) -> None:
        rl_env = getattr(env_wrapper, "env", env_wrapper)
        if rl_env is None:
            return
        if not hasattr(rl_env, "_rlvla_lchol_base_instruction_types"):
            try:
                setattr(
                    rl_env,
                    "_rlvla_lchol_base_instruction_types",
                    tuple(getattr(rl_env, "instruction_types", ()) or ()),
                )
            except Exception:
                pass
        try:
            rl_env._task_metadata = self.current_task_metadata()
        except Exception:
            pass

        try:
            if self.dense_gate_active() and self.dense_instruction_types:
                rl_env.instruction_types = tuple(self.dense_instruction_types)
            elif self.sparse_instruction_types:
                rl_env.instruction_types = tuple(self.sparse_instruction_types)
        except Exception:
            pass

    def record_dense_validation(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        run_dir: Any | None = None,
        update: int | None = None,
    ) -> None:
        for item in results:
            instruction = str(item.get("instruction_id") or item.get("instruction_type") or "").strip()
            if not instruction:
                continue
            self.dense_gate_success[instruction] = float(np.clip(float(item.get("success_rate", 0.0)), 0.0, 1.0))
            self.dense_gate_rollouts[instruction] = int(item.get("rollouts", item.get("episodes", 0)) or 0)
            reward_value = item.get("mean_reward", item.get("mean_env_return"))
            if reward_value is not None:
                try:
                    self.dense_gate_rewards[instruction] = float(reward_value)
                except (TypeError, ValueError):
                    pass

        if self.dense_gate_active() and self._dense_gate_passed():
            self._open_dense_gate(reason="success_threshold")
            print(
                "[lchol] dense-to-sparse gate opened "
                f"mean_success={self.dense_gate_mean_success():.4f} "
                f"threshold={self.dense_success_threshold:.4f}",
                flush=True,
            )

        if run_dir is not None:
            self._save_dense_gate_state(run_dir=run_dir, update=update)

    def dense_gate_mean_success(self) -> float:
        if not self.dense_instruction_types:
            return 0.0
        values = [float(self.dense_gate_success.get(instruction, 0.0)) for instruction in self.dense_instruction_types]
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0

    def dense_gate_mean_reward(self) -> float:
        if not self.dense_instruction_types:
            return 0.0
        values = [
            float(self.dense_gate_rewards[instruction])
            for instruction in self.dense_instruction_types
            if instruction in self.dense_gate_rewards
        ]
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0

    def before_update(self, *, update: int) -> None:
        del update
        if (
            self.dense_gate_active()
            and self.dense_stage_open_on_max_updates
            and self.dense_stage_max_updates > 0
            and self.dense_updates_completed >= self.dense_stage_max_updates
        ):
            self._open_dense_gate(reason="dense_stage_max_updates")
            print(
                "[lchol] dense-to-sparse gate opened by dense update limit "
                f"dense_updates_completed={int(self.dense_updates_completed)} "
                f"limit={int(self.dense_stage_max_updates)}",
                flush=True,
            )
        self._stage_at_update_start = "dense" if self.dense_gate_active() else "sparse"

    def after_update(
        self,
        *,
        update: int,
        global_step: int | None = None,
        run_dir: Any | None = None,
    ) -> None:
        del global_step
        stage = str(self._stage_at_update_start or ("dense" if self.dense_gate_active() else "sparse"))
        if stage == "dense":
            self.dense_updates_completed += 1
        else:
            self.sparse_updates_completed += 1

        if (
            self.dense_gate_active()
            and self.dense_stage_open_on_max_updates
            and self.dense_stage_max_updates > 0
            and self.dense_updates_completed >= self.dense_stage_max_updates
        ):
            self._open_dense_gate(reason="dense_stage_max_updates")
            print(
                "[lchol] dense-to-sparse gate opened by dense update limit "
                f"dense_updates_completed={int(self.dense_updates_completed)} "
                f"limit={int(self.dense_stage_max_updates)}",
                flush=True,
            )

        if run_dir is not None:
            self._save_dense_gate_state(run_dir=run_dir, update=update)

    def should_stop_training(self) -> bool:
        return bool(
            not self.dense_gate_active()
            and self.sparse_stage_max_updates > 0
            and self.sparse_updates_completed >= self.sparse_stage_max_updates
        )

    def sample_instruction_type(self) -> str | None:
        if self.dense_gate_active():
            if not self.dense_instruction_types:
                return None
            return str(self.dense_instruction_types[int(self.rng.integers(0, len(self.dense_instruction_types)))])
        if self.reverse_scheduler is not None:
            return self.sample_reset_options().get("instruction_type")
        if self._curriculum_name() != "strict_staged":
            return None
        return self.curriculum.sample_option(rng=self.rng, available_options=self.sparse_instruction_types)

    def sample_reset_options(self) -> dict[str, Any]:
        if self.dense_gate_active():
            sampled = self.sample_instruction_type()
            return {"instruction_type": sampled, "lchol_dense_stage": True} if sampled else {}
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
        if self.dense_gate_active():
            return float(fallback)
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
        if self.dense_gate_active():
            self.source_counts["pg"] += 1
            return
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
        if self.dense_gate_active():
            return []
        if self._curriculum_name() == "strict_staged":
            allowed = self.curriculum.allowed_options(self.sparse_instruction_types)
        else:
            allowed = self.sparse_instruction_types
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

        if self.dense_gate_active():
            return torch.zeros((), dtype=torch.float32, device=device)

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
        out["dense_gate/active"] = float(self.dense_gate_active())
        out["dense_gate/armed"] = float(self.dense_gate_armed)
        out["dense_gate/threshold"] = float(self.dense_success_threshold)
        out["dense_gate/mean_success"] = float(self.dense_gate_mean_success())
        out["dense_gate/mean_reward"] = float(self.dense_gate_mean_reward())
        out["dense_stage/active"] = float(self.dense_gate_active())
        out["dense_stage/complete"] = float(self.dense_gate_armed)
        out["dense_stage/threshold"] = float(self.dense_success_threshold)
        out["dense_stage/mean_success"] = float(self.dense_gate_mean_success())
        out["dense_stage/mean_reward"] = float(self.dense_gate_mean_reward())
        out["dense_stage/updates_completed"] = float(self.dense_updates_completed)
        out["dense_stage/max_updates"] = float(self.dense_stage_max_updates)
        out["sparse_stage/updates_completed"] = float(self.sparse_updates_completed)
        out["sparse_stage/max_updates"] = float(self.sparse_stage_max_updates)
        for instruction in self.dense_instruction_types:
            out[f"dense_gate/success_rate/{instruction}"] = float(self.dense_gate_success.get(instruction, 0.0))
            out[f"dense_gate/rollouts/{instruction}"] = float(self.dense_gate_rollouts.get(instruction, 0))
            out[f"dense_gate/reward/{instruction}"] = float(self.dense_gate_rewards.get(instruction, 0.0))
            out[f"dense_stage/success_rate/{instruction}"] = float(self.dense_gate_success.get(instruction, 0.0))
            out[f"dense_stage/rollouts/{instruction}"] = float(self.dense_gate_rollouts.get(instruction, 0))
            out[f"dense_stage/reward/{instruction}"] = float(self.dense_gate_rewards.get(instruction, 0.0))
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
            f"dense_gate={'active' if self.dense_gate_active() else 'armed'} "
            f"dense_mean={metrics.get('dense_gate/mean_success', 0.0):.4f} "
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
                if key.startswith("dense_stage/"):
                    tb_writer.add_scalar(f"stage/dense/{key[len('dense_stage/'):]}", float(value), int(global_step))
                elif key.startswith("sparse_stage/"):
                    tb_writer.add_scalar(f"stage/sparse/{key[len('sparse_stage/'):]}", float(value), int(global_step))
                else:
                    tb_writer.add_scalar(f"lchol/{key}", float(value), int(global_step))
            tb_writer.flush()
        self.phase_scores.clear()
        self.source_counts["hindsight_replay"] = 0
        self.grpo_non_pg_count = 0
        self.grpo_batch_count = 0

    def after_rollout(self, *, update: int) -> None:
        del update
        if self.dense_gate_active():
            return
        if self.reverse_scheduler is not None:
            self.reverse_scheduler.record_train_update()

    def sync_dense_gate_state(self, *, run_dir: Any | None) -> None:
        if run_dir is None:
            return
        from pathlib import Path

        state_path = Path(run_dir) / "lchol_dense_gate" / "state_latest.json"
        if not state_path.is_file():
            return
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, Mapping):
            return

        success = payload.get("success")
        if isinstance(success, Mapping):
            self.dense_gate_success = {
                str(key): float(np.clip(float(value), 0.0, 1.0))
                for key, value in success.items()
            }
        rollouts = payload.get("rollouts")
        if isinstance(rollouts, Mapping):
            self.dense_gate_rollouts = {
                str(key): int(value)
                for key, value in rollouts.items()
            }
        rewards = payload.get("rewards")
        if isinstance(rewards, Mapping):
            self.dense_gate_rewards = {
                str(key): float(value)
                for key, value in rewards.items()
            }
        try:
            self.dense_updates_completed = max(
                0,
                int(payload.get("dense_updates_completed", self.dense_updates_completed)),
            )
        except (TypeError, ValueError):
            pass
        try:
            self.sparse_updates_completed = max(
                0,
                int(payload.get("sparse_updates_completed", self.sparse_updates_completed)),
            )
        except (TypeError, ValueError):
            pass
        reason = payload.get("open_reason")
        if reason:
            self._dense_gate_open_reason = str(reason)

        remote_armed = bool(payload.get("armed", not bool(payload.get("active", True))))
        if self.dense_gate_active() and remote_armed:
            self._open_dense_gate(reason=str(reason or "synced_state"))

    def consume_grpo_stats_reset_request(self) -> bool:
        pending = bool(self._grpo_stats_reset_pending)
        self._grpo_stats_reset_pending = False
        return pending

    def reverse_validation_options(self, instruction_id: str, shell_id: int) -> dict[str, Any]:
        return {
            "instruction_type": str(instruction_id),
            "curriculum_mode": "reverse_frontier",
            "curriculum_shell": int(shell_id),
        }

    def reverse_validation_plan(self) -> list[tuple[str, int]]:
        if self.dense_gate_active():
            return []
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

        specs = get_cdpr_reverse_shell_specs(self.sparse_instruction_types)
        if not specs:
            raise ValueError("No CDPR reverse shell specs match the available LC-HOL options.")
        return specs

    def _open_dense_gate(self, *, reason: str = "") -> None:
        if self.dense_gate_armed:
            return
        self.dense_gate_armed = True
        self._dense_gate_open_reason = str(reason or "unspecified")
        self._grpo_stats_reset_pending = True

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

    def _dense_gate_passed(self) -> bool:
        if not self.dense_instruction_types:
            return True
        for instruction in self.dense_instruction_types:
            if int(self.dense_gate_rollouts.get(instruction, 0)) < int(self.dense_min_success_samples):
                return False
        return bool(self.dense_gate_mean_success() > float(self.dense_success_threshold) + 1e-12)

    def _save_dense_gate_state(self, *, run_dir: Any, update: int | None = None) -> None:
        from pathlib import Path

        state_dir = Path(run_dir) / "lchol_dense_gate"
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "armed": bool(self.dense_gate_armed),
            "active": bool(self.dense_gate_active()),
            "open_reason": str(self._dense_gate_open_reason),
            "threshold": float(self.dense_success_threshold),
            "min_success_samples": int(self.dense_min_success_samples),
            "mean_success": float(self.dense_gate_mean_success()),
            "mean_reward": float(self.dense_gate_mean_reward()),
            "dense_stage_max_updates": int(self.dense_stage_max_updates),
            "sparse_stage_max_updates": int(self.sparse_stage_max_updates),
            "dense_updates_completed": int(self.dense_updates_completed),
            "sparse_updates_completed": int(self.sparse_updates_completed),
            "instruction_types": list(self.dense_instruction_types),
            "sparse_instruction_types": list(self.sparse_instruction_types),
            "grpo_stats_reset_pending": bool(self._grpo_stats_reset_pending),
            "success": dict(sorted(self.dense_gate_success.items())),
            "rollouts": {key: int(value) for key, value in sorted(self.dense_gate_rollouts.items())},
            "rewards": dict(sorted(self.dense_gate_rewards.items())),
        }
        if update is not None:
            (state_dir / f"state_update_{int(update):05d}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        (state_dir / "state_latest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _load_task_metadata_from_env() -> dict[str, Any]:
    raw = os.environ.get("RLVLA_TASK_METADATA_JSON")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _metadata_name_tuple(metadata: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    for key in keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, Sequence):
            continue
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        if out:
            return tuple(out)
    return ()


def _metadata_bool(metadata: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = metadata.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    return bool(default)
