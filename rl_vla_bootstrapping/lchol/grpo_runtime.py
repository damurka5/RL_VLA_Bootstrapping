from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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


_SPARSE_EPISODE_OUTCOME_FIELDS = (
    "timestamp_utc",
    "rank",
    "update",
    "global_step",
    "episode_key",
    "env_instance_id",
    "episode_index",
    "instruction_type",
    "curriculum_shell",
    "scene",
    "target_object_catalog",
    "binary_reward",
    "success",
    "failure",
    "terminal_reason",
    "observed_selected_steps",
    "episode_return_env",
    "episode_return_shaped",
    "terminal_env_reward",
    "terminal_shaped_reward",
    "replay_records_added",
    "replay_options",
    "stored_in_replay",
)

_SPARSE_EPISODE_SUMMARY_FIELDS = (
    "generated_at_utc",
    "scope",
    "instruction_type",
    "curriculum_shell",
    "episodes",
    "reward_1_count",
    "reward_0_count",
    "reward_1_ratio",
    "reward_0_ratio",
    "episodes_with_replay_records",
    "episodes_with_replay_ratio",
    "replay_records_added",
    "last_update",
)


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
        rank: int = 0,
    ):
        self.config = config
        self.spec = spec
        self.available_options = tuple(str(option) for option in available_options if str(option))
        self.rng = np.random.default_rng(int(seed))
        self.rank = int(rank)
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
        self.dense_min_instruction_success = float(
            self.base_task_metadata.get("dense_to_sparse_min_instruction_success", 0.0)
            or 0.0
        )
        self.dense_required_consecutive_passes = max(
            1,
            int(
                self.base_task_metadata.get(
                    "dense_to_sparse_required_consecutive_passes",
                    1,
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
        requested_start_stage = str(
            self.base_task_metadata.get(
                "lchol_start_stage",
                self.base_task_metadata.get("start_stage", ""),
            )
            or ""
        ).strip().lower()
        self.start_sparse = requested_start_stage in {
            "2",
            "second",
            "second_stage",
            "sparse",
            "sparse_stage",
            "stage2",
            "stage_2",
        }
        self.dense_gate_armed = self.start_sparse or not (
            self.dense_instruction_types
            and np.isfinite(self.dense_success_threshold)
            and self.dense_success_threshold > 0.0
        )
        self.dense_gate_success: dict[str, float] = {}
        self.dense_gate_rollouts: dict[str, int] = {}
        self.dense_gate_rewards: dict[str, float] = {}
        self.dense_gate_consecutive_passes = 0
        self.dense_updates_completed = 0
        self.sparse_updates_completed = 0
        self._stage_at_update_start = "sparse" if self.dense_gate_armed else "dense"
        if self.start_sparse:
            self._dense_gate_open_reason = "configured_sparse_start"
        else:
            self._dense_gate_open_reason = "initially_armed" if self.dense_gate_armed else ""
        # A sparse-only continuation still needs a stage-specific exploration
        # distribution instead of silently keeping dense-stage actor statistics.
        self._grpo_stats_reset_pending = bool(self.start_sparse)
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
        self._replay_episode_telemetry: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._selected_episode_telemetry: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._completed_sparse_episode_keys: set[tuple[Any, ...]] = set()
        self._pending_sparse_episode_outcomes: list[dict[str, Any]] = []
        self._sparse_outcome_counts = self._empty_outcome_counts()
        self._sparse_outcome_update_counts = self._empty_outcome_counts()
        self._sparse_outcome_by_instruction: dict[str, dict[str, int]] = {}
        self._sparse_outcome_by_instruction_shell: dict[tuple[str, str], dict[str, int]] = {}
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

        if self.dense_gate_active():
            if self._dense_gate_snapshot_passed():
                self.dense_gate_consecutive_passes += 1
            else:
                self.dense_gate_consecutive_passes = 0
            if self._dense_gate_passed():
                self._open_dense_gate(reason="success_threshold")
                print(
                    "[lchol] dense-to-sparse gate opened "
                    f"mean_success={self.dense_gate_mean_success():.4f} "
                    f"threshold={self.dense_success_threshold:.4f} "
                    f"min_instruction_success={self.dense_min_instruction_success:.4f} "
                    f"consecutive_passes={self.dense_gate_consecutive_passes}",
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
        is_main: bool = True,
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
            self._flush_sparse_episode_outcomes(run_dir=run_dir)
        if run_dir is not None and bool(is_main):
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
        allowed_options = {
            str(option)
            for option in (self.sparse_instruction_types or self.available_options)
        }
        for record in records:
            if allowed_options and str(record.option_name) not in allowed_options:
                continue
            self.replay.add(record.option_name, record)
            episode_key = self._episode_key(info)
            self.replay_episode_keys.add(episode_key)
            self.replay_episode_keys_by_option.setdefault(str(record.option_name), set()).add(episode_key)
            telemetry = self._replay_episode_telemetry.setdefault(
                episode_key,
                {"records_added": 0, "options": set()},
            )
            telemetry["records_added"] = int(telemetry["records_added"]) + 1
            telemetry["options"].add(str(record.option_name))
            self.source_counts["hindsight_new"] += 1

    def record_selected_step(
        self,
        *,
        step_info: Mapping[str, Any],
        env_reward: float,
        shaped_reward: float,
        done: bool,
        env_done: bool,
        forced_scene_refresh: bool,
        forced_unstable_reset: bool,
        update: int,
        global_step: int,
    ) -> None:
        """Record the actual selected rollout branch, not every GRPO candidate."""
        if self.dense_gate_active():
            return
        info = dict(step_info)
        episode_key = self._episode_key(info)
        if episode_key in self._completed_sparse_episode_keys:
            return

        selected = self._selected_episode_telemetry.setdefault(
            episode_key,
            {
                "observed_selected_steps": 0,
                "episode_return_env": 0.0,
                "episode_return_shaped": 0.0,
            },
        )
        selected["observed_selected_steps"] = int(selected["observed_selected_steps"]) + 1
        selected["episode_return_env"] = float(selected["episode_return_env"]) + self._finite_float(env_reward)
        selected["episode_return_shaped"] = float(selected["episode_return_shaped"]) + self._finite_float(
            shaped_reward
        )
        if not bool(done):
            return

        binary_reward = self._binary_success(info)
        replay = self._replay_episode_telemetry.pop(
            episode_key,
            {"records_added": 0, "options": set()},
        )
        instruction = str(
            info.get("curriculum_instruction_id")
            or info.get("instruction_type")
            or "unknown"
        )
        shell = self._shell_token(info.get("curriculum_shell"))
        records_added = max(0, int(replay.get("records_added", 0)))
        options = sorted(str(option) for option in replay.get("options", set()) if str(option))
        row = {
            "timestamp_utc": self._utc_now(),
            "rank": int(self.rank),
            "update": int(update),
            "global_step": int(global_step),
            "episode_key": json.dumps(list(episode_key), ensure_ascii=False, default=str),
            "env_instance_id": info.get("env_instance_id", ""),
            "episode_index": info.get("episode_index", ""),
            "instruction_type": instruction,
            "curriculum_shell": shell,
            "scene": info.get("scene", ""),
            "target_object_catalog": info.get(
                "target_object_catalog",
                info.get("target_object_name", ""),
            ),
            "binary_reward": int(binary_reward),
            "success": int(binary_reward),
            "failure": int(1 - binary_reward),
            "terminal_reason": self._terminal_reason(
                info,
                binary_reward=binary_reward,
                env_done=bool(env_done),
                forced_scene_refresh=bool(forced_scene_refresh),
                forced_unstable_reset=bool(forced_unstable_reset),
            ),
            "observed_selected_steps": int(selected["observed_selected_steps"]),
            "episode_return_env": float(selected["episode_return_env"]),
            "episode_return_shaped": float(selected["episode_return_shaped"]),
            "terminal_env_reward": self._finite_float(env_reward),
            "terminal_shaped_reward": self._finite_float(shaped_reward),
            "replay_records_added": records_added,
            "replay_options": "|".join(options),
            "stored_in_replay": int(records_added > 0),
        }
        self._pending_sparse_episode_outcomes.append(row)
        self._record_sparse_outcome_counts(
            binary_reward=binary_reward,
            instruction=instruction,
            shell=shell,
            records_added=records_added,
        )
        self._selected_episode_telemetry.pop(episode_key, None)
        self._completed_sparse_episode_keys.add(episode_key)

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
        out["dense_gate/min_instruction_success"] = float(self.dense_min_instruction_success)
        out["dense_gate/consecutive_passes"] = float(self.dense_gate_consecutive_passes)
        out["dense_gate/required_consecutive_passes"] = float(self.dense_required_consecutive_passes)
        out["dense_stage/active"] = float(self.dense_gate_active())
        out["dense_stage/complete"] = float(self.dense_gate_armed)
        out["dense_stage/threshold"] = float(self.dense_success_threshold)
        out["dense_stage/mean_success"] = float(self.dense_gate_mean_success())
        out["dense_stage/mean_reward"] = float(self.dense_gate_mean_reward())
        out["dense_stage/min_instruction_success"] = float(self.dense_min_instruction_success)
        out["dense_stage/consecutive_passes"] = float(self.dense_gate_consecutive_passes)
        out["dense_stage/required_consecutive_passes"] = float(self.dense_required_consecutive_passes)
        out["dense_stage/updates_completed"] = float(self.dense_updates_completed)
        out["dense_stage/max_updates"] = float(self.dense_stage_max_updates)
        out["sparse_stage/updates_completed"] = float(self.sparse_updates_completed)
        out["sparse_stage/max_updates"] = float(self.sparse_stage_max_updates)
        out["sparse_stage/configured_start"] = float(self.start_sparse)
        self._append_sparse_outcome_metrics(out)
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
            f"replay_episodes={len(self.replay_episode_keys)} "
            f"sparse_reward_1_ratio="
            f"{metrics.get('sparse_stage/buffer_episode_outcomes/rank_local/cumulative/reward_1_ratio', 0.0):.4f} "
            f"sparse_episodes="
            f"{int(metrics.get('sparse_stage/buffer_episode_outcomes/rank_local/cumulative/episodes_total', 0.0))}",
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
        self._sparse_outcome_update_counts = self._empty_outcome_counts()

    def after_rollout(self, *, update: int) -> None:
        del update
        if self.dense_gate_active():
            return
        if self.reverse_scheduler is not None:
            self.reverse_scheduler.record_train_update()

    def log_persisted_sparse_outcomes(
        self,
        *,
        run_dir: Any,
        tb_writer: Any,
        global_step: int,
    ) -> None:
        if tb_writer is None:
            return
        summary_path = Path(run_dir) / "lchol_episode_stats" / "sparse_episode_outcome_summary.csv"
        if not summary_path.is_file():
            return
        with summary_path.open(newline="", encoding="utf-8") as summary_fp:
            rows = list(csv.DictReader(summary_fp))
        for row in rows:
            scope = str(row.get("scope") or "")
            instruction = self._metric_token(row.get("instruction_type") or "unknown")
            shell = self._metric_token(row.get("curriculum_shell") or "")
            if scope == "all":
                prefix = "stage/sparse/buffer_episode_outcomes/global/cumulative"
            elif scope == "instruction":
                prefix = f"stage/sparse/buffer_episode_outcomes/global/instruction/{instruction}"
            elif scope == "instruction_shell" and shell:
                prefix = (
                    "stage/sparse/buffer_episode_outcomes/global/instruction_shell/"
                    f"{instruction}/shell_{shell}"
                )
            else:
                continue
            for csv_key, metric_name in (
                ("episodes", "episodes_total"),
                ("reward_1_count", "reward_1_count"),
                ("reward_0_count", "reward_0_count"),
                ("reward_1_ratio", "reward_1_ratio"),
                ("reward_0_ratio", "reward_0_ratio"),
                ("episodes_with_replay_ratio", "episodes_with_replay_ratio"),
                ("replay_records_added", "replay_records_added"),
            ):
                tb_writer.add_scalar(
                    f"{prefix}/{metric_name}",
                    self._finite_float(row.get(csv_key), 0.0),
                    int(global_step),
                )
        tb_writer.flush()

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
        try:
            self.dense_gate_consecutive_passes = max(
                0,
                int(payload.get("consecutive_passes", self.dense_gate_consecutive_passes)),
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
    def _empty_outcome_counts() -> dict[str, int]:
        return {
            "episodes": 0,
            "reward_1": 0,
            "reward_0": 0,
            "with_replay": 0,
            "replay_records": 0,
        }

    @staticmethod
    def _finite_float(value: Any, fallback: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return float(numeric) if np.isfinite(numeric) else float(fallback)

    @staticmethod
    def _binary_success(info: Mapping[str, Any]) -> int:
        for key in ("success", "sparse_success"):
            if key not in info:
                continue
            raw = info.get(key)
            try:
                return int(float(raw) >= 0.5)
            except (TypeError, ValueError):
                return int(bool(raw))
        return 0

    @staticmethod
    def _shell_token(value: Any) -> str:
        if value is None or value == "":
            return ""
        try:
            return str(max(0, int(value)))
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _terminal_reason(
        info: Mapping[str, Any],
        *,
        binary_reward: int,
        env_done: bool,
        forced_scene_refresh: bool,
        forced_unstable_reset: bool,
    ) -> str:
        if int(binary_reward) == 1:
            return "success"
        if forced_unstable_reset or bool(info.get("forced_unstable_reset")):
            return "unstable_reset"
        if forced_scene_refresh:
            return "scene_refresh"
        if bool(info.get("episode_timeout")) or bool(info.get("truncated")):
            return "timeout"
        if bool(info.get("terminated")):
            return "terminated_failure"
        if env_done or bool(info.get("env_done")):
            return "env_done_failure"
        return "rollout_reset"

    def _record_sparse_outcome_counts(
        self,
        *,
        binary_reward: int,
        instruction: str,
        shell: str,
        records_added: int,
    ) -> None:
        for counts in (self._sparse_outcome_counts, self._sparse_outcome_update_counts):
            self._increment_outcome_counts(
                counts,
                binary_reward=binary_reward,
                records_added=records_added,
            )
        instruction_counts = self._sparse_outcome_by_instruction.setdefault(
            str(instruction),
            self._empty_outcome_counts(),
        )
        self._increment_outcome_counts(
            instruction_counts,
            binary_reward=binary_reward,
            records_added=records_added,
        )
        if shell:
            shell_counts = self._sparse_outcome_by_instruction_shell.setdefault(
                (str(instruction), str(shell)),
                self._empty_outcome_counts(),
            )
            self._increment_outcome_counts(
                shell_counts,
                binary_reward=binary_reward,
                records_added=records_added,
            )

    @staticmethod
    def _increment_outcome_counts(
        counts: dict[str, int],
        *,
        binary_reward: int,
        records_added: int,
    ) -> None:
        counts["episodes"] += 1
        counts["reward_1" if int(binary_reward) == 1 else "reward_0"] += 1
        counts["with_replay"] += int(int(records_added) > 0)
        counts["replay_records"] += max(0, int(records_added))

    @staticmethod
    def _count_ratio(counts: Mapping[str, int], key: str) -> float:
        episodes = max(0, int(counts.get("episodes", 0)))
        return float(counts.get(key, 0)) / float(episodes) if episodes else 0.0

    def _append_sparse_outcome_metrics(self, out: dict[str, float]) -> None:
        for window, counts in (
            ("cumulative", self._sparse_outcome_counts),
            ("update", self._sparse_outcome_update_counts),
        ):
            prefix = f"sparse_stage/buffer_episode_outcomes/rank_local/{window}"
            out[f"{prefix}/episodes_total"] = float(counts["episodes"])
            out[f"{prefix}/reward_1_count"] = float(counts["reward_1"])
            out[f"{prefix}/reward_0_count"] = float(counts["reward_0"])
            out[f"{prefix}/reward_1_ratio"] = self._count_ratio(counts, "reward_1")
            out[f"{prefix}/reward_0_ratio"] = self._count_ratio(counts, "reward_0")
            out[f"{prefix}/episodes_with_replay_ratio"] = self._count_ratio(counts, "with_replay")
            out[f"{prefix}/replay_records_added"] = float(counts["replay_records"])

        for instruction, counts in sorted(self._sparse_outcome_by_instruction.items()):
            instruction_tag = self._metric_token(instruction)
            prefix = (
                "sparse_stage/buffer_episode_outcomes/rank_local/instruction/"
                f"{instruction_tag}"
            )
            out[f"{prefix}/episodes_total"] = float(counts["episodes"])
            out[f"{prefix}/reward_1_ratio"] = self._count_ratio(counts, "reward_1")
            out[f"{prefix}/reward_0_ratio"] = self._count_ratio(counts, "reward_0")
            out[f"{prefix}/episodes_with_replay_ratio"] = self._count_ratio(counts, "with_replay")

        for (instruction, shell), counts in sorted(self._sparse_outcome_by_instruction_shell.items()):
            instruction_tag = self._metric_token(instruction)
            shell_tag = self._metric_token(shell)
            prefix = (
                "sparse_stage/buffer_episode_outcomes/rank_local/instruction_shell/"
                f"{instruction_tag}/shell_{shell_tag}"
            )
            out[f"{prefix}/episodes_total"] = float(counts["episodes"])
            out[f"{prefix}/reward_1_ratio"] = self._count_ratio(counts, "reward_1")
            out[f"{prefix}/reward_0_ratio"] = self._count_ratio(counts, "reward_0")

    def _flush_sparse_episode_outcomes(self, *, run_dir: Any) -> None:
        if not self._pending_sparse_episode_outcomes:
            return
        rows = list(self._pending_sparse_episode_outcomes)
        stats_dir = Path(run_dir) / "lchol_episode_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        outcomes_path = stats_dir / "sparse_episode_outcomes.csv"
        summary_path = stats_dir / "sparse_episode_outcome_summary.csv"
        lock_path = stats_dir / ".sparse_episode_outcomes.lock"

        lock_fp = lock_path.open("a+", encoding="utf-8")
        try:
            self._lock_file(lock_fp)
            write_header = not outcomes_path.is_file() or outcomes_path.stat().st_size == 0
            with outcomes_path.open("a", newline="", encoding="utf-8") as outcomes_fp:
                writer = csv.DictWriter(outcomes_fp, fieldnames=_SPARSE_EPISODE_OUTCOME_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
                outcomes_fp.flush()
                os.fsync(outcomes_fp.fileno())
            summary_rows = self._summarize_sparse_episode_outcomes(outcomes_path)
            tmp_path = summary_path.with_name(
                f".{summary_path.name}.rank{int(self.rank):02d}.{os.getpid()}.tmp"
            )
            with tmp_path.open("w", newline="", encoding="utf-8") as summary_fp:
                writer = csv.DictWriter(summary_fp, fieldnames=_SPARSE_EPISODE_SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerows(summary_rows)
                summary_fp.flush()
                os.fsync(summary_fp.fileno())
            os.replace(tmp_path, summary_path)
        finally:
            self._unlock_file(lock_fp)
            lock_fp.close()
        del self._pending_sparse_episode_outcomes[: len(rows)]

    @classmethod
    def _summarize_sparse_episode_outcomes(cls, outcomes_path: Path) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], dict[str, int]] = {}
        with outcomes_path.open(newline="", encoding="utf-8") as outcomes_fp:
            for row in csv.DictReader(outcomes_fp):
                instruction = str(row.get("instruction_type") or "unknown")
                shell = str(row.get("curriculum_shell") or "")
                binary_reward = int(cls._finite_float(row.get("binary_reward"), 0.0) >= 0.5)
                records_added = max(0, int(cls._finite_float(row.get("replay_records_added"), 0.0)))
                update = max(0, int(cls._finite_float(row.get("update"), 0.0)))
                keys = [("all", "", ""), ("instruction", instruction, "")]
                if shell:
                    keys.append(("instruction_shell", instruction, shell))
                for key in keys:
                    counts = grouped.setdefault(
                        key,
                        {
                            **cls._empty_outcome_counts(),
                            "last_update": 0,
                        },
                    )
                    cls._increment_outcome_counts(
                        counts,
                        binary_reward=binary_reward,
                        records_added=records_added,
                    )
                    counts["last_update"] = max(int(counts["last_update"]), update)

        generated_at = cls._utc_now()
        summary_rows: list[dict[str, Any]] = []
        for (scope, instruction, shell), counts in sorted(grouped.items()):
            summary_rows.append(
                {
                    "generated_at_utc": generated_at,
                    "scope": scope,
                    "instruction_type": instruction,
                    "curriculum_shell": shell,
                    "episodes": int(counts["episodes"]),
                    "reward_1_count": int(counts["reward_1"]),
                    "reward_0_count": int(counts["reward_0"]),
                    "reward_1_ratio": cls._count_ratio(counts, "reward_1"),
                    "reward_0_ratio": cls._count_ratio(counts, "reward_0"),
                    "episodes_with_replay_records": int(counts["with_replay"]),
                    "episodes_with_replay_ratio": cls._count_ratio(counts, "with_replay"),
                    "replay_records_added": int(counts["replay_records"]),
                    "last_update": int(counts["last_update"]),
                }
            )
        return summary_rows

    @staticmethod
    def _lock_file(file_obj: Any) -> None:
        try:
            import fcntl

            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            return

    @staticmethod
    def _unlock_file(file_obj: Any) -> None:
        try:
            import fcntl

            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            return

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

    def _dense_gate_snapshot_passed(self) -> bool:
        if not self.dense_instruction_types:
            return True
        for instruction in self.dense_instruction_types:
            if int(self.dense_gate_rollouts.get(instruction, 0)) < int(self.dense_min_success_samples):
                return False
            if (
                self.dense_min_instruction_success > 0.0
                and float(self.dense_gate_success.get(instruction, 0.0))
                + 1e-12
                < float(self.dense_min_instruction_success)
            ):
                return False
        return bool(
            self.dense_gate_mean_success() + 1e-12
            >= float(self.dense_success_threshold)
        )

    def _dense_gate_passed(self) -> bool:
        return bool(
            self._dense_gate_snapshot_passed()
            and self.dense_gate_consecutive_passes >= self.dense_required_consecutive_passes
        )

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
            "min_instruction_success": float(self.dense_min_instruction_success),
            "required_consecutive_passes": int(self.dense_required_consecutive_passes),
            "consecutive_passes": int(self.dense_gate_consecutive_passes),
            "mean_success": float(self.dense_gate_mean_success()),
            "mean_reward": float(self.dense_gate_mean_reward()),
            "dense_stage_max_updates": int(self.dense_stage_max_updates),
            "sparse_stage_max_updates": int(self.sparse_stage_max_updates),
            "configured_sparse_start": bool(self.start_sparse),
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
