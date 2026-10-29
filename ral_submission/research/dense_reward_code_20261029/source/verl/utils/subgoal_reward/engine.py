from __future__ import annotations

from typing import Any, Mapping

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from .dense_reward import DenseRewardManager, RewardWeights
from .libero_state import LiberoStateExtractor
from .phases import Thresholds
from .task_specs import infer_task_spec
from .tracker import OnlineSubgoalTracker


def _to_container(config: Any) -> dict:
    if config is None:
        return {}
    if OmegaConf is not None and OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def _get(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    try:
        return {str(item).strip() for item in value if str(item).strip()}
    except TypeError:
        return {str(value).strip()}


class LiberoSubgoalRewardEngine:
    def __init__(self, config: Any = None):
        cfg = _to_container(config)
        self.enabled = bool(_get(cfg, "enabled", False))
        self.mode = str(_get(cfg, "mode", "log_only"))
        self.log = bool(_get(cfg, "log", True))
        self.unsupported_task_behavior = str(_get(cfg, "unsupported_task_behavior", "terminal_only"))
        self.use_best_progress = bool(_get(cfg, "use_best_progress", True))
        self.clip_dense_reward = _get(cfg, "clip_dense_reward", 0.05)
        self.excluded_task_ids = _as_string_set(_get(cfg, "excluded_task_ids", None))
        self.excluded_task_names = _as_string_set(_get(cfg, "excluded_task_names", None))

        thresholds_cfg = _to_container(cfg.get("thresholds"))
        self.thresholds = Thresholds(
            reach_distance=float(_get(thresholds_cfg, "reach_distance", 0.05)),
            target_distance=float(_get(thresholds_cfg, "target_distance", 0.06)),
            lift_height=float(_get(thresholds_cfg, "lift_height", 0.08)),
        )

        weights_cfg = _to_container(cfg.get("weights"))
        self.reward_manager = DenseRewardManager(
            RewardWeights(
                subgoal_progress=float(_get(weights_cfg, "subgoal_progress", 0.2)),
                phase_transition=float(_get(weights_cfg, "phase_transition", 0.05)),
                terminal_success=float(_get(weights_cfg, "terminal_success", 1.0)),
                smoothness=float(_get(weights_cfg, "smoothness", 0.0)),
            ),
            clip_dense_reward=self.clip_dense_reward,
        )
        self.extractor = LiberoStateExtractor()
        self.trackers: dict[int, OnlineSubgoalTracker] = {}

    def reset(self, env_index: int | None = None):
        if env_index is None:
            self.trackers.clear()
        else:
            self.trackers.pop(int(env_index), None)

    def step(
        self,
        env_index: int,
        env: Any,
        obs: Any,
        next_obs: Any,
        action: Any,
        env_reward: float,
        done: bool,
        info: Mapping[str, Any] | None,
        task_metadata: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        if not self.enabled:
            return (
                {
                    "subgoal_supported": 0.0,
                    "subgoal_phase_id": -1.0,
                    "subgoal_has_object": 0.0,
                    "subgoal_has_target": 0.0,
                    "subgoal_has_gripper": 0.0,
                    "subgoal_progress": 0.0,
                    "subgoal_best_progress": 0.0,
                    "subgoal_positive_delta": 0.0,
                    "subgoal_phase_completed": 0.0,
                    "success": float(done),
                    "action_delta_l2": 0.0,
                    "phase_name": "disabled",
                },
                {
                    "reward_env": float(env_reward),
                    "reward_subgoal": 0.0,
                    "reward_phase": 0.0,
                    "reward_terminal": 0.0,
                    "reward_smoothness": 0.0,
                    "reward_total": float(env_reward),
                },
            )

        if self._is_excluded_task(task_metadata):
            return self._excluded_step(env, obs, next_obs, done, info, task_metadata)

        state = self.extractor.extract(
            env=env,
            obs=next_obs,
            info=info,
            task_metadata=task_metadata,
            done=done,
        )
        tracker = self._tracker_for(env_index, state)
        step_info = tracker.update(state, action=action)
        reward_parts = self.reward_manager.compute(
            positive_delta=step_info.positive_delta,
            phase_completed=step_info.phase_completed,
            terminal_success=state.success,
            action_delta_l2=step_info.action_delta_l2,
            env_reward=env_reward,
        )

        subgoal_info = step_info.as_numeric_dict()
        subgoal_info["subgoal_has_object"] = float(state.object_position is not None)
        subgoal_info["subgoal_has_target"] = float(state.target_position is not None)
        subgoal_info["subgoal_has_gripper"] = float(state.gripper_position is not None)
        subgoal_info["phase_name"] = step_info.phase_name
        if done:
            self.reset(env_index)
        return subgoal_info, reward_parts.as_dict()

    def _tracker_for(self, env_index: int, state) -> OnlineSubgoalTracker:
        env_index = int(env_index)
        tracker = self.trackers.get(env_index)
        if tracker is not None:
            return tracker

        task_spec = infer_task_spec(state, self.thresholds)
        if task_spec is None and self.unsupported_task_behavior == "error":
            name = state.task_name or state.instruction or "unknown"
            raise ValueError(f"Unsupported LIBERO subgoal task: {name}")
        tracker = OnlineSubgoalTracker(task_spec=task_spec, use_best_progress=self.use_best_progress)
        self.trackers[env_index] = tracker
        return tracker

    def _is_excluded_task(self, task_metadata: Mapping[str, Any] | None) -> bool:
        task_metadata = task_metadata or {}
        task_id = task_metadata.get("task_id")
        if task_id is not None and str(task_id) in self.excluded_task_ids:
            return True
        for key in ("task_name", "task_suite_name", "instruction", "task_description"):
            value = task_metadata.get(key)
            if value is not None and str(value) in self.excluded_task_names:
                return True
        return False

    def _excluded_step(
        self,
        env: Any,
        obs: Any,
        next_obs: Any,
        done: bool,
        info: Mapping[str, Any] | None,
        task_metadata: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        state = self.extractor.extract(
            env=env,
            obs=next_obs,
            info=info,
            task_metadata=task_metadata,
            done=done,
        )
        return (
            {
                "subgoal_supported": 0.0,
                "subgoal_phase_id": -1.0,
                "subgoal_has_object": float(state.object_position is not None),
                "subgoal_has_target": float(state.target_position is not None),
                "subgoal_has_gripper": float(state.gripper_position is not None),
                "subgoal_progress": 0.0,
                "subgoal_best_progress": 0.0,
                "subgoal_positive_delta": 0.0,
                "subgoal_phase_completed": 0.0,
                "success": float(state.success),
                "action_delta_l2": 0.0,
                "phase_name": "excluded",
            },
            {
                "reward_env": 0.0,
                "reward_subgoal": 0.0,
                "reward_phase": 0.0,
                "reward_terminal": 0.0,
                "reward_smoothness": 0.0,
                "reward_total": 0.0,
            },
        )
