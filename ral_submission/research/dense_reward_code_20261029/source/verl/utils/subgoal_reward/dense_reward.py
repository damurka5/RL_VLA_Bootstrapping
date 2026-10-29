from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    subgoal_progress: float = 0.2
    phase_transition: float = 0.05
    terminal_success: float = 1.0
    smoothness: float = 0.0


@dataclass
class RewardParts:
    reward_env: float = 0.0
    reward_subgoal: float = 0.0
    reward_phase: float = 0.0
    reward_terminal: float = 0.0
    reward_smoothness: float = 0.0
    reward_total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "reward_env": self.reward_env,
            "reward_subgoal": self.reward_subgoal,
            "reward_phase": self.reward_phase,
            "reward_terminal": self.reward_terminal,
            "reward_smoothness": self.reward_smoothness,
            "reward_total": self.reward_total,
        }


class DenseRewardManager:
    def __init__(self, weights: RewardWeights | None = None, clip_dense_reward: float | None = 0.05):
        self.weights = weights or RewardWeights()
        self.clip_dense_reward = clip_dense_reward

    def compute(
        self,
        positive_delta: float,
        phase_completed: bool,
        terminal_success: bool,
        action_delta_l2: float,
        env_reward: float = 0.0,
    ) -> RewardParts:
        reward_subgoal = self.weights.subgoal_progress * float(positive_delta)
        reward_phase = self.weights.phase_transition * float(phase_completed)
        reward_terminal = self.weights.terminal_success * float(terminal_success)
        reward_smoothness = -self.weights.smoothness * float(action_delta_l2)

        shaping = reward_subgoal + reward_phase + reward_smoothness
        if self.clip_dense_reward is not None:
            shaping = float(np.clip(shaping, -self.clip_dense_reward, self.clip_dense_reward))
        total = shaping + reward_terminal

        return RewardParts(
            reward_env=float(env_reward),
            reward_subgoal=float(reward_subgoal),
            reward_phase=float(reward_phase),
            reward_terminal=float(reward_terminal),
            reward_smoothness=float(reward_smoothness),
            reward_total=float(total),
        )
