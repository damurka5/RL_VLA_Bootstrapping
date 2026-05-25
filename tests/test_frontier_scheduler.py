from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from rl_vla_bootstrapping.lchol.frontier_scheduler import (
    FrontierScheduler,
    FrontierSchedulerConfig,
)


@dataclass(frozen=True)
class _Spec:
    instruction_id: str
    shell_count: int
    instruction_template: str = "<instruction>"

    def sample_scene(self, rng):
        return {}

    def sample_reset(self, shell_id, scene, rng, **kwargs):
        return {}

    def success(self, state, instruction_binding):
        return bool(state)


class FrontierSchedulerTests(unittest.TestCase):
    def _scheduler(self) -> FrontierScheduler:
        return FrontierScheduler(
            specs=[_Spec("put_into_plate", 6), _Spec("grab_object", 5)],
            config=FrontierSchedulerConfig(
                promotion_success=0.50,
                demotion_success=0.20,
                min_train_updates_before_validation=0,
                saturation_abort_threshold=0.30,
                sample_frontier_probability=1.0,
            ),
        )

    def test_validation_success_promotes_active_shell(self):
        scheduler = self._scheduler()

        scheduler.update(
            [
                {
                    "instruction_id": "put_into_plate",
                    "shell_id": 0,
                    "success_rate": 0.50,
                    "rollouts": 50,
                }
            ]
        )

        self.assertEqual(scheduler.active_shells["put_into_plate"], 1)

    def test_validation_failure_demotes_shell(self):
        scheduler = self._scheduler()
        scheduler.state["put_into_plate"].active_shell = 2

        scheduler.update(
            [
                {
                    "instruction_id": "put_into_plate",
                    "shell_id": 2,
                    "success_rate": 0.20,
                    "rollouts": 50,
                }
            ]
        )

        self.assertEqual(scheduler.active_shells["put_into_plate"], 1)

    def test_saturation_prevents_promotion(self):
        scheduler = self._scheduler()

        scheduler.update(
            [
                {
                    "instruction_id": "put_into_plate",
                    "shell_id": 0,
                    "success_rate": 1.0,
                    "rollouts": 50,
                    "action_saturation_rate": 0.30,
                }
            ]
        )

        self.assertEqual(scheduler.active_shells["put_into_plate"], 0)

    def test_sampling_returns_active_frontier_shell(self):
        scheduler = self._scheduler()
        scheduler.state["put_into_plate"].active_shell = 2

        sample = scheduler.sample(rng=np.random.default_rng(2))

        self.assertIn(sample.instruction_id, {"put_into_plate", "grab_object"})
        self.assertEqual(sample.source, "frontier")
        self.assertEqual(sample.shell_id, scheduler.active_shells[sample.instruction_id])


if __name__ == "__main__":
    unittest.main()
