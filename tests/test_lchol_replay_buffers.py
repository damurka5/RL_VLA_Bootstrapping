from __future__ import annotations

import unittest

import numpy as np

from rl_vla_bootstrapping.lchol.replay_buffers import PerOptionReplayBuffer
from rl_vla_bootstrapping.lchol.grpo_runtime import LCHOLGRPOConfig, LCHOLGRPORuntime
from robots.cdpr.cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec


class LCHOLReplayBufferTests(unittest.TestCase):
    def test_sampling_respects_allowed_options_and_capacity(self):
        replay = PerOptionReplayBuffer(capacity_per_option=2)
        replay.add("grab_object", "grab-old")
        replay.add("grab_object", "grab-new")
        replay.add("grab_object", "grab-newest")
        replay.add("push_left", "push")

        self.assertEqual(replay.sizes()["grab_object"], 2)

        samples = replay.sample_balanced(
            batch_size=8,
            rng=np.random.default_rng(0),
            allowed_options=["push_left"],
        )

        self.assertEqual(samples, ["push"] * 8)

    def test_runtime_metrics_include_replay_episode_counts(self):
        runtime = LCHOLGRPORuntime(
            config=LCHOLGRPOConfig(enabled=True),
            spec=CDPRLCHOLSpec(),
            available_options=("grab_object",),
            seed=0,
        )

        runtime.capture_candidate(
            obs={"instruction": "put apple into plate"},
            step_info={
                "env_instance_id": 0,
                "episode_index": 7,
                "instruction_type": "put_into_plate",
                "target_object_catalog": "ycb_apple",
                "source_instruction": "put apple into plate",
                "distance_ee_to_object_xy": 0.02,
                "gripper_closed": 1.0,
                "caught_object_is_target": 1.0,
            },
            sampled_action=np.zeros((4,), dtype=np.float32),
            group_score=1.0,
            update=1,
            global_step=1,
        )

        metrics = runtime.metrics()

        self.assertEqual(metrics["replay/total_records"], 1.0)
        self.assertEqual(metrics["replay/episodes_total"], 1.0)
        self.assertEqual(metrics["replay/episodes/grab_object"], 1.0)


if __name__ == "__main__":
    unittest.main()
