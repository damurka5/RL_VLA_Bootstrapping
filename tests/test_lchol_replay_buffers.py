from __future__ import annotations

import unittest

import numpy as np

from rl_vla_bootstrapping.lchol.replay_buffers import PerOptionReplayBuffer


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


if __name__ == "__main__":
    unittest.main()
