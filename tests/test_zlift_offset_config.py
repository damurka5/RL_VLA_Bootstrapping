"""The z-lift run must differ from its baseline in exactly one thing.

Half of this campaign's wasted GPU time has been comparisons between runs that
differed in more than the variable under test -- a horizon mix, a caught
fraction, a cap. This run exists to answer whether a sustained post-grasp z
bias recovers `pick_up`'s lift, and the answer is only readable if nothing else
moved. So the config is asserted to be its parent plus two settings, by
comparing the parsed documents rather than by trusting a diff.

WHY THESE TWO SETTINGS

Measured on the bowl_peak checkpoint, over the steps the object is held:

    pick_up          mean a_z p10 -0.020  p50 +0.232   lift|grasp 0.2584
    put_into_plate                +0.181       +0.371              0.6747
    put_into_bowl                 +0.095       +0.272              0.6334

§4.2 puts the loaded plant's ineffective->reliable transition at a_z 0.20-0.30.
`pick_up`'s median sits inside that band and the containers clear it, which is
the ordering of their lift rates. Its median grasp lifts 0.0334 m against a
0.05 m threshold -- short by 1.66 cm on a response curve that is steep exactly
there.

The z channel is the only one that has never carried an episode offset, and
per-step noise cannot explore a sustained bias: `_marginal_log_std` is what
makes the offset visible to the gradient at all (measured score +1.43 against
-0.015 for the conditional form), and the group of eight then becomes a probe
along sustained-bias directions.

The gate matters beyond tidiness. Bowl's binding constraint is its grasp
(0.6003; 54% of its failures never grasp) and its lift is already fine, so
gating the offset on holding spends the exploration only where the deficit is.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/examples/cdpr_smolvla_release_recovery_pilot.yaml"
CONFIG = ROOT / "configs/examples/cdpr_smolvla_zlift_offset.yaml"

# Action channel order is (x, y, z, yaw, gripper); the lift is channel 2.
Z_CHANNEL = 2
GRIPPER_CHANNEL = 4


def flatten(document, prefix=()):
    if isinstance(document, dict):
        out = {}
        for key, value in document.items():
            out.update(flatten(value, prefix + (str(key),)))
        return out
    return {".".join(prefix): document}


class ZLiftOffsetConfigTests(unittest.TestCase):
    def setUp(self):
        self.parent = flatten(yaml.safe_load(PARENT.read_text()))
        self.config = flatten(yaml.safe_load(CONFIG.read_text()))

    def _offset(self, flat):
        keys = sorted(k for k in flat if k.endswith("episode_offset_std"))
        self.assertEqual(len(keys), 1, keys)
        return keys[0], flat[keys[0]]

    def test_the_z_channel_carries_a_sustained_offset(self):
        _, std = self._offset(self.config)
        self.assertEqual(len(std), 5)
        self.assertGreater(std[Z_CHANNEL], 0.0)
        # Below the gripper's value, which has a recorded cost: it stalled the
        # curriculum and collapsed entropy when it carried this magnitude.
        self.assertLess(std[Z_CHANNEL], std[GRIPPER_CHANNEL])

    def test_the_offset_is_gated_on_holding(self):
        keys = [k for k in self.config if k.endswith("episode_offset_after_grasp")]
        self.assertEqual(len(keys), 1, keys)
        self.assertTrue(self.config[keys[0]])

    def test_the_parent_had_no_z_offset_and_no_gate(self):
        """Without this the run is not a change, and the test is vacuous."""

        _, std = self._offset(self.parent)
        self.assertEqual(std[Z_CHANNEL], 0.0)
        keys = [k for k in self.parent if k.endswith("episode_offset_after_grasp")]
        self.assertFalse(self.parent[keys[0]])

    def test_nothing_else_differs_from_the_parent(self):
        """One variable, or the run answers a question nobody asked."""

        allowed = {
            key for key in set(self.parent) | set(self.config)
            if key.endswith(("episode_offset_std", "episode_offset_after_grasp"))
        }
        changed = {
            key for key in set(self.parent) | set(self.config)
            if self.parent.get(key, object()) != self.config.get(key, object())
        }
        self.assertEqual(changed - allowed, set())

    def test_the_fixed_caps_did_not_drift(self):
        """The pilot's one-rung ladders are what made pick_up learnable."""

        prefix = "random_workspace_start_distance_ladder_by_instruction."
        ladder = {
            key.split(prefix)[1]: value
            for key, value in self.config.items() if prefix in key
        }
        self.assertTrue(ladder, "per-instruction ladders are missing")
        self.assertEqual(ladder.get("pick_up"), [0.06])
        self.assertEqual(ladder.get("move_to_object"), [0.08])
        # One rung each: a ladder that can promote is the failure §7.13
        # describes, where pick_up climbed to 0.13 on a grasp rate while its
        # success sat at 0.0091 and its GRPO groups carried no gradient.
        for name, rungs in ladder.items():
            self.assertEqual(len(rungs), 1, (name, rungs))


if __name__ == "__main__":
    unittest.main()
