"""The discriminator in ``tools/audit/sil_action_stats.py``.

Built against two synthetic policies whose answer is known by construction: a
servo that commands the direction to its own target, and a drift that commands
one fixed direction regardless. If the statistics cannot separate those, they
cannot separate the real thing either, and a reassuring histogram would be
worse than no analysis at all.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.sil_action_stats import _aiming, _per_axis

RNG = np.random.default_rng(0)
N = 4000

# Objects scattered around the end effector, as the workspace scatters them.
RELATIVE = np.concatenate(
    [RNG.uniform(-0.19, 0.19, size=(N, 2)), np.zeros((N, 1))], axis=1
)
SHUFFLED = RELATIVE[RNG.permutation(N)]


def _unit_xy(vectors: np.ndarray) -> np.ndarray:
    return vectors[:, :2] / np.linalg.norm(
        vectors[:, :2], axis=-1, keepdims=True
    )


def _actions(command_xy: np.ndarray) -> np.ndarray:
    action = np.zeros((N, 5))
    action[:, :2] = command_xy
    return action


class ServoTests(unittest.TestCase):
    """Commands the direction to its own target."""

    def setUp(self) -> None:
        command = 0.5 * _unit_xy(RELATIVE) + RNG.normal(0, 0.05, size=(N, 2))
        self.report = _aiming(_actions(command), RELATIVE, SHUFFLED)

    def test_it_aims(self) -> None:
        self.assertGreater(self.report["target_cosine"], 0.9)

    def test_the_shuffled_control_collapses(self) -> None:
        # Same commands, another world's geometry. Near zero is what makes the
        # raw cosine above mean anything.
        self.assertLess(abs(self.report["shuffled_cosine"]), 0.1)

    def test_no_preferred_world_frame_direction(self) -> None:
        self.assertLess(self.report["direction_concentration"], 0.1)

    def test_the_mean_command_is_near_zero_despite_large_commands(self) -> None:
        # The giveaway that a marginal distribution cannot answer this: the
        # mean is ~0 and the spread is large, which is also what a zero-mean
        # noise policy looks like.
        self.assertLess(self.report["command_mean_norm"], 0.05)
        self.assertGreater(self.report["command_spread"], 0.4)


class ConstantDriftTests(unittest.TestCase):
    """Commands one fixed direction regardless of the object."""

    def setUp(self) -> None:
        command = np.tile([0.4, -0.3], (N, 1)) + RNG.normal(
            0, 0.05, size=(N, 2)
        )
        self.report = _aiming(_actions(command), RELATIVE, SHUFFLED)

    def test_it_does_not_aim(self) -> None:
        self.assertLess(abs(self.report["target_cosine"]), 0.1)

    def test_direction_concentration_exposes_it(self) -> None:
        self.assertGreater(self.report["direction_concentration"], 0.9)

    def test_the_shuffled_control_matches_the_real_one(self) -> None:
        # Both near zero: the geometry explains nothing either way, which is
        # the signature of a command that ignores it.
        self.assertLess(
            abs(
                self.report["target_cosine"] - self.report["shuffled_cosine"]
            ),
            0.1,
        )


class NoisePolicyTests(unittest.TestCase):
    """Random commands: wide spread, no aiming, no fixed direction."""

    def setUp(self) -> None:
        command = RNG.uniform(-1.0, 1.0, size=(N, 2))
        self.report = _aiming(_actions(command), RELATIVE, SHUFFLED)

    def test_spread_alone_does_not_imply_aiming(self) -> None:
        # This is the case the histograms would flatter: a broad, well-filled
        # distribution that contains no information about the object at all.
        self.assertGreater(self.report["command_spread"], 0.4)
        self.assertLess(abs(self.report["target_cosine"]), 0.1)
        self.assertLess(self.report["direction_concentration"], 0.1)


class CosineGapTests(unittest.TestCase):
    """The gap, not the raw cosine, is what survives a confounded control."""

    def test_a_workspace_centre_policy_scores_a_gap_of_zero(self) -> None:
        # Commands point at the middle of the workspace, never at the object.
        # Its raw target cosine is positive -- the middle is vaguely toward
        # everything -- and only the gap exposes that it aims at nothing.
        rng = np.random.default_rng(7)
        ee = rng.uniform(-0.15, 0.15, size=(N, 2))
        target = rng.uniform(-0.19, 0.19, size=(N, 2))
        relative = np.concatenate([target - ee, np.zeros((N, 1))], axis=1)
        shuffled = np.concatenate(
            [target[rng.permutation(N)] - ee, np.zeros((N, 1))], axis=1
        )
        command = -ee / np.linalg.norm(ee, axis=-1, keepdims=True)

        report = _aiming(_actions(command), relative, shuffled)
        self.assertGreater(report["target_cosine"], 0.3)
        self.assertLess(abs(report["cosine_gap"]), 0.1)

    def test_a_servo_keeps_a_large_gap(self) -> None:
        rng = np.random.default_rng(8)
        ee = rng.uniform(-0.15, 0.15, size=(N, 2))
        target = rng.uniform(-0.19, 0.19, size=(N, 2))
        relative = np.concatenate([target - ee, np.zeros((N, 1))], axis=1)
        shuffled = np.concatenate(
            [target[rng.permutation(N)] - ee, np.zeros((N, 1))], axis=1
        )
        command = _unit_xy(relative)

        report = _aiming(_actions(command), relative, shuffled)
        self.assertGreater(report["cosine_gap"], 0.5)


class PerAxisTests(unittest.TestCase):
    def test_saturation_is_reported(self) -> None:
        action = np.zeros((100, 5))
        action[:40, 0] = 1.0  # 40% pinned at the positive rail
        stats = _per_axis(action)
        self.assertAlmostEqual(stats["x"]["saturated_fraction"], 0.4, places=5)
        self.assertEqual(stats["y"]["saturated_fraction"], 0.0)

    def test_every_axis_is_named(self) -> None:
        stats = _per_axis(np.zeros((10, 5)))
        self.assertEqual(
            sorted(stats), sorted(["x", "y", "z", "yaw", "gripper"])
        )


if __name__ == "__main__":
    unittest.main()
