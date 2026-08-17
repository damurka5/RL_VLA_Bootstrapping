"""The discriminator in ``tools/audit/sil_action_stats.py``.

Built against synthetic policies whose answer is known by construction. The
two nulls this file previously used both certified a pure fixed drift -- one
carrying no goal information whatsoever -- as aiming, so the drift case here
is not a formality: it is the case that broke the metric twice.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.sil_action_stats import _aiming, _per_axis

N = 40000


def _world(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Goals scattered, arm loosely tracking its own goal but offset in +x.

    The offset is what makes this hard: it gives any fixed -x command a large
    spurious cosine against the goal direction, because the goals really do
    lie that way on average.
    """

    rng = np.random.default_rng(seed)
    goal = rng.uniform(-0.19, 0.19, size=(N, 2))
    ee = goal * 0.6 + np.array([0.10, 0.0]) + rng.normal(0, 0.03, size=(N, 2))
    relative = np.concatenate([goal - ee, np.zeros((N, 1))], axis=1)
    return relative, rng


def _actions(command_xy: np.ndarray) -> np.ndarray:
    action = np.zeros((command_xy.shape[0], 5))
    action[:, :2] = command_xy
    return action


class FixedDriftTests(unittest.TestCase):
    """One command for every world. Knows nothing. Must score zero."""

    def setUp(self) -> None:
        relative, rng = _world(11)
        command = np.array([-0.5, 0.0]) + rng.normal(0, 0.15, size=(N, 2))
        self.report = _aiming(_actions(command), relative, rng)

    def test_the_raw_cosine_is_large_and_meaningless(self) -> None:
        # The trap: the drift points where the goals happen to be, so a bare
        # cosine reads like excellent aiming.
        self.assertGreater(self.report["target_cosine"], 0.6)

    def test_aim_is_zero(self) -> None:
        self.assertLess(abs(self.report["aim"]), 0.02)

    def test_direction_concentration_exposes_it(self) -> None:
        self.assertGreater(self.report["direction_concentration"], 0.9)


class ServoTests(unittest.TestCase):
    def setUp(self) -> None:
        relative, rng = _world(12)
        unit = relative[:, :2] / np.linalg.norm(
            relative[:, :2], axis=-1, keepdims=True
        )
        command = 0.6 * unit + rng.normal(0, 0.05, size=(N, 2))
        self.report = _aiming(_actions(command), relative, rng)

    def test_aim_is_clearly_positive(self) -> None:
        self.assertGreater(self.report["aim"], 0.25)

    def test_a_servo_can_also_look_concentrated(self) -> None:
        # Not a bug. The arm sits offset in +x, so pointing at the goals means
        # pointing -x most of the time, and a genuine servo scores 0.82 here.
        # direction_concentration therefore cannot separate aiming from drift
        # on its own either -- only `aim` can, and this is why.
        self.assertGreater(self.report["direction_concentration"], 0.5)
        self.assertGreater(self.report["aim"], 0.25)


class MixtureTests(unittest.TestCase):
    """Half the rows servo, half drift. Should land between the two."""

    def setUp(self) -> None:
        relative, rng = _world(13)
        unit = relative[:, :2] / np.linalg.norm(
            relative[:, :2], axis=-1, keepdims=True
        )
        servo = rng.random((N, 1)) < 0.5
        command = np.where(servo, 0.6 * unit, np.array([-0.5, 0.0]))
        command = command + rng.normal(0, 0.05, size=(N, 2))
        self.report = _aiming(_actions(command), relative, rng)

    def test_aim_is_between_drift_and_servo(self) -> None:
        self.assertGreater(self.report["aim"], 0.05)
        self.assertLess(self.report["aim"], 0.25)


class NoiseTests(unittest.TestCase):
    def setUp(self) -> None:
        relative, rng = _world(14)
        self.report = _aiming(
            _actions(rng.uniform(-1, 1, size=(N, 2))), relative, rng
        )

    def test_spread_alone_does_not_imply_aiming(self) -> None:
        # The case a histogram would flatter: broad, well filled, uninformed.
        self.assertGreater(self.report["command_spread"], 0.4)
        self.assertLess(abs(self.report["aim"]), 0.02)


class PermutationStabilityTests(unittest.TestCase):
    def test_the_null_is_reported_with_its_own_spread(self) -> None:
        relative, rng = _world(15)
        report = _aiming(
            _actions(rng.normal(0, 0.3, size=(N, 2))), relative, rng
        )
        # Eight permutations; if their spread is large the aim is unreadable,
        # so it is published rather than hidden behind a mean.
        self.assertIn("permutation_spread", report)
        self.assertLess(report["permutation_spread"], 0.02)


class PerAxisTests(unittest.TestCase):
    def test_saturation_is_reported(self) -> None:
        action = np.zeros((100, 5))
        action[:40, 0] = 1.0
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
