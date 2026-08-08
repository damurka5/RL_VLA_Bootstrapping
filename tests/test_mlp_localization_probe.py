"""The nonlinear probe has to find what the linear one misses, and no more.

Every localization number in this campaign came from a LINEAR probe, and the
thing that has to learn the map is a two-hidden-layer MLP. Those are different
claims, and the difference now matters: a 5.2M-step run fed a feature measured
at +0.389 linear direction cosine never moved the residual's own aim off 0.055.
Either the map is harder to learn than the linear number suggests, or it is
learnable and RL is not finding it -- and only a probe with the residual's own
hypothesis class can tell those apart.

So the probe is checked against synthetic features whose answer is known:

* a LINEARLY decodable object position -- both probes must find it;
* a position recoverable only through a nonlinearity -- the MLP must find it and
  the ridge must not, or the MLP adds nothing;
* pure noise -- BOTH must score near zero, and the swap control must too.

The last one is the one that matters most. An MLP with 1024 hidden units on a
few thousand samples can memorize its way to a high training score, and a probe
that reports a strong result on noise would have ended this investigation with a
confident wrong answer.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tools" / "audit" / "grasp_feature_probe.py"

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "grasp_feature_probe", PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

# The effective sample size for "features -> object XY" is the number of
# EPISODES, not steps: the object does not move within an episode, so its twelve
# steps are one training point wearing twelve hats. Measured on the synthetic
# task below, an MLP scores 0.55 at 30 episodes (memorizing, train R2 1.00) and
# 0.94 at 120. Anything smaller tests the probe's capacity to overfit, not its
# ability to find the map.
EPISODES = 120
PER_EPISODE = 12


def _episode_ids() -> np.ndarray:
    return np.repeat(np.arange(EPISODES), PER_EPISODE)


def _targets(rng: np.random.RandomState) -> np.ndarray:
    """One object position per episode, held across its steps."""

    per_episode = rng.uniform(-0.25, 0.25, size=(EPISODES, 2))
    return np.repeat(per_episode, PER_EPISODE, axis=0)


def _swapped(targets: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    per_episode = targets[::PER_EPISODE]
    permuted = per_episode[rng.permutation(EPISODES)]
    return np.repeat(permuted, PER_EPISODE, axis=0)


@unittest.skipIf(torch is None, "torch is unavailable")
class MlpProbeDiscriminatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.RandomState(0)
        self.episodes = _episode_ids()
        self.targets = _targets(self.rng)
        self.control = _swapped(self.targets, np.random.RandomState(1))

    def _mlp(self, features, **kwargs):
        return probe._mlp_r2(
            features,
            self.targets,
            self.control,
            self.episodes,
            seed=0,
            hidden=64,
            epochs=400,
            **kwargs,
        )

    def _ridge(self, features):
        return probe._dual_ridge_r2(
            features, self.targets, self.control, self.episodes, seed=0
        )

    def test_a_linear_feature_is_found_by_both(self) -> None:
        noise = self.rng.normal(0.0, 0.02, size=(len(self.episodes), 24))
        features = noise.copy()
        features[:, :2] += self.targets
        mlp = self._mlp(features)
        ridge = self._ridge(features)
        self.assertGreater(mlp["direction_cosine"], 0.8)
        self.assertGreater(ridge["direction_cosine"], 0.7)

    def test_a_nonlinear_feature_is_found_only_by_the_mlp(self) -> None:
        """The reason for having this probe at all.

        The position is encoded through a squaring and a product, which no
        linear map can invert. If the MLP cannot beat ridge here it is adding
        nothing and a negative result from it would mean nothing either.
        """

        # A phase encoding: x -> (cos 12x, sin 12x). It is INJECTIVE over the
        # sampled range (|12x| < pi, so atan2 recovers x uniquely) but a linear
        # map cannot invert it. Squares and products would not do -- x^2, y^2
        # and xy are identical for (x, y) and (-x, -y), so the target would not
        # be a function of the features and BOTH probes would fail, which says
        # nothing about either.
        x, y = self.targets[:, 0], self.targets[:, 1]
        features = np.stack(
            [
                np.cos(12.0 * x),
                np.sin(12.0 * x),
                np.cos(12.0 * y),
                np.sin(12.0 * y),
            ],
            axis=1,
        )
        features = features + self.rng.normal(0.0, 1e-3, size=features.shape)
        mlp = self._mlp(features)
        ridge = self._ridge(features)
        self.assertGreater(mlp["r2"], 0.5)
        self.assertGreater(
            mlp["r2"],
            ridge["r2"] + 0.2,
            msg=(
                f"MLP {mlp['r2']:.3f} did not beat ridge {ridge['r2']:.3f} on "
                "a feature only a nonlinearity can decode"
            ),
        )

    def test_pure_noise_scores_near_zero_on_held_out_data(self) -> None:
        """The guard against a confident wrong answer.

        A 1024-wide MLP on a few thousand samples can memorize. If that showed
        up as a real held-out score, this probe would report that the feature
        carries the object position when it carries nothing.
        """

        features = self.rng.normal(0.0, 1.0, size=(len(self.episodes), 64))
        mlp = self._mlp(features)
        self.assertLess(mlp["r2"], 0.2)
        self.assertLess(abs(mlp["direction_cosine"]), 0.35)

    def test_the_swap_control_does_not_fire_on_a_real_feature(self) -> None:
        """A control that scores would mean the folds leak, not that it works."""

        noise = self.rng.normal(0.0, 0.02, size=(len(self.episodes), 24))
        features = noise.copy()
        features[:, :2] += self.targets
        mlp = self._mlp(features)
        self.assertLess(mlp["control_r2"], 0.2)
        self.assertGreater(mlp["r2"] - mlp["control_r2"], 0.4)

    def test_memorization_is_visible_as_a_train_test_gap(self) -> None:
        """train_r2 is printed next to r2 so this is readable, not inferred."""

        features = self.rng.normal(0.0, 1.0, size=(len(self.episodes), 64))
        mlp = self._mlp(features)
        self.assertGreater(mlp["train_r2"], mlp["r2"])

    def test_it_reports_the_same_keys_as_the_ridge_probe(self) -> None:
        """They are printed as two rows of one table and must stay comparable."""

        features = self.rng.normal(0.0, 1.0, size=(len(self.episodes), 8))
        mlp = self._mlp(features)
        ridge = self._ridge(features)
        for key in (
            "r2",
            "control_r2",
            "direction_cosine",
            "direction_spread",
            "episodes",
        ):
            self.assertIn(key, mlp)
            self.assertIn(key, ridge)

    def test_too_few_episodes_returns_nan_rather_than_a_number(self) -> None:
        episodes = np.repeat(np.arange(3), 4)
        targets = np.repeat(np.random.RandomState(2).uniform(size=(3, 2)), 4, axis=0)
        out = probe._mlp_r2(
            np.random.RandomState(3).normal(size=(12, 8)),
            targets,
            targets,
            episodes,
            seed=0,
            hidden=16,
            epochs=10,
        )
        self.assertTrue(np.isnan(out["r2"]))


class PoolingPassthroughTest(unittest.TestCase):
    """The probe scored flat_random no matter what the config asked for.

    _build_runtime never forwarded residual_vision_pooling, so a run configured
    for per_token_random or dual_random was measured on the feature it had
    replaced -- and the measurement would have looked perfectly reasonable.
    """

    def test_the_runtime_is_built_with_the_configured_pooling(self) -> None:
        import inspect

        source = inspect.getsource(probe._build_runtime)
        self.assertIn("vision_pooling=", source)
        self.assertIn("residual_vision_pooling", source)

    def test_the_pooling_is_returned_so_the_report_can_split_the_halves(
        self,
    ) -> None:
        import inspect

        source = inspect.getsource(probe._build_runtime)
        self.assertIn("return runtime, vision_dim, state_dim,", source)


if __name__ == "__main__":
    unittest.main()
