"""The residual's vision feature must keep the object's position.

A fixed random projection retains only ``d_out/d_in`` of any linearly decodable
signal. At the shipped 512 of 30720 that is 1.7%, and measured on MJWarp it took
the object's direction from cosine +0.41 (un-projected) to +0.09 -- which is why
the policy servos to the wrong place. ``per_token_random`` reduces channels only
and keeps the 4x4 spatial grid where position lives, at the same 512 width.
"""

from __future__ import annotations

import unittest

from rl_vla_bootstrapping.policy import smolvla_cdpr
from rl_vla_bootstrapping.policy.smolvla_cdpr import SmolVLARuntime

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _runtime(mode: str) -> SmolVLARuntime:
    """A bare instance: _pool_vision touches no model state."""

    runtime = SmolVLARuntime.__new__(SmolVLARuntime)
    runtime.vision_pooling = mode
    return runtime


@unittest.skipIf(torch is None, "torch is not installed")
class VisionPoolingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Another module in this suite sets smolvla_cdpr.torch to None to check
        # the no-torch error path and does not restore it, so _vision_projection
        # (which reads the module global) sees None when the suite runs in one
        # process. Pin it for the duration rather than depend on file order.
        self._saved_torch = smolvla_cdpr.torch
        smolvla_cdpr.torch = torch

    def tearDown(self) -> None:
        smolvla_cdpr.torch = self._saved_torch

    def _cameras(self, batch: int = 3):
        torch.manual_seed(0)
        return [torch.randn(batch, 16, 960) for _ in range(3)]

    def test_both_modes_produce_the_configured_width(self):
        for mode in ("flat_random", "per_token_random"):
            out = _runtime(mode)._pool_vision(self._cameras(), 512)
            self.assertEqual(tuple(out.shape), (3, 512), mode)

    def test_only_the_two_real_cameras_are_used(self):
        """The third image is the masked aux view and must not contribute."""

        for mode in ("flat_random", "per_token_random"):
            cams = self._cameras()
            baseline = _runtime(mode)._pool_vision(cams, 512)
            cams[2] = torch.randn_like(cams[2]) * 100.0
            after = _runtime(mode)._pool_vision(cams, 512)
            self.assertTrue(torch.allclose(baseline, after, atol=1e-5), mode)

    def test_per_token_keeps_the_spatial_grid_separable(self):
        """Each output block must come from ONE token.

        This is the property the whole change rests on: if a token's output
        block depended on other tokens, position would be mixed away again.
        """

        cams = self._cameras()
        runtime = _runtime("per_token_random")
        baseline = runtime._pool_vision(cams, 512)
        # Perturb exactly one spatial place (camera 0, token 5).
        cams[0][:, 5, :] += 10.0
        after = runtime._pool_vision(cams, 512)
        changed = (after - baseline).abs().reshape(3, 32, 16).sum(dim=(0, 2))
        moved = (changed > 1e-4).nonzero().reshape(-1).tolist()
        self.assertEqual(moved, [5])

    def test_flat_random_mixes_every_token_together(self):
        """The failure mode, asserted so the contrast is not theoretical."""

        cams = self._cameras()
        runtime = _runtime("flat_random")
        baseline = runtime._pool_vision(cams, 512)
        cams[0][:, 5, :] += 10.0
        after = runtime._pool_vision(cams, 512)
        # One token moved and essentially the whole 512-d output responds.
        touched = ((after - baseline).abs() > 1e-4).float().mean()
        self.assertGreater(float(touched), 0.9)

    def test_indivisible_width_is_refused(self):
        with self.assertRaises(ValueError):
            _runtime("per_token_random")._pool_vision(self._cameras(), 500)

    def test_unknown_mode_is_refused_at_construction(self):
        runtime = SmolVLARuntime.__new__(SmolVLARuntime)
        with self.assertRaises(ValueError):
            SmolVLARuntime.__init__(
                runtime,
                policy=None,
                checkpoint="x",
                device="cpu",
                dtype=None,
                obs_spec=None,
                action_spec=None,
                tokenizer=object(),
                vision_pooling="nonsense",
            )


if __name__ == "__main__":
    unittest.main()
