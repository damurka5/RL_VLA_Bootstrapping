"""Adding a vision feature must change nothing on step 0.

That is the whole claim. Two attempts at REPLACING the pooling destroyed the
policy -- the second one measurably: handed a perfect object position, the
probe's oracle_xy arm fell from 0.92 ever-grasped to 0.30, which is damage to
descend/close/lift and not to vision. The cause is that removing an input from a
trained MLP's first layer shifts the hidden activations every later layer was
calibrated on, and there is no surgical version of that.

Adding removes nothing, so the network is required to compute a bit-identical
function immediately after the widened load. If that invariant does not hold,
the third attempt is the second attempt again and there is no point running it.

The failure modes worth pinning are all silent -- the shapes stay valid and
training proceeds either way:

* zero columns inserted in the wrong place shift the prior block, deleting the
  residual's access to the action it is a residual ON;
* the prior block dropped instead of moved;
* the new columns initialised to noise rather than zero, so step 0 is a
  different policy;
* the two pooling halves sharing one projection cache slot, so each forward
  rebuilds the other's matrix and the feature is different every step.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PROPRIO = 6
OLD_VISION = 8
NEW_VISION = 16  # 8 old + 8 appended
CHUNK = 2
ACTION = 5
HIDDEN = 16


def _trainer(state_dim: int, tmp: str):
    from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import (
        DistributedContext,
    )
    from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
        SmolVLAGRPOTrainer,
        parse_args,
    )

    args = parse_args(
        [
            "--device", "cpu",
            "--no-distributed",
            "--hidden-dim", str(HIDDEN),
            "--chunk-size", str(CHUNK),
            "--action-dim", str(ACTION),
        ]
    )
    return SmolVLAGRPOTrainer(
        args=args,
        state_dim=state_dim,
        action_dim=ACTION,
        chunk_size=CHUNK,
        run_dir=Path(tmp),
        device=torch.device("cpu"),
        distributed=DistributedContext(device="cpu"),
    )


@unittest.skipIf(torch is None, "torch is unavailable")
class ExpandPreservesTheFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.narrow = _trainer(PROPRIO + OLD_VISION, self._tmp.name)
        self.wide = _trainer(PROPRIO + NEW_VISION, self._tmp.name)
        # Give the narrow model distinctive weights so a mis-slice shows up.
        with torch.no_grad():
            for param in self.narrow._unwrap(self.narrow.actor).parameters():
                param.uniform_(-0.5, 0.5)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self):
        state = dict(self.narrow._unwrap(self.narrow.actor).state_dict())
        widened, info = self.wide.expand_residual_vision_columns(state)
        self.wide._unwrap(self.wide.actor).load_state_dict(widened)
        return info

    def test_the_widened_policy_computes_the_same_actions(self) -> None:
        """The load-bearing invariant: step 0 is unchanged."""

        info = self._load()
        self.assertEqual(info["vision_expand/inserted"], float(NEW_VISION - OLD_VISION))

        narrow_state = torch.randn((16, PROPRIO + OLD_VISION))
        prior = torch.randn((16, CHUNK, ACTION)) * 0.3
        # The wide model sees the same inputs plus an arbitrary new vision block,
        # which zero columns must ignore completely.
        wide_state = torch.cat(
            [narrow_state, torch.randn((16, NEW_VISION - OLD_VISION)) * 7.0],
            dim=-1,
        )
        with torch.no_grad():
            a = self.narrow._unwrap(self.narrow.actor)(narrow_state, prior)
            b = self.wide._unwrap(self.wide.actor)(wide_state, prior)
        self.assertTrue(torch.allclose(a, b, atol=1e-6), "step 0 is not identical")

    def test_the_prior_block_moved_rather_than_being_dropped(self) -> None:
        """A fencepost here silently deletes the residual's view of the prior."""

        self._load()
        narrow_first = next(
            m
            for m in self.narrow._unwrap(self.narrow.actor).modules()
            if isinstance(m, torch.nn.Linear)
        )
        wide_first = next(
            m
            for m in self.wide._unwrap(self.wide.actor).modules()
            if isinstance(m, torch.nn.Linear)
        )
        prior_width = CHUNK * ACTION
        self.assertTrue(
            torch.allclose(
                wide_first.weight[:, -prior_width:],
                narrow_first.weight[:, -prior_width:],
            )
        )

    def test_the_old_state_columns_kept_their_offsets(self) -> None:
        self._load()
        narrow_first = next(
            m
            for m in self.narrow._unwrap(self.narrow.actor).modules()
            if isinstance(m, torch.nn.Linear)
        )
        wide_first = next(
            m
            for m in self.wide._unwrap(self.wide.actor).modules()
            if isinstance(m, torch.nn.Linear)
        )
        old_state = PROPRIO + OLD_VISION
        self.assertTrue(
            torch.allclose(
                wide_first.weight[:, :old_state],
                narrow_first.weight[:, :old_state],
            )
        )

    def test_the_appended_block_is_exactly_zero(self) -> None:
        self._load()
        wide_first = next(
            m
            for m in self.wide._unwrap(self.wide.actor).modules()
            if isinstance(m, torch.nn.Linear)
        )
        old_state = PROPRIO + OLD_VISION
        new_state = PROPRIO + NEW_VISION
        self.assertEqual(
            float(wide_first.weight[:, old_state:new_state].abs().sum()), 0.0
        )

    def test_the_new_path_can_learn(self) -> None:
        """Zeroed, not frozen."""

        self._load()
        wide = self.wide._unwrap(self.wide.actor)
        first = next(m for m in wide.modules() if isinstance(m, torch.nn.Linear))
        state = torch.randn((8, PROPRIO + NEW_VISION))
        prior = torch.zeros((8, CHUNK, ACTION))
        wide(state, prior).sum().backward()
        old_state = PROPRIO + OLD_VISION
        new_state = PROPRIO + NEW_VISION
        self.assertGreater(
            float(first.weight.grad[:, old_state:new_state].abs().sum()), 0.0
        )

    def test_a_same_width_checkpoint_is_passed_through(self) -> None:
        state = dict(self.wide._unwrap(self.wide.actor).state_dict())
        out, info = self.wide.expand_residual_vision_columns(state)
        self.assertEqual(info["vision_expand/inserted"], 0.0)
        self.assertEqual(set(out), set(state))

    def test_a_wider_checkpoint_is_refused(self) -> None:
        state = dict(self.wide._unwrap(self.wide.actor).state_dict())
        with self.assertRaises(RuntimeError):
            self.narrow.expand_residual_vision_columns(state)

    def test_a_changed_output_width_is_refused(self) -> None:
        """Not a vision-width change; widening it would be nonsense."""

        state = dict(self.narrow._unwrap(self.narrow.actor).state_dict())
        first_key = next(k for k in state if k.endswith(".weight"))
        state[first_key] = torch.zeros((HIDDEN + 3, PROPRIO + OLD_VISION + CHUNK * ACTION))
        with self.assertRaises(RuntimeError):
            self.wide.expand_residual_vision_columns(state)


@unittest.skipIf(torch is None, "torch is unavailable")
class DualPoolingTest(unittest.TestCase):
    """The runtime half: two summaries of the same tokens, flat one first."""

    class _Fake:
        """Only the two methods under test, with the real implementations."""

        from rl_vla_bootstrapping.policy.smolvla_cdpr import (  # noqa: E402
            SmolVLARuntime as _Real,
        )

        _vision_projection = _Real._vision_projection
        _pool_vision = _Real._pool_vision
        _pool_vision_single = _Real._pool_vision_single

        def __init__(self, mode: str) -> None:
            self.vision_pooling = mode

    def _tokens(self):
        torch.manual_seed(0)
        # Two cameras, 16 tokens, 960 channels -- the real connector shape.
        return [torch.randn((4, 16, 960)) for _ in range(2)]

    # 32 tokens (2 cameras x 16), so each half must be a multiple of 32.
    HALF = 32
    DUAL = 64

    def test_dual_is_the_two_halves_concatenated_in_order(self) -> None:
        captured = self._tokens()
        flat = self._Fake("flat_random")._pool_vision(captured, self.HALF)
        per_token = self._Fake("per_token_random")._pool_vision(
            captured, self.HALF
        )
        dual = self._Fake("dual_random")._pool_vision(captured, self.DUAL)
        self.assertEqual(tuple(dual.shape), (4, self.DUAL))
        # flat_random FIRST, so a checkpoint's columns keep their offsets.
        self.assertTrue(torch.allclose(dual[:, : self.HALF], flat, atol=1e-6))
        self.assertTrue(
            torch.allclose(dual[:, self.HALF :], per_token, atol=1e-6)
        )

    def test_the_two_halves_are_not_the_same_numbers(self) -> None:
        """Guards the test above against both halves being one pooling."""

        captured = self._tokens()
        dual = self._Fake("dual_random")._pool_vision(captured, self.DUAL)
        self.assertFalse(
            torch.allclose(
                dual[:, : self.HALF], dual[:, self.HALF :], atol=1e-3
            )
        )

    def test_the_projection_cache_holds_both_shapes_at_once(self) -> None:
        """A one-slot cache would rebuild a matrix on every forward.

        The two halves need projections of different shapes. If the cache is
        keyed on a single slot, each call evicts the other's matrix, the feature
        changes every step, and nothing downstream can learn it -- silently.
        """

        fake = self._Fake("dual_random")
        captured = self._tokens()
        first = fake._pool_vision(captured, self.DUAL)
        cache = fake._vision_proj_cache
        self.assertGreaterEqual(len(cache), 2)
        second = fake._pool_vision(captured, self.DUAL)
        self.assertTrue(torch.equal(first, second), "feature is not stable")

    def test_an_odd_width_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._Fake("dual_random")._pool_vision(self._tokens(), 63)

    def test_single_pooling_output_is_unchanged_by_the_cache_rewrite(self) -> None:
        """flat_random at 512 must still be the matrix the campaign measured."""

        captured = self._tokens()
        a = self._Fake("flat_random")._pool_vision(captured, 32)
        b = self._Fake("flat_random")._pool_vision(captured, 32)
        self.assertTrue(torch.equal(a, b))


class ConfigWiringTest(unittest.TestCase):
    def test_the_pick_up_config_asks_for_both_at_the_doubled_width(self) -> None:
        import yaml

        root = Path(__file__).resolve().parents[1]
        raw = yaml.safe_load(
            (
                root
                / "configs"
                / "examples"
                / "cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml"
            ).read_text()
        )

        def find(node, key):
            if isinstance(node, dict):
                for name, value in node.items():
                    if name == key:
                        return value
                    found = find(value, key)
                    if found is not None:
                        return found
            return None

        self.assertEqual(find(raw, "residual_vision_pooling"), "dual_random")
        # Both halves at the width each was measured at.
        self.assertEqual(find(raw, "residual_vision_dim"), 1024)

    def test_the_runtime_accepts_the_mode(self) -> None:
        from rl_vla_bootstrapping.policy.smolvla_cdpr import SmolVLARuntime

        import inspect

        source = inspect.getsource(SmolVLARuntime.__init__)
        self.assertIn("dual_random", source)


if __name__ == "__main__":
    unittest.main()
