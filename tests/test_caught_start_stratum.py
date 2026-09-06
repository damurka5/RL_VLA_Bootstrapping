"""`physical_grasp` is live state, and reading it late says the opposite thing.

`BatchedReset.physical_grasp` is not a record of how the episode started.
`_update_physical_grasp` calls `.copy_()` into it on every env step, and the
value additionally requires `active_mask` -- so after a rollout it reads "was
still running AND still holding at the final step", which is False for every
episode that terminated.

Two consumers read it after the fact and both were wrong:

  sil_record wrote it into every recording's `physical_grasp_at_reset` column.
  Measured on the pooled bank, that column called 494 of pick_up's 25 550
  decisions caught -- the ~2% still running and holding at the round's last
  step -- and essentially none of put_into's, because a successful placement
  ends RELEASED. So the composed-fraction quota classified nearly the whole
  bank as composed, and the 0.2/0.4/0.6 sweep realising 0.981 three times was
  that, not a fact about the bank.

  The collector's caught-start group mask, which made the uncaught-only
  approach gate a silent no-op: with the mask almost always False, `~caught`
  is almost always True and the gate read the same blended rate as before.

The fix is a snapshot at reset in both places, plus deriving the stratum from
`caught_target[0]` so recordings ALREADY on disk are read correctly.
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from tools.audit.sil_record import _Recording


def _recording(*, caught_first_step, stored_column, steps=6, per=2):
    """A recording whose stored column disagrees with its own step-0 truth."""

    worlds = len(caught_first_step)
    caught = np.zeros((steps, worlds), dtype=bool)
    caught[0] = np.asarray(caught_first_step, dtype=bool)
    return _Recording(
        actions=np.zeros((steps, worlds, 5), np.float32),
        active=np.ones((steps, worlds), bool),
        success=np.zeros((steps, worlds), bool),
        terminated=np.zeros((steps, worlds), bool),
        caught_target=caught,
        ee_xyz=np.zeros((steps, worlds, 3), np.float32),
        gripper_opening=np.zeros((steps, worlds), np.float32),
        object_xyz=np.zeros((steps, worlds, 2, 3), np.float32),
        instruction_ids=np.full((worlds,), 4, np.int64),
        target_slots=np.zeros((worlds,), np.int64),
        reference_slots=np.ones((worlds,), np.int64),
        second_reference_slots=np.full((worlds,), -1, np.int64),
        horizons=np.full((worlds,), steps // per, np.int64),
        initial_target_xyz=np.zeros((worlds, 3), np.float32),
        support_surface_z=np.zeros((worlds,), np.float32),
        release_threshold=np.full((worlds,), 0.55, np.float32),
        target_rest_height=np.zeros((worlds,), np.float32),
        physical_grasp_at_reset=np.asarray(stored_column, dtype=bool),
        instructions=np.array(["put the apple into the plate"] * worlds),
        actions_per_decision=per, round_index=0, diverged_worlds=0,
        pick_lift_success_height=0.05,
    )


class TheStratumComesFromStepZeroTests(unittest.TestCase):
    def test_it_ignores_the_stored_column_entirely(self) -> None:
        """Old recordings carry the wrong value there and must still read right."""

        rec = _recording(
            caught_first_step=[True, False, True],
            # What a post-rollout read produces: all False, because every
            # episode terminated.
            stored_column=[False, False, False],
        )
        np.testing.assert_array_equal(
            rec.starts_grasped, np.array([True, False, True])
        )

    def test_the_failure_mode_reproduced(self) -> None:
        # A successful placement ends released, so the stored column says
        # "composed" for a caught carry. That is the bug, stated as a case.
        rec = _recording(caught_first_step=[True], stored_column=[False])
        self.assertTrue(bool(rec.starts_grasped[0]))
        self.assertFalse(bool(rec.physical_grasp_at_reset[0]))

    def test_a_pick_up_that_ends_holding_is_still_an_uncaught_start(self) -> None:
        # The other direction: pick_up succeeds while HOLDING, so a late read
        # calls it caught. It began with nothing in the gripper.
        rec = _recording(caught_first_step=[False], stored_column=[True])
        self.assertFalse(bool(rec.starts_grasped[0]))

    def test_the_dataset_builder_uses_it(self) -> None:
        from tools.audit.sil_record import _build_dataset

        source = inspect.getsource(_build_dataset)
        self.assertIn("rec.starts_grasped[world]", source)
        self.assertNotIn("rec.physical_grasp_at_reset[world]", source)

    def test_the_decomposition_uses_it(self) -> None:
        from tools.audit import placement_failure_decomposition as module

        source = inspect.getsource(module._episode_terms)
        self.assertIn("recording.starts_grasped[world]", source)
        self.assertNotIn("recording.physical_grasp_at_reset[world]", source)


class TheSnapshotIsTakenBeforeTheRolloutTests(unittest.TestCase):
    def test_the_collector_clones_at_reset(self) -> None:
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            RankLocalMJWarpGRPOCollector,
        )

        source = inspect.getsource(RankLocalMJWarpGRPOCollector.collect_round)
        clone = source.index("caught_at_reset = reset.physical_grasp.detach().clone()")
        loop = source.index("for decision in range(max_decisions):")
        use = source.index("caught_at_reset.to(dtype=torch.bool)")
        self.assertLess(clone, loop, "the snapshot must precede the rollout")
        self.assertLess(loop, use)
        # And the live value must not be read for the mask any more.
        mask = source[use - 200 : use + 200]
        self.assertNotIn("reset.physical_grasp.to(dtype=torch.bool)", mask)

    def test_the_recorder_clones_in_its_reset_hook(self) -> None:
        from tools.audit.sil_record import _RoundRecorder

        source = inspect.getsource(_RoundRecorder)
        self.assertIn("self.caught_at_reset = _host_bool(reset.physical_grasp).copy()", source)
        self.assertIn("physical_grasp_at_reset=self.caught_at_reset,", source)

    def test_the_detector_still_owns_the_live_value(self) -> None:
        """The fix must not stop the grasp detector writing its own state."""

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            RankLocalMJWarpGRPOCollector,
        )

        source = inspect.getsource(
            RankLocalMJWarpGRPOCollector._update_physical_grasp
        )
        self.assertIn("reset.physical_grasp.copy_(physical_grasp)", source)


if __name__ == "__main__":
    unittest.main()
