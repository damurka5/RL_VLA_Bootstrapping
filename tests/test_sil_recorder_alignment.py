"""Recorder/collector handoff when containment resets a world inside step."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rl_vla_bootstrapping.policy import mjwarp_rank_local_collector as collector_module
from tools.audit.sil_record import _Recording, _RoundRecorder


class Backend:
    def __init__(self, diverge):
        self.diverge = diverge
        self.seen = torch.zeros(3, dtype=torch.bool)
        self.events = 5  # Stale events from a previous round must be ignored.
        self.step_index = 0

    def low_dim_observations(self):
        return SimpleNamespace(ee_position=torch.zeros(3, 3),
                               object_positions=torch.zeros(3, 2, 3),
                               gripper_opening=torch.ones(3))

    def step(self, actions, active_mask):
        if self.diverge:
            # World 1 diverges live, world 2 after its episode ended. World 1
            # diverges again on step 2: event counts differ from world counts.
            self.seen[1:] = True
            self.events += 2 if self.step_index == 0 else 1
        self.step_index += 1
        return self.low_dim_observations()

    def nonfinite_world_mask(self):
        return self.seen.clone()

    def pop_nonfinite_world_events(self):
        # Same delegation as the real backend, with a count-only fallback.
        report = self.pop_nonfinite_world_report
        if report is not None:
            return report()[0]
        count, self.events = self.events, 0
        self.seen.zero_()
        return count

    def pop_nonfinite_world_report(self):
        count, self.events = self.events, 0
        mask = self.seen.numpy().copy()
        self.seen.zero_()
        return count, mask


class Resetter:
    def __init__(self, backend):
        self.backend = backend

    def reset(self):
        self.backend.step_index = 0
        return SimpleNamespace(
            horizons=torch.full((3,), 2), physical_grasp=torch.zeros(3, dtype=torch.bool),
            instructions=["move to apple"] * 3,
            task_state=SimpleNamespace(
                instruction_ids=torch.zeros(3, dtype=torch.int64),
                target_slots=torch.zeros(3, dtype=torch.int64),
                reference_slots=torch.ones(3, dtype=torch.int64),
                second_reference_slots=torch.full((3,), -1, dtype=torch.int64),
                initial_target_positions=torch.zeros(3, 3), support_surface_z=torch.zeros(3),
                release_threshold=torch.ones(3), target_rest_height=torch.zeros(3),
            ),
        )


class Trainer:
    def deterministic_action_chunks_tensor(self, *, states, priors, action_count):
        return torch.zeros(3, action_count, 5)


class Collector:
    actions_per_policy_decision = 1

    def __init__(self, backend, mismatch):
        self.backend = backend
        self.mismatch = mismatch
        self.resetter = Resetter(backend)
        self.trainer = Trainer()

    def validate_round(self, *, round_index):
        guard = collector_module.NonFiniteEpisodeGuard(self.backend, torch, 3, "cpu")
        reset = self.resetter.reset()
        active = torch.tensor([True, True, False])
        for _ in range(2):
            actions = self.trainer.deterministic_action_chunks_tensor(
                states=torch.zeros(3, 4), priors=torch.zeros(3, 1, 5), action_count=1)
            low = self.backend.step(actions[:, 0], active.clone())
            scored = guard.after_step(active)
            active &= ~guard.live
            if self.mismatch == "drop_healthy":
                scored[0] = False
            elif self.mismatch == "score_idle":
                scored[2] = True
            collector_module.evaluate_active_sparse_tasks(
                state=reset.task_state, active_mask=scored, caught_target=reset.physical_grasp,
                ee_position=low.ee_position, object_positions=low.object_positions,
                gripper_opening=low.gripper_opening)
        guard.finish()  # Consumes both count and mask before recorder.run returns.


def world(*, diverge=True, mismatch=None, legacy=False):
    backend = Backend(diverge)
    if legacy:
        backend.nonfinite_world_mask = None
        backend.pop_nonfinite_world_report = None
    return SimpleNamespace(torch=torch, backend=backend, collector=Collector(backend, mismatch),
                           layout=SimpleNamespace(worlds_per_rank=3))


class RecorderAlignmentTests(unittest.TestCase):
    def run_recording(self, fixture):
        def predicate(**kwargs):
            active = kwargs["active_mask"]
            return SimpleNamespace(success=active.clone(), terminated=torch.zeros_like(active))

        with patch.object(collector_module, "evaluate_active_sparse_tasks", side_effect=predicate) as original:
            try:
                with _RoundRecorder(fixture) as recorder:
                    result = recorder.run(round_index=4)
                    self.assertEqual(collector_module.evaluate_active_sparse_tasks.__name__, "patched_predicate")
            finally:
                self.assertIs(collector_module.evaluate_active_sparse_tasks, original)
                self.assertNotIn("step", vars(fixture.backend))
                self.assertNotIn("pop_nonfinite_world_events", vars(fixture.backend))
                self.assertNotIn("reset", vars(fixture.collector.resetter))
                self.assertNotIn("deterministic_action_chunks_tensor", vars(fixture.collector.trainer))
        return result

    def test_divergence_is_allowed_and_report_survives_collectors_pop(self):
        rec = self.run_recording(world())
        np.testing.assert_array_equal(rec.active, [[True, True, False], [True, False, False]])
        np.testing.assert_array_equal(rec.success, [[True, False, False], [True, False, False]])
        np.testing.assert_array_equal(rec.diverged_world_mask, [False, True, True])
        self.assertEqual(rec.diverged_worlds, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.npz"
            rec.to_npz(path)
            saved = _Recording.from_npz(path)
        np.testing.assert_array_equal(saved.diverged_world_mask, rec.diverged_world_mask)
        self.assertEqual(saved.diverged_worlds, 3)

    def test_equal_masks_without_divergence_remain_valid(self):
        rec = self.run_recording(world(diverge=False))
        self.assertFalse(rec.diverged_world_mask.any())
        self.assertEqual(rec.diverged_worlds, 0)

    def test_unexplained_mask_changes_still_fail_even_in_a_diverged_round(self):
        for mismatch in ("drop_healthy", "score_idle"):
            with self.subTest(mismatch=mismatch), self.assertRaisesRegex(RuntimeError, "misaligned"):
                self.run_recording(world(mismatch=mismatch))

    def test_legacy_count_only_backend_retains_events_and_unknown_mask(self):
        rec = self.run_recording(world(legacy=True))
        self.assertEqual(rec.diverged_worlds, 3)
        self.assertIsNone(rec.diverged_world_mask)

    def test_repeated_rounds_do_not_inherit_divergence_reports(self):
        fixture = world()
        self.assertEqual(self.run_recording(fixture).diverged_worlds, 3)
        fixture.backend.diverge = False
        rec = self.run_recording(fixture)
        self.assertEqual(rec.diverged_worlds, 0)
        self.assertFalse(rec.diverged_world_mask.any())


if __name__ == "__main__":
    unittest.main()
