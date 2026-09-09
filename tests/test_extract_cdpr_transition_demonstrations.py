import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.audit.extract_cdpr_transition_demonstrations import main, select_episodes
from tools.audit.sil_record import _Recording
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID


def recording():
    steps, worlds = 8, 3
    xyz = np.zeros((steps, worlds, 2, 3), dtype=np.float32)
    xyz[..., 2] = .10
    xyz[3, :2, 0, 2] = .12
    xyz[4:, :2, 0, 2] = .16
    xyz[3:, 2, 0, 2] = .30  # Airborne without a grasp must not count.
    held = np.zeros((steps, worlds), dtype=bool)
    held[2:6, :2] = True
    success = np.zeros_like(held)
    success[6, 0] = True
    active = np.ones_like(held)
    active[7, 0] = False
    return _Recording(
        actions=np.zeros((steps, worlds, 5)), active=active, success=success,
        terminated=success.copy(), caught_target=held, ee_xyz=np.zeros((steps, worlds, 3)),
        gripper_opening=np.zeros((steps, worlds)), object_xyz=xyz,
        instruction_ids=np.array([INSTRUCTION_TO_ID['put_into_plate'],
                                  INSTRUCTION_TO_ID['put_into_bowl'], INSTRUCTION_TO_ID['put_into_plate']]),
        target_slots=np.zeros(worlds, dtype=int), reference_slots=np.ones(worlds, dtype=int),
        second_reference_slots=np.full(worlds, -1), horizons=np.full(worlds, 40),
        initial_target_xyz=xyz[0, :, 0].copy(), support_surface_z=np.zeros(worlds),
        release_threshold=np.full(worlds, .55), target_rest_height=np.full(worlds, .10),
        physical_grasp_at_reset=np.zeros(worlds, dtype=bool),
        instructions=np.array(['put apple into plate', 'put tomato into bowl', 'put apple into plate']),
        actions_per_decision=4, round_index=0, diverged_worlds=0, pick_lift_success_height=.05,
    )


class TransitionDemonstrationTests(unittest.TestCase):
    def test_keeps_successful_lift_from_failed_placement_and_excludes_toss(self):
        episodes, rejected = select_episodes(recording())
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]['pickup_success_env_step'], 4)
        self.assertEqual(episodes[0]['placement_success_env_step'], 6)
        self.assertEqual(episodes[1]['pickup_prompt'], 'pick up tomato')
        self.assertTrue(episodes[1]['placement_prefix_only'])
        self.assertEqual(rejected['no_pickup_success'], 1)

    def test_caught_and_raised_starts_are_excluded(self):
        r = recording()
        r.caught_target[0, 0] = True
        r.object_xyz[0, 1, 0, 2] = .15
        episodes, rejected = select_episodes(r)
        self.assertEqual(episodes, [])
        self.assertEqual(rejected['starts_grasped'], 1)
        self.assertEqual(rejected['not_a_desk_start'], 1)

    def test_missing_receptacle_is_excluded(self):
        r = recording()
        r.reference_slots[1] = r.target_slots[1]
        _, rejected = select_episodes(r)
        self.assertEqual(rejected['missing_distinct_receptacle'], 1)

    def test_a_stale_production_baseline_is_recorded_not_rejected(self):
        """`initial_target_positions` is stale for uncaught_container starts.

        The resetter updates it for `held_group` and `grasp_learning` and not
        for composed container starts, so on the population this file exists to
        harvest it holds a pre-repositioning lattice point. Rejecting on it
        threw away every composed episode -- the guard excluded exactly what it
        was written to protect. The lift is scored from the recorded start pose
        instead, and the production datum's disagreement is reported.
        """

        r = recording()
        r.initial_target_xyz[0, 2] = -.2
        episodes, rejected = select_episodes(r)
        self.assertNotIn('inconsistent_lift_baseline', rejected)
        chosen = [e for e in episodes if e['world'] == 0]
        self.assertEqual(len(chosen), 1)
        # object_xyz[0, 0, target, 2] is 0.10, so the disagreement is 0.30 m.
        self.assertAlmostEqual(
            chosen[0]['production_lift_baseline_delta_m'], .30, places=5
        )

    def test_the_lift_is_measured_from_the_recorded_start_pose(self):
        """A stale baseline must not decide the verdict in either direction.

        Set it 0.2 m ABOVE the desk: scored against it, a real 0.06 m lift
        reads as no lift at all and the episode disappears. Scored against
        where the object actually was, it is the same demonstration it was
        before the field was corrupted.
        """

        r = recording()
        before = [e['world'] for e in select_episodes(r)[0]]
        r.initial_target_xyz[:, 2] += .2
        after = [e['world'] for e in select_episodes(r)[0]]
        self.assertEqual(before, after)
        self.assertTrue(before)

    def test_inactive_success_and_open_gripper_do_not_count(self):
        r = recording()
        r.active[4:, 0] = False
        r.gripper_opening[4:, 1] = .95
        episodes, rejected = select_episodes(r)
        self.assertEqual(episodes, [])

    def test_reported_divergence_quarantines_recording(self):
        r = recording()
        r.diverged_worlds = 1
        self.assertEqual(select_episodes(r), ([], {'recording_reported_divergence': 3}))

    def test_export_truncates_actions_and_keeps_link_between_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'record_00.npz'
            recording().to_npz(source)
            duplicate = root / 'copied.npz'
            duplicate.write_bytes(source.read_bytes())
            output = root / 'bank'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--recordings', str(source), str(duplicate), '--output', str(output)]), 0)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['counts']['pickup_clips'], 2)
            self.assertEqual(manifest['counts']['complete_placement_clips'], 1)
            self.assertEqual(len(manifest['sources']), 1)
            episode = manifest['episodes'][0]
            with np.load(output / episode['pickup_clip']) as clip:
                self.assertEqual(clip['executed_actions'].shape, (5, 5))
                self.assertEqual(clip['instruction_text'].item(), 'pick up apple')
                self.assertNotIn('prior', clip.files)
                self.assertEqual(clip['source_episode_uid'].item(), episode['episode_uid'])
            with np.load(output / episode['placement_clip']) as clip:
                self.assertEqual(clip['executed_actions'].shape, (7, 5))
                self.assertEqual(clip['instruction_text'].item(), 'put apple into plate')


if __name__ == '__main__':
    unittest.main()
