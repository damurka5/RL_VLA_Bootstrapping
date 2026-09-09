import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.audit.compare_release_recovery_checkpoints import (
    cap_report, comparison_exit_code, inspect_existing, main, outcome_counts,
    pairing_issues, resolve_run, select_checkpoints, summarize,
)
import json
from tools.audit.sil_record import _Recording, _instruction_name
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID


def recording():
    names = ['move_to_object', 'pick_up', 'put_into_plate', 'put_into_bowl']
    ids = np.repeat([INSTRUCTION_TO_ID[name] for name in names], 8)
    success = np.zeros((2, 32), dtype=bool)
    success[1, [0, 8, 16, 24]] = True
    return _Recording(
        actions=np.zeros((2, 32, 5)), active=np.ones((2, 32), dtype=bool),
        success=success, terminated=success.copy(), caught_target=np.zeros((2, 32), dtype=bool),
        ee_xyz=np.zeros((2, 32, 3)), gripper_opening=np.zeros((2, 32)),
        object_xyz=np.zeros((2, 32, 2, 3)), instruction_ids=ids,
        target_slots=np.zeros(32, dtype=int), reference_slots=np.ones(32, dtype=int),
        second_reference_slots=np.full(32, -1), horizons=np.full(32, 40),
        initial_target_xyz=np.zeros((32, 3)), support_surface_z=np.zeros(32),
        release_threshold=np.full(32, .55), target_rest_height=np.full(32, .03),
        physical_grasp_at_reset=np.zeros(32, dtype=bool), instructions=np.repeat(names, 8),
        actions_per_decision=4, round_index=0, diverged_worlds=0, pick_lift_success_height=.05,
        target_catalog_ids=np.zeros(32, dtype=int),
    )


class CompareReleaseRecoveryTests(unittest.TestCase):
    def test_numerically_latest_checkpoint_and_two_independent_peaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for step in (999999, 1505251, 2117145, 3540000):
                p = run / 'rl' / f'step_{step}' / 'smolvla_grpo_adapter.pt'
                p.parent.mkdir(parents=True)
                p.touch()
            selected = select_checkpoints(run, 1505251, 2117145, 3527307)
            self.assertEqual(selected['final'].parent.name, 'step_3540000')
            self.assertNotEqual(selected['plate_peak'], selected['bowl_peak'])
            with self.assertRaisesRegex(ValueError, 'below'):
                select_checkpoints(run, 1505251, 2117145, 4000000)
            with self.assertRaisesRegex(ValueError, 'Missing peak'):
                select_checkpoints(run, 123, 2117145, 3527307)

    def test_ambiguous_run_requires_explicit_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix in ('one', 'two'):
                (root / 'runs' / f'release_recovery_continue_3m_{suffix}' / 'rl').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                resolve_run(None, root)

    def test_dry_run_preserves_config_location_and_does_not_write_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for step in (1505251, 2117145, 3540000):
                path = run / 'rl' / f'step_{step}' / 'smolvla_grpo_adapter.pt'
                path.parent.mkdir(parents=True)
                path.touch()
            output = run / 'evaluation'
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                self.assertEqual(main(['--run-dir', str(run), '--output', str(output), '--dry-run']), 0)
            self.assertFalse(output.exists())
            text = capture.getvalue()
            self.assertEqual(text.count('--mode record'), 3)
            self.assertNotIn('--start-distance-cap', text)
            self.assertNotIn('--horizon-decisions', text)
            self.assertNotIn('evaluation_config.yaml', text)
            self.assertEqual(text.count('--devices cuda:0,cuda:1'), 3)

    def test_scene_check_uses_actual_layout_and_not_stale_initial_target(self):
        a, b = recording(), recording()
        b.initial_target_xyz += 1
        self.assertEqual(pairing_issues(a, b), [])
        b.object_xyz[0, 0, 0, 0] += .01
        self.assertIn('object_xyz_at_step_0', pairing_issues(a, b))

    def test_horizon_and_assisted_starts_are_rejected(self):
        a = recording()
        a.horizons[16:] = 48
        with self.assertRaisesRegex(ValueError, '40 decisions'):
            outcome_counts(a, _instruction_name)
        a = recording()
        a.caught_target[0, 16] = True
        with self.assertRaisesRegex(ValueError, 'uncaught'):
            outcome_counts(a, _instruction_name)

    def test_three_arm_summary_and_mismatch_suppresses_paired_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = recording()
            for arm in ('final', 'plate_peak', 'bowl_peak'):
                r = copy.deepcopy(base)
                if arm == 'final':
                    r.success[1, 9:12] = True
                if arm == 'bowl_peak':
                    r.object_xyz[0, 0, 0, 0] = .02
                r.to_npz(root / arm / 'record_00.npz')
            with contextlib.redirect_stdout(io.StringIO()):
                report = summarize(root, 1, 0)
            self.assertEqual(report['arms']['final']['pick_up']['rate'], .5)
            delta = report['final_minus_peak']['plate_peak']['pick_up']
            self.assertEqual(delta['delta'], .375)
            self.assertEqual(delta['paired_verdicts'], {'final_only': 3, 'peak_only': 0})
            self.assertIsNone(report['final_minus_peak']['bowl_peak']['pick_up']['paired_verdicts'])
            self.assertEqual(comparison_exit_code(report), 0)
            report['pairing']['bowl_peak']['rounds'][0]['issues'].append('horizons')
            self.assertEqual(comparison_exit_code(report), 2)
            self.assertTrue((root / 'comparison.json').exists())
            before = (root / 'comparison.json').read_bytes()
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                self.assertEqual(inspect_existing(root), 0)
            self.assertIn('POST-ACTION', capture.getvalue())
            self.assertEqual(before, (root / 'comparison.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'expected 2 recordings'):
                summarize(root, 2, 0)
            # No summary.json in this fixture, so no arm can certify its caps.
            for arm in ('final', 'plate_peak', 'bowl_peak'):
                self.assertFalse(report['caps'][arm]['cap_check_available'])
                self.assertIsNone(report['caps'][arm]['applied_start_distance_cap'])
            self.assertIn('UNVERIFIED', (root / 'comparison.md').read_text())


class CapRecordingTests(unittest.TestCase):
    """A rate whose cap is unknown cannot be read; §7.7.

    `--start-distance-cap` applies to every instruction while the approach
    ladders end at different rungs, so one evaluation is routinely at the right
    cap for the families it was aimed at and above the earned cap for the rest.
    That has inverted a conclusion twice, which is why the cap travels with the
    number rather than with the console log.
    """

    def _arm(self, root, cap, verdicts=None):
        r = recording()
        r.start_distance_cap = cap
        r.to_npz(root / 'final' / 'record_00.npz')
        for other in ('plate_peak', 'bowl_peak'):
            copy.deepcopy(r).to_npz(root / other / 'record_00.npz')
            if verdicts is not None:
                (root / other / 'summary.json').write_text(json.dumps({'cap_check': verdicts}))
        if verdicts is not None:
            (root / 'final' / 'summary.json').write_text(json.dumps({'cap_check': verdicts}))

    def test_the_applied_cap_is_read_from_the_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._arm(root, .17)
            with contextlib.redirect_stdout(io.StringIO()):
                report = summarize(root, 1, 0)
            self.assertEqual(report['caps']['final']['applied_start_distance_cap'], .17)

    def test_an_above_earned_cap_instruction_is_named(self):
        verdicts = {'pick_up': {'earned_cap': .06, 'requested_cap': .2,
                                'verdict': 'above_earned_cap'},
                    'put_into_plate': {'earned_cap': .2, 'requested_cap': .2,
                                       'verdict': 'at_earned_cap'}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._arm(root, .2, verdicts)
            with contextlib.redirect_stdout(io.StringIO()):
                report = summarize(root, 1, 0)
            self.assertEqual(report['caps']['final']['above_earned_cap'], ['pick_up'])
            self.assertTrue(report['caps']['final']['cap_check_available'])
            markdown = (root / 'comparison.md').read_text()
            self.assertIn('ABOVE EARNED CAP: pick_up', markdown)

    def test_rounds_disagreeing_on_the_cap_are_fatal(self):
        """Two caps in one arm is two reset distributions, not one rate."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ('final', 'plate_peak', 'bowl_peak'):
                for index, cap in enumerate((.10, .20)):
                    r = recording()
                    r.round_index = index
                    r.start_distance_cap = cap
                    r.to_npz(root / arm / f'record_{index:02d}.npz')
            with self.assertRaisesRegex(ValueError, 'disagree on the applied start-distance cap'):
                with contextlib.redirect_stdout(io.StringIO()):
                    summarize(root, 2, 0)

    def test_a_missing_cap_check_is_reported_absent_not_assumed_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'final').mkdir(parents=True)
            entry = cap_report(root / 'final', .2)
            self.assertFalse(entry['cap_check_available'])
            self.assertEqual(entry['above_earned_cap'], [])
            self.assertEqual(entry['applied_start_distance_cap'], .2)


if __name__ == '__main__':
    unittest.main()
