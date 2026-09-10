import contextlib
import copy
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID
from tools.audit.sil_record import _Recording
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import vla_capture_world_indices
from tools.audit.extract_cdpr_transition_demonstrations import main as extract
from tools.audit.probe_cdpr_demonstration_handoff import (
    clone_reset_groups, collect_suffix_once, main, plan_boundaries, run_job, sha256,
)


def grouped_recording():
    # Two scene groups of eight, plate and bowl: eight accepted candidates
    # from one scene must not become eight demo groups.
    # Keep the fixture local: importing another test via "tests.*" depends
    # on which unrelated package named "tests" happens to be installed.
    steps, worlds = 8, 16
    xyz = np.zeros((steps, worlds, 2, 3), dtype=np.float32)
    xyz[..., 2] = .10
    xyz[3, :, 0, 2] = .12
    xyz[4:, :, 0, 2] = .16
    held = np.zeros((steps, worlds), dtype=bool)
    held[2:6] = True
    success = np.zeros_like(held)
    success[6, :8] = True
    active = np.ones_like(held)
    active[7, :8] = False
    opening = np.zeros((steps, worlds))
    opening[6:] = .8
    return _Recording(
        actions=np.zeros((steps, worlds, 5)), active=active, success=success,
        terminated=success.copy(), caught_target=held, ee_xyz=np.zeros((steps, worlds, 3)),
        gripper_opening=opening, object_xyz=xyz,
        instruction_ids=np.repeat([INSTRUCTION_TO_ID['put_into_plate'], INSTRUCTION_TO_ID['put_into_bowl']], 8),
        target_slots=np.zeros(worlds, dtype=int), reference_slots=np.ones(worlds, dtype=int),
        second_reference_slots=np.full(worlds, -1), horizons=np.full(worlds, 40),
        initial_target_xyz=xyz[0, :, 0].copy(), support_surface_z=np.zeros(worlds),
        release_threshold=np.full(worlds, .55), target_rest_height=np.full(worlds, .10),
        physical_grasp_at_reset=np.zeros(worlds, dtype=bool),
        instructions=np.repeat(['put apple into plate', 'put tomato into bowl'], 8),
        actions_per_decision=4, round_index=0, diverged_worlds=0, pick_lift_success_height=.05,
    )


def long_recording():
    # Same two scene groups, stretched in time so several decision boundaries
    # precede the release. One boundary cannot exercise a backoff.
    steps, worlds = 16, 16
    xyz = np.zeros((steps, worlds, 2, 3), dtype=np.float32)
    xyz[..., 2] = .10
    xyz[3, :, 0, 2] = .12
    xyz[4:, :, 0, 2] = .16
    held = np.zeros((steps, worlds), dtype=bool)
    held[2:14] = True
    success = np.zeros_like(held)
    success[14, :8] = True
    active = np.ones_like(held)
    active[15, :8] = False
    opening = np.zeros((steps, worlds))
    opening[14:] = .8
    return _Recording(
        actions=np.zeros((steps, worlds, 5)), active=active, success=success,
        terminated=success.copy(), caught_target=held, ee_xyz=np.zeros((steps, worlds, 3)),
        gripper_opening=opening, object_xyz=xyz,
        instruction_ids=np.repeat([INSTRUCTION_TO_ID['put_into_plate'], INSTRUCTION_TO_ID['put_into_bowl']], 8),
        target_slots=np.zeros(worlds, dtype=int), reference_slots=np.ones(worlds, dtype=int),
        second_reference_slots=np.full(worlds, -1), horizons=np.full(worlds, 40),
        initial_target_xyz=xyz[0, :, 0].copy(), support_surface_z=np.zeros(worlds),
        release_threshold=np.full(worlds, .55), target_rest_height=np.full(worlds, .10),
        physical_grasp_at_reset=np.zeros(worlds, dtype=bool),
        instructions=np.repeat(['put apple into plate', 'put tomato into bowl'], 8),
        actions_per_decision=4, round_index=0, diverged_worlds=0, pick_lift_success_height=.05,
    )


def episodes(r):
    from tools.audit.extract_cdpr_transition_demonstrations import select_episodes
    return select_episodes(r)[0]


@dataclass(frozen=True)
class ResetFixture:
    horizons: object
    instructions: tuple
    physical_grasp: object
    bilateral_contact_steps: object
    previous_relative_position: object
    task_state: object
    group_instruction_ids: object
    prelifted: object = None


@dataclass
class TaskFixture:
    grasped: object
    ever_grasped: object
    step_count: object
    initial_target_positions: object


class HandoffPlanningTests(unittest.TestCase):
    def test_prefix_is_a_count_at_boundary_before_success(self):
        r = grouped_recording()
        jobs, rejected = plan_boundaries(r, episodes(r), 'pick_up')
        self.assertEqual(rejected, {})
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['prefix_steps'], 4)
        self.assertEqual([e['world'] for e in jobs[0]['episodes']], [8, 0])
        self.assertTrue(all(e['pickup_success_env_step'] == 4 for e in jobs[0]['episodes']))
        # The replay stops after row 3; success-producing action row 4 is fresh.
        self.assertEqual(len(jobs[0]['episodes']), 2)

    def test_placement_requires_complete_placement_and_held_before_release(self):
        r = grouped_recording()
        jobs, rejected = plan_boundaries(r, episodes(r), 'placement')
        self.assertEqual(rejected, {'no_complete_placement': 8})
        self.assertEqual(jobs[0]['prefix_steps'], 4)
        self.assertEqual([e['world'] for e in jobs[0]['episodes']], [0])

    def test_boundary_backoff_steps_through_validated_boundaries_only(self):
        r = long_recording()
        stride = int(r.actions_per_decision)
        seen = [plan_boundaries(r, episodes(r), 'placement', boundary_backoff=b)[0][0]['prefix_steps']
                for b in (0, 1, 2, 99)]
        # Each step back lands on the previous validated boundary, and backing
        # off past the first clamps rather than inventing one.
        self.assertEqual(seen, [3 * stride, 2 * stride, stride, stride])
        for boundary in seen:
            self.assertTrue(r.caught_target[boundary - 1, 0])
            self.assertTrue(r.active[:boundary, 0].all())
            self.assertFalse(r.terminated[:boundary, 0].any())
        with self.assertRaises(ValueError):
            plan_boundaries(r, episodes(r), 'placement', boundary_backoff=-1)

    def test_lift_before_first_boundary_is_not_an_admissible_start(self):
        r = grouped_recording()
        eps = episodes(r)
        for ep in eps:
            ep['pickup_success_env_step'] = 3
        jobs, rejected = plan_boundaries(r, eps, 'pick_up')
        self.assertFalse(jobs)
        self.assertEqual(rejected['no_held_decision_boundary_before_event'], 16)

    def test_terminal_prefix_and_exhausted_budget_are_excluded(self):
        r = grouped_recording()
        eps = episodes(r)
        r.terminated[2, :8] = True
        r.horizons[8:] = 1
        jobs, rejected = plan_boundaries(r, eps, 'pick_up')
        self.assertFalse(jobs)
        self.assertEqual(sum(rejected.values()), 16)

    def test_mixed_instruction_group_is_rejected(self):
        r = grouped_recording()
        eps = episodes(r)
        r.instruction_ids[1] = 8
        with self.assertRaisesRegex(ValueError, 'mixes instruction'):
            plan_boundaries(r, eps, 'pick_up')

    def test_dry_run_checks_hashes_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot, bank, output = root / 'pilot', root / 'bank', root / 'output'
            source = pilot / 'baseline' / 'record_00.npz'
            grouped_recording().to_npz(source)
            checkpoint, config = root / 'donor.pt', root / 'config.yaml'
            checkpoint.write_bytes(b'not loaded in dry run')
            config.write_text('test: true\n')
            (pilot / 'pilot_manifest.json').write_text(json.dumps({
                'source': str(checkpoint), 'source_sha256': sha256(checkpoint),
                'config': str(config), 'config_sha256': sha256(config)}))
            with contextlib.redirect_stdout(io.StringIO()):
                extract(['--recordings', str(source), '--output', str(bank)])
                result = main(['--manifest', str(bank / 'manifest.json'),
                               '--pilot-run', str(pilot), '--output', str(output), '--dry-run'])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                main(['--manifest', str(bank / 'manifest.json'), '--pilot-run', str(pilot),
                      '--output', str(output), '--dry-run', '--source-instruction', 'put_into_bowl'])
            self.assertEqual(json.loads(capture.getvalue())['jobs'][0]['groups'], 1)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Provenance hash mismatch'):
                main(['--manifest', str(bank / 'manifest.json'), '--pilot-run', str(pilot),
                      '--output', str(output), '--dry-run'])


class VLACaptureTests(unittest.TestCase):
    def test_sparse_handoffs_after_world_127_are_captured(self):
        horizons = torch.zeros(512, dtype=torch.long)
        groups = [17, 46, 60]
        for group in groups:
            horizons[group * 8:group * 8 + 8] = 29
        indices = vla_capture_world_indices(horizons, 8, 128)
        self.assertEqual(indices.tolist(), [w for g in groups for w in range(g * 8, g * 8 + 8)])
        self.assertTrue(torch.all(horizons[indices] > 0))

    def test_ordinary_all_active_collection_keeps_previous_order_and_cap(self):
        for cap in (1, 9, 128, 1000):
            expected = max(8, min(cap, 512) // 8 * 8)
            actual = vla_capture_world_indices(torch.ones(512, dtype=torch.long), 8, cap)
            self.assertTrue(torch.equal(actual, torch.arange(expected)))

    def test_empty_or_disabled_capture_has_no_inactive_rows(self):
        self.assertEqual(vla_capture_world_indices(torch.zeros(512), 8, 128).numel(), 0)
        self.assertEqual(vla_capture_world_indices(torch.ones(512), 8, 0).numel(), 0)

    def test_partial_scene_group_is_rejected(self):
        horizons = torch.zeros(16)
        horizons[3] = 10
        with self.assertRaisesRegex(ValueError, 'mixes active'):
            vla_capture_world_indices(horizons, 8, 128)


class LiveHandoffTests(unittest.TestCase):
    def test_task_and_contact_history_are_cloned_with_the_physics_group(self):
        values = torch.arange(16)
        task = TaskFixture(values.clone(), values.clone(), values.clone(),
                           values[:, None].repeat(1, 3))
        reset = ResetFixture(values.clone(), tuple(map(str, range(16))), values.clone(),
                             values.clone(), values[:, None].repeat(1, 3), task,
                             torch.tensor([3, 4]))
        result = clone_reset_groups(reset, [3], 8, torch)
        self.assertEqual(result.instructions[:8], ('3',) * 8)
        self.assertTrue(torch.all(result.bilateral_contact_steps[:8] == 3))
        self.assertTrue(torch.all(result.previous_relative_position[:8] == 3))
        self.assertTrue(torch.all(result.task_state.step_count[:8] == 3))
        self.assertTrue(torch.equal(result.horizons[8:], values[8:]))
        self.assertTrue(torch.equal(result.group_instruction_ids, torch.tensor([3, 4])))

    def test_suffix_collection_begins_at_prepared_state_and_restores_resetter(self):
        class Resetter:
            def reset(self, **kwargs):
                raise AssertionError('ordinary reset would destroy handoff')
        resetter = Resetter()
        handoff = object()
        seen = []

        def collect(**kwargs):
            seen.append(resetter.reset(**kwargs))
            return {'fresh_records': [1, 2]}

        collector = SimpleNamespace(resetter=resetter, collect_round=collect)
        result = collect_suffix_once(collector, handoff, round_index=0)
        self.assertEqual(seen, [handoff])
        self.assertEqual(result, {'fresh_records': [1, 2]})
        self.assertNotIn('reset', vars(resetter))

    def test_failure_restores_an_existing_reset_override(self):
        original = lambda **kwargs: 'original'
        resetter = SimpleNamespace(reset=original)

        def collect(**kwargs):
            resetter.reset(**kwargs)
            resetter.reset(**kwargs)

        collector = SimpleNamespace(resetter=resetter, collect_round=collect)
        with self.assertRaisesRegex(RuntimeError, 'once'):
            collect_suffix_once(collector, object(), round_index=0)
        self.assertIs(resetter.reset, original)

    def test_replay_handoff_and_fresh_collection_with_real_task_predicate(self):
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedTaskState, BatchedCatchReleaseDenseReward,
        )
        r = grouped_recording()
        jobs, _ = plan_boundaries(r, episodes(r), 'pick_up')
        task = BatchedTaskState(
            instruction_ids=torch.tensor(r.instruction_ids), target_slots=torch.tensor(r.target_slots),
            reference_slots=torch.tensor(r.reference_slots), second_reference_slots=torch.tensor(r.second_reference_slots),
            initial_target_positions=torch.tensor(r.initial_target_xyz), ever_grasped=torch.zeros(16, dtype=torch.bool),
            grasped=torch.zeros(16, dtype=torch.bool), step_count=torch.zeros(16, dtype=torch.long),
            release_threshold=torch.tensor(r.release_threshold), support_surface_z=torch.tensor(r.support_surface_z),
            target_rest_height=torch.tensor(r.target_rest_height))
        reset = ResetFixture(torch.tensor(r.horizons), tuple(r.instructions),
                             torch.zeros(16, dtype=torch.bool), torch.zeros(16, dtype=torch.long),
                             torch.zeros(16, 3), task, torch.tensor([4, 3]))
        # No catalogs in this source fixture, so the live optional field is unused.
        low = SimpleNamespace(object_positions=torch.tensor(r.object_xyz[0]),
                              ee_position=torch.tensor(r.ee_xyz[0]), gripper_opening=torch.tensor(r.gripper_opening[0]))
        class Backend:
            _qpos = torch.zeros(16, 5)
            _qvel = torch.zeros(16, 5)
            steps = 0
            pose_offset = 0.
            def low_dim_observations(self):
                return low
            def step(self, actions, active):
                index = self.steps
                self.steps += 1
                low.object_positions = torch.tensor(r.object_xyz[index]) + self.pose_offset
                low.ee_position = torch.tensor(r.ee_xyz[index])
                low.gripper_opening = torch.tensor(r.gripper_opening[index])
                return low
            def pop_nonfinite_world_events(self):
                return 0
            def pop_nonfinite_world_report(self):
                return 0, np.zeros(16, dtype=bool)
            def broadcast_group_state(self, representatives):
                for w in representatives.tolist():
                    start = w // 8 * 8
                    for v in (low.object_positions, low.ee_position, low.gripper_opening):
                        v[start:start + 8] = v[w].clone()
            def controller_state(self):
                return {'target': np.zeros((16, 3)), 'gripper': np.zeros(16)}
        backend = Backend()
        resetter = SimpleNamespace(reset=lambda **kwargs: copy.deepcopy(reset))
        def grasp(current, obs, active):
            caught = torch.tensor(r.caught_target[backend.steps - 1])
            current.physical_grasp.copy_(caught)
            return obs, caught, {'bilateral_contact': caught}
        calls = []
        def collect(**kwargs):
            prepared = resetter.reset(**kwargs)
            calls.append(prepared)
            self.assertEqual(backend.steps, 4)  # All teacher steps precede record collection.
            if len(calls) == 1:  # The canonical two-group pick_up handoff.
                self.assertTrue(torch.all(prepared.task_state.instruction_ids == 8))
                self.assertTrue(torch.all(prepared.horizons == 39))  # Four actions consume one decision.
                self.assertTrue(torch.all(prepared.task_state.ever_grasped))
            return SimpleNamespace(metrics={}, loss_mask=torch.ones(16), vla_records=None,
                                   candidate_rewards=torch.zeros(2, 8), candidate_success=torch.zeros(2, 8, dtype=torch.bool))
        collector = SimpleNamespace(actions_per_policy_decision=4, resetter=resetter,
                                    _update_physical_grasp=grasp, _task_thresholds=lambda: None,
                                    move_to_distance_reward=None,
                                    catch_release_dense_reward=BatchedCatchReleaseDenseReward(), collect_round=collect)
        world = SimpleNamespace(torch=torch, backend=backend, collector=collector,
                                resetter=resetter, device=torch.device('cpu'))
        report = run_job(world, r, jobs[0], 'pick_up', group_size=8,
                         position_tolerance=.002, opening_tolerance=.03, suffix_decisions=40, seed_torch=0)
        self.assertEqual(report['status'], 'fresh_suffixes_collected_no_training')
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(len(report['groups']), 2)
        self.assertEqual(len(calls), 1)
        # Replay drift must stop BEFORE any fresh suffix is collected.
        backend.steps = 0
        backend.pose_offset = .01
        low.object_positions = torch.tensor(r.object_xyz[0])
        rejected = run_job(world, r, jobs[0], 'pick_up', group_size=8,
                           position_tolerance=.002, opening_tolerance=.03, suffix_decisions=40, seed_torch=0)
        self.assertEqual(rejected['status'], 'no_verified_handoffs')
        self.assertTrue(all('replay_pose_mismatch' in e['rejected'] for e in rejected['episodes']))
        self.assertEqual(len(calls), 1)
        # A destination success at admission must never become a free reward.
        backend.steps = 0
        backend.pose_offset = 0.
        low.object_positions = torch.tensor(r.object_xyz[0])
        collector.catch_release_dense_reward = BatchedCatchReleaseDenseReward(pick_lift_success_height=.01)
        terminal = run_job(world, r, jobs[0], 'pick_up', group_size=8,
                           position_tolerance=.002, opening_tolerance=.03, suffix_decisions=40, seed_torch=0)
        self.assertEqual(terminal['status'], 'destination_task_already_terminal')
        self.assertEqual(len(calls), 1)
        # One already-satisfied destination must not abandon the whole batch:
        # its group is dropped and the remaining groups still collect.
        collector.catch_release_dense_reward = BatchedCatchReleaseDenseReward()
        backend.steps = 0
        low.object_positions = torch.tensor(r.object_xyz[0])
        # Only the second group's datum sits far enough below the object for
        # its relabelled pick_up to be satisfied at the handoff. The offset is
        # chosen so the 5 cm crossing happens AT the last prefix state (z .12
        # against a .07 datum) and not before it: an earlier crossing is the
        # teacher earning the lift, which the per-episode guard rejects, while
        # a crossing that stands at the handoff is what the group drop is for.
        low.object_positions[8:, 0, 2] -= .035
        partial = run_job(world, r, jobs[0], 'pick_up', group_size=8,
                          position_tolerance=.002, opening_tolerance=.03,
                          suffix_decisions=40, seed_torch=0, lift_datum_tolerance=.06)
        self.assertEqual(partial['status'], 'fresh_suffixes_collected_no_training')
        self.assertEqual(partial['destination_terminal_groups'], [1])
        self.assertEqual([g['group'] for g in partial['groups']], [0])
        self.assertIn('destination_task_already_terminal',
                      [reason for e in partial['episodes'] if e['group'] == 1
                       for reason in e['rejected']])
        # The recording's row 0 is stored AFTER the first env step, so a gap
        # between it and the reset pose is the object's settle, not replay
        # drift. It must not be judged by the 2 mm replay tolerance, and the
        # live datum is what pick_up actually lifts against.
        collector.catch_release_dense_reward = BatchedCatchReleaseDenseReward()
        for offset, tolerance, expected, gated in (
            (-.008, .02, 'fresh_suffixes_collected_no_training', False),
            (-.030, .02, 'no_verified_handoffs', True),
        ):
            backend.steps = 0
            low.object_positions = torch.tensor(r.object_xyz[0]) + offset
            result = run_job(world, r, jobs[0], 'pick_up', group_size=8,
                             position_tolerance=.002, opening_tolerance=.03,
                             suffix_decisions=40, seed_torch=0,
                             lift_datum_tolerance=tolerance)
            self.assertEqual(result['status'], expected)
            for entry in result['episodes']:
                self.assertAlmostEqual(entry['baseline_error_m'], abs(offset), places=6)
                self.assertEqual('reset_vs_first_post_action_lift_datum' in entry['rejected'], gated)
        # A lift the teacher earned INSIDE the prefix is rejected per episode,
        # before any state is broadcast: the destination would otherwise be
        # handed a pick_up its own actions had already solved.
        collector.catch_release_dense_reward = BatchedCatchReleaseDenseReward()
        backend.steps = 0
        collected_so_far = len(calls)
        low.object_positions = torch.tensor(r.object_xyz[0]) - .06
        early = run_job(world, r, jobs[0], 'pick_up', group_size=8,
                        position_tolerance=.002, opening_tolerance=.03,
                        suffix_decisions=40, seed_torch=0, lift_datum_tolerance=.08)
        self.assertEqual(early['status'], 'no_verified_handoffs')
        self.assertTrue(all('pickup_already_succeeded_in_prefix' in e['rejected']
                            for e in early['episodes']))
        self.assertTrue(all(e['live_datum_lift_env_step'] < jobs[0]['prefix_steps'] - 1
                            for e in early['episodes']))
        self.assertEqual(len(calls), collected_so_far)
        # Placement success reads no Z lift datum, so it is never gated on one.
        placement, _ = plan_boundaries(r, episodes(r), 'placement')
        backend.steps = 0
        low.object_positions = torch.tensor(r.object_xyz[0]) - .030
        result = run_job(world, r, placement[0], 'placement', group_size=8,
                         position_tolerance=.002, opening_tolerance=.03,
                         suffix_decisions=40, seed_torch=0, lift_datum_tolerance=.02)
        self.assertTrue(all('reset_vs_first_post_action_lift_datum' not in e['rejected']
                            for e in result['episodes']))


if __name__ == '__main__':
    unittest.main()
