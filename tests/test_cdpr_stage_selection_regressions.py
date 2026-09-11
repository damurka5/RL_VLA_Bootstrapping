"""Regressions behind the 0.058 m pickup gap and mixed-stage teacher scores."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import (
    sample_staged_teacher_actions, projected_grasp_xy_offset,
    TeacherBank, TeacherEntry, PickupReadiness, PickupYawCalibration,
    StageMachine, StageBudgets, STAGE_ALIGN,
    STAGE_PICK_UP, YawTailController, _apply_overrides, SOURCE_YAW_TAIL,
    StagedRound,
)
from tools.audit.select_cdpr_stage_teachers import score_rounds, main
from tools.audit.xy_approach_probe import _set_config_controller_workspace


class PhysicalFloorTests(unittest.TestCase):
    def test_config_overrides_missing_or_high_donor_floor(self):
        project = load_project_config(Path(__file__).resolve().parents[1] /
            'configs/examples/cdpr_smolvla_three_stage_put_into.yaml')
        for saved in (None, [0.25, 0.6], [0.3, 0.5]):
            args = Namespace(controller_workspace_z_bounds=saved)
            _set_config_controller_workspace(args, project.training.rl.args)
            self.assertEqual(args.controller_workspace_z_bounds, [0.18, 0.6])
            self.assertLess(args.controller_workspace_z_bounds[0], 0.192)

    def test_invalid_explicit_bounds_fail_instead_of_falling_back(self):
        for bounds in (None, [], [0.3], [0.6, 0.18], [float('nan'), 0.6]):
            with self.assertRaises(ValueError):
                _set_config_controller_workspace(Namespace(),
                    {'controller_workspace_z_bounds': bounds})


class TeacherRoutingTests(unittest.TestCase):
    def sample(self, *, noisy=False):
        bank = SimpleNamespace(active_role=None, runtime=object())
        def activate(role):
            bank.active_role = role
        bank.activate = activate
        values = {'move_to': 1., 'pick_up': 2., 'placement': 3.}
        def prior(runtime, *, indices, **kwargs):
            value = values[bank.active_role]
            p = torch.full((indices.numel(), 8, 5), value)
            if noisy:
                p += torch.rand_like(p)
            return p, torch.full((indices.numel(), 2), value)
        def actor(*, states, priors, action_count):
            value = values[bank.active_role]
            # Verify that the matching teacher's vision features reached its actor.
            self.assertTrue(torch.all(states[:, -2:] == value))
            return priors[:, :action_count] + 10 * value
        bank.trainer = SimpleNamespace(deterministic_action_chunks_tensor=actor)
        config = SimpleNamespace(chunk_size=8, state_dim=8,
            actions_per_decision=4, vision_feature_dim=2, microbatch_size=8)
        with patch('rl_vla_bootstrapping.policy.cdpr_staged_demonstrations._sample_role', prior):
            return sample_staged_teacher_actions(torch=torch, bank=bank, cameras=None,
                proprio=torch.zeros(4, 6), roles=torch.tensor([2, 0, 1, 0]),
                active=torch.tensor([True, True, True, False]),
                role_texts={role: [role] * 4 for role in values}, config=config,
                sampling_seed=55)

    def test_mixed_worlds_use_their_own_residual_and_prior(self):
        state, prior, actions, switches = self.sample()
        self.assertEqual(actions[:, 0, 0].tolist(), [33., 11., 22., 0.])
        self.assertEqual(prior[:, 0, 0].tolist(), [3., 1., 2., 0.])
        self.assertEqual(switches, 3)
        self.assertTrue(torch.all(state[3] == 0))

    def test_role_noise_is_reproducible_without_consuming_caller_rng(self):
        before = torch.get_rng_state().clone()
        first = self.sample(noisy=True)[2]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.rand(100)
        self.assertTrue(torch.equal(first, self.sample(noisy=True)[2]))

    def test_bank_restores_non_tensor_residual_scale(self):
        actor = torch.nn.Linear(1, 1)
        trainer = SimpleNamespace(actor=actor, _unwrap=lambda x: x, device='cpu')
        entries = [TeacherEntry(role, Path(role), role, {}, actor.state_dict(), {}, scale)
                   for role, scale in [('move_to', .5), ('pick_up', 1.), ('placement', 2.)]]
        bank = TeacherBank(torch=torch, runtime=SimpleNamespace(policy=torch.nn.Linear(1, 1)),
                           trainer=trainer, entries=entries)
        for entry in entries:
            bank.activate(entry.role)
            self.assertEqual(actor.residual_scale, entry.residual_scale)


class PresentationTests(unittest.TestCase):
    def test_potato_can_fit_in_one_fixed_yaw_presentation(self):
        along_y = projected_grasp_xy_offset('robocasa_potato', (1, 0, 0, 0), 0)
        across_x = projected_grasp_xy_offset('robocasa_potato',
            (math.sqrt(.5), 0, 0, math.sqrt(.5)), 0)
        self.assertGreater(along_y, 0)
        self.assertLess(across_x, 0)

    def test_sphere_slack_does_not_depend_on_yaw(self):
        for yaw in (0, .7, math.pi):
            slack = projected_grasp_xy_offset('robocasa_orange',
                (math.cos(yaw/2), 0, 0, math.sin(yaw/2)), .3)
            self.assertAlmostEqual(slack, .0475 - .029 - .003)

    def test_alignment_cannot_handoff_after_lateral_drift(self):
        calibration = PickupYawCalibration(target_yaw=0, source='unit-test')
        machine = StageMachine(torch=torch, device='cpu', worlds=1,
            budgets=StageBudgets(8, 8, 8), calibration=calibration, readiness=PickupReadiness())
        machine.stage[:] = STAGE_ALIGN
        false = torch.tensor([False])
        for decision in range(2):
            machine.advance(decision=decision, reach_success=~false, pickup_success=false,
                placement_success=false, placement_geometry_ok=false, wrong_place_settled=false,
                physical_grasp=false, released=false, gripper_opening=torch.ones(1),
                ee_position=torch.tensor([[0., 0., .26]]), grasp_point_z=torch.tensor([.192]),
                target_xy_error=torch.tensor([.03]), max_grasp_xy_offset=torch.tensor([.013]),
                target_lift=torch.zeros(1), yaw_aligned=~false, diverged=false)
        self.assertEqual(int(machine.align_event[0]), -1)
        self.assertEqual(int(machine.stage[0]), STAGE_ALIGN)


class PickupHeightBridgeTests(unittest.TestCase):
    def servo(self):
        return YawTailController(torch=torch,
            calibration=PickupYawCalibration(target_yaw=0, source='unit-test'),
            action_step_yaw=.08, action_step_xyz=.015,
            pickup_height_above_grasp=.01)

    def test_rotation_clearance_is_not_a_valid_pickup_handoff(self):
        servo = self.servo()
        pose = torch.tensor([[0., 0., .26], [0., 0., .202]])
        ready = servo.handoff_ready(ee_yaw=torch.zeros(2), ee_position=pose,
                                   grasp_point_z=torch.full((2,), .192))
        self.assertEqual(ready.tolist(), [False, True])

    def test_bridge_climbs_before_rotating_then_descends_with_open_hand(self):
        servo = self.servo()
        pose = torch.tensor([[.01, -.02, .20]])
        yaw = torch.tensor([1.])
        opening = torch.ones(1)
        grasp = torch.tensor([.192])
        saw_rotation = saw_descent = False
        for _ in range(100):
            a = servo.actions(ee_position=pose, ee_yaw=yaw,
                              gripper_opening=opening, grasp_point_z=grasp)
            self.assertTrue(torch.equal(a[:, :2], torch.zeros(1, 2)))
            self.assertGreaterEqual(float(a[0, 4]), 0.)
            if not bool(servo.aligned(yaw)[0]) and float(pose[0, 2]) < .257:
                self.assertEqual(float(a[0, 3]), 0.)
            if float(a[0, 2]) < 0:
                self.assertTrue(bool(servo.aligned(yaw)[0]))
                saw_descent = True
            saw_rotation |= abs(float(a[0, 3])) > 0
            pose += a[:, :3] * .015
            yaw += a[:, 3] * .08
        self.assertTrue(saw_rotation and saw_descent)
        self.assertAlmostEqual(float(pose[0, 2]), .202, places=5)
        self.assertTrue(bool(servo.handoff_ready(ee_yaw=yaw, ee_position=pose,
                                               grasp_point_z=grasp)[0]))

    def test_descent_is_recorded_override_and_pickup_retains_xyz_gripper(self):
        low = SimpleNamespace(ee_position=torch.tensor([[0., 0., .26]] * 2),
                              ee_yaw=torch.zeros(2), gripper_opening=torch.ones(2))
        raw = torch.full((2, 5), .4)
        applied, source = _apply_overrides(torch, raw=raw,
            stage=torch.tensor([STAGE_ALIGN, STAGE_PICK_UP]), low_dim=low,
            servo=self.servo(), hold_pickup=True, hold_placement=False,
            grasp_point_z=torch.full((2,), .192))
        self.assertEqual(float(applied[0, 2]), -1.)
        self.assertEqual(int(source[0]), SOURCE_YAW_TAIL)
        self.assertTrue(torch.equal(applied[1, [0, 1, 2, 4]], raw[1, [0, 1, 2, 4]]))

    def test_stage_waits_for_height_and_requires_consecutive_ready_boundaries(self):
        machine = StageMachine(torch=torch, device='cpu', worlds=1,
            budgets=StageBudgets(8, 8, 8), calibration=self.servo().calibration)
        machine.stage[:] = STAGE_ALIGN
        false = torch.tensor([False])
        for decision, ready in enumerate([False, False, True, True]):
            machine.advance(decision=decision, reach_success=~false, pickup_success=false,
                placement_success=false, placement_geometry_ok=false, wrong_place_settled=false,
                physical_grasp=false, released=false, gripper_opening=torch.ones(1),
                ee_position=torch.tensor([[0., 0., .202 if ready else .26]]),
                grasp_point_z=torch.tensor([.192]), target_xy_error=torch.zeros(1),
                max_grasp_xy_offset=torch.tensor([.013]), target_lift=torch.zeros(1),
                yaw_aligned=~false, diverged=false, alignment_ready=torch.tensor([ready]))
            self.assertEqual(int(machine.stage[0]), STAGE_PICK_UP if decision == 3 else STAGE_ALIGN)

    def test_report_uses_last_alignment_pose_before_pickup_action(self):
        steps = 4
        record = SimpleNamespace(config_json='{}', actions_per_decision=2,
            active=np.ones((steps, 1), bool), step_stage=np.array([[STAGE_ALIGN]] * 2 + [[STAGE_PICK_UP]] * 2),
            align_event=np.array([0]), pickup_event=np.array([-1]),
            object_xyz=np.zeros((steps, 1, 1, 3)), ee_xyz=np.zeros((steps, 1, 3)),
            target_lift=np.zeros((steps, 1)), physical_grasp=np.zeros((steps, 1), bool),
            pickup_success=np.zeros((steps, 1), bool), actions=np.zeros((steps, 1, 5)),
            pickup_regrasp_total=lambda: 0)
        record.ee_xyz[:, 0, 2] = [.0675, .0175, .04, .05]
        report = StagedRound.pickup_diagnostics(record)
        self.assertEqual(report['handoff_height_above_grasp_m']['median'], .01)


class ScoreTests(unittest.TestCase):
    def round(self, picked):
        return SimpleNamespace(scene_uid=np.array(['a', 'b']),
            destination=np.array(['plate', 'bowl']), reach_event=np.array([0, 0]),
            align_event=np.array([1, 1]), pickup_event=np.array(picked),
            placement_event=np.array([-1, -1]), acceptance=lambda: (np.array([False, False]), {}),
            summary=lambda: {'rejection_reasons': {}, 'failure_counts': {},
                'reach_diagnostics': {}, 'pickup_diagnostics': {}, 'placement_diagnostics': {}})

    def test_pickup_printed_metric_does_not_report_alignment_as_success(self):
        score = score_rounds([self.round([-1, -1])], phase='pick_up')
        self.assertEqual(score['aligned_given_upstream']['rate'], 1.)
        self.assertEqual(score['conditional']['rate'], 0.)
        self.assertEqual(score['conditional']['chains'], 2)

    def test_placement_without_pickup_has_no_conditional_trials(self):
        score = score_rounds([self.round([-1, -1])], phase='placement')
        self.assertIsNone(score['conditional']['rate'])
        self.assertEqual(score['conditional']['chains'], 0)

    def test_selector_stops_at_failed_pickup_and_does_not_export_teachers(self):
        result = self.round([-1, -1])
        result.summary = lambda: {'rejection_reasons': {}, 'failure_counts': {},
            'reach_diagnostics': {'predicate_fired_worlds': 2, 'worlds': 2,
                'ready_worlds': 2, 'closest_xy_distance_m': {}, 'height_above_grasp_m': {}},
            'pickup_diagnostics': {}, 'placement_diagnostics': {}}
        world = SimpleNamespace(backend=None, collector=None, runtime=None, trainer=None,
            task_metadata={}, payload={'state_dim': 6, 'chunk_size': 8},
            args=Namespace(replan_every=4, action_step_xyz=.015, action_step_yaw=.08,
                           action_step_gripper=.05, controller_workspace_z_bounds=[.18, .6]))
        entries = {role: SimpleNamespace(checkpoint=Path(role), sha256=role)
                   for role in ('move_to', 'pick_up', 'placement')}
        teacher_manifest = {
            role: {'checkpoint': role, 'sha256': role}
            for role in ('move_to', 'pick_up', 'placement')
        }
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = Path(tmp)
            yaw = output / 'yaw.json'
            yaw.write_text(json.dumps(PickupYawCalibration(target_yaw=0, source='unit-test').to_json()))
            # A failed rerun must not leave an old promotable manifest behind.
            (output / 'selected_teachers.json').write_text('{}')
            mocks = {'_build_world': world, 'FullTaskSceneResetter': None,
                'read_manifest': ([object(), object()], {}),
                'select_split': [object(), object()], 'load_teacher_entries': [],
                'TeacherBank': SimpleNamespace(entries=entries,
                    manifest=lambda: teacher_manifest),
                'run_staged_chains': result}
            handles = {}
            for name, value in mocks.items():
                handles[name] = stack.enter_context(patch(
                    'tools.audit.select_cdpr_stage_teachers.' + name, return_value=value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = main(['--config', 'config.yaml', '--scene-manifest', 'scenes.json',
                '--yaw-calibration', str(yaw), '--worlds', '2', '--rounds', '1',
                '--candidate', 'move_to=move.pt', '--candidate', 'pick_up=pick.pt',
                '--candidate', 'placement=place.pt', '--output', str(output)])
            self.assertEqual(code, 2)
            # One candidate per role is scored from one shared continuous run.
            # Pickup's zero must not require rerunning the stochastic prefix.
            self.assertEqual(handles['run_staged_chains'].call_count, 1)
            self.assertFalse((output / 'selected_teachers.json').exists())
            report = json.loads((output / 'teacher_selection.json').read_text())
            self.assertEqual(report['blocked_phase'], 'pick_up')
            self.assertIsNone(report['phases']['pick_up']['chosen'])

    def test_one_candidate_per_role_reuses_full_chain_evidence(self):
        result = self.round([2, 2])
        result.placement_event = np.array([3, 3])
        result.acceptance = lambda: (np.array([True, True]), {})
        result.summary = lambda: {
            'rejection_reasons': {'accepted': 2}, 'failure_counts': {},
            'reach_diagnostics': {'predicate_fired_worlds': 2, 'worlds': 2,
                'ready_worlds': 2, 'closest_xy_distance_m': {},
                'height_above_grasp_m': {}},
            'align_diagnostics': {}, 'pickup_diagnostics': {},
            'placement_diagnostics': {},
        }
        world = SimpleNamespace(backend=None, collector=None, runtime=None, trainer=None,
            task_metadata={}, payload={'state_dim': 6, 'chunk_size': 8},
            args=Namespace(replan_every=4, action_step_xyz=.015, action_step_yaw=.08,
                           action_step_gripper=.05,
                           controller_workspace_z_bounds=[.18, .6]))
        entries = {role: SimpleNamespace(checkpoint=Path(role), sha256=role)
                   for role in ('move_to', 'pick_up', 'placement')}
        teacher_manifest = {
            role: {'checkpoint': role, 'sha256': role}
            for role in ('move_to', 'pick_up', 'placement')
        }
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = Path(tmp)
            yaw = output / 'yaw.json'
            yaw.write_text(json.dumps(
                PickupYawCalibration(target_yaw=0, source='unit-test').to_json()
            ))
            mocks = {'_build_world': world, 'FullTaskSceneResetter': None,
                'read_manifest': ([object(), object()], {'manifest_sha256': 'scenes'}),
                'select_split': [object(), object()], 'load_teacher_entries': [],
                'TeacherBank': SimpleNamespace(entries=entries,
                    manifest=lambda: teacher_manifest),
                'run_staged_chains': result}
            handles = {}
            for name, value in mocks.items():
                handles[name] = stack.enter_context(patch(
                    'tools.audit.select_cdpr_stage_teachers.' + name,
                    return_value=value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = main(['--config', 'config.yaml', '--scene-manifest', 'scenes.json',
                '--yaw-calibration', str(yaw), '--worlds', '2', '--rounds', '1',
                '--candidate', 'move_to=move.pt', '--candidate', 'pick_up=pick.pt',
                '--candidate', 'placement=place.pt', '--align-xy-centring',
                '--align-decisions', '48', '--pickup-prompt', 'destination',
                '--output', str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(handles['run_staged_chains'].call_count, 1)
            selected = json.loads((output / 'selected_teachers.json').read_text())
            self.assertEqual(selected['confirmation_source'],
                             'shared_full_chain_screen')
            self.assertEqual(selected['accepted']['rate'], 1.)
            self.assertEqual(selected['teachers']['move_to']['checkpoint'],
                             'move_to')
            self.assertNotIn('path', selected['teachers']['move_to'])
            self.assertTrue(selected['protocol']['align_xy_centring'])
            self.assertEqual(selected['protocol']['align_decisions'], 48)
            self.assertEqual(selected['protocol']['pickup_prompt'], 'destination')


if __name__ == '__main__':
    unittest.main()
