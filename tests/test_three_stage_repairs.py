"""Regression cases taken from the reward-starved September 13 run."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    ThreeStageMilestones, advance_three_stage_milestones,
    concatenate_collector_rounds, three_stage_count_metrics, ValidationRound,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule, global_stage_loss_weights
from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
    _synchronize_update_metrics_once, _synchronize_validation_rounds,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID
from rl_vla_bootstrapping.simulation.cdpr_full_task_outcome import FullTaskOutcome
from test_grpo_episode_offset_exploration import _args, _trainer


class MilestoneRepairTests(unittest.TestCase):
    def setUp(self):
        self.reset = SimpleNamespace(task_state=SimpleNamespace(
            target_slots=torch.tensor([0]), reference_slots=torch.tensor([1]),
            initial_target_positions=torch.tensor([[0., 0., .15]])))
        self.low = SimpleNamespace(object_positions=torch.tensor([[[0., 0., .20], [.1, 0., .15]]]),
                                   gripper_opening=torch.tensor([.5]), ee_position=torch.tensor([[0., 0., .22]]))
        self.result = SimpleNamespace(success=torch.tensor([False]), diagnostics={
            'pick_grasp_distance': torch.tensor([.04]), 'grasped': torch.tensor([True]),
            'ever_grasped': torch.tensor([True]), 'pick_success': torch.tensor([True]), 'target_lift': torch.tensor([.05]),
            'credited_lift': torch.tensor([.10]), 'released': torch.tensor([False]),
            'container_xy_radius': torch.tensor([.091]), 'wrong_place_drop': torch.tensor([False])})

    def advance(self, state=None, physical=True, active=True):
        return advance_three_stage_milestones(
            state or ThreeStageMilestones.zeros(torch, 1, 'cpu'), reset=self.reset,
            low_dim=self.low, result=self.result, physical_grasp=torch.tensor([physical]),
            gripper_command=torch.tensor([0.]), previous_opening=self.low.gripper_opening,
            active_mask=torch.tensor([active]))

    def test_real_grasp_backfills_approach_after_closing_and_displacement(self):
        s = self.advance()
        self.assertTrue(s.approached.item())
        self.assertTrue(s.picked_up.item())
        self.assertFalse(s.diagnostics['legacy_approach'].item())
        self.assertTrue(s.diagnostics['lift_without_legacy_approach'].item())

    def test_historical_peak_does_not_award_current_pickup(self):
        self.result.diagnostics['target_lift'].zero_()
        self.assertFalse(self.advance().picked_up.item())
        self.result.diagnostics['target_lift'].fill_(.05)
        self.assertFalse(self.advance(physical=False).picked_up.item())

    def test_distance_alone_is_accessible_even_with_closed_hand(self):
        self.result.diagnostics['pick_grasp_distance'].fill_(.02)
        self.result.diagnostics['target_lift'].zero_()
        s = self.advance(physical=False)
        self.assertTrue(s.approached.item())
        self.assertFalse(s.picked_up.item())
        self.assertTrue(s.diagnostics['reach_closed_hand'].item())

    def test_inactive_world_cannot_earn_events(self):
        s = self.advance(active=False)
        self.assertFalse(s.approached.item())
        self.assertFalse(s.picked_up.item())

    def test_strict_outcome_survives_missed_legacy_gate(self):
        s = self.advance()
        self.low.object_positions[0, 0, 0] = .1
        self.low.gripper_opening.fill_(.8)
        self.result.success.fill_(True)
        self.result.diagnostics['released'].fill_(True)
        s = self.advance(s, physical=False)
        self.assertTrue(s.outcome.native.item())
        self.assertTrue(s.placed.item())
        self.assertTrue(s.diagnostics['strict_without_legacy_approach'].item())

    def test_real_drop_is_not_excused_by_later_success(self):
        s = self.advance()
        s = self.advance(s, physical=False)
        self.result.success.fill_(True)
        self.result.diagnostics['released'].fill_(True)
        s = self.advance(s, physical=False)
        self.assertTrue(s.outcome.native.item())
        self.assertTrue(s.carry_slip.item())
        self.assertFalse(s.placed.item())

    def test_strict_success_backfills_pickup_when_lift_definitions_disagree(self):
        # The task predicate lifts at a lower configured pick height than M2's
        # 5 cm. A held carry at 4 cm is lifted for strict but never M2; a strict
        # success must not then carry pickup return 0.
        self.result.diagnostics['target_lift'].fill_(.04)
        s = self.advance(physical=True)
        self.assertFalse(s.picked_up.item())
        self.assertTrue(s.outcome.lifted.item())
        self.result.diagnostics['pick_grasp_distance'].fill_(.2)
        self.result.success.fill_(True)
        self.result.diagnostics['released'].fill_(True)
        s = self.advance(s, physical=False)
        self.assertTrue(s.placed.item())
        self.assertTrue(s.picked_up.item())
        self.assertTrue(s.approached.item())
        self.assertTrue(s.diagnostics['strict_without_pickup_milestone'].item())

    def test_milestone_returns_are_ordered(self):
        s = self.advance()
        self.assertTrue(bool((s.approached >= s.picked_up).all()))
        self.assertTrue(bool((s.picked_up >= s.placed).all()))

    def test_count_diagnostics_keep_destination_denominators(self):
        s = self.advance()
        metrics = three_stage_count_metrics(s, torch.tensor([INSTRUCTION_TO_ID['put_into_bowl']]))
        self.assertEqual(metrics['three_stage/bowl_episodes'], 1)
        self.assertEqual(metrics['three_stage/plate_episodes'], 0)
        self.assertEqual(metrics['three_stage/bowl_pickup_count'], 1)


class WrongPlaceTerminationTests(unittest.TestCase):
    def evaluate(self, *, requires_lift, peak_lift):
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedCatchReleaseDenseReward, BatchedTaskState, evaluate_active_sparse_tasks)
        objects = torch.zeros((1, 2, 3))
        # Set down outside the bowl, at rest, no longer held.
        objects[0, 0] = torch.tensor([0.10, 0.00, 0.18])
        objects[0, 1] = torch.tensor([0.00, 0.00, 0.176])
        state = BatchedTaskState(
            instruction_ids=torch.tensor([INSTRUCTION_TO_ID['put_into_bowl']]),
            target_slots=torch.tensor([0]), reference_slots=torch.tensor([1]),
            second_reference_slots=torch.tensor([-1]),
            initial_target_positions=torch.tensor([[0.10, 0.00, 0.18]]),
            ever_grasped=torch.tensor([True]), grasped=torch.tensor([False]),
            step_count=torch.zeros(1, dtype=torch.int64), release_threshold=torch.tensor([0.55]),
            support_surface_z=torch.tensor([0.15]), target_rest_height=torch.tensor([0.03]),
            peak_lift=torch.tensor([peak_lift]))
        return evaluate_active_sparse_tasks(
            state=state, ee_position=torch.tensor([[0.10, 0.00, 0.26]]), object_positions=objects,
            gripper_opening=torch.tensor([0.90]), caught_target=torch.tensor([False]),
            active_mask=torch.tensor([True]), max_steps=128,
            catch_release_dense_reward=BatchedCatchReleaseDenseReward(wrong_place_requires_lift=requires_lift))

    def test_default_keeps_the_unconditional_termination(self):
        result = self.evaluate(requires_lift=False, peak_lift=0.0)
        self.assertTrue(result.terminated.item())
        self.assertTrue(result.diagnostics['wrong_place_drop'].item())

    def test_failed_tabletop_grasp_is_not_terminated_when_gated(self):
        result = self.evaluate(requires_lift=True, peak_lift=0.0)
        self.assertFalse(result.terminated.item())
        self.assertFalse(result.diagnostics['wrong_place_drop'].item())
        self.assertTrue(result.diagnostics['wrong_place_drop_legacy'].item())

    def test_dropped_carry_still_terminates_when_gated(self):
        result = self.evaluate(requires_lift=True, peak_lift=0.06)
        self.assertTrue(result.terminated.item())
        self.assertTrue(result.diagnostics['wrong_place_drop'].item())

    def test_config_key_arms_the_gate(self):
        from rl_vla_bootstrapping.core.config import load_project_config
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import BatchedCatchReleaseDenseReward
        metadata = dict(load_project_config(
            'configs/examples/cdpr_smolvla_three_stage_put_into.yaml').task.metadata or {})
        reward = BatchedCatchReleaseDenseReward.from_metadata(metadata)
        self.assertTrue(reward.wrong_place_requires_lift)

    def test_legacy_termination_verdict_is_logged(self):
        base = MilestoneRepairTests('test_inactive_world_cannot_earn_events')
        base.setUp()
        base.result.diagnostics['target_lift'].zero_()
        base.result.diagnostics['pick_success'].fill_(False)
        base.result.diagnostics['wrong_place_drop_legacy'] = torch.tensor([True])
        s = base.advance(physical=False)
        self.assertTrue(s.diagnostics['wrong_place_legacy_without_lift'].item())
        self.assertFalse(s.diagnostics['wrong_place_legacy_after_lift'].item())
        self.assertFalse(s.wrong_place.item())
        # A later strict success is a failure under the old protocol.
        base.result.diagnostics['wrong_place_drop_legacy'] = torch.tensor([False])
        base.result.diagnostics['target_lift'].fill_(.05)
        base.result.diagnostics['pick_success'].fill_(True)
        s = base.advance(s, physical=True)
        base.result.success.fill_(True)
        base.result.diagnostics['released'].fill_(True)
        s = base.advance(s, physical=False)
        self.assertTrue(s.placed.item())
        self.assertFalse(s.diagnostics['strict_under_legacy_termination'].item())


class EvaluatorMilestoneTests(unittest.TestCase):
    def test_summary_reports_milestones_beside_strict(self):
        import numpy as np
        from tools.audit.evaluate_cdpr_full_put_into import summarize
        n = 4
        row = {name: np.zeros(n, dtype=bool) for name in (
            'native', 'strict', 'approached', 'grasped', 'lifted', 'released', 'carry_slip',
            'wrong_place', 'final_geometry_ok', 'milestone_approach', 'milestone_pickup',
            'milestone_placement', 'milestone_legacy_approach')}
        row['milestone_approach'][:3] = True
        row['milestone_legacy_approach'][:1] = True
        row.update(completion_steps=np.ones(n), min_grasp_distance=np.ones(n), peak_lift=np.zeros(n),
                   scene_uid=np.arange(n), destination=np.array(['plate', 'bowl'] * 2),
                   target_catalog=np.array(['apple'] * n))
        report = summarize([row])
        self.assertEqual(report['milestones']['approach'], .75)
        self.assertEqual(report['milestones']['legacy_approach'], .25)
        self.assertEqual(report['milestones']['plate_approach'], 1.)
        self.assertIn('strict', report)


class CreditRepairTests(unittest.TestCase):
    def make_records(self, trainer, n=5, stage=None):
        states = torch.randn(n, 6)
        priors = torch.zeros(n, 2, 5)
        actions, log_probs, _ = trainer.sample_action_chunks_tensor(
            states=states, priors=priors, action_count=2,
            generator=torch.Generator().manual_seed(4))
        return {'state': states, 'prior': priors, 'action': actions[:, 0],
                'action_index': torch.zeros(n, dtype=torch.long), 'old_log_prob': log_probs[:, 0],
                'advantage': torch.full((n,), 7 ** .5),
                'credit_stage': torch.ones(n, dtype=torch.long) if stage is None else stage}

    def test_single_successful_entrant_keeps_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _trainer(_args('--entropy-coef', '0', '--action-l2', '0', '--microbatch-size', '2'), Path(directory))
            records = self.make_records(trainer)
            metrics = trainer.update_tensor_records(records, loss_mask=torch.ones(5),
                schedule=EqualDDPSchedule(records_per_minibatch=4, ppo_epochs=1, global_max_records=5))
            self.assertGreater(metrics['gradient_norm_max'], 0)
            self.assertAlmostEqual(metrics['advantage_mean'], 7 ** .5, places=5)
            self.assertEqual(metrics['three_stage/pickup_positive_records'], 5)
            # Five records, minibatch 4: two minibatches, one step each.
            self.assertEqual(metrics['optimizer_steps'], 2)

    def test_microbatch_partition_does_not_change_update(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(16)
            a = _trainer(_args('--entropy-coef', '0', '--action-l2', '0', '--microbatch-size', '1'), Path(directory))
            b = _trainer(_args('--entropy-coef', '0', '--action-l2', '0', '--microbatch-size', '4'), Path(directory))
            b.actor.load_state_dict(a.actor.state_dict())
            records = self.make_records(a, stage=torch.tensor([0, 0, 0, 1, 2]))
            for trainer in (a, b):
                torch.manual_seed(9)
                trainer.update_tensor_records(records, loss_mask=torch.ones(5),
                    schedule=EqualDDPSchedule(records_per_minibatch=4, ppo_epochs=1, global_max_records=5))
            for x, y in zip(a.actor.parameters(), b.actor.parameters()):
                self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_empty_update_does_not_advance_adam_momentum(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _trainer(_args('--microbatch-size', '2'), Path(directory))
            records = self.make_records(trainer, n=4)
            trainer.update_tensor_records(records, loss_mask=torch.ones(4),
                schedule=EqualDDPSchedule(4, 1, 4))
            before = [p.detach().clone() for p in trainer.actor.parameters()]
            steps = [state['step'].clone() for state in trainer.optimizer.state.values()]
            metrics = trainer.update_tensor_records(records, loss_mask=torch.zeros(4),
                schedule=EqualDDPSchedule(4, 1, 0))
            self.assertEqual(metrics['optimizer_steps'], 0)
            self.assertEqual(metrics['backward_collectives'], 0)
            for x, y in zip(before, trainer.actor.parameters()):
                self.assertTrue(torch.equal(x, y))
            for x, state in zip(steps, trainer.optimizer.state.values()):
                self.assertTrue(torch.equal(x, state['step']))

    def test_minibatch_steps_and_single_minibatch_matches_exact_mean(self):
        # M minibatches each carry a 1/M sample scaled by M; with one minibatch
        # the step is the exact global stage mean, whatever the microbatching.
        with tempfile.TemporaryDirectory() as directory:
            trainer = _trainer(_args('--entropy-coef', '0', '--action-l2', '0', '--microbatch-size', '2'), Path(directory))
            records = self.make_records(trainer, n=8, stage=torch.tensor([0, 0, 0, 0, 1, 1, 2, 2]))
            metrics = trainer.update_tensor_records(records, loss_mask=torch.ones(8),
                schedule=EqualDDPSchedule(records_per_minibatch=2, ppo_epochs=3, global_max_records=8))
            self.assertEqual(metrics['optimizer_steps'], 12)
            for name in ('approach', 'pickup', 'placement'):
                self.assertAlmostEqual(metrics[f'three_stage/{name}_loss_mass'], 1/3, places=5)

    def test_candidate_mean_ignores_how_long_a_candidate_stayed(self):
        stages = torch.tensor([0, 0, 0, 0, 1])
        candidates = torch.tensor([7, 7, 7, 9, 7])
        weights = global_stage_loss_weights(stages, torch.ones(5, dtype=torch.bool),
                                            candidate_id=candidates)
        # Candidate 7 spent 3 rows in approach, candidate 9 one; equal mass.
        self.assertAlmostEqual(float(weights[:3].sum()), 1/4, places=6)
        self.assertAlmostEqual(float(weights[3]), 1/4, places=6)
        self.assertAlmostEqual(float(weights[4]), 1/2, places=6)

    def test_refill_rounds_get_disjoint_candidate_ids(self):
        def round(ids):
            return SimpleNamespace(records={'credit_stage': torch.zeros(2, dtype=torch.long),
                                            'candidate_id': torch.tensor(ids)},
                loss_mask=torch.tensor([True, True]), candidate_rewards=torch.zeros(1, 4),
                candidate_success=torch.zeros(1, 4, dtype=torch.bool), candidate_ever_grasped=None,
                group_instruction_ids=torch.tensor([0]), group_shell_ids=torch.tensor([0]),
                group_prelifted=None, group_caught_start=None, group_skips_approach=None, metrics={})
        records = concatenate_collector_rounds([round([0, 3]), round([0, 3])])[0]
        self.assertEqual(records['candidate_id'].tolist(), [0, 3, 4, 7])

    def test_weighted_stage_mass_is_exact_and_duration_free(self):
        stages = torch.tensor([0, 0, 0, 1, 1, 2])
        candidates = torch.tensor([1, 1, 2, 1, 3, 3])
        weights = global_stage_loss_weights(stages, torch.ones(6, dtype=torch.bool),
                                            candidate_id=candidates, stage_mass=[0.2, 0.2, 0.6])
        for stage, mass in enumerate((0.2, 0.2, 0.6)):
            self.assertAlmostEqual(float(weights[stages == stage].sum()), mass, places=6)
        # Within approach, candidate 1 (two rows) and candidate 2 (one row) get equal mass.
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]), places=6)

    def test_weighted_mass_renormalizes_over_represented_stages(self):
        stages = torch.tensor([0, 0, 1])
        weights = global_stage_loss_weights(stages, torch.ones(3, dtype=torch.bool),
                                            stage_mass=[0.2, 0.2, 0.6])
        self.assertAlmostEqual(float(weights[:2].sum()), 0.5, places=6)
        self.assertAlmostEqual(float(weights[2]), 0.5, places=6)
        only_placement = global_stage_loss_weights(torch.tensor([2, 2]), torch.ones(2, dtype=torch.bool),
                                                   stage_mass=[0.2, 0.2, 0.6])
        self.assertAlmostEqual(float(only_placement.sum()), 1.0, places=6)

    def test_zero_mass_stage_receives_no_gradient_weight(self):
        weights = global_stage_loss_weights(torch.tensor([0, 1, 2]), torch.ones(3, dtype=torch.bool),
                                            stage_mass=[0.0, 1.0, 1.0])
        self.assertEqual(float(weights[0]), 0.0)
        empty = global_stage_loss_weights(torch.tensor([0, 0]), torch.ones(2, dtype=torch.bool),
                                          stage_mass=[0.0, 1.0, 1.0])
        self.assertEqual(float(empty.sum()), 0.0)

    def test_stage_mass_arguments_are_validated_and_normalized(self):
        from rl_vla_bootstrapping.policy.rank_local_grpo import normalized_stage_loss_mass
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args
        self.assertIsNone(parse_args([]).three_stage_stage_loss_weights)
        args = parse_args(['--three-stage-stage-loss-weights', '1', '1', '3'])
        for got, want in zip(args.three_stage_stage_loss_weights, (0.2, 0.2, 0.6)):
            self.assertAlmostEqual(got, want)
        for bad in ([1, -1, 1], [0, 0, 0], [1, float('nan'), 1], [1, 1]):
            with self.assertRaises(ValueError):
                normalized_stage_loss_mass(bad)
        with self.assertRaises(SystemExit):
            parse_args(['--three-stage-stage-loss-weights', '0', '0', '0'])

    def test_config_sets_placement_weighted_mass(self):
        import yaml
        from rl_vla_bootstrapping.core.commands import append_cli_arg
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args
        raw = yaml.safe_load(open('configs/examples/cdpr_smolvla_three_stage_put_into.yaml', encoding='utf-8'))
        argv = []
        for key, value in raw['training']['rl']['args'].items():
            append_cli_arg(argv, key, value)
        weights = parse_args(argv).three_stage_stage_loss_weights
        for got, want in zip(weights, (0.2, 0.2, 0.6)):
            self.assertAlmostEqual(got, want)

    def test_trainer_applies_weighted_stage_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _trainer(_args('--entropy-coef', '0', '--action-l2', '0', '--microbatch-size', '2',
                                     '--three-stage-stage-loss-weights', '0.2', '0.2', '0.6'), Path(directory))
            records = self.make_records(trainer, n=8, stage=torch.tensor([0, 0, 0, 0, 1, 1, 2, 2]))
            metrics = trainer.update_tensor_records(records, loss_mask=torch.ones(8),
                schedule=EqualDDPSchedule(records_per_minibatch=4, ppo_epochs=1, global_max_records=8))
            for name, mass in (('approach', 0.2), ('pickup', 0.2), ('placement', 0.6)):
                self.assertAlmostEqual(metrics[f'three_stage/{name}_loss_mass'], mass, places=5)
            self.assertGreater(metrics['gradient_norm_max'], 0)

    def test_stage_mean_ignores_duration_and_padding(self):
        stages = torch.tensor([0, 0, 0, 1, 2, 2])
        valid = torch.tensor([True, True, True, True, True, False])
        weights = global_stage_loss_weights(stages, valid)
        for stage in range(3):
            self.assertAlmostEqual(float(weights[stages == stage].sum()), 1/3, places=6)
        self.assertEqual(weights[-1], 0)


class TelemetryRepairTests(unittest.TestCase):
    def test_refill_sums_counts_instead_of_half_success(self):
        def round(success):
            return SimpleNamespace(records={'credit_stage': torch.tensor([0, 1])},
                loss_mask=torch.tensor([True, True]), candidate_rewards=torch.tensor([[success, 0.]]),
                candidate_success=torch.tensor([[bool(success), False]]), candidate_ever_grasped=None,
                group_instruction_ids=torch.tensor([INSTRUCTION_TO_ID['put_into_bowl']]),
                group_shell_ids=torch.tensor([0]), group_prelifted=None,
                group_caught_start=None, group_skips_approach=None,
                metrics={'three_stage/approach_successes': success, 'three_stage/approach_usable_groups': success,
                         'three_stage/approach_count': success, 'three_stage/episodes': 2,
                         'three_stage/bowl_episodes': 2, 'three_stage/bowl_approach_count': success,
                         'non_finite_ee_worlds': success})
        merged = concatenate_collector_rounds([round(1), round(0)])[6]
        self.assertEqual(merged['three_stage/approach_successes'], 1)
        self.assertEqual(merged['three_stage/approach_usable_groups'], 1)
        self.assertEqual(merged['three_stage/episodes'], 4)
        self.assertEqual(merged['three_stage/approach_rate'], .25)
        self.assertEqual(merged['three_stage/bowl_approach_rate'], .25)

    def test_validation_forwards_stage_counts_and_rates(self):
        rounds = []
        for success in (1., 0.):
            rounds.append(ValidationRound(candidate_rewards=torch.tensor([[success, 0.]]),
                candidate_success=torch.tensor([[bool(success), False]]), final_xy_distance=torch.ones(1, 2),
                final_ee_z=torch.ones(1, 2)*.2, min_ee_z=torch.ones(1, 2)*.2,
                group_target_catalog_ids=torch.tensor([0]), group_shell_ids=torch.tensor([0]),
                group_instruction_ids=torch.tensor([INSTRUCTION_TO_ID['put_into_bowl']]),
                metrics={'validation/three_stage_episodes': 2,
                         'validation/three_stage_pickup_count': success,
                         'validation/three_stage_bowl_episodes': 2,
                         'validation/three_stage_bowl_pickup_count': success}))
        metrics = _synchronize_validation_rounds(rounds, device=torch.device('cpu'))
        self.assertEqual(metrics['validation/three_stage_pickup_count'], 1)
        self.assertEqual(metrics['validation/three_stage_pickup_rate'], .25)
        self.assertEqual(metrics['validation/three_stage_bowl_pickup_rate'], .25)


if __name__ == '__main__':
    unittest.main()
