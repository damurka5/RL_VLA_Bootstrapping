"""Bounded demonstration-start GRPO pilot: bank provenance, quarantine, schedule.

CPU only. Nothing here trains; the parts under test are the ones that decide
WHICH rows reach an optimizer and whether the two evaluation arms are even
scoring the same episodes.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID
from rl_vla_bootstrapping.policy.cdpr_demonstration_training import (
    FAMILIES, balanced_vla_batches, quarantine_suffix, reset_seed,
    scheduled_family, scheduled_job_index, scheduled_stage,
    validate_bank, validate_seed_partition,
)
from tools.audit.probe_cdpr_demonstration_handoff import sha256
from tools.train.build_cdpr_training_handoffs import build_bank
from tools.train.run_cdpr_demo_grpo_pilot import compare_evaluations
from tools.audit.sil_record import _Recording


TRAIN_SEED, VALIDATION_SEED = 17000000, 2000000


def source_recording(round_index, *, worlds=512, steps=24):
    """One reserved training round: grasp at 6, lift at 12, release at 20.

    The gaps are deliberate. A boundary before the grasp exists (step 4), a
    held boundary before the lift exists (step 8), and a held boundary before
    the release exists (step 20), so the three families and both stages each
    plan a DIFFERENT prefix out of the same source.
    """
    groups = worlds // 8
    xyz = np.zeros((steps, worlds, 2, 3), dtype=np.float32)
    xyz[..., 2] = .10
    xyz[12:20, :, 0, 2] = .16  # carried above the 5 cm lift, then set down
    xyz[:, :, 1, 0] = .30  # receptacle offset in x, so it is a distinct slot
    caught = np.zeros((steps, worlds), dtype=bool)
    caught[6:20] = True
    opening = np.zeros((steps, worlds), dtype=np.float32)
    opening[20:] = .8
    success = np.zeros((steps, worlds), dtype=bool)
    success[21] = True
    terminated = success.copy()
    active = np.ones((steps, worlds), dtype=bool)
    active[22:] = False
    plate = np.repeat(np.arange(groups) % 2 == 0, 8)
    instruction_ids = np.where(plate, INSTRUCTION_TO_ID['put_into_plate'],
                               INSTRUCTION_TO_ID['put_into_bowl'])
    instructions = np.where(plate, 'put apple into plate', 'put tomato into bowl')
    return _Recording(
        actions=np.zeros((steps, worlds, 5), dtype=np.float32), active=active,
        success=success, terminated=terminated, caught_target=caught,
        ee_xyz=np.zeros((steps, worlds, 3), dtype=np.float32),
        gripper_opening=opening, object_xyz=xyz,
        instruction_ids=instruction_ids,
        target_slots=np.zeros(worlds, dtype=int), reference_slots=np.ones(worlds, dtype=int),
        second_reference_slots=np.full(worlds, -1), horizons=np.full(worlds, 40),
        initial_target_xyz=xyz[0, :, 0].copy(), support_surface_z=np.zeros(worlds),
        release_threshold=np.full(worlds, .55), target_rest_height=np.full(worlds, .10),
        physical_grasp_at_reset=np.zeros(worlds, dtype=bool), instructions=instructions,
        actions_per_decision=4, round_index=round_index, diverged_worlds=0,
        pick_lift_success_height=.05,
        diverged_world_mask=np.zeros(worlds, dtype=bool),
        reset_object_xyz=xyz[0].copy(), reset_ee_xyz=np.zeros((worlds, 3), dtype=np.float32),
    )


def evaluation_recording(round_index, *, successes, worlds=16):
    """An ordinary uncaught container round, half of it starting inside goal."""
    steps = 4
    xyz = np.zeros((steps, worlds, 2, 3), dtype=np.float32)
    xyz[..., 2] = .10
    reset_xyz = xyz[0].copy()
    # Worlds 0..7 start with the object well outside the plate radius; 8..15
    # start inside it, which is the slice the composed rate must be split on.
    reset_xyz[:8, 0, 0] = .50
    success = np.zeros((steps, worlds), dtype=bool)
    success[-1, list(successes)] = True
    return _Recording(
        actions=np.zeros((steps, worlds, 5), dtype=np.float32),
        active=np.ones((steps, worlds), dtype=bool), success=success,
        terminated=np.zeros((steps, worlds), dtype=bool),
        caught_target=np.zeros((steps, worlds), dtype=bool),
        ee_xyz=np.zeros((steps, worlds, 3), dtype=np.float32),
        gripper_opening=np.zeros((steps, worlds), dtype=np.float32), object_xyz=xyz,
        instruction_ids=np.full(worlds, INSTRUCTION_TO_ID['put_into_plate']),
        target_slots=np.zeros(worlds, dtype=int), reference_slots=np.ones(worlds, dtype=int),
        second_reference_slots=np.full(worlds, -1), horizons=np.full(worlds, 40),
        initial_target_xyz=reset_xyz[:, 0].copy(), support_surface_z=np.zeros(worlds),
        release_threshold=np.full(worlds, .55), target_rest_height=np.full(worlds, .10),
        physical_grasp_at_reset=np.zeros(worlds, dtype=bool),
        instructions=np.full(worlds, 'put apple into plate'),
        actions_per_decision=4, round_index=round_index, diverged_worlds=0,
        pick_lift_success_height=.05,
        diverged_world_mask=np.zeros(worlds, dtype=bool),
        reset_object_xyz=reset_xyz, reset_ee_xyz=np.zeros((worlds, 3), dtype=np.float32),
    )


def write_bank(root, *, rounds=2):
    """Record two reserved rounds, extract them, and build the training bank."""
    from tools.audit.extract_cdpr_transition_demonstrations import main as extract

    sources = root / 'training_sources'
    for offset in range(rounds):
        source_recording(1000 + offset).to_npz(sources / f'record_{offset:02d}.npz')
    checkpoint, config = root / 'donor.pt', root / 'config.yaml'
    torch.save({'policy': {}, 'args': {'validation_seed': VALIDATION_SEED,
                                       'seed': TRAIN_SEED}}, checkpoint)
    config.write_text(yaml.safe_dump({'training': {'rl': {'args': {
        'seed': TRAIN_SEED, 'validation_seed': VALIDATION_SEED}}}}))
    with contextlib.redirect_stdout(io.StringIO()):
        extract(['--recordings', str(sources / 'record_*.npz'), '--output', str(root / 'demonstrations')])
        bank = build_bank(manifest=root / 'demonstrations' / 'manifest.json',
                          checkpoint=checkpoint, config=config,
                          output=root / 'training_bank.json', rounds=rounds)
    return bank, checkpoint, config


class SeedPartitionTests(unittest.TestCase):
    def test_reserved_rounds_touch_neither_training_nor_evaluation(self):
        validate_seed_partition(VALIDATION_SEED, list(range(1000, 1012)),
                                TRAIN_SEED, VALIDATION_SEED)

    def test_a_bank_cut_from_evaluation_rounds_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_seed_partition(VALIDATION_SEED, [0, 1, 2],
                                    TRAIN_SEED, VALIDATION_SEED)

    def test_a_bank_cut_from_the_training_stream_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_seed_partition(TRAIN_SEED, [0], TRAIN_SEED, VALIDATION_SEED)

    def test_the_seed_formula_is_the_collector_s_own(self):
        self.assertEqual(reset_seed(VALIDATION_SEED, 1, 3, 7),
                         VALIDATION_SEED + 1_000_003 + 3 * 10_000_019 + 7 * 100_003)


class ScheduleTests(unittest.TestCase):
    def test_pickup_receives_half_of_the_assisted_attempts(self):
        drawn = [scheduled_family(i) for i in range(len(FAMILIES) * 4)]
        self.assertEqual(drawn.count('pick_up'), len(drawn) // 2)
        self.assertEqual(set(drawn), {'pick_up', 'put_into_plate', 'put_into_bowl'})

    def test_earlier_boundaries_only_appear_after_half_the_budget(self):
        budget = 1_000_000
        self.assertEqual({scheduled_stage(i, 0, budget) for i in range(64)}, {0})
        later = [scheduled_stage(i, budget // 2, budget) for i in range(16)]
        # Half of the cycles keep the easier handoff; the retention is the point.
        self.assertEqual(later, [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)

    def test_the_two_ranks_walk_different_scenes(self):
        for update in range(8):
            self.assertNotEqual(scheduled_job_index(update, 0, 16),
                                scheduled_job_index(update, 1, 16))
        self.assertEqual(scheduled_job_index(0, 0, 1), 0)
        with self.assertRaisesRegex(ValueError, 'No jobs'):
            scheduled_job_index(0, 0, 0)


def collector_round(*, group_successes, worlds=32, group_size=8, decisions=2):
    """A minimal CollectorRound whose masks match the collector's own layout."""
    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import CollectorRound

    groups = worlds // group_size
    success = torch.zeros(groups, group_size, dtype=torch.bool)
    for group, count in enumerate(group_successes):
        success[group, :count] = True
    advantage = torch.where(success.reshape(-1), 1., -1.)
    degenerate = success.all(dim=1) | (~success).any(dim=1).logical_not()
    usable = ~(success.all(dim=1) | (~success).all(dim=1))
    advantage = torch.where(usable.repeat_interleave(group_size), advantage, 0.)
    return CollectorRound(
        records={'advantage': advantage.repeat(decisions)},
        # Records are appended one full world block per decision, so the mask
        # is world-major inside each block and the block repeats.
        loss_mask=torch.ones(worlds * decisions),
        candidate_rewards=success.to(torch.float32),
        candidate_success=success,
        group_instruction_ids=torch.full((groups,), INSTRUCTION_TO_ID['pick_up']),
        group_shell_ids=torch.zeros(groups, dtype=torch.int64), metrics={},
        group_skips_approach=torch.zeros(groups, dtype=torch.bool),
        usable_groups=usable,
        vla_records={'advantage': advantage.clone(),
                     'world_index': torch.arange(worlds),
                     'instruction': ['pick up apple'] * worlds},
    )


class QuarantineTests(unittest.TestCase):
    def test_a_diverged_group_reaches_neither_optimizer(self):
        item = collector_round(group_successes=[4, 4, 4, 4])
        report = {'groups': [{'group': 0, 'contains_divergence': False},
                             {'group': 1, 'contains_divergence': True}]}
        clean = quarantine_suffix(item, report)
        mask = clean.loss_mask.reshape(2, 4, 8)
        self.assertTrue(torch.all(mask[:, 0] == 1.))
        # Group 1 diverged; groups 2 and 3 were never planned into this job.
        self.assertTrue(torch.all(mask[:, 1:] == 0.))
        self.assertEqual(clean.usable_groups.tolist(), [True, False, False, False])
        self.assertEqual(clean.vla_records['world_index'].tolist(), list(range(8)))

    def test_assisted_groups_never_reach_an_approach_promotion_gate(self):
        item = collector_round(group_successes=[4, 4, 4, 4])
        clean = quarantine_suffix(item, {'groups': [{'group': 2, 'contains_divergence': False}]})
        # Every assisted start skips some approach, so none of them may be
        # counted by the start-distance curriculum's pass rate.
        self.assertTrue(torch.all(clean.group_skips_approach))
        self.assertEqual(clean.group_instruction_ids.tolist(),
                         [-1, -1, int(INSTRUCTION_TO_ID['pick_up']), -1])

    def test_a_degenerate_group_carries_no_lora_rows(self):
        item = collector_round(group_successes=[8, 4, 0, 4])
        report = {'groups': [{'group': g, 'contains_divergence': False} for g in range(4)]}
        clean = quarantine_suffix(item, report)
        # Groups 0 and 2 are uniform, so their advantage is exactly zero.
        self.assertEqual(sorted(set(clean.vla_records['world_index'].tolist())),
                         list(range(8, 16)) + list(range(24, 32)))
        self.assertEqual(len(clean.vla_records['instruction']),
                         clean.vla_records['advantage'].numel())

    def test_an_unexpected_record_layout_is_refused(self):
        item = collector_round(group_successes=[4, 4, 4, 4])
        item.loss_mask.resize_(33)
        with self.assertRaisesRegex(ValueError, 'record layout'):
            quarantine_suffix(item, {'groups': []})


class BalancedLoRATests(unittest.TestCase):
    def batch(self, groups, *, offset=0):
        advantage = torch.tensor([1. if live else 0. for live in groups]).repeat_interleave(8)
        return {'advantage': advantage,
                'world_index': torch.arange(len(groups) * 8) + offset,
                'instruction': [f'row {i + offset}' for i in range(len(groups) * 8)]}

    def test_the_ordinary_arm_cannot_exhaust_the_cap_alone(self):
        ordinary = self.batch([True] * 8)
        assisted = self.batch([True] * 8, offset=100)
        merged = balanced_vla_batches([ordinary, assisted], 32)
        self.assertEqual([b['advantage'].numel() for b in merged], [16, 16])
        self.assertEqual(merged[1]['world_index'].tolist(), list(range(100, 116)))

    def test_zero_advantage_groups_consume_no_budget(self):
        merged = balanced_vla_batches([self.batch([False, True, False, True])], 16)
        self.assertEqual(merged[0]['world_index'].tolist(),
                         list(range(8, 16)) + list(range(24, 32)))

    def test_one_arm_takes_the_whole_cap_when_the_other_is_empty(self):
        merged = balanced_vla_batches([self.batch([True] * 4), None], 16)
        self.assertEqual([b['advantage'].numel() for b in merged], [16])

    def test_rows_and_instructions_stay_aligned(self):
        merged = balanced_vla_batches([self.batch([True, True])], 8)
        self.assertEqual(merged[0]['instruction'], [f'row {i}' for i in range(8)])

    def test_a_partial_group_is_refused(self):
        batch = self.batch([True])
        batch['advantage'] = batch['advantage'][:7]
        with self.assertRaisesRegex(ValueError, 'complete groups'):
            balanced_vla_batches([batch], 16)


class TrainingBankTests(unittest.TestCase):
    def test_every_family_and_stage_gets_its_own_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank, checkpoint, config = write_bank(Path(tmp))
            self.assertEqual(bank['split'], 'training_only')
            self.assertEqual(bank['source_rounds'], [1000, 1001])
            self.assertEqual(bank['reset_seed'], VALIDATION_SEED)
            prefixes = {}
            for job in bank['jobs']:
                prefixes.setdefault((job['family'], job['stage']), set()).add(
                    job['job']['prefix_steps'])
            self.assertEqual(prefixes[('pick_up', 0)], {8})
            # Stage 1 pick_up hands over BEFORE the grasp, not after it.
            self.assertEqual(prefixes[('pick_up', 1)], {4})
            self.assertFalse(any(job['job']['require_held'] for job in bank['jobs']
                                 if (job['family'], job['stage']) == ('pick_up', 1)))
            self.assertEqual(prefixes[('put_into_plate', 0)], {20})
            self.assertEqual(prefixes[('put_into_plate', 1)], {8})
            self.assertEqual(sorted(bank['unique_scene_counts']),
                             ['pick_up/stage0', 'pick_up/stage1',
                              'put_into_bowl/stage0', 'put_into_bowl/stage1',
                              'put_into_plate/stage0', 'put_into_plate/stage1'])
            validate_bank(bank_path := Path(tmp) / 'training_bank.json', config=config)
            self.assertEqual(json.loads(bank_path.read_text())['checkpoint_sha256'],
                             sha256(checkpoint))

    def test_too_few_source_scenes_names_the_knob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'increase DEMO_ROUNDS'):
                with contextlib.redirect_stdout(io.StringIO()):
                    write_bank_with_min_scenes(root, 200)

    def test_a_source_changed_after_extraction_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank, _, config = write_bank(root)
            source = Path(bank['sources'][0]['path'])
            source_recording(1000, steps=24).to_npz(source.with_name('other.npz'))
            source.write_bytes(source.with_name('other.npz').read_bytes() + b'\x00')
            with self.assertRaisesRegex(ValueError, 'source recording hash'):
                validate_bank(root / 'training_bank.json', config=config)

    def test_the_evaluation_prototype_is_refused_as_a_training_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, config = write_bank(root)
            path = root / 'training_bank.json'
            payload = json.loads(path.read_text())
            payload['split'] = 'evaluation'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'evaluation prototype refused'):
                validate_bank(path, config=config)

    def test_a_config_edited_after_the_bank_was_built_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, config = write_bank(root)
            config.write_text(config.read_text() + '\n# edited\n')
            with self.assertRaisesRegex(ValueError, 'config hash mismatch'):
                validate_bank(root / 'training_bank.json', config=config)


def write_bank_with_min_scenes(root, min_scenes):
    from tools.audit.extract_cdpr_transition_demonstrations import main as extract

    sources = root / 'training_sources'
    source_recording(1000).to_npz(sources / 'record_00.npz')
    checkpoint, config = root / 'donor.pt', root / 'config.yaml'
    torch.save({'policy': {}, 'args': {'validation_seed': VALIDATION_SEED}}, checkpoint)
    config.write_text(yaml.safe_dump({'training': {'rl': {'args': {
        'seed': TRAIN_SEED, 'validation_seed': VALIDATION_SEED}}}}))
    extract(['--recordings', str(sources / 'record_*.npz'), '--output', str(root / 'demonstrations')])
    return build_bank(manifest=root / 'demonstrations' / 'manifest.json',
                      checkpoint=checkpoint, config=config,
                      output=root / 'training_bank.json', rounds=1, min_scenes=min_scenes)


class EvaluationComparisonTests(unittest.TestCase):
    def arms(self, root, *, baseline, final):
        for arm, winners in (('baseline', baseline), ('final_eval', final)):
            for index in range(3):
                evaluation_recording(index, successes=winners).to_npz(
                    root / arm / f'record_{index:02d}.npz')

    def test_before_and_after_split_the_starts_that_begin_inside_the_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'final.pt'
            checkpoint.write_bytes(b'weights')
            self.arms(root, baseline=[8, 9], final=[0, 8, 9, 10])
            with contextlib.redirect_stdout(io.StringIO()):
                result = compare_evaluations(root, checkpoint)
            plate = result['arms']['baseline']['put_into_plate']
            self.assertEqual((plate['successes'], plate['episodes']), (6, 48))
            # Both baseline winners start inside the plate radius, so the
            # outside-goal rate -- the composed question -- is still zero.
            self.assertEqual(plate['outside_goal_episodes'], 24)
            self.assertEqual(plate['outside_goal_rate'], 0.)
            after = result['arms']['final_eval']['put_into_plate']
            self.assertAlmostEqual(after['outside_goal_rate'], 3 / 24)
            self.assertTrue(result['reset_identity_checked'])
            self.assertEqual(result['reset_mismatches'], [])
            self.assertEqual(json.loads((root / 'pilot_comparison.json').read_text())['arms'],
                             result['arms'])

    def test_arms_that_scored_different_episodes_are_not_called_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'final.pt'
            checkpoint.write_bytes(b'weights')
            self.arms(root, baseline=[0], final=[0])
            moved = evaluation_recording(1, successes=[0])
            moved.reset_object_xyz[0, 0, 0] += .05
            moved.to_npz(root / 'final_eval' / 'record_01.npz')
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'reset identity differs'):
                    compare_evaluations(root, checkpoint)
            # The raw scores are still written, so the run is not lost.
            saved = json.loads((root / 'pilot_comparison.json').read_text())
            self.assertFalse(saved['reset_identity_checked'])
            self.assertIn('round 1: reset_object_xyz', saved['reset_mismatches'])

    def test_a_caught_start_container_round_is_not_an_ordinary_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'final.pt'
            checkpoint.write_bytes(b'weights')
            self.arms(root, baseline=[0], final=[0])
            caught = evaluation_recording(0, successes=[0])
            caught.caught_target[0, :] = True
            caught.to_npz(root / 'baseline' / 'record_00.npz')
            with self.assertRaisesRegex(ValueError, 'not ordinary uncaught'):
                compare_evaluations(root, checkpoint)

    def test_a_missing_round_is_not_quietly_averaged_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'final.pt'
            checkpoint.write_bytes(b'weights')
            self.arms(root, baseline=[0], final=[0])
            (root / 'final_eval' / 'record_02.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'exactly three rounds'):
                compare_evaluations(root, checkpoint)


if __name__ == '__main__':
    unittest.main()
