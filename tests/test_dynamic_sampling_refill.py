"""DAPO refill: collect until enough groups carry gradient, not until enough records.

`grpo_dynamic_sampling` was already true in every phase-5/6 config and did only
half the job. It MARKS groups whose candidates do not separate and drops their
records; nothing refills the batch, so the update shrinks to the informative
fraction. With a dense reward that fraction is near 1 and the difference is
invisible. With a binary reward it is the whole problem: a group of eight gives
gradient only if it holds both a success and a failure, which is
1 - p^8 - (1-p)^8 -- 0.99 at the measured move_to p=0.480, and 0.11 at the
measured composed-bowl p=0.0147.

The refill loop existed and could never take a second round, for two
independent reasons, which is why it was invisible:

  max_rounds = ceil(grpo_max_groups_per_update / groups_per_rank)
             = ceil(64 / (512 worlds / group 8)) = ceil(64/64) = 1

  and the records target could not have refilled either: 512 worlds at up to
  128 records each is ~65k against grpo_target_records_per_update 1024,
  satisfied by round 0 every time.

These cover the arithmetic, the new group target, and the backwards
compatibility that keeps every existing run byte-identical.
"""

from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _max_rounds(max_groups: int, groups_per_rank: int) -> int:
    """The loop's own formula, restated."""

    return max(1, (max_groups + groups_per_rank - 1) // groups_per_rank)


class TheOldBudgetCouldNotRefillTests(unittest.TestCase):
    def test_the_compose_config_pinned_max_rounds_at_one(self) -> None:
        # 512 worlds at group size 8 is 64 groups per rank, against a 64-group
        # budget. This is the bug, stated as arithmetic.
        self.assertEqual(_max_rounds(max_groups=64, groups_per_rank=64), 1)

    def test_the_new_config_can_take_four(self) -> None:
        self.assertEqual(_max_rounds(max_groups=256, groups_per_rank=64), 4)

    def test_the_formula_matches_the_source(self) -> None:
        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as module

        source = inspect.getsource(module.main)
        self.assertIn("int(args.grpo_max_groups_per_update)", source)
        self.assertIn("// int(layout.groups_per_rank)", source)


class InformativeFractionTests(unittest.TestCase):
    """The numbers the budget has to be sized against."""

    @staticmethod
    def _informative(p: float, group: int = 8) -> float:
        return 1.0 - (1.0 - p) ** group - p**group

    def test_the_measured_legs_land_where_the_config_says(self) -> None:
        # Success rates from the sft_phase6 evaluation, three rounds of 512.
        measured = {
            "move_to_object": (0.4798, 0.992),
            "put_into_plate_caught": (0.7150, 0.932),
            "put_into_bowl_caught": (0.4794, 0.992),
            "pick_up": (0.1491, 0.725),
            "put_into_plate_composed": (0.0689, 0.435),
            "put_into_bowl_composed": (0.0147, 0.112),
        }
        for name, (p, expected) in measured.items():
            self.assertAlmostEqual(
                self._informative(p), expected, places=3, msg=name
            )

    def test_four_rounds_covers_every_leg_except_composed_bowl(self) -> None:
        # The budget is deliberately sized to 4 rounds: it covers composed
        # plate's 1/0.43 = 2.3x and leaves composed bowl's 1/0.11 = 9x
        # uncovered, because paying nine rollouts an update for one family is
        # the wrong answer to a family whose scripted ORACLE reaches only 0.455.
        budget = _max_rounds(max_groups=256, groups_per_rank=64)
        for p in (0.4798, 0.7150, 0.4794, 0.1491, 0.0689):
            self.assertLessEqual(math.ceil(1.0 / self._informative(p)), budget)
        self.assertGreater(math.ceil(1.0 / self._informative(0.0147)), budget)


class CollectorExposesUsableGroupsTests(unittest.TestCase):
    def test_the_field_exists_and_defaults_to_none(self) -> None:
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            CollectorRound,
        )

        self.assertIn("usable_groups", CollectorRound.__dataclass_fields__)
        # None keeps rounds built by anything that predates this loadable, and
        # the trainer guards on it.
        self.assertIsNone(
            CollectorRound.__dataclass_fields__["usable_groups"].default
        )

    def test_it_unions_both_return_streams(self) -> None:
        """With split credit a group can separate one stream and not the other.

        Counting only the terminal stream would discard rollouts that did
        contribute approach gradient, and the refill would then over-collect.
        """

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            RankLocalMJWarpGRPOCollector,
        )

        source = inspect.getsource(RankLocalMJWarpGRPOCollector.collect_round)
        start = source.index("usable_groups = informative_group")
        block = source[start : start + 400]
        self.assertIn("self.split_credit_at_grasp", block)
        self.assertIn("degenerate_pre", block)

    def test_degenerate_pre_is_only_read_where_it_is_defined(self) -> None:
        # `degenerate_pre` is bound inside `if split_credit_at_grasp:` and then
        # inside `if min_group_reward_std > 0.0:`. Reading it under weaker
        # conditions is a NameError that only fires in a configuration nobody
        # runs locally.
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            RankLocalMJWarpGRPOCollector,
        )

        source = inspect.getsource(RankLocalMJWarpGRPOCollector.collect_round)
        start = source.index("usable_groups = informative_group")
        block = source[start : source.index("vla_records = None", start)]
        self.assertIn("self.split_credit_at_grasp and self.min_group_reward_std > 0.0", block)


class TheLoopUsesTheGroupTargetTests(unittest.TestCase):
    def test_the_flag_exists_and_is_off_by_default(self) -> None:
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        self.assertEqual(parse_args([]).grpo_target_informative_groups, 0)
        self.assertEqual(
            parse_args(["--grpo-target-informative-groups", "64"]).grpo_target_informative_groups,
            64,
        )

    def test_the_group_target_overrides_the_records_target(self) -> None:
        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as module

        source = inspect.getsource(module.main)
        self.assertIn("if target_groups > 0:", source)
        self.assertIn("if local_usable_groups >= target_groups:", source)
        # The records path stays reachable, as the `elif`, so a run without the
        # new flag keeps its exact previous break condition.
        self.assertIn("elif (\n", source)
        self.assertIn("int(args.grpo_target_records_per_update) <= 0", source)

    def test_the_wall_clock_budget_still_applies(self) -> None:
        # The refill can now cost four rollouts an update; without the seconds
        # cap a thin family would stall the run rather than train slowly.
        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as module

        source = inspect.getsource(module.main)
        self.assertIn("grpo_max_collection_seconds_per_update", source)

    def test_the_oversample_is_reported(self) -> None:
        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as module

        source = inspect.getsource(module.main)
        for metric in ("rounds_collected", "usable_groups_collected"):
            self.assertIn(f'"{metric}"', source)


class TheSparseJointConfigTests(unittest.TestCase):
    CONFIG = ROOT / "configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml"

    def _args(self):
        import yaml

        from rl_vla_bootstrapping.core.commands import append_cli_arg
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        raw = yaml.safe_load(self.CONFIG.read_text(encoding="utf-8"))
        section = next(
            value["args"]
            for value in raw["training"].values()
            if isinstance(value, dict) and "args" in value
        )
        argv: list[str] = []
        for key, value in section.items():
            append_cli_arg(argv, key, value)
        return parse_args(argv)

    def test_every_key_in_it_reaches_the_parser(self) -> None:
        # A YAML key that does not match a flag becomes an unknown argument;
        # parse_args raising here is the point of the test.
        self._args()

    def test_the_budgets_are_consistent_with_each_other(self) -> None:
        import yaml

        args = self._args()
        raw = yaml.safe_load(self.CONFIG.read_text(encoding="utf-8"))
        worlds = int(raw["simulator"]["worlds_per_rank"])
        groups_per_rank = worlds // int(args.grpo_group_size)
        rounds = _max_rounds(int(args.grpo_max_groups_per_update), groups_per_rank)
        self.assertGreater(
            rounds,
            1,
            "grpo_max_groups_per_update leaves max_rounds at 1; the refill "
            "cannot take a second round and dynamic sampling is masking only",
        )
        self.assertGreater(int(args.grpo_target_informative_groups), 0)
        self.assertLessEqual(
            int(args.grpo_target_informative_groups),
            rounds * groups_per_rank,
            "the group target cannot be reached inside the round budget",
        )

    def test_it_is_sparse_and_joint_and_measures_the_composed_task(self) -> None:
        from rl_vla_bootstrapping.core.config import load_project_config
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            sparse_binary_reward_requested,
        )

        project = load_project_config(self.CONFIG)
        self.assertTrue(
            sparse_binary_reward_requested(dict(project.task.metadata or {}))
        )
        self.assertEqual(
            set(project.task.instruction_types),
            {"move_to_object", "pick_up", "put_into_plate", "put_into_bowl"},
        )
        self.assertGreater(
            self._args().composed_validation_episodes_per_instruction, 0
        )

    def test_every_instruction_has_its_own_ladder(self) -> None:
        # Each instruction runs its own approach gate; a family without a
        # ladder falls back to the shared initial, which for placement sits
        # INSIDE the success radius and makes the first rung solvable by
        # opening the gripper.
        from rl_vla_bootstrapping.core.config import load_project_config

        metadata = dict(load_project_config(self.CONFIG).task.metadata or {})
        ladders = metadata["random_workspace_start_distance_ladder_by_instruction"]
        for name in ("move_to_object", "pick_up", "put_into_plate", "put_into_bowl"):
            self.assertIn(name, ladders)
            self.assertTrue(ladders[name] == sorted(ladders[name]), name)


if __name__ == "__main__":
    unittest.main()
