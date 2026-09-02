"""Sparse must change the REWARD and nothing else.

The sparse binary reward is already the base: `evaluate_active_sparse_tasks`
builds `where(success, sparse_success_reward, sparse_failure_reward)` and the
dense terms overwrite it per instruction. So the obvious way to "turn dense
off" -- pass None for the reward objects -- is a trap, because both objects
also own SUCCESS PREDICATE geometry:

    plate radius          0.091 -> 0.03   (cfg.container_xy)
    bowl radius           0.057 -> 0.03
    target_has_settled    required -> not computed at all
    minimum_target_motion 0.0 -> 0.04
    wrong_place_settled   terminates -> never fires
    move_to xy window     from the config -> the 0.02 default

Every number in the campaign report was measured under the first column. A
sparse arm that silently moved to the second would be scoring a different task
and its success rate would not be comparable to anything.

So the claim `sparse_binary_reward` makes is narrow and testable: identical
trajectories produce IDENTICAL success verdicts under both settings, and only
the reward differs. These test exactly that, by running the real predicate.
"""

from __future__ import annotations

import unittest
from dataclasses import fields

import torch

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    INSTRUCTION_TO_ID,
    BatchedCatchReleaseDenseReward,
    BatchedMoveToDistanceReward,
    BatchedTaskState,
    BatchedTaskThresholds,
    evaluate_active_sparse_tasks,
    sparse_binary_reward_requested,
)

# The predicate-bearing fields. If sparse changes any of these the arms are
# scoring different tasks, and every comparison drawn between them is void.
PREDICATE_FIELDS = {
    BatchedCatchReleaseDenseReward: (
        "plate_radius",
        "bowl_radius",
        "container_z_tolerance",
        "wrong_place_settle_margin",
        "release_max_height",
        "plate_release_max_height",
        "bowl_release_max_height",
        "pick_lift_success_height",
        "pick_grasp_height_offset",
    ),
    BatchedMoveToDistanceReward: (
        "xy_window_low",
        "xy_window_high",
        "z_window_low",
        "z_window_high",
        "require_z_window",
    ),
}

def _campaign_metadata() -> dict:
    """The real compose-loop metadata, not a hand-written stand-in.

    A fixture with fewer shaping terms than production would let the flag look
    correct while leaving a weight the real config sets. Read from the file the
    dense arm of the A/B actually runs.
    """

    from pathlib import Path

    from rl_vla_bootstrapping.core.config import load_project_config

    config = Path(__file__).resolve().parents[1] / (
        "configs/examples/cdpr_smolvla_phase5_compose_loop.yaml"
    )
    metadata = dict(load_project_config(config).task.metadata or {})
    metadata.pop("sparse_binary_reward", None)
    return metadata


BASE_METADATA = _campaign_metadata()

PLATE_Z = 0.15
REST = 0.02


class FlagParsingTests(unittest.TestCase):
    def test_the_flag_reads_yaml_truthiness(self) -> None:
        self.assertFalse(sparse_binary_reward_requested(None))
        self.assertFalse(sparse_binary_reward_requested({}))
        self.assertFalse(sparse_binary_reward_requested({"sparse_binary_reward": False}))
        for truthy in (True, "true", "True", "yes", "on", "1"):
            self.assertTrue(
                sparse_binary_reward_requested({"sparse_binary_reward": truthy}),
                truthy,
            )
        for falsy in (False, "false", "no", "off", "0"):
            self.assertFalse(
                sparse_binary_reward_requested({"sparse_binary_reward": falsy}),
                falsy,
            )


class GeometryIsUntouchedTests(unittest.TestCase):
    def test_every_predicate_field_survives(self) -> None:
        sparse = {**BASE_METADATA, "sparse_binary_reward": True}
        for cls, names in PREDICATE_FIELDS.items():
            dense_cfg = cls.from_metadata(BASE_METADATA)
            sparse_cfg = cls.from_metadata(sparse)
            for name in names:
                self.assertEqual(
                    getattr(dense_cfg, name),
                    getattr(sparse_cfg, name),
                    f"{cls.__name__}.{name} moved; the two arms would score "
                    "different tasks",
                )

    def test_the_shaping_weights_actually_go_to_zero(self) -> None:
        # The other half: a flag that preserved everything would be a no-op.
        sparse = {**BASE_METADATA, "sparse_binary_reward": True}
        dense_cfg = BatchedCatchReleaseDenseReward.from_metadata(BASE_METADATA)
        sparse_cfg = BatchedCatchReleaseDenseReward.from_metadata(sparse)
        for name in (
            "distance_reward_weight",
            "fine_distance_reward_weight",
            "placement_failure_penalty",
            "pick_contact_bonus",
            "pick_grasp_bonus",
            "pick_lift_reward_weight",
        ):
            self.assertNotEqual(getattr(dense_cfg, name), 0.0, f"{name} was already 0")
            self.assertEqual(getattr(sparse_cfg, name), 0.0, name)
        # Success bonuses go to 1.0, not 0.0: GRPO normalises within the group,
        # but a reward that is zero everywhere has no spread and EVERY group
        # would be filtered as degenerate.
        self.assertEqual(sparse_cfg.placement_success_bonus, 1.0)
        self.assertEqual(sparse_cfg.pick_success_bonus, 1.0)
        move_sparse = BatchedMoveToDistanceReward.from_metadata(sparse)
        self.assertEqual(move_sparse.distance_reward_weight, 0.0)
        self.assertEqual(move_sparse.z_penalty_weight, 0.0)
        self.assertEqual(move_sparse.success_bonus, 1.0)

    def test_nothing_outside_the_named_sets_changed_unnoticed(self) -> None:
        """Catch a future field that is neither zeroed nor asserted stable."""

        sparse = {**BASE_METADATA, "sparse_binary_reward": True}
        expected_to_move = {
            BatchedCatchReleaseDenseReward: {
                "distance_reward_weight",
                "fine_distance_reward_weight",
                "placement_success_bonus",
                "placement_failure_penalty",
                "pick_contact_bonus",
                "pick_grasp_bonus",
                "pick_lift_reward_weight",
                "pick_success_bonus",
            },
            BatchedMoveToDistanceReward: {
                "distance_reward_weight",
                "success_bonus",
                "z_penalty_weight",
                "excess_distance_penalty_weight",
                "too_close_penalty_weight",
            },
        }
        for cls, allowed in expected_to_move.items():
            dense_cfg = cls.from_metadata(BASE_METADATA)
            sparse_cfg = cls.from_metadata(sparse)
            moved = {
                f.name
                for f in fields(cls)
                if getattr(dense_cfg, f.name) != getattr(sparse_cfg, f.name)
            }
            self.assertTrue(
                moved <= allowed,
                f"{cls.__name__}: unexpected fields changed by the sparse "
                f"flag: {sorted(moved - allowed)}",
            )


def _thresholds(catch_release: BatchedCatchReleaseDenseReward,
                move_to: BatchedMoveToDistanceReward) -> BatchedTaskThresholds:
    """Built exactly as RankLocalMJWarpGRPOCollector._task_thresholds builds it."""

    return BatchedTaskThresholds(
        move_to_xy_low=float(move_to.xy_window_low),
        move_to_xy=float(move_to.xy_window_high),
        container_xy=max(float(catch_release.plate_radius), float(catch_release.bowl_radius)),
        container_z=float(catch_release.container_z_tolerance),
        minimum_target_motion=0.0,
    )


class ThePredicateIsIdenticalTests(unittest.TestCase):
    """Run the real predicate over the same trajectories under both settings."""

    def _state(self, instruction: str, count: int) -> BatchedTaskState:
        return BatchedTaskState(
            instruction_ids=torch.full(
                (count,), INSTRUCTION_TO_ID[instruction], dtype=torch.int64
            ),
            target_slots=torch.zeros((count,), dtype=torch.int64),
            reference_slots=torch.ones((count,), dtype=torch.int64),
            second_reference_slots=torch.full((count,), -1, dtype=torch.int64),
            initial_target_positions=torch.tensor(
                [[0.0, 0.0, PLATE_Z + 0.10]] * count
            ),
            ever_grasped=torch.zeros((count,), dtype=torch.bool),
            grasped=torch.zeros((count,), dtype=torch.bool),
            step_count=torch.zeros((count,), dtype=torch.int64),
            release_threshold=torch.full((count,), 0.55),
            support_surface_z=torch.full((count,), PLATE_Z),
            target_rest_height=torch.full((count,), REST),
            peak_lift=torch.zeros((count,), dtype=torch.float32),
            release_clearance=torch.full((count,), float("nan")),
        )

    def _sweep(self, instruction: str, metadata: dict):
        """Step a spread of geometries and collect verdicts and rewards."""

        catch_release = BatchedCatchReleaseDenseReward.from_metadata(metadata)
        move_to = BatchedMoveToDistanceReward.from_metadata(metadata)
        thresholds = _thresholds(catch_release, move_to)

        # A spread that straddles every threshold: inside/outside the radius,
        # above/below the settle margin, gripper closed and open.
        offsets = [0.0, 0.03, 0.06, 0.09, 0.12]
        heights = [PLATE_Z + REST, PLATE_Z + 0.05, PLATE_Z + 0.12]
        grippers = [0.1, 0.9]
        # The end-effector is varied INDEPENDENTLY of the object. Pinning it at
        # the grasp point (object + pick_grasp_height_offset) makes
        # pick_grasp_distance identically zero, and pick_up's dense shaping is
        # then saturated -- the sweep reports two reward values and would pass a
        # "sparse looks binary" test while never exercising the shaping it is
        # supposed to be switching off.
        ee_offsets = [0.0, 0.025, 0.06]
        count = len(offsets) * len(heights) * len(grippers) * len(ee_offsets)
        state = self._state(instruction, count)

        verdicts, rewards, terminated = [], [], []
        for caught in (True, False, False):
            objects = torch.zeros((count, 2, 3))
            gripper = torch.zeros((count,))
            ee_extra = torch.zeros((count,))
            index = 0
            for dx in offsets:
                for z in heights:
                    for g in grippers:
                        for extra in ee_offsets:
                            objects[index, 0, 0] = dx
                            objects[index, 0, 2] = z
                            objects[index, 1, 2] = PLATE_Z
                            gripper[index] = g
                            ee_extra[index] = extra
                            index += 1
            ee = objects[:, 0].clone()
            ee[:, 2] += 0.0075 + ee_extra
            result = evaluate_active_sparse_tasks(
                state=state,
                ee_position=ee,
                object_positions=objects,
                gripper_opening=gripper,
                caught_target=torch.full((count,), caught),
                active_mask=torch.ones((count,), dtype=torch.bool),
                max_steps=128,
                thresholds=thresholds,
                catch_release_dense_reward=catch_release,
                move_to_distance_reward=move_to,
            )
            verdicts.append(result.success.clone())
            terminated.append(result.terminated.clone())
            rewards.append(result.rewards.clone())
        return verdicts, terminated, rewards

    def test_success_and_termination_are_bit_identical(self) -> None:
        sparse = {**BASE_METADATA, "sparse_binary_reward": True}
        for instruction in (
            "put_into_plate",
            "put_into_bowl",
            "pick_up",
            "move_to_object",
        ):
            dense_v, dense_t, _ = self._sweep(instruction, BASE_METADATA)
            sparse_v, sparse_t, _ = self._sweep(instruction, sparse)
            for step, (a, b) in enumerate(zip(dense_v, sparse_v)):
                torch.testing.assert_close(
                    a.to(torch.int32),
                    b.to(torch.int32),
                    msg=f"{instruction} step {step}: success verdicts differ",
                )
            for step, (a, b) in enumerate(zip(dense_t, sparse_t)):
                torch.testing.assert_close(
                    a.to(torch.int32),
                    b.to(torch.int32),
                    msg=f"{instruction} step {step}: termination differs",
                )

    def test_the_sweep_is_not_vacuous(self) -> None:
        # A sweep where nothing ever succeeds would pass the test above while
        # testing nothing at all.
        verdicts, _, _ = self._sweep("put_into_plate", BASE_METADATA)
        self.assertTrue(
            any(bool(v.any()) for v in verdicts),
            "the geometry sweep never produced a success",
        )
        self.assertTrue(
            any(bool((~v).any()) for v in verdicts),
            "the geometry sweep never produced a failure",
        )

    def test_sparse_rewards_take_two_values_and_dense_do_not(self) -> None:
        sparse = {**BASE_METADATA, "sparse_binary_reward": True}
        for instruction in ("put_into_plate", "pick_up", "move_to_object"):
            _, _, sparse_r = self._sweep(instruction, sparse)
            values = torch.unique(torch.cat(sparse_r))
            self.assertLessEqual(
                values.numel(),
                2,
                f"{instruction} sparse rewards took {values.tolist()}",
            )
            _, _, dense_r = self._sweep(instruction, BASE_METADATA)
            self.assertGreater(
                torch.unique(torch.cat(dense_r)).numel(),
                2,
                f"{instruction} dense reward was already binary; the sweep "
                "does not exercise the shaping",
            )

    def test_a_sparse_success_still_outscores_a_sparse_failure(self) -> None:
        # The group needs spread or every group is filtered as degenerate.
        sparse = {**BASE_METADATA, "sparse_binary_reward": True}
        verdicts, _, rewards = self._sweep("put_into_plate", sparse)
        wins = torch.cat([r[v] for r, v in zip(rewards, verdicts)])
        losses = torch.cat([r[~v] for r, v in zip(rewards, verdicts)])
        self.assertGreater(wins.numel(), 0)
        self.assertGreater(losses.numel(), 0)
        self.assertGreater(float(wins.min()), float(losses.max()))


if __name__ == "__main__":
    unittest.main()
