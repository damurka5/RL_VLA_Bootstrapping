"""The decomposition must reproduce the verdict it is decomposing.

`placement_failure_decomposition` recomputes terms that
`evaluate_active_sparse_tasks` owns, because the predicate is a stateful torch
function and there is no `BatchedTaskState` beside a stored npz. That is the
duplication this campaign has paid for twice, so the tool ships with the
comparison wired in: every world's recomputed verdict is checked against the
recording's own latched `success`.

These tests drive synthetic episodes through the REAL predicate, package the
same trajectories into a `_Recording` exactly as `_RoundRecorder` would, and
assert both halves: that the recomputed verdict agrees, and that the blocking
stage names the term that was actually withheld. A tool that agreed on the
verdict while mislabelling the cause would still produce a wrong table, which
is the whole output.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    INSTRUCTION_TO_ID,
    BatchedCatchReleaseDenseReward,
    BatchedTaskState,
    BatchedTaskThresholds,
    evaluate_active_sparse_tasks,
)
from tools.audit.placement_failure_decomposition import _Thresholds, _episode_terms
from tools.audit.sil_record import _Recording

BOWL_Z = 0.15
REST = 0.02
RADIUS_BOWL = 0.057
RADIUS_PLATE = 0.091
Z_TOL = 0.12
SETTLE_MARGIN = 0.025

# The metadata a compose-loop config carries, so the tool and the predicate are
# handed the same numbers from the two ends.
METADATA = {
    "put_plate_xy_tolerance": RADIUS_PLATE,
    "put_bowl_xy_tolerance": RADIUS_BOWL,
    "put_container_z_tolerance": Z_TOL,
    "placement_wrong_drop_settle_margin": SETTLE_MARGIN,
}

SETTLED_Z = BOWL_Z + REST
CARRY_Z = BOWL_Z + 0.10


def _dense() -> BatchedCatchReleaseDenseReward:
    return BatchedCatchReleaseDenseReward(
        plate_radius=RADIUS_PLATE,
        bowl_radius=RADIUS_BOWL,
        container_z_tolerance=Z_TOL,
        wrong_place_settle_margin=SETTLE_MARGIN,
    )


def _thresholds() -> BatchedTaskThresholds:
    # Built the way RankLocalMJWarpGRPOCollector._task_thresholds builds it:
    # container_z comes FROM the dense reward's z tolerance.
    return BatchedTaskThresholds(
        container_xy=max(RADIUS_PLATE, RADIUS_BOWL),
        container_z=Z_TOL,
        minimum_target_motion=0.0,
    )


class _Episode:
    """One world's scripted trajectory, in the units the predicate reads."""

    def __init__(
        self,
        *,
        instruction: str,
        object_xy: list[tuple[float, float]],
        object_z: list[float],
        gripper: list[float],
        caught: list[bool],
    ) -> None:
        lengths = {len(object_xy), len(object_z), len(gripper), len(caught)}
        assert len(lengths) == 1, f"ragged episode: {lengths}"
        self.instruction = instruction
        self.object_xy = object_xy
        self.object_z = object_z
        self.gripper = gripper
        self.caught = caught

    def __len__(self) -> int:
        return len(self.object_z)


def _run(episodes: list[_Episode]) -> tuple[_Recording, np.ndarray, BatchedTaskState]:
    """Step the real predicate over the batch, then build the recording.

    Mirrors `_RoundRecorder`: the arrays captured here are the same keyword
    arguments the patched predicate captures, stacked over env steps.
    """

    worlds = len(episodes)
    steps = max(len(episode) for episode in episodes)
    instruction_ids = torch.tensor(
        [INSTRUCTION_TO_ID[e.instruction] for e in episodes], dtype=torch.int64
    )
    state = BatchedTaskState(
        instruction_ids=instruction_ids,
        target_slots=torch.zeros((worlds,), dtype=torch.int64),
        reference_slots=torch.ones((worlds,), dtype=torch.int64),
        second_reference_slots=torch.full((worlds,), -1, dtype=torch.int64),
        initial_target_positions=torch.tensor(
            [[e.object_xy[0][0], e.object_xy[0][1], CARRY_Z] for e in episodes]
        ),
        ever_grasped=torch.zeros((worlds,), dtype=torch.bool),
        grasped=torch.zeros((worlds,), dtype=torch.bool),
        step_count=torch.zeros((worlds,), dtype=torch.int64),
        release_threshold=torch.full((worlds,), 0.55),
        support_surface_z=torch.full((worlds,), BOWL_Z),
        target_rest_height=torch.full((worlds,), REST),
        peak_lift=torch.zeros((worlds,), dtype=torch.float32),
        release_clearance=torch.full((worlds,), float("nan")),
    )

    captured: dict[str, list[np.ndarray]] = {
        key: [] for key in ("success", "active", "caught", "gripper", "objects", "ee")
    }
    latched = torch.zeros((worlds,), dtype=torch.bool)
    for step_index in range(steps):
        objects = torch.zeros((worlds, 2, 3))
        gripper = torch.zeros((worlds,))
        caught = torch.zeros((worlds,), dtype=torch.bool)
        active = torch.zeros((worlds,), dtype=torch.bool)
        for world, episode in enumerate(episodes):
            index = min(step_index, len(episode) - 1)
            active[world] = step_index < len(episode)
            objects[world, 0, 0] = episode.object_xy[index][0]
            objects[world, 0, 1] = episode.object_xy[index][1]
            objects[world, 0, 2] = episode.object_z[index]
            objects[world, 1, 2] = BOWL_Z
            gripper[world] = episode.gripper[index]
            caught[world] = episode.caught[index]
        ee = objects[:, 0].clone()
        ee[:, 2] += 0.0075
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=gripper,
            caught_target=caught,
            active_mask=active,
            max_steps=128,
            thresholds=_thresholds(),
            catch_release_dense_reward=_dense(),
        )
        # `validate_round` latches with logical_or over the round.
        latched |= result.success
        captured["success"].append(result.success.numpy().copy())
        captured["active"].append(active.numpy().copy())
        captured["caught"].append(caught.numpy().copy())
        captured["gripper"].append(gripper.numpy().copy())
        captured["objects"].append(objects.numpy().copy())
        captured["ee"].append(ee.numpy().copy())

    recording = _Recording(
        actions=np.zeros((steps, worlds, 5), dtype=np.float32),
        active=np.stack(captured["active"]),
        success=np.stack(captured["success"]),
        terminated=np.stack(captured["success"]),
        caught_target=np.stack(captured["caught"]),
        ee_xyz=np.stack(captured["ee"]),
        gripper_opening=np.stack(captured["gripper"]),
        object_xyz=np.stack(captured["objects"]),
        instruction_ids=instruction_ids.numpy(),
        target_slots=np.zeros((worlds,), dtype=np.int64),
        reference_slots=np.ones((worlds,), dtype=np.int64),
        second_reference_slots=np.full((worlds,), -1, dtype=np.int64),
        horizons=np.full((worlds,), 32, dtype=np.int64),
        initial_target_xyz=state.initial_target_positions.numpy(),
        support_surface_z=np.full((worlds,), BOWL_Z, dtype=np.float32),
        release_threshold=np.full((worlds,), 0.55, dtype=np.float32),
        target_rest_height=np.full((worlds,), REST, dtype=np.float32),
        physical_grasp_at_reset=np.zeros((worlds,), dtype=bool),
        instructions=np.array([e.instruction for e in episodes]),
        actions_per_decision=4,
        round_index=0,
        diverged_worlds=0,
        pick_lift_success_height=0.05,
    )
    return recording, latched.numpy(), state


def carry(xy: tuple[float, float], *, steps: int = 4) -> _Episode:
    """Grasped, carried over `xy`, never released. The timeout case."""

    return _Episode(
        instruction="put_into_bowl",
        object_xy=[xy] * steps,
        object_z=[CARRY_Z] * steps,
        gripper=[0.2] * steps,
        caught=[True] * steps,
    )


def place(
    xy_release: tuple[float, float],
    xy_rest: tuple[float, float],
    *,
    release_z: float = BOWL_Z + 0.03,
    rest_z: float = SETTLED_Z,
    instruction: str = "put_into_bowl",
) -> _Episode:
    """Grasp, carry, open over `xy_release`, come to rest at `xy_rest`."""

    return _Episode(
        instruction=instruction,
        object_xy=[xy_release, xy_release, xy_release, xy_rest, xy_rest],
        object_z=[CARRY_Z, CARRY_Z, release_z, rest_z, rest_z],
        gripper=[0.2, 0.2, 0.9, 0.9, 0.9],
        caught=[True, True, False, False, False],
    )


class DecompositionAgreesWithThePredicate(unittest.TestCase):
    def _terms(self, episodes: list[_Episode]) -> list[dict]:
        recording, latched, state = _run(episodes)
        np.testing.assert_array_equal(recording.episode_success, latched)
        rows = _episode_terms(recording, _Thresholds(METADATA))
        self.assertEqual(len(rows), len(episodes))
        latched_clearance = state.release_clearance.numpy()
        for row in rows:
            self.assertEqual(
                row["recomputed_success"],
                row["recorded_success"],
                f"world {row['world']} disagrees: {row}",
            )
            # The clearance the tool reports must be the value the predicate
            # LATCHED, not a recomputation that happens to look similar. The
            # latch is written once, at the first release; a tool that read the
            # clearance at termination would report ~0 for every episode and
            # the release-height sweep would be flat by construction.
            expected = float(latched_clearance[row["world"]])
            reported = float(row["release_clearance_m"])
            if np.isnan(expected):
                self.assertTrue(
                    np.isnan(reported),
                    f"world {row['world']} never released but reports {reported}",
                )
            else:
                self.assertAlmostEqual(reported, expected, places=4)
        return rows

    def test_a_clean_placement_succeeds_and_is_not_classified(self) -> None:
        rows = self._terms([place((0.0, 0.0), (0.0, 0.0))])
        self.assertTrue(rows[0]["recomputed_success"])
        self.assertEqual(rows[0]["blocking_stage"], "")

    def test_never_grasped_blocks_at_the_grasp(self) -> None:
        never = _Episode(
            instruction="put_into_bowl",
            object_xy=[(0.0, 0.0)] * 4,
            object_z=[SETTLED_Z] * 4,
            gripper=[0.9] * 4,
            caught=[False] * 4,
        )
        rows = self._terms([never])
        self.assertFalse(rows[0]["recomputed_success"])
        self.assertEqual(rows[0]["blocking_stage"], "no_grasp")

    def test_carried_and_never_released_blocks_at_the_release(self) -> None:
        rows = self._terms([carry((0.0, 0.0))])
        self.assertEqual(rows[0]["blocking_stage"], "no_release")

    def test_released_far_from_the_receptacle_blocks_on_xy(self) -> None:
        far = (RADIUS_BOWL + 0.05, 0.0)
        rows = self._terms([place(far, far)])
        self.assertEqual(rows[0]["blocking_stage"], "xy_miss")

    def test_landing_inside_but_never_settling_blocks_on_settle(self) -> None:
        # Perched on the rim: XY and z both inside tolerance, but the object
        # never falls to its resting height.
        # Released from the carry height -- above the settle margin, so the
        # release step itself is not already a success -- and perched on the
        # rim rather than falling into the receptacle.
        perched = place(
            (0.0, 0.0),
            (0.0, 0.0),
            release_z=CARRY_Z,
            rest_z=BOWL_Z + REST + SETTLE_MARGIN + 0.02,
        )
        rows = self._terms([perched])
        self.assertEqual(rows[0]["blocking_stage"], "not_settled")

    def test_bounce_is_the_xy_travelled_after_the_release(self) -> None:
        # THE bowl hypothesis: released over the centre, ends outside the
        # radius. The tool must call this xy_miss and report a positive bounce,
        # because a policy fix and a release-height fix are different actions.
        out = (RADIUS_BOWL + 0.03, 0.0)
        rows = self._terms([place((0.0, 0.0), out, release_z=CARRY_Z)])
        # xy_miss, NOT not_settled: the object did come to rest, and it came to
        # rest in the wrong place. An earlier version of the funnel asked each
        # term independently over the episode and called this not_settled,
        # because the object was over the centre while still in the gripper.
        self.assertEqual(rows[0]["blocking_stage"], "xy_miss")
        self.assertAlmostEqual(rows[0]["xy_at_release_m"], 0.0, places=4)
        self.assertAlmostEqual(rows[0]["bounce_m"], RADIUS_BOWL + 0.03, places=4)
        self.assertAlmostEqual(
            rows[0]["min_xy_at_rest_m"], RADIUS_BOWL + 0.03, places=4
        )

    def test_release_clearance_is_measured_against_the_receptacle(self) -> None:
        rows = self._terms([place((0.0, 0.0), (0.0, 0.0), release_z=BOWL_Z + 0.03)])
        self.assertAlmostEqual(rows[0]["release_clearance_m"], 0.03, places=4)

    def test_plate_and_bowl_get_their_own_radius(self) -> None:
        # Same geometry, different receptacle: inside the plate's 0.091 and
        # outside the bowl's 0.057. A tool that applied one radius to both
        # would move the funnel's collapse point by a whole stage.
        between = (0.07, 0.0)
        rows = self._terms(
            [
                place(between, between, instruction="put_into_plate"),
                place(between, between, instruction="put_into_bowl"),
            ]
        )
        self.assertTrue(rows[0]["recomputed_success"])
        self.assertEqual(rows[1]["blocking_stage"], "xy_miss")

    def test_a_mixed_batch_classifies_every_world_independently(self) -> None:
        far = (RADIUS_BOWL + 0.05, 0.0)
        rows = self._terms(
            [
                place((0.0, 0.0), (0.0, 0.0)),
                carry((0.0, 0.0)),
                place(far, far),
                _Episode(
                    instruction="put_into_bowl",
                    object_xy=[(0.0, 0.0)] * 4,
                    object_z=[SETTLED_Z] * 4,
                    gripper=[0.9] * 4,
                    caught=[False] * 4,
                ),
            ]
        )
        self.assertEqual(
            [row["blocking_stage"] for row in rows],
            ["", "no_release", "xy_miss", "no_grasp"],
        )

    def test_grasp_timing_and_timeout_separate_the_no_release_causes(self) -> None:
        """`no_release` is 25-33% of every failure population measured so far.

        Either the episode grasped too late to carry and open -- a horizon
        finding that points at placement_grasp_horizon_min_decisions -- or it
        grasped in good time and the release never fired, which is a different
        piece of work entirely. The funnel cannot tell them apart; these two
        columns can.
        """

        # Grasps at step 1, never releases, and uses every step of its budget.
        stalled = carry((0.0, 0.0), steps=8)
        rows = self._terms([stalled])
        self.assertEqual(rows[0]["blocking_stage"], "no_release")
        self.assertEqual(rows[0]["first_grasp_env_step"], 0)
        # horizons is 32 decisions at 4 actions each in the fixture, and the
        # episode ran 8 steps, so it did NOT exhaust the budget.
        self.assertEqual(rows[0]["env_step_budget"], 32 * 4)
        self.assertFalse(rows[0]["timed_out"])

    def test_an_episode_that_never_grasps_reports_minus_one(self) -> None:
        never = _Episode(
            instruction="put_into_bowl",
            object_xy=[(0.0, 0.0)] * 4,
            object_z=[SETTLED_Z] * 4,
            gripper=[0.9] * 4,
            caught=[False] * 4,
        )
        rows = self._terms([never])
        self.assertEqual(rows[0]["first_grasp_env_step"], -1)

    def test_spawn_separation_comes_from_the_first_observed_position(self) -> None:
        """NOT from initial_target_positions, which is stale for this task.

        The resetter updates `initial_target_group` for `held_group` and for
        `grasp_learning` and NOT for `uncaught_container`, so a composed
        episode's entry still holds the pre-repositioning lattice point. Reading
        it reported medians of 0.26-0.35 m against a spawn the config fixes at
        0.06-0.10 m, and the only reason that was caught is that the number
        disagreed with the knob that produces it.
        """

        episode = place((0.08, 0.0), (0.0, 0.0), release_z=CARRY_Z)
        recording, _, _ = _run([episode])
        # Poison the stale field the tool must NOT be reading. If it is, the
        # measurement comes back 0.5 instead of 0.08.
        recording.initial_target_xyz = np.array(
            [[0.5, 0.0, CARRY_Z]], dtype=np.float32
        )
        rows = _episode_terms(recording, _Thresholds(METADATA))
        self.assertAlmostEqual(
            rows[0]["object_to_receptacle_at_reset_m"], 0.08, places=4
        )

    def test_grip_retention_splits_no_release_a_second_way(self) -> None:
        """Ended holding, versus dropped it.

        Orthogonal to `timed_out`. An episode that ran the budget still holding
        never reached the release -- a horizon problem. One that terminated
        early having lost the object grasped and then dropped it -- a grip
        retention problem. The oracle's no_release is the first kind (100% of
        plate's ran the whole budget); the policy's is mostly the second.
        """

        # Grasps, holds to the end, never opens.
        holder = carry((0.0, 0.0), steps=6)
        # Grasps at step 0-1, loses the object at step 2, never opens.
        dropper = _Episode(
            instruction="put_into_bowl",
            object_xy=[(0.0, 0.0)] * 6,
            object_z=[CARRY_Z, CARRY_Z, CARRY_Z, SETTLED_Z, SETTLED_Z, SETTLED_Z],
            gripper=[0.2] * 6,
            caught=[True, True, False, False, False, False],
        )
        rows = self._terms([holder, dropper])
        self.assertEqual(
            [row["blocking_stage"] for row in rows], ["no_release", "no_release"]
        )
        self.assertTrue(rows[0]["ended_holding"])
        self.assertEqual(rows[0]["lost_grip_env_step"], -1)
        self.assertFalse(rows[1]["ended_holding"])
        self.assertEqual(rows[1]["lost_grip_env_step"], 2)

    def test_non_container_worlds_are_dropped(self) -> None:
        # A composed harvest is a mixed batch; move_to and pick_up worlds have
        # no receptacle and must not be counted into a placement denominator.
        recording, _, _ = _run([place((0.0, 0.0), (0.0, 0.0))])
        recording.instruction_ids = np.array(
            [INSTRUCTION_TO_ID["pick_up"]], dtype=np.int64
        )
        self.assertEqual(_episode_terms(recording, _Thresholds(METADATA)), [])


if __name__ == "__main__":
    unittest.main()
