"""The P0 instrument, and the reset field it reads.

These guard the two ways this measurement can be technically correct and mean
something else: reading the cap against the wrong goal point, and reading a
pass out of a verdict that was never applied to the rungs it claims.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _metadata(**overrides):
    base = {
        "random_workspace_gripper_start": True,
        "placement_start_with_caught_object": True,
        "curriculum_cap_includes_z": True,
        "random_workspace_min_goal_xy_distance": 0.10,
        "ee_workspace_x_bounds": [-0.19, 0.19],
        "ee_workspace_y_bounds": [-0.19, 0.19],
        "ee_workspace_z_bounds": [0.19, 0.32],
        "put_plate_release_height": 0.10,
        "put_bowl_release_height": 0.10,
        "pick_grasp_height_offset": 0.0075,
        "curriculum_horizon_coupling_enabled": True,
        "curriculum_horizon_min": 16,
        "curriculum_horizon_max": 32,
        "random_workspace_start_distance_initial": 0.03,
        "random_workspace_start_distance_final": 0.34,
    }
    base.update(overrides)
    return base


_OBJECTS = (
    "robocasa_apple",
    "robocasa_tomato",
    "robocasa_plate",
    "robocasa_bowl",
)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required for the start-distance probe",
)
class StartDistanceProbeTests(unittest.TestCase):
    def _reset(self, *, instructions, caps, metadata=None, rounds=1):
        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
            RankLocalCurriculum,
        )
        from rl_vla_bootstrapping.policy.rank_local_grpo import (
            RankLocalGroupLayout,
        )
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
        )
        from tools.audit.start_distance_probe import build_recording_backend

        backend = build_recording_backend(torch)
        resetter = BatchedReverseFrontierResetter(
            backend=backend,
            layout=RankLocalGroupLayout(
                worlds_per_rank=64, groups_per_rank=8, group_size=8
            ),
            curriculum=RankLocalCurriculum(device=backend.device),
            rank=0,
            base_seed=11,
            instruction_types=tuple(instructions),
            allowed_objects=_OBJECTS,
            frontier_probability=1.0,
            rehearsal_probability=0.0,
            balanced_target_catalogs=True,
            task_metadata=_metadata(**(metadata or {})),
        )
        resetter.set_random_start_max_goal_distance(
            {int(INSTRUCTION_TO_ID[name]): float(cap) for name, cap in caps.items()}
        )
        resets = [
            resetter.reset(update_index=index, round_index=index)
            for index in range(rounds)
        ]
        return backend, resets

    def test_reset_publishes_the_point_the_cap_is_a_radius_around(self):
        """XY of the published goal is the goal slot; Z is the hover height.

        The two halves are checked separately because only the Z half is the one
        that was got wrong: measuring the cap against the receptacle's own
        centre reported a 100% 3-D violation on a reset that was correct, since
        for a placement task the centre sits a release height plus the gripper
        hang below the point the curriculum uses.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            goal_slots_for_reset,
        )

        backend, (reset,) = self._reset(
            instructions=("put_into_plate",), caps={"put_into_plate": 0.06}
        )
        published = reset.curriculum_goal_xyz
        self.assertIsNotNone(published)
        objects = backend.object_positions
        rows = torch.arange(objects.shape[0], dtype=torch.int64)
        goal = objects[rows, goal_slots_for_reset(torch, reset)]
        self.assertLess(
            float(
                torch.linalg.vector_norm(
                    published[:, :2] - goal[:, :2], dim=-1
                ).max()
            ),
            1.0e-5,
        )
        # Release height 0.10 plus the 0.0075 gripper hang above the receptacle.
        self.assertLess(
            float((published[:, 2] - (goal[:, 2] + 0.1075)).abs().max()), 1.0e-5
        )

    def test_no_curriculum_goal_without_the_random_workspace_start(self):
        """No cap, no point. Publishing a plausible zero would be worse."""

        _, (reset,) = self._reset(
            instructions=("put_into_plate",),
            caps={"put_into_plate": 0.06},
            metadata={"random_workspace_gripper_start": False},
        )
        self.assertIsNone(reset.curriculum_goal_xyz)

    def test_collector_goal_slots_delegate_to_the_module_function(self):
        """One definition of the goal slot, not two.

        The collector's ``_goal_slots`` is the consumer that matters and the
        probe reads the module function; if they ever diverge the probe measures
        a different goal than the reward does.
        """

        import inspect

        from rl_vla_bootstrapping.policy import mjwarp_rank_local_collector

        source = inspect.getsource(
            mjwarp_rank_local_collector.RankLocalMJWarpGRPOCollector._goal_slots
        )
        self.assertIn("goal_slots_for_reset(self.torch, reset)", source)

    def test_realized_start_distance_tracks_the_cap(self):
        """The measurement the whole phase-4 trigger rests on."""

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            goal_slots_for_reset,
        )

        medians = []
        for cap in (0.03, 0.06, 0.10):
            backend, resets = self._reset(
                instructions=("put_into_bowl",),
                caps={"put_into_bowl": cap},
                rounds=3,
            )
            # One resetter per cap, so only the last round's poses are on the
            # backend; re-read per round instead of pooling stale state.
            reset = resets[-1]
            objects = backend.object_positions
            rows = torch.arange(objects.shape[0], dtype=torch.int64)
            goal = objects[rows, goal_slots_for_reset(torch, reset)]
            planar = torch.linalg.vector_norm(
                backend.ee_positions[:, :2] - goal[:, :2], dim=-1
            )
            medians.append(float(planar.median()))
            self.assertLessEqual(float(planar.max()), cap + 1.0e-3)
        self.assertLess(medians[0], medians[1])
        self.assertLess(medians[1], medians[2])


class VerdictTests(unittest.TestCase):
    """The verdict has to fail the things it says it fails."""

    @staticmethod
    def _rung(cap, median, over_3d=0.0, over_xy=0.0, spatial_max=None):
        return {
            "cap": cap,
            "planar_median": median,
            "over_cap_fraction": over_xy,
            "over_cap_fraction_3d": over_3d,
            "spatial_max": cap if spatial_max is None else spatial_max,
        }

    def test_a_cap_that_does_not_move_the_start_fails(self):
        from tools.audit.start_distance_probe import verdict

        decision = verdict(
            {
                "pick_up": [
                    self._rung(0.03, 0.020),
                    self._rung(0.06, 0.0201),
                    self._rung(0.10, 0.0202),
                ]
            },
            includes_z=False,
        )
        self.assertFalse(decision["pick_up"]["pass"])
        self.assertIn(
            "medians_do_not_track_cap", decision["pick_up"]["failures"]
        )

    def test_a_responsive_cap_passes(self):
        from tools.audit.start_distance_probe import verdict

        decision = verdict(
            {
                "pick_up": [
                    self._rung(0.03, 0.023),
                    self._rung(0.06, 0.046),
                    self._rung(0.10, 0.076),
                ]
            },
            includes_z=False,
        )
        self.assertTrue(decision["pick_up"]["pass"], decision)

    def test_the_far_rung_is_exempt_from_the_3d_bound_and_reported(self):
        """It is released on purpose, so failing it reports a design decision.

        The exemption is narrow: containers only, 3-D only, at or above
        placement_far_rung_min_cap. The overshoot still travels in the report,
        because it is a real fact about what the top rung guarantees.
        """

        from tools.audit.start_distance_probe import verdict

        decision = verdict(
            {
                "put_into_bowl": [
                    self._rung(0.03, 0.023),
                    self._rung(0.06, 0.046),
                    self._rung(
                        0.10, 0.076, over_3d=0.33, spatial_max=0.127
                    ),
                ]
            },
            includes_z=True,
            far_rung_min_cap=0.09,
        )
        item = decision["put_into_bowl"]
        self.assertTrue(item["pass"], item)
        self.assertEqual(
            item["far_rungs_exempt_from_3d_bound"],
            [
                {
                    "cap": 0.10,
                    "over_cap_fraction_3d": 0.33,
                    "spatial_max": 0.127,
                }
            ],
        )

    def test_the_exemption_does_not_reach_a_bounded_rung(self):
        from tools.audit.start_distance_probe import verdict

        decision = verdict(
            {
                "put_into_bowl": [
                    self._rung(0.03, 0.023, over_3d=0.20),
                    self._rung(0.06, 0.046),
                    self._rung(0.10, 0.076, over_3d=0.33),
                ]
            },
            includes_z=True,
            far_rung_min_cap=0.09,
        )
        self.assertFalse(decision["put_into_bowl"]["pass"])

    def test_the_exemption_does_not_reach_pick_up(self):
        """pick_up has no far-rung z band; its 3-D bound is unconditional."""

        from tools.audit.start_distance_probe import verdict

        decision = verdict(
            {
                "pick_up": [
                    self._rung(0.03, 0.023),
                    self._rung(0.06, 0.046),
                    self._rung(0.10, 0.076, over_3d=0.33),
                ]
            },
            includes_z=True,
            far_rung_min_cap=0.09,
        )
        self.assertFalse(decision["pick_up"]["pass"])


if __name__ == "__main__":
    unittest.main()
