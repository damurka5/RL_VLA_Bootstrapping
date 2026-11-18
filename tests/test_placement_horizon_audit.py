"""Extra time must not get credit for rerun variation or post-termination motion."""
import copy
import unittest

import numpy as np

from tools.audit.placement_horizon_audit import compare_rounds, episode_rows, summarize


RADII = {"put_into_plate": .073}


def fixture(steps=4, success_step=None):
    xyz = np.zeros((steps, 1, 2, 3), dtype=np.float32)
    xyz[:, 0, 0, 0] = np.linspace(.10, .02, steps)
    success = np.zeros((steps, 1), dtype=bool)
    active = np.ones_like(success)
    terminated = np.zeros_like(success)
    if success_step is not None:
        success[success_step, 0] = terminated[success_step, 0] = True
        active[success_step + 1:] = False
    return dict(
        active=active, success=success, terminated=terminated,
        actions=np.zeros((steps, 1, 5)), ee_xyz=np.zeros((steps, 1, 3)), object_xyz=xyz,
        caught_target=np.ones((steps, 1), dtype=bool), live_diverged=np.array([False]),
        metadata=dict(instructions=["put_into_plate"], catalogs=["orange"], group_size=1),
        target_slots=np.array([0]), reference_slots=np.array([1]), second_reference_slots=np.array([-1]),
        target_catalog_ids=np.array([0]), instruction_ids=np.array([0]), instructions=np.array(["put orange on plate"]),
        horizons=np.array([steps // 2]), actions_per_decision=2, round_index=1000,
        start_distance_cap=.18, contact_based_grasp=True,
        release_threshold=np.array([.8]), target_rest_height=np.array([.02]), support_surface_z=np.array([.15]),
        reset_object_xyz=np.zeros((1, 2, 3)), reset_ee_xyz=np.zeros((1, 3)),
        reset_object_quaternions=np.zeros((1, 2, 4)), reset_ee_quaternion=np.zeros((1, 4)),
    )


def paired():
    before = fixture()
    after = fixture(8, success_step=6)
    after["object_xyz"][:4] = before["object_xyz"]
    return before, after


def compare(before, after):
    return compare_rounds(before, after, episode_rows(before, RADII), episode_rows(after, RADII))[0]


class HorizonTests(unittest.TestCase):
    def test_held_over_plate_is_a_failed_timeout_not_success(self):
        row = episode_rows(fixture(), RADII)[0]
        self.assertEqual(row["category"], "timeout_holding")
        self.assertTrue(row["within_radius_final"])
        self.assertFalse(row["success"])
        self.assertEqual(summarize([row])[0]["timeout_holding_within_radius"], 1)

    def test_uses_last_active_step_and_preserves_early_failure(self):
        data = fixture()
        data["active"][2:] = False
        data["caught_target"][1] = False
        data["object_xyz"][2:, 0, 0] = 999
        row = episode_rows(data, RADII)[0]
        self.assertEqual(row["category"], "early_failure")
        self.assertFalse(row["held_final"])
        self.assertLess(row["final_xy_m"], 1)

    def test_live_divergence_is_not_a_timeout_candidate(self):
        data = fixture()
        data["live_diverged"][:] = True
        row = episode_rows(data, RADII)[0]
        self.assertEqual(row["category"], "live_divergence")
        self.assertEqual(summarize([row])[0]["timeout_holding_within_radius"], 0)

    def test_matching_prefix_and_later_success_is_a_budget_rescue(self):
        row = compare(*paired())
        self.assertTrue(row["prefix_matches"])
        self.assertTrue(row["rescued_after_original_budget"])

    def test_changed_actions_are_not_a_budget_rescue(self):
        before, after = paired()
        after["actions"][1, 0, 0] = .1
        row = compare(before, after)
        self.assertTrue(row["extended_success"])
        self.assertFalse(row["prefix_matches"])
        self.assertFalse(row["rescued_after_original_budget"])

    def test_success_before_old_limit_is_rerun_variation(self):
        before, after = paired()
        after["success"][2] = True
        after["active"][3:] = False
        row = compare(before, after)
        self.assertFalse(row["prefix_matches"])
        self.assertFalse(row["rescued_after_original_budget"])

    def test_different_reset_or_contract_refuses_paired_claims(self):
        before, after = paired()
        after["reset_object_xyz"][0, 0, 0] = .01
        with self.assertRaisesRegex(ValueError, "reset_object_xyz"):
            compare(before, after)
        before, after = paired()
        after["metadata"] = copy.deepcopy(after["metadata"])
        after["metadata"]["group_size"] = 2
        with self.assertRaisesRegex(ValueError, "contracts"):
            compare(before, after)

    def test_unchanged_horizon_is_rejected(self):
        before, after = paired()
        after["horizons"][:] = before["horizons"]
        with self.assertRaisesRegex(ValueError, "exceed"):
            compare(before, after)


if __name__ == "__main__":
    unittest.main()
