"""Diagnostic samples preserve reset identity, verdicts and termination boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.audit.policy_outcome_videos import episode_frames, outcome_rows, pipeline_selection, sliced_rates, tracked_worlds


class OutcomeTests(unittest.TestCase):
    def test_resolves_sharded_eval_and_separates_rl_from_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "eval/final/put_into_composed"
            folder.mkdir(parents=True)
            (root / "pipeline_state.json").write_text(json.dumps({"stages": {"s6": {
                "selected_checkpoint": "chosen.pt", "best_checkpoint": "best.pt", "policy": "sil.pt"}}}))
            (root / "final_report.json").write_text(json.dumps({"policy": "joint.pt"}))
            (folder / "summary.json").write_text(json.dumps({"shard_summaries": ["summary_cuda0.json"]}))
            (folder / "summary_cuda0.json").write_text(json.dumps({"config": "original.yaml", "start_distance_cap": .18}))
            final = pipeline_selection(root, "put_into_composed", "final", "handoff")
            self.assertEqual(final, {"checkpoint": Path("joint.pt"), "config": Path("original.yaml"), "start_distance_cap": .18})
            self.assertEqual(pipeline_selection(root, "put_into_composed", "s6", "rl")["checkpoint"], Path("chosen.pt"))
            self.assertEqual(pipeline_selection(root, "put_into_composed", "s6", "handoff")["checkpoint"], Path("sil.pt"))
            (folder / "summary.json").write_text(json.dumps({"failed_shards": ["cuda:1"]}))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                pipeline_selection(root, "put_into_composed", "final", "handoff")

    def test_tracks_distinct_scenes_reproducibly_without_copy_duplicates(self):
        worlds = tracked_worlds(512, 8, 16, 7)
        self.assertEqual(worlds, tracked_worlds(512, 8, 16, 7))
        self.assertEqual(len(set(w // 8 for w in worlds)), 16)
        self.assertTrue(all(w % 8 == 0 and 0 <= w < 512 for w in worlds))
        self.assertNotEqual(worlds, tracked_worlds(512, 8, 16, 8))
        self.assertEqual(tracked_worlds(16, 8, 99, 0), [0, 8])
        with self.assertRaises(ValueError):
            tracked_worlds(15, 8, 2, 0)

    def test_stops_at_last_active_decision_and_appends_terminal_image(self):
        rec = SimpleNamespace(episode_length=np.array([5, 0, 12]), actions_per_decision=4)
        # Three decision observations, then an extra final-state render.
        self.assertEqual(episode_frames([10, 20, 30, 99], rec, 0), [10, 20, 99])
        self.assertEqual(episode_frames([10, 20, 30, 99], rec, 1), [])
        self.assertEqual(episode_frames([10, 20, 30, 99], rec, 2), [10, 20, 30, 99])

    def test_rates_include_unfilmed_failures_and_count_shared_reset_groups(self):
        common = dict(instruction="pick_up", object="apple", target_reset_x=.123456,
                      target_reset_y=.2, round=0, scene=0)
        rows = [{**common, "success": True, "ever_grasped": True},
                {**common, "success": False, "ever_grasped": True},
                {**common, "round": 1, "success": False, "ever_grasped": False}]
        summary = sliced_rates(rows)
        result = summary["by_instruction"][0]
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["distinct_reset_groups"], 2)
        self.assertAlmostEqual(result["ever_grasped_rate"], 2 / 3)
        self.assertEqual(summary["by_spawn"][0]["target_reset_x"], .123)

    def test_rows_use_pre_action_positions_and_keep_failed_verdict(self):
        from tools.audit.sil_record import _Recording

        steps, worlds = 2, 2
        record = _Recording(
            actions=np.zeros((steps, worlds, 5)), active=np.ones((steps, worlds), dtype=bool),
            success=np.array([[False, False], [True, False]]),
            terminated=np.zeros((steps, worlds), dtype=bool),
            caught_target=np.zeros((steps, worlds), dtype=bool),
            ee_xyz=np.ones((steps, worlds, 3)), gripper_opening=np.ones((steps, worlds)),
            object_xyz=np.ones((steps, worlds, 1, 3)), instruction_ids=np.array([8, 8]),
            target_slots=np.zeros(worlds, dtype=int), reference_slots=np.full(worlds, -1),
            second_reference_slots=np.full(worlds, -1), horizons=np.ones(worlds, dtype=int),
            initial_target_xyz=np.zeros((worlds, 3)), support_surface_z=np.zeros(worlds),
            release_threshold=np.ones(worlds), target_rest_height=np.zeros(worlds),
            physical_grasp_at_reset=np.zeros(worlds, dtype=bool), instructions=np.array(["pick", "pick"]),
            actions_per_decision=2, round_index=0, diverged_worlds=0, pick_lift_success_height=.05,
            reset_object_xyz=np.full((worlds, 1, 3), .123), reset_ee_xyz=np.full((worlds, 3), .456),
            start_distance_cap=.2, diverged_world_mask=np.array([False, True]),
        )
        rows = outcome_rows(record)
        self.assertEqual([r["success"] for r in rows], [True, False])
        self.assertAlmostEqual(rows[0]["target_reset_x"], .123)
        self.assertAlmostEqual(rows[0]["ee_reset_z"], .456)
        self.assertIsNone(rows[0]["container_reset_x"])
        self.assertTrue(rows[1]["diverged"])
        self.assertEqual(rows[0]["object"], "unknown")


if __name__ == "__main__":
    unittest.main()
