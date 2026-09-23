"""Strict policy-only full-task collection and balanced dataset tests."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit.build_cdpr_full_put_into_dataset import (
    DEFAULT_DESTINATIONS,
    DEFAULT_TARGETS,
    build_rows,
    main as build_main,
    select_balanced,
)
from tools.audit.collect_cdpr_full_put_into import (
    SCHEMA as RECORD_SCHEMA,
    FullTaskTrace,
    plan_batches,
)


class _Scene:
    def __init__(self, index: int) -> None:
        self.scene_uid = f"scene_{index:04d}"


class BatchPlanTests(unittest.TestCase):
    def test_shards_are_distinct_and_offset_is_global(self):
        scenes = [_Scene(index) for index in range(40)]
        first = plan_batches(
            scenes, worlds=4, rounds=2, shard=0, num_shards=2,
            scene_offset=8,
        )
        second = plan_batches(
            scenes, worlds=4, rounds=2, shard=1, num_shards=2,
            scene_offset=8,
        )
        a = {scene.scene_uid for batch in first for scene in batch}
        b = {scene.scene_uid for batch in second for scene in batch}
        self.assertFalse(a & b)
        self.assertTrue(all(int(uid[-4:]) >= 8 for uid in a | b))

    def test_rejects_more_work_than_distinct_scenes(self):
        with self.assertRaisesRegex(ValueError, "required"):
            plan_batches(
                [_Scene(index) for index in range(8)],
                worlds=4, rounds=2, shard=0, num_shards=2,
            )


class TraceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_records_policy_inputs_actions_and_ordered_events(self):
        import torch

        trace = FullTaskTrace(
            decisions=2, worlds=2, actions_per_decision=2,
            record_frames=True,
        )
        cameras = type(
            "Cameras", (),
            {
                "overview": torch.full((2, 3, 2, 3), 0.5),
                "wrist": torch.full((2, 3, 2, 3), 0.25),
            },
        )()
        trace.record_decision(
            decision_index=0, cameras=cameras,
            state=torch.ones((2, 5)), prior=torch.zeros((2, 4, 5)),
            active=torch.tensor([True, True]),
        )
        trace.record_step(
            decision_index=0, action_index=0,
            action=torch.ones((2, 5)), active=torch.tensor([True, True]),
            physical_grasp=torch.tensor([False, True]),
            held_lift=torch.tensor([False, False]),
            released=torch.tensor([False, False]),
            native=torch.tensor([False, False]),
            strict=torch.tensor([False, False]),
        )
        trace.record_step(
            decision_index=0, action_index=1,
            action=torch.ones((2, 5)), active=torch.tensor([True, True]),
            physical_grasp=torch.tensor([False, True]),
            held_lift=torch.tensor([False, True]),
            released=torch.tensor([False, True]),
            native=torch.tensor([False, True]),
            strict=torch.tensor([False, True]),
        )
        payload = trace.selected_payload(np.asarray([1]))
        self.assertEqual(payload["state"].shape, (1, 1, 5))
        self.assertEqual(payload["action"].shape, (1, 1, 2, 5))
        self.assertEqual(int(payload["first_grasp_step"][0]), 0)
        self.assertEqual(int(payload["first_lift_step"][0]), 1)
        self.assertEqual(int(payload["first_release_step"][0]), 1)
        frames = trace.frame_payload(
            np.asarray([1]), episode_uid=["strict_s0_r0/r0w1"]
        )
        self.assertEqual(frames["overview"].dtype, np.uint8)
        self.assertEqual(int(frames["overview"][0, 0, 0, 0, 0]), 128)


class BalancedSelectionTests(unittest.TestCase):
    def test_exact_quota_for_every_object_destination_cell(self):
        candidates = []
        for catalog in DEFAULT_TARGETS:
            for destination in DEFAULT_DESTINATIONS:
                for index in range(3):
                    candidates.append(
                        {
                            "path": Path(f"record_{index}.npz"),
                            "column": index,
                            "episode_uid": f"{catalog}_{destination}_{index}",
                            "scene_uid": f"scene_{catalog}_{destination}_{index}",
                            "target_catalog": catalog,
                            "destination": destination,
                        }
                    )
        selected, report = select_balanced(
            candidates,
            target_catalogs=DEFAULT_TARGETS,
            destinations=DEFAULT_DESTINATIONS,
            successes_per_cell=2,
            seed=7,
        )
        self.assertEqual(len(selected), 16)
        self.assertFalse(report["shortages"])
        self.assertTrue(
            all(cell["selected"] == 2 for cell in report["cells"].values())
        )
        self.assertEqual(len({row["scene_uid"] for row in selected}), 16)

    def test_short_cell_is_reported_not_spilled(self):
        candidates = [
            {
                "path": Path("record.npz"), "column": 0,
                "episode_uid": "one", "scene_uid": "scene_one",
                "target_catalog": DEFAULT_TARGETS[0],
                "destination": DEFAULT_DESTINATIONS[0],
            }
        ]
        selected, report = select_balanced(
            candidates,
            target_catalogs=DEFAULT_TARGETS,
            destinations=DEFAULT_DESTINATIONS,
            successes_per_cell=2,
            seed=7,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(report["shortages"]), 8)


class DatasetRowsTests(unittest.TestCase):
    def _record(self, path: Path, *, release_step: int = 12) -> None:
        decisions, worlds, per = 4, 1, 4
        np.savez_compressed(
            path,
            schema=np.asarray(RECORD_SCHEMA),
            state=np.arange(decisions * worlds * 3, dtype=np.float32).reshape(
                decisions, worlds, 3
            ),
            prior=np.zeros((decisions, worlds, 8, 5), dtype=np.float32),
            action=np.ones((decisions, worlds, per, 5), dtype=np.float32),
            action_mask=np.ones((decisions, worlds, per), dtype=bool),
            decision_active=np.ones((decisions, worlds), dtype=bool),
            first_grasp_step=np.asarray([5]),
            first_lift_step=np.asarray([7]),
            first_release_step=np.asarray([release_step]),
            first_strict_step=np.asarray([13]),
            instruction_id=np.asarray([2]),
            instruction_text=np.asarray(["put apple into plate"]),
            episode_uid=np.asarray(["strict_s0_r0000/r0w0"]),
            scene_uid=np.asarray(["scene_a"]),
            split=np.asarray("collection"),
            destination=np.asarray(["plate"]),
            target_catalog=np.asarray(["robocasa_apple"]),
            strict=np.asarray([True]),
            native=np.asarray([True]),
            grasped=np.asarray([True]),
            lifted=np.asarray([True]),
            released=np.asarray([True]),
            carry_slip=np.asarray([False]),
            wrong_place=np.asarray([False]),
            starts_grasped=np.asarray([False]),
            checkpoint_sha256=np.asarray("abc"),
            scene_manifest_sha256=np.asarray("manifest"),
            config=np.asarray("config.yaml"),
        )

    def test_whole_strict_trajectory_becomes_three_semantic_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "record.npz")
            self._record(path)
            dataset = build_rows(
                [
                    {
                        "path": path,
                        "column": 0,
                        "episode_uid": "strict_s0_r0000/r0w0",
                        "scene_uid": "scene_a",
                        "target_catalog": "robocasa_apple",
                        "destination": "plate",
                    }
                ]
            )
        self.assertEqual(dataset["state"].shape, (4, 3))
        self.assertEqual(
            dataset["stage_name"].tolist(),
            ["move_to", "pick_up", "placement", "placement"],
        )
        self.assertTrue(dataset["full_chain_success"].all())
        self.assertFalse(dataset["starts_grasped"].any())
        self.assertFalse(dataset["release_event_repaired"].any())
        self.assertEqual(len(np.unique(dataset["scene_uid"])), 1)

    def test_repairs_only_the_legacy_pre_grasp_release_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "record.npz")
            self._record(path, release_step=0)
            dataset = build_rows(
                [
                    {
                        "path": path,
                        "column": 0,
                        "episode_uid": "strict_s0_r0000/r0w0",
                        "scene_uid": "scene_a",
                        "target_catalog": "robocasa_apple",
                        "destination": "plate",
                    }
                ]
            )
        self.assertTrue(dataset["release_event_repaired"].all())
        self.assertEqual(
            dataset["stage_name"].tolist(),
            ["move_to", "pick_up", "placement", "placement"],
        )

    def test_action_only_cli_writes_auditable_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "record.npz"
            output = root / "dataset"
            self._record(record)
            self.assertEqual(
                build_main(
                    [
                        "--records", str(record),
                        "--output", str(output),
                        "--successes-per-cell", "1",
                        "--target-catalogs", "robocasa_apple",
                        "--destinations", "plate",
                        "--allow-missing-frames",
                    ]
                ),
                0,
            )
            report = json.loads((output / "dataset.json").read_text("utf-8"))
            self.assertFalse(report["priors_stale"])
            self.assertEqual(report["selection"]["selected_total"], 1)
            self.assertEqual(report["dataset"]["episodes"], 1)
            self.assertTrue((output / "demonstrations.npz").is_file())


class LauncherTests(unittest.TestCase):
    def test_remote_launcher_pins_checkpoint_and_strict_quota(self):
        script = Path(
            __file__
        ).resolve().parents[1] / "scripts/collect_cdpr_strict_success_dataset_remote.sh"
        text = script.read_text("utf-8")
        self.assertIn("step_52791642/smolvla_grpo_adapter.pt", text)
        self.assertIn('SUCCESSES_PER_CELL="${SUCCESSES_PER_CELL:-64}"', text)
        self.assertIn("--split collection", text)
        self.assertNotIn("--settle-decisions", text)


if __name__ == "__main__":
    unittest.main()
