"""Step 4, stage 1: deterministic failures, stochastic attempts, discovery yield."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit.collect_cdpr_full_put_into import plan_repeated_batches, read_scene_uids
from tools.audit.discovery_scenes import (
    discovery_yield,
    failure_mode,
    main,
    select_scenes,
)


class _Scene:
    def __init__(self, uid: str) -> None:
        self.scene_uid = uid


def _row(uid, *, strict=False, grasped=False, lifted=False, slip=False, dest="plate", mode="deterministic", repeat=0, native=None):
    return {
        "scene_uid": uid, "strict": strict, "native": strict if native is None else native,
        "grasped": grasped, "lifted": lifted, "carry_slip": slip, "non_finite": False,
        "destination": dest, "target_catalog": "robocasa_apple",
        "rollout_mode": mode, "repeat_index": repeat, "source": "x",
    }


class PlannerTests(unittest.TestCase):
    def test_repeat_major_so_a_capped_run_covers_every_scene_once(self):
        scenes = [_Scene(f"s{i}") for i in range(4)]
        batches, repeats = plan_repeated_batches(scenes, worlds=2, repeats=3, shard=0, num_shards=1)
        flat = [scene.scene_uid for batch in batches for scene in batch]
        self.assertEqual(flat[:4], ["s0", "s1", "s2", "s3"])
        self.assertEqual(len(flat), 12)
        self.assertEqual([r for batch in repeats for r in batch][:4], [0, 0, 0, 0])
        capped, _ = plan_repeated_batches(scenes, worlds=2, repeats=3, shard=0, num_shards=1, max_rounds=2)
        self.assertEqual({s.scene_uid for batch in capped for s in batch}, {"s0", "s1", "s2", "s3"})

    def test_shards_split_attempts_without_overlap(self):
        scenes = [_Scene(f"s{i}") for i in range(4)]
        a, ra = plan_repeated_batches(scenes, worlds=2, repeats=2, shard=0, num_shards=2)
        b, rb = plan_repeated_batches(scenes, worlds=2, repeats=2, shard=1, num_shards=2)
        pairs_a = {(s.scene_uid, r) for batch, reps in zip(a, ra) for s, r in zip(batch, reps)}
        pairs_b = {(s.scene_uid, r) for batch, reps in zip(b, rb) for s, r in zip(batch, reps)}
        self.assertFalse(pairs_a & pairs_b)
        self.assertEqual(len(pairs_a | pairs_b), 8)

    def test_scene_list_reads_both_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps({"scene_uids": ["a", "b"]}))
            self.assertEqual(read_scene_uids(path), ["a", "b"])
            path.write_text(json.dumps(["c"]))
            self.assertEqual(read_scene_uids(path), ["c"])


class SelectionTests(unittest.TestCase):
    def test_failure_modes(self):
        self.assertIsNone(failure_mode(_row("a", strict=True, grasped=True, lifted=True)))
        self.assertEqual(failure_mode(_row("a")), "no_grasp")
        self.assertEqual(failure_mode(_row("a", grasped=True)), "failed_lift")
        self.assertEqual(failure_mode(_row("a", grasped=True, lifted=True, slip=True)), "carry_slip")
        self.assertEqual(failure_mode(_row("a", grasped=True, lifted=True)), "placement")

    def test_rare_modes_are_oversampled_by_round_robin(self):
        rows = [_row(f"ng{i}") for i in range(50)]
        rows += [_row(f"fl{i}", grasped=True) for i in range(3)]
        rows += [_row(f"ok{i}", strict=True, grasped=True, lifted=True) for i in range(5)]
        result = select_scenes(rows, modes=["no_grasp", "failed_lift"], max_scenes=10, seed=1)
        self.assertEqual(len(result["scene_uids"]), 10)
        self.assertEqual(result["selected_by_mode_destination"], {"failed_lift/plate": 3, "no_grasp/plate": 7})
        self.assertEqual(result["deterministic_strict"], 5)
        self.assertFalse(any(uid.startswith("ok") for uid in result["scene_uids"]))


class YieldTests(unittest.TestCase):
    def test_solved_fraction_counts_scenes_not_attempts(self):
        selection = {
            "scene_uids": ["a", "b"],
            "failure_mode": {"a": "failed_lift", "b": "placement"},
            "destination": {"a": "plate", "b": "bowl"},
            "target_catalog": {"a": "robocasa_apple", "b": "robocasa_apple"},
        }
        attempts = [
            _row("a", mode="stochastic", repeat=0, strict=True),
            _row("a", mode="stochastic", repeat=1, strict=True),
            _row("b", mode="stochastic", repeat=0),
            _row("b", mode="stochastic", repeat=1),
        ]
        report = discovery_yield(selection, attempts)
        self.assertEqual(report["all"]["strict_successes"], 2)
        self.assertEqual(report["all"]["scenes_solved_at_least_once"], 1)
        self.assertEqual(report["by_mode"]["failed_lift"]["strict_per_attempt"], 1.0)
        self.assertEqual(report["by_mode"]["placement"]["solved_scene_fraction"], 0.0)

    def test_deterministic_attempts_are_refused_as_discovery(self):
        selection = {"scene_uids": ["a"], "failure_mode": {"a": "no_grasp"},
                     "destination": {"a": "plate"}, "target_catalog": {"a": "x"}}
        with self.assertRaises(SystemExit):
            discovery_yield(selection, [_row("a")])

    def test_select_cli_reads_a_legacy_attempts_file(self):
        # Attempts written before repeat_index/rollout_mode existed.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts_x.npz"
            np.savez(
                path, schema=np.asarray("s"),
                scene_uid=np.asarray(["a", "b", "c"]), strict=np.asarray([True, False, False]),
                native=np.asarray([True, False, False]), grasped=np.asarray([True, True, False]),
                lifted=np.asarray([True, False, False]), carry_slip=np.zeros(3, bool),
                non_finite=np.zeros(3, bool), destination=np.asarray(["plate", "bowl", "plate"]),
                target_catalog=np.asarray(["robocasa_apple"] * 3),
            )
            output = Path(tmp) / "hard.json"
            self.assertEqual(main(["select", "--attempts", str(path), "--output", str(output), "--max-scenes", "0"]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(sorted(result["scene_uids"]), ["b", "c"])
            self.assertEqual(result["failure_mode"], {"b": "failed_lift", "c": "no_grasp"})


if __name__ == "__main__":
    unittest.main()
