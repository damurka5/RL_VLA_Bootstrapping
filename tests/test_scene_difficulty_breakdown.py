"""Scene-level breakdown: separates scene-determined failure from chaos."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit.scene_difficulty_breakdown import main, rankdata, scene_icc, spearman


def _write_eval(directory: Path, rows: dict[str, dict], checkpoint: str) -> None:
    directory.mkdir(parents=True)
    report = {
        "checkpoint": checkpoint, "scene_manifest_sha256": "abc", "split": "student_validation",
        "worlds": 4, "rounds": 1, "distinct_scene_rounds": True, "decisions": 128,
        "results": {"episodes": [{"scene_uid": uid, **flags} for uid, flags in rows.items()]},
    }
    (directory / "evaluation.json").write_text(json.dumps(report))


def _episode(uid: str, *, grasp: bool, lift: bool, strict: bool, destination: str, target: str) -> dict:
    return {"destination": destination, "target_catalog": target, "grasped": grasp,
            "lifted": lift, "strict": strict, "native": strict, "carry_slip": False,
            "wrong_place": False, "non_finite": False}


def _write_root(root: Path, uids, probability, *, seed=0, checkpoints=2, repeats=3, destination=None, target=None):
    rng = np.random.default_rng(seed)
    for c in range(checkpoints):
        for r in range(repeats):
            rows = {}
            for i, uid in enumerate(uids):
                grasp = bool(rng.random() < 0.9)
                strict = bool(grasp and rng.random() < probability[i])
                rows[uid] = _episode(uid, grasp=grasp, lift=bool(strict or (grasp and rng.random() < 0.5)), strict=strict,
                                     destination=(destination[i] if destination else ("plate" if i % 2 else "bowl")),
                                     target=(target[i] if target else "apple"))
            _write_eval(root / f"ckpt{c}" / f"rep{r + 1}", rows, checkpoint=f"ckpt{c}")


class StatsTests(unittest.TestCase):
    def test_rankdata_and_spearman(self):
        self.assertEqual(rankdata(np.array([3.0, 1.0, 1.0, 2.0])).tolist(), [4.0, 1.5, 1.5, 3.0])
        x = np.arange(10.0)
        self.assertAlmostEqual(spearman(x, x ** 3), 1.0)
        self.assertIsNone(spearman(x, np.ones(10)))

    def test_icc_separates_systematic_from_chaos(self):
        rng = np.random.default_rng(1)
        chaos = (rng.random((12, 2000)) < 0.4).astype(float)
        self.assertLess(scene_icc(chaos)["icc"], 0.03)
        fixed = np.tile((np.arange(2000) < 800).astype(float), (12, 1))
        self.assertGreater(scene_icc(fixed)["icc"], 0.99)


class CliTests(unittest.TestCase):
    def test_end_to_end_with_systematic_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repeats"
            uids = [f"s{i:03d}" for i in range(200)]
            # Half the scenes are impossible; the rest are coin flips.
            probability = [0.0 if i < 100 else 0.6 for i in range(200)]
            _write_root(root, uids, probability)
            out = Path(tmp) / "out"
            main(["--eval-root", str(root), "--output", str(out)])
            result = json.loads((out / "breakdown.json").read_text())
            strict = result["stages"]["strict"]
            self.assertEqual(result["draws_per_scene"], 6)
            self.assertGreaterEqual(strict["never"], 100)
            self.assertGreater(strict["never"], 2 * strict["expected_never"])
            self.assertGreater(strict["icc"], 0.3)
            classes = json.loads((out / "scene_classes.json").read_text())
            self.assertTrue(set(uids[:100]) <= set(classes["never"]))
            never = result["failure_by_class"]["never"]
            self.assertGreater(never["share_of_all_failures"], 0.6)
            with (out / "scenes.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 200)

    def test_geometry_join_on_a_real_manifest(self):
        from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (
            DEFAULT_SPLIT_WEIGHTS,
            SCENE_MANIFEST_VERSION,
            SceneGeometryConfig,
            generate_scenes,
            manifest_payload,
            write_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            config = SceneGeometryConfig()
            scenes = generate_scenes(count=64, seed=3, config=config)
            manifest = Path(tmp) / "scenes.json"
            write_manifest(manifest, manifest_payload(
                scenes, config=config, seed=3, split_weights=DEFAULT_SPLIT_WEIGHTS,
                split_salt=SCENE_MANIFEST_VERSION))
            uids = [s.scene_uid for s in scenes]
            # Success falls with transport distance: the tool must see it.
            transport = np.array([s.transport_xy_distance for s in scenes])
            probability = (transport < np.median(transport)).astype(float) * 0.9
            _write_root(Path(tmp) / "repeats", uids, probability, repeats=4,
                        destination=[s.destination for s in scenes], target=[s.target_catalog for s in scenes])
            out = Path(tmp) / "out"
            main(["--eval-root", str(Path(tmp) / "repeats"), "--scene-manifest", str(manifest), "--output", str(out)])
            result = json.loads((out / "breakdown.json").read_text())
            self.assertLess(result["geometry"]["transport_xy_distance"]["spearman"]["strict"], -0.5)
            self.assertIn("nearest_distractor_m", result["geometry"])
            self.assertIn("receptacle_y", result["geometry"])
            # Detail tables: 4 quartiles x (all, plate, bowl), episode-pooled.
            table = result["detail"]["target_y"]
            self.assertEqual(len(table), 12)
            self.assertEqual(sum(r["scenes"] for r in table if r["destination"] == "all"), 64)
            self.assertEqual(sum(c["scenes"] for c in result["detail"]["grid"]["cells"]), 64)


class SceneListTests(unittest.TestCase):
    def test_select_scene_list_keeps_order_and_whole_rounds(self):
        from types import SimpleNamespace

        from tools.audit.evaluate_cdpr_full_put_into import select_scene_list

        scenes = [SimpleNamespace(scene_uid=f"u{i}") for i in range(10)]
        picked, rounds = select_scene_list(scenes, ["u7", "u2", "u5", "u1", "u9"], worlds=2)
        self.assertEqual(rounds, 2)
        self.assertEqual([s.scene_uid for s in picked], ["u7", "u2", "u5", "u1"])
        with self.assertRaises(SystemExit):
            select_scene_list(scenes, ["u1", "nope"], worlds=2)
        with self.assertRaises(SystemExit):
            select_scene_list(scenes, ["u1", "u1"], worlds=2)
        with self.assertRaises(SystemExit):
            select_scene_list(scenes, ["u1"], worlds=2)

    def test_refuses_mismatched_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_eval(root / "a" / "rep1", {"s1": _episode("s1", grasp=True, lift=True, strict=True,
                                                             destination="plate", target="apple")}, "a")
            _write_eval(root / "b" / "rep1", {"s2": _episode("s2", grasp=True, lift=True, strict=True,
                                                             destination="plate", target="apple")}, "b")
            with self.assertRaises(SystemExit):
                main(["--eval-root", str(root)])


if __name__ == "__main__":
    unittest.main()
