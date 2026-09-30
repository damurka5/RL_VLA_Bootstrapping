"""Scene-clustered comparison of repeated put_into evaluations."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.audit.compare_put_into_repeats import holm, main, permutation_p, repeat_discordance
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import grouped_full_task_scenes


def _write_eval(directory: Path, outcomes: dict[str, dict], *, checkpoint: str, **protocol) -> None:
    directory.mkdir(parents=True)
    episodes = [
        {"scene_uid": uid, "destination": "plate" if int(uid[1:]) % 2 else "bowl",
         "target_catalog": "apple", **flags}
        for uid, flags in outcomes.items()
    ]
    report = {
        "checkpoint": checkpoint, "scene_manifest_sha256": "abc", "split": "student_validation",
        "worlds": 4, "rounds": 1, "distinct_scene_rounds": True, "decisions": 128,
        "results": {"episodes": episodes}, **protocol,
    }
    (directory / "evaluation.json").write_text(json.dumps(report))


def _flags(strict: bool) -> dict:
    return {"strict": strict, "native": strict, "grasped": True, "lifted": strict,
            "carry_slip": False, "wrong_place": False}


class PermutationTests(unittest.TestCase):
    def test_null_is_calibrated(self):
        rng = np.random.default_rng(1)
        rates = rng.uniform(0.1, 0.9, size=200)
        pvalues = []
        for _ in range(200):
            base = (rng.random((3, 200)) < rates).astype(float)
            cand = (rng.random((3, 200)) < rates).astype(float)
            pvalues.append(permutation_p(base, cand, rng=rng, resamples=500))
        rejected = np.mean(np.asarray(pvalues) < 0.05)
        self.assertLess(rejected, 0.10)

    def test_detects_a_large_shift(self):
        rng = np.random.default_rng(2)
        base = (rng.random((4, 300)) < 0.3).astype(float)
        cand = (rng.random((4, 300)) < 0.5).astype(float)
        self.assertLess(permutation_p(base, cand, rng=rng, resamples=2000), 0.001)

    def test_settled_scenes_are_uninformative(self):
        rng = np.random.default_rng(3)
        same = np.ones((2, 10))
        self.assertEqual(permutation_p(same, same, rng=rng, resamples=100), 1.0)

    def test_discordance_and_holm(self):
        self.assertEqual(repeat_discordance(np.asarray([[1, 0, 1, 1], [1, 1, 0, 1]])), 0.5)
        self.assertIsNone(repeat_discordance(np.ones((1, 4))))
        adjusted = holm({"a": 0.01, "b": 0.04})
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["b"], 0.04)


class CliTests(unittest.TestCase):
    def test_compares_and_guards_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = [f"s{i}" for i in range(40)]
            for rep in range(2):
                _write_eval(root / f"a{rep}", {uid: _flags(i < 10) for i, uid in enumerate(scenes)}, checkpoint="A")
                _write_eval(root / f"b{rep}", {uid: _flags(i < 30) for i, uid in enumerate(scenes)}, checkpoint="B")
            out = root / "cmp.json"
            main([
                "--checkpoint", f"A={root / 'a0'},{root / 'a1'}",
                "--checkpoint", f"B={root / 'b0'},{root / 'b1'}",
                "--resamples", "2000", "--output", str(out),
            ])
            result = json.loads(out.read_text())
            pair = result["comparisons"]["B"]
            self.assertAlmostEqual(pair["metrics"]["strict"]["difference"], 0.5)
            self.assertEqual(pair["verdict"], "candidate_better")
            self.assertEqual(result["checkpoints"]["A"]["strict_repeat_discordance"], 0.0)

            _write_eval(root / "c0", {uid: _flags(False) for uid in scenes}, checkpoint="C", decisions=64)
            _write_eval(root / "c1", {uid: _flags(False) for uid in scenes}, checkpoint="C", decisions=64)
            with self.assertRaises(SystemExit):
                main(["--checkpoint", f"A={root / 'a0'},{root / 'a1'}",
                      "--checkpoint", f"C={root / 'c0'},{root / 'c1'}"])
            with self.assertRaises(SystemExit):
                main(["--checkpoint", f"A={root / 'a0'},{root / 'b1'}",
                      "--checkpoint", f"B={root / 'b0'}"])


class ValidationPanelTests(unittest.TestCase):
    def test_panel_matches_the_resetter_windows(self):
        scenes = [
            SimpleNamespace(scene_uid=f"u{i}", scene_index=i, destination="plate" if i % 2 else "bowl")
            for i in reversed(range(300))
        ]
        rank0 = grouped_full_task_scenes(scenes, groups=8, rank=0, update_index=0, round_index=0, base_seed=2_000_000)
        rank1 = grouped_full_task_scenes(scenes, groups=8, rank=1, update_index=0, round_index=0, base_seed=2_000_000)
        self.assertEqual(len(rank0), 8)
        # Destinations alternate and the two ranks get disjoint windows.
        self.assertEqual([s.destination for s in rank0], ["plate", "bowl"] * 4)
        self.assertFalse({s.scene_uid for s in rank0} & {s.scene_uid for s in rank1})
        # Deterministic: the panel is the same at every validation.
        again = grouped_full_task_scenes(scenes, groups=8, rank=0, update_index=0, round_index=0, base_seed=2_000_000)
        self.assertEqual([s.scene_uid for s in rank0], [s.scene_uid for s in again])


if __name__ == "__main__":
    unittest.main()
