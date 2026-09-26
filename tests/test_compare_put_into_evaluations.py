"""Paired McNemar comparison of two evaluations on the same scenes."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.audit.compare_put_into_evaluations import compare, main, mcnemar_exact


class McNemarTests(unittest.TestCase):
    def test_exact_values(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)
        self.assertAlmostEqual(mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(mcnemar_exact(2, 8), mcnemar_exact(8, 2))

    def test_only_discordant_scenes_count(self):
        base = {"a": True, "b": True, "c": False, "d": False}
        cand = {"a": True, "b": False, "c": True, "d": True}
        result = compare(base, cand)
        self.assertEqual((result["both"], result["baseline_only"], result["candidate_only"], result["neither"]), (1, 1, 2, 0))

    def test_legacy_video_index_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = []
            for name, kept in (("base", ["s1", "s2"]), ("cand", ["s2", "s3", "s4"])):
                path = Path(tmp) / name
                (path / "videos").mkdir(parents=True)
                (path / "videos" / "videos.json").write_text(json.dumps([{"scene_uid": uid} for uid in kept]))
                (path / "evaluation.json").write_text(json.dumps({
                    "checkpoint": name, "scene_manifest_sha256": "x", "split": "v", "worlds": 4,
                    "rounds": 2, "distinct_scene_rounds": True, "decisions": 128,
                    "videos": {"outcome_filter": "strict"},
                    "results": {"chains": 8, "scenes": 8, "strict": {"rate": len(kept) / 8}},
                }))
                dirs.append(str(path))
            self.assertEqual(main(dirs), 0)


if __name__ == "__main__":
    unittest.main()
