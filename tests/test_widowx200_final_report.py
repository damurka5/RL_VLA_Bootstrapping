"""Final reports expose the retention failure and distinguish policy selections."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.train.run_widowx200_pipeline import Pipeline, parse_args


ROOT = Path(__file__).resolve().parents[1]


class FinalReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.p = Pipeline(parse_args(["--pipeline-dir", str(self.directory)]))
        # Reproduce the supplied 2026-11-01 evaluation, including unequal
        # destination counts and a pickup policy improved over its reference.
        self.report = json.loads((ROOT / "docs/artifacts/widowx200_pipeline_20261103/final_report.json").read_text())
        self.p.state["references"] = copy.deepcopy(self.report["references"])
        self.p.state["current_policy"] = str(self.directory / "policy.pt")
        (self.directory / "policy.pt").touch()
        for stage in self.p.spec["stages"]:
            self.p.state["stages"][stage["name"]] = {
                "status": "done", "metric": "validation/success_rate", "best_value": .0,
                "best_step": 100, "best_checkpoint": "best_rl.pt", "policy": "handoff.pt",
                "early_stop": {"reason": "plateau"},
            }

    def finish(self):
        with patch.object(self.p, "write_config", return_value=self.directory / "config.yaml"), \
             patch.object(self.p, "record", side_effect=[copy.deepcopy(self.report[key]) for key in
                 ("move_to", "pick_up", "put_into", "put_into_composed")]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.p.final_report()
        return json.loads((self.directory / "final_report.json").read_text()), (self.directory / "final_report.md").read_text()

    def test_completed_pipeline_reports_observed_retention_failure(self):
        report, markdown = self.finish()
        self.assertEqual(report["retention"]["status"], "failed")
        self.assertEqual(set(report["retention"]["degraded_instructions"]),
                         {"move_to_object", "put_into_bowl", "put_into_plate"})
        self.assertIn("Retention check: failed", markdown)
        self.assertIn("32/520", markdown)
        self.assertIn("does not measure a full approach", markdown)

    def test_low_composed_success_is_not_compared_with_caught_reference(self):
        for label in ("move_to", "pick_up", "put_into"):
            for name, entry in self.report[label].items():
                self.p.state["references"][name]["rate"] = entry["rate"]
        report, _ = self.finish()
        self.assertEqual(report["retention"]["status"], "passed")
        self.assertEqual(report["retention"]["degraded_instructions"], [])

    def test_best_rl_manual_selection_and_sil_handoff_remain_distinct(self):
        entry = self.p.state["stages"]["s6_put_into_binary"]
        entry.update(best_value=.6582, best_step=1016667, selected_step=1764679,
                     selected_checkpoint="manual_rl.pt", selected_value=.6,
                     checkpoint_selection="user_checkpoint", policy="accepted_sil.pt",
                     sil_cycles=[{"accepted": True}, {"accepted": False}])
        report, markdown = self.finish()
        stage = next(row for row in report["stages"] if row["name"] == "s6_put_into_binary")
        self.assertEqual(stage["best_checkpoint"], "best_rl.pt")
        self.assertEqual(stage["selected_checkpoint"], "manual_rl.pt")
        self.assertEqual(stage["policy"], "accepted_sil.pt")
        self.assertEqual((stage["sil_cycles_accepted"], stage["sil_cycles_attempted"]), (1, 2))
        self.assertIn("selected RL (user_checkpoint) `manual_rl.pt` at step 1764679", markdown)
        self.assertIn("SIL accepted/attempted 1/2", markdown)

    def test_missing_evaluation_cannot_pass_retention(self):
        self.p.state["references"] = {"pick_up": self.report["references"]["pick_up"]}
        self.report["pick_up"] = {}
        report, _ = self.finish()
        self.assertEqual(report["retention"]["status"], "incomplete")
        self.assertEqual(report["retention"]["missing_instructions"], ["pick_up"])


if __name__ == "__main__":
    unittest.main()
