"""Stop diagnosis must identify the monitored protocol and keep NaNs visible."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit.inspect_widowx200_stage7 import inspect


class Stage7InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.rl = self.run / "rl"
        self.rl.mkdir()
        self.write("config.json", {"early_stop_metric": "min_by_instruction_composed",
                                   "early_stop_min_steps": 5000000, "max_train_steps": 30000000})

    def write(self, name, value):
        (self.rl / name).write_text(json.dumps(value))

    def row(self, step, composed, pickup=.7, cap=.18):
        return {"global_step": step, "validation/fixed_start_distance": 1.,
                "validation/start_distance_cap/put_into_bowl": cap,
                "validation/by_instruction/move_to_object/success_rate": .8,
                "validation/by_instruction/pick_up/success_rate": pickup,
                "validation/by_instruction/put_into_bowl/success_rate": .9,
                "validation_composed/by_instruction/put_into_bowl/success_rate": composed,
                "validation_composed/by_instruction/put_into_bowl/episodes": 512}

    def curve(self, rows):
        (self.rl / "validation.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def test_selects_composed_score_and_the_actual_best_checkpoint(self):
        self.curve([self.row(250000, .05357142857142857), self.row(5006757, .044642857142857144)])
        self.write("early_stop.json", {"reason": "plateau", "global_step": 5006757})
        checkpoint = self.rl / "step_0250000" / "smolvla_grpo_adapter.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"fixture, never loaded")
        report = inspect(self.run)
        self.assertEqual(report["best"]["step"], 250000)
        self.assertEqual(report["best"]["bottleneck"], "put_into_bowl")
        self.assertEqual(report["best"]["checkpoint"], str(checkpoint.resolve()))
        self.assertEqual(report["validation_curve"][-1]["episodes"]["composed_bowl"], 512)
        self.assertEqual(report["stop"]["reason"], "plateau")

    def test_pickup_can_be_the_bottleneck_even_with_composed_metric(self):
        self.curve([self.row(100, .1, pickup=.02)])
        best = inspect(self.run)["best"]
        self.assertEqual(best["bottleneck"], "pick_up")
        self.assertEqual(best["score"], .02)

    def test_ignores_validation_from_other_caps(self):
        self.write("config.json", {"early_stop_metric": "min_by_instruction_composed",
                                   "validation_start_distance_caps": ["put_into_bowl=0.18"]})
        self.curve([self.row(100, .9, cap=.02), self.row(200, .05)])
        report = inspect(self.run)
        self.assertEqual(report["validation_rows"], 2)
        self.assertEqual(report["matching_validation_rows"], 1)
        self.assertEqual(report["best"]["step"], 200)

    def test_nonfinite_force_is_counted_and_report_stays_standard_json(self):
        metrics = {"global_step": 5006757, "left_pad_normal_force_mean_n": float("nan"),
                   "right_pad_normal_force_mean_n": float("inf"), "candidate_reward_mean": .5,
                   "curriculum/placement_caught_fraction": .5}
        path = self.rl / "metrics.jsonl"
        path.write_text(json.dumps(metrics) + "\n")
        before = path.read_bytes()
        report = inspect(self.run)
        self.assertEqual(report["non_finite_metric_updates"]["left_pad_normal_force_mean_n"], 1)
        self.assertIsNone(report["last_training_metrics"]["left_pad_normal_force_mean_n"])
        json.dumps(report, allow_nan=False)
        self.assertEqual(path.read_bytes(), before)
        self.assertIsNone(report["stop"])

    def test_incomplete_last_validation_is_reported_without_losing_good_rows(self):
        self.curve([self.row(100, .05)])
        with (self.rl / "validation.jsonl").open("a") as handle:
            handle.write('{"global_step":')
        report = inspect(self.run)
        self.assertEqual(report["matching_validation_rows"], 1)
        self.assertEqual(len(report["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()
