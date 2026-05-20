from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from rl_vla_bootstrapping.cli.plot_cdpr_tensorboard_metrics import (
    ScalarPoint,
    _METRIC_SPECS,
    _render_report_markdown,
    _dedupe_scalar_points,
    _resolve_metric_specs,
    _resolve_run_specs,
    _summarize_metric_points,
)


class PlotCDPRTensorBoardMetricsTests(unittest.TestCase):
    def test_resolve_metric_specs_accepts_metric_aliases(self):
        metrics = _resolve_metric_specs(
            ["action daturation rate", "env reward mean", "success"]
        )

        self.assertEqual(
            [metric.key for metric in metrics],
            ["action_saturation_rate", "env_reward_mean", "success_rate"],
        )

    def test_dedupe_scalar_points_keeps_latest_point_per_step(self):
        points = _dedupe_scalar_points(
            [
                ScalarPoint(step=10, value=1.0, wall_time=100.0),
                ScalarPoint(step=10, value=2.0, wall_time=101.0),
                ScalarPoint(step=20, value=3.0, wall_time=102.0),
            ]
        )

        self.assertEqual(points, [ScalarPoint(step=10, value=2.0, wall_time=101.0), ScalarPoint(step=20, value=3.0, wall_time=102.0)])

    def test_resolve_run_specs_combines_named_and_direct_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                ppo_dir=str(root / "ppo"),
                grpo_dir=str(root / "grpo"),
                run=[f"baseline={root / 'baseline'}"],
            )

            runs = _resolve_run_specs(args)

            self.assertEqual([run.label for run in runs], ["ppo", "grpo", "baseline"])
            self.assertTrue(all(run.path.is_absolute() for run in runs))

    def test_summarize_metric_points_tracks_windowed_trend(self):
        metric = _METRIC_SPECS["action_saturation_rate"]
        points = [
            ScalarPoint(step=100, value=0.50, wall_time=1.0),
            ScalarPoint(step=200, value=0.40, wall_time=2.0),
            ScalarPoint(step=300, value=0.30, wall_time=3.0),
            ScalarPoint(step=400, value=0.20, wall_time=4.0),
            ScalarPoint(step=500, value=0.10, wall_time=5.0),
        ]

        summary = _summarize_metric_points(metric, points)

        self.assertEqual(summary.point_count, 5)
        self.assertEqual(summary.step_start, 100)
        self.assertEqual(summary.step_end, 500)
        self.assertEqual(summary.window_points, 2)
        self.assertAlmostEqual(summary.first_value, 0.50)
        self.assertAlmostEqual(summary.last_value, 0.10)
        self.assertAlmostEqual(summary.start_window_mean, 0.45)
        self.assertAlmostEqual(summary.end_window_mean, 0.15)
        self.assertAlmostEqual(summary.delta, -0.30)
        self.assertEqual(summary.trend, "improving")

    def test_render_report_markdown_includes_overall_takeaways(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [Namespace(label="grpo", path=root / "tensorboard")]
            metric = _METRIC_SPECS["success_rate"]
            summary = _summarize_metric_points(
                metric,
                [
                    ScalarPoint(step=128, value=0.0, wall_time=1.0),
                    ScalarPoint(step=256, value=0.03125, wall_time=2.0),
                    ScalarPoint(step=384, value=0.06250, wall_time=3.0),
                    ScalarPoint(step=512, value=0.09375, wall_time=4.0),
                    ScalarPoint(step=640, value=0.12500, wall_time=5.0),
                ],
            )

            report = _render_report_markdown(
                runs=runs,
                metrics=[metric],
                metric_summaries={metric.key: {"grpo": summary}},
                metric_tags={metric.key: {"grpo": "rollout_step/success_rate_mean"}},
                output_dir=root / "plots",
                combined_path=None,
            )

            self.assertIn("# TensorBoard Metrics Report", report)
            self.assertIn("## Overall Takeaways", report)
            self.assertIn("Best final success rate: `grpo` at 0.1250.", report)
            self.assertIn("`grpo` finishes with the highest final success rate (0.1250).", report)


if __name__ == "__main__":
    unittest.main()
