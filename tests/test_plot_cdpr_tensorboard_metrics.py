from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from rl_vla_bootstrapping.cli.plot_cdpr_tensorboard_metrics import (
    ScalarPoint,
    _dedupe_scalar_points,
    _resolve_metric_specs,
    _resolve_run_specs,
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


if __name__ == "__main__":
    unittest.main()
