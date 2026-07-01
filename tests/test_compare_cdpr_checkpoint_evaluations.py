from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_vla_bootstrapping.cli.compare_cdpr_checkpoint_evaluations import (
    _render_report_markdown,
    _summarize_run,
    main,
)


def _write_manifest(
    run_dir: Path,
    *,
    checkpoint_dir: str,
    instruction_summaries: list[dict],
    normal_scene_canonical_summaries: list[dict] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_dir": checkpoint_dir,
        "generated_at": "2026-07-01T12:00:00",
        "instruction_summaries": instruction_summaries,
        "normal_scene_canonical_summaries": normal_scene_canonical_summaries or [],
        "video_validation": [{"valid": True}],
        "video_coverage": [{"complete": True}, {"complete": False}],
        "total_reset_retries": 3,
        "simulation_instability_episodes": 1,
    }
    (run_dir / "validation_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


class CompareCDPRCheckpointEvaluationsTests(unittest.TestCase):
    def test_summarize_run_computes_weighted_stage_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "step_0192000"
            _write_manifest(
                run_dir,
                checkpoint_dir="/runs/rl/step_0192000",
                instruction_summaries=[
                    {
                        "instruction_type": "move_left",
                        "successes": 7,
                        "episodes": 10,
                    },
                    {
                        "instruction_type": "move_to_object",
                        "successes": 3,
                        "episodes": 10,
                    },
                    {
                        "instruction_type": "grab_object",
                        "successes": 1,
                        "episodes": 10,
                    },
                ],
                normal_scene_canonical_summaries=[
                    {
                        "instruction_type": "move_left",
                        "successes": 6,
                        "episodes": 10,
                    }
                ],
            )

            summary = _summarize_run("step_0192000", run_dir)

            self.assertEqual(summary.overall.successes, 11)
            self.assertEqual(summary.overall.episodes, 30)
            self.assertAlmostEqual(summary.dense_simple.success_rate, 0.5)
            self.assertAlmostEqual(summary.sparse_complex.success_rate, 0.2)
            self.assertAlmostEqual(summary.dense_normal_canonical.success_rate, 0.6)
            self.assertEqual(summary.incomplete_video_coverage, 1)
            self.assertEqual(summary.reset_retries, 3)

    def test_render_report_includes_sparse_gate_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "a"
            run_b = root / "b"
            _write_manifest(
                run_a,
                checkpoint_dir="/runs/rl/step_0192000",
                instruction_summaries=[
                    {
                        "instruction_type": "move_left",
                        "successes": 4,
                        "episodes": 10,
                    }
                ],
            )
            _write_manifest(
                run_b,
                checkpoint_dir="/runs/rl/step_0336000",
                instruction_summaries=[
                    {
                        "instruction_type": "move_left",
                        "successes": 3,
                        "episodes": 10,
                    }
                ],
            )
            run_specs = [("step_0192000", run_a), ("step_0336000", run_b)]
            summaries = [
                _summarize_run(label, path)
                for label, path in run_specs
            ]

            report = _render_report_markdown(
                summaries,
                run_specs=run_specs,
                dense_gate_threshold=0.70,
            )

            self.assertIn("below the 70% gate", report)
            self.assertIn("step_0336000", report)
            self.assertIn("-10.0%", report)

    def test_main_writes_report_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "a"
            run_b = root / "b"
            output_dir = root / "out"
            _write_manifest(
                run_a,
                checkpoint_dir="/runs/rl/step_0192000",
                instruction_summaries=[
                    {
                        "instruction_type": "move_left",
                        "successes": 4,
                        "episodes": 10,
                    }
                ],
            )
            _write_manifest(
                run_b,
                checkpoint_dir="/runs/rl/step_0336000",
                instruction_summaries=[
                    {
                        "instruction_type": "move_left",
                        "successes": 3,
                        "episodes": 10,
                    }
                ],
            )

            result = main(
                [
                    f"step_0192000={run_a}",
                    f"step_0336000={run_b}",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "checkpoint_comparison_report.md").exists())
            self.assertTrue((output_dir / "checkpoint_comparison_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
