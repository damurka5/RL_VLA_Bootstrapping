"""Retention recorded under the three-stage contract, balanced and scene-disjoint.

The 2026-09-25 retention run mixed in phase4_bank, recorded before the
fixed-world-yaw contract, and its rows asked the residual for the opposite yaw
correction. This bank is harvested by the same checkpoint and config as the
strict SFT bank, in the same rollouts, and only from stages the episode
actually completed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit.build_cdpr_full_put_into_dataset import (
    DEFAULT_TARGETS,
    main as build_main,
)
from tools.audit.collect_cdpr_full_put_into import (
    FullTaskTrace,
    _write_round,
    spread_evenly,
    stage_decisions,
)

CATALOG = DEFAULT_TARGETS[0]
DECISIONS, PER = 10, 2


class _Scene:
    def __init__(self, index: int, destination: str) -> None:
        self.scene_uid = f"scene_{index:04d}"
        self.destination = destination
        self.target_catalog = CATALOG


class HelperTests(unittest.TestCase):
    def test_spread_keeps_ends_and_caps_count(self):
        self.assertEqual(spread_evenly(np.arange(10), 3).tolist(), [0, 4, 9])
        self.assertEqual(spread_evenly(np.arange(2), 5).tolist(), [0, 1])
        self.assertEqual(spread_evenly(np.arange(5), 0).size, 0)

    def test_only_completed_stages_are_returned(self):
        self.assertEqual(stage_decisions(active_decisions=10, grasp_step=-1, lift_step=-1, per=2, include_placement=True), {})
        grasp_only = stage_decisions(active_decisions=10, grasp_step=4, lift_step=-1, per=2, include_placement=True)
        self.assertEqual(sorted(grasp_only), [0])
        self.assertEqual(grasp_only[0].tolist(), [0, 1])
        full = stage_decisions(active_decisions=9, grasp_step=6, lift_step=10, per=2, include_placement=True)
        self.assertEqual(full[0].tolist(), [0, 1, 2])
        self.assertEqual(full[1].tolist(), [3, 4, 5])
        self.assertEqual(full[2].tolist(), [6, 7, 8])
        self.assertNotIn(2, stage_decisions(active_decisions=9, grasp_step=6, lift_step=10, per=2, include_placement=False))


def _round(output: Path) -> None:
    """Eight worlds: four strict, three grasped-and-lifted failures, one miss."""

    worlds = 8
    destinations = ["plate", "bowl"] * 4
    scenes = [_Scene(index, destinations[index]) for index in range(worlds)]
    trace = FullTaskTrace(decisions=DECISIONS, worlds=worlds, actions_per_decision=PER, record_frames=True)
    generator = np.random.default_rng(0)
    trace.state = generator.normal(size=(DECISIONS, worlds, 6)).astype(np.float32)
    trace.prior = generator.normal(size=(DECISIONS, worlds, 4, 5)).astype(np.float32)
    trace.overview = generator.integers(1, 255, size=(DECISIONS, worlds, 2, 3, 3), dtype=np.uint8)
    trace.wrist = generator.integers(1, 255, size=(DECISIONS, worlds, 2, 3, 3), dtype=np.uint8)
    trace.action = generator.normal(size=trace.action.shape).astype(np.float32)
    trace.used_decisions = DECISIONS
    strict = np.array([True] * 4 + [False] * 4)
    for world in range(worlds):
        active = 9 if strict[world] else 10
        trace.decision_active[:active, world] = True
        trace.action_mask[:active, world] = True
    trace.first_grasp_step[:] = [6, 6, 6, 6, 4, 4, 4, -1]
    trace.first_lift_step[:] = [10, 10, 10, 10, 8, 8, -1, -1]
    trace.first_release_step[:4] = 14
    trace.first_strict_step[:4] = 16
    grasped = trace.first_grasp_step >= 0
    lifted = trace.first_lift_step >= 0
    result = {
        "strict": strict,
        "native": strict,
        "approached": grasped,
        "grasped": grasped,
        "lifted": lifted,
        "released": strict,
        "carry_slip": ~strict & lifted,
        "wrong_place": np.zeros(worlds, dtype=bool),
        "non_finite": np.zeros(worlds, dtype=bool),
    }
    report = _write_round(
        output=output, stem="tag_s0_r0000", scenes=scenes, trace=trace, result=result,
        checkpoint=Path("ckpt.pt"), checkpoint_sha256="abc", config=Path("config.yaml"),
        manifest_sha256="m", split="collection", shard=0, round_index=0,
        record_frames=True, retention_rows_per_stage=2,
    )
    # Worlds 4 and 5 lifted (approach + pickup), world 6 only grasped (approach),
    # world 7 never grasped (nothing). Two rows each at most per stage.
    assert report["retention_episodes"] == 3, report
    assert report["retention_rows"] == 2 + 2 + 2 + 2 + 2, report


class RetentionBankTests(unittest.TestCase):
    def test_collect_then_build_both_banks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "bank_shard0"
            shard.mkdir()
            _round(shard)
            records = sorted(shard.glob("record_*.npz"))
            frames = sorted(shard.glob("frames_*.npz"))
            retention = sorted(shard.glob("retention_tag*.npz"))
            retention_frames = sorted(shard.glob("retention_frames_*.npz"))
            self.assertEqual((len(records), len(frames), len(retention), len(retention_frames)), (1, 1, 1, 1))

            common = ["--target-catalogs", CATALOG, "--destinations", "plate", "bowl"]
            self.assertEqual(build_main([
                "--records", *map(str, records), "--frames", *map(str, frames),
                "--output", str(root / "dataset"), "--successes-per-cell", "1", *common,
            ]), 0)
            with np.load(root / "dataset" / "demonstrations.npz") as bank:
                sft_scenes = set(bank["scene_uid"].tolist())
            self.assertEqual(len(sft_scenes), 2)

            self.assertEqual(build_main([
                "--mode", "retention",
                "--records", *map(str, records),
                "--retention-records", *map(str, retention),
                "--exclude-dataset", str(root / "dataset" / "demonstrations.npz"),
                "--frames", *map(str, frames + retention_frames),
                "--output", str(root / "retention"),
                "--rows-per-cell-stage", "2", "--retention-rows-per-stage", "2", *common,
            ]), 0)
            with np.load(root / "retention" / "demonstrations.npz") as bank:
                self.assertEqual(bank["state"].shape[0], 2 * 2 * 3)
                self.assertFalse(set(bank["scene_uid"].tolist()) & sft_scenes)
                for destination in ("plate", "bowl"):
                    for stage in ("move_to", "pick_up", "placement"):
                        cell = (bank["destination"] == destination) & (bank["stage_name"] == stage)
                        self.assertEqual(int(cell.sum()), 2, (destination, stage))
                placement = bank["stage_name"] == "placement"
                self.assertTrue(set(bank["source_group"][placement]) == {"retention_strict_surplus"})
                self.assertTrue(bank["full_chain_success"][placement].all())
            report = json.loads((root / "retention" / "dataset.json").read_text())
            self.assertEqual(report["frames"]["resolved_fraction"], 1.0)
            self.assertTrue(report["contract"]["retention"])

    def test_short_cell_fails_instead_of_spilling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "bank_shard0"
            shard.mkdir()
            _round(shard)
            records = sorted(shard.glob("record_*.npz"))
            common = ["--target-catalogs", CATALOG, "--destinations", "plate", "bowl"]
            build_main([
                "--records", *map(str, records), "--allow-missing-frames",
                "--output", str(root / "dataset"), "--successes-per-cell", "1", *common,
            ])
            with self.assertRaises(SystemExit) as caught:
                build_main([
                    "--mode", "retention", "--records", *map(str, records),
                    "--retention-records", *map(str, shard.glob("retention_tag*.npz")),
                    "--exclude-dataset", str(root / "dataset" / "demonstrations.npz"),
                    "--allow-missing-frames", "--output", str(root / "retention"),
                    "--rows-per-cell-stage", "3", "--retention-rows-per-stage", "2", *common,
                ])
            self.assertIn("placement", str(caught.exception))


class LauncherWiringTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_harvest_launcher_builds_retention_after_the_strict_bank(self):
        source = (self.ROOT / "scripts" / "collect_cdpr_strict_success_dataset_remote.sh").read_text()
        self.assertIn("--retention-rows-per-stage", source)
        self.assertIn("--mode retention", source)
        self.assertIn('--exclude-dataset "$RUN_DIR/dataset/demonstrations.npz"', source)

    def test_sft_launcher_refreshes_retention_and_gates_promotion(self):
        source = (self.ROOT / "scripts" / "train_cdpr_strict_success_sft_remote.sh").read_text()
        self.assertIn('--dataset "$RETENTION_SOURCE"', source)
        self.assertIn("compare_put_into_evaluations.py", source)
        self.assertIn('"verdict": "candidate_better"', source)
        self.assertIn('"selected": "initializer"', source)


if __name__ == "__main__":
    unittest.main()
