"""The untouched checkpoint is a selection candidate, and retention is measured.

Two defects of the 2026-09-23/25 strict-success SFT runs:

1. ``best`` started at infinity, so epoch 0 was always saved -- a residual
   already worse than its initializer was written and evaluated.
2. Retention rows were 20% of each batch and nothing reported their share of
   the loss or the gradient, or their held-out error. Selection saw put_into
   rows only.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_vla_bootstrapping.policy.octo_finetune_cdpr import ResidualChunkActor
from tools.audit import sil_sft

STATE_DIM, CHUNK, ACTION_DIM, SLOTS, HIDDEN = 6, 4, 5, 2, 16


def _checkpoint(root: Path) -> tuple[Path, ResidualChunkActor]:
    torch.manual_seed(0)
    actor = ResidualChunkActor(
        state_dim=STATE_DIM, chunk_size=CHUNK, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN, residual_scale=1.0,
    )
    policy = {f"actor.{key}": value for key, value in actor.state_dict().items()}
    policy["log_std"] = torch.zeros(ACTION_DIM)
    payload = {
        "policy": policy,
        "args": {"hidden_dim": HIDDEN, "residual_scale": 1.0},
        "state_dim": STATE_DIM,
        "chunk_size": CHUNK,
        "action_dim": ACTION_DIM,
        "hidden_dim": HIDDEN,
        "residual_scale": 1.0,
        "global_step": 11,
    }
    path = root / "source.pt"
    torch.save(payload, path)
    return path, actor


def _bank(actor, *, scenes: int, offset: float, noise: float, seed: int, prefix: str) -> dict:
    generator = np.random.default_rng(seed)
    rows = scenes * 12
    state = generator.normal(size=(rows, STATE_DIM)).astype(np.float32)
    prior = generator.normal(scale=0.3, size=(rows, CHUNK, ACTION_DIM)).astype(np.float32)
    with torch.no_grad():
        own = actor(torch.as_tensor(state), torch.as_tensor(prior))[:, :SLOTS].numpy()
    action = own + offset + generator.normal(scale=noise, size=own.shape).astype(np.float32)
    stage = np.tile(np.repeat(np.arange(3), 4), scenes)
    scene = np.repeat([f"{prefix}_scene_{i:03d}" for i in range(scenes)], 12)
    return {
        "state": state,
        "prior": prior,
        "action": action.astype(np.float32),
        "action_mask": np.ones((rows, SLOTS), dtype=bool),
        "scene_uid": scene,
        "episode_uid": np.asarray([f"{name}/r0w0" for name in scene]),
        "stage_id": stage.astype(np.int8),
        "stage_name": np.asarray(sil_sft.STAGE_NAMES)[stage],
        "destination": np.asarray(["plate", "bowl"] * (rows // 2)),
        "target_catalog": np.asarray(["robocasa_apple"] * rows),
        "instruction_id": np.zeros(rows, dtype=np.int64),
    }


def _save(root: Path, name: str, bank: dict) -> Path:
    directory = root / name
    directory.mkdir()
    path = directory / "demonstrations.npz"
    np.savez(path, **bank)
    return path


def _run(root: Path, checkpoint: Path, dataset: Path, *extra: str) -> dict:
    output = root / "model"
    code = sil_sft.main([
        "--dataset", str(dataset), "--checkpoint", str(checkpoint),
        "--output", str(output), "--device", "cpu", "--epochs", "4",
        "--batch-size", "64", "--lr", "3e-2", "--val-fraction", "0.25",
        "--seed", "3", "--split-by", "scene", "--progress", "never", *extra,
    ])
    assert code == 0
    return json.loads((output / "sft_report.json").read_text())


class InitializerSelectionTests(unittest.TestCase):
    def test_noise_only_bank_selects_the_initializer_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, actor = _checkpoint(root)
            # The untouched actor is already the best predictor of held-out
            # rows: targets are its own output plus independent noise.
            dataset = _save(root, "main", _bank(actor, scenes=24, offset=0.0, noise=0.3, seed=1, prefix="m"))
            report = _run(root, checkpoint, dataset)
            self.assertEqual(report["selected"], "initializer")
            self.assertFalse(report["adapter_written"])
            self.assertFalse((root / "model" / "sil_sft_adapter.pt").exists())
            self.assertEqual(report["best_epoch"], -1)
            self.assertIsNone(report["best_val_mse"])
            self.assertEqual(report["initializer_selection_score"], 1.0)
            self.assertGreaterEqual(min(row["selection_score"] for row in report["history"]), 1.0)

    def test_learnable_bank_selects_an_epoch_and_writes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, actor = _checkpoint(root)
            dataset = _save(root, "main", _bank(actor, scenes=24, offset=0.2, noise=0.01, seed=1, prefix="m"))
            report = _run(root, checkpoint, dataset)
            self.assertTrue(report["selected"].startswith("residual_epoch_"))
            self.assertTrue(report["adapter_written"])
            self.assertLess(report["best_selection_score"], 1.0)


class RetentionMeasurementTests(unittest.TestCase):
    def test_conflicting_retention_is_visible_in_loss_and_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, actor = _checkpoint(root)
            main = _save(root, "main", _bank(actor, scenes=24, offset=0.2, noise=0.01, seed=1, prefix="m"))
            # Asks the same weights for the OPPOSITE correction.
            retention = _save(root, "retention", _bank(actor, scenes=24, offset=-0.2, noise=0.01, seed=2, prefix="r"))
            report = _run(
                root, checkpoint, main,
                "--retention-dataset", str(retention), "--retention-fraction", "0.2",
            )
            retention_report = report["retention"]
            self.assertGreater(retention_report["rows_val"], 0)
            self.assertIsNotNone(retention_report["baseline_val"])
            self.assertLess(retention_report["baseline_gradients"]["gradient_cosine"], 0.0)
            epoch = report["history"][0]
            for key in (
                "retention_val_mse", "retention_loss_share", "main_move_to_loss_share",
                "retention_placement_loss_share", "retention_gradient_share",
                "gradient_cosine",
            ):
                self.assertIn(key, epoch)
            shares = [value for key, value in epoch.items() if key.endswith("_loss_share") and key != "retention_loss_share"]
            self.assertAlmostEqual(sum(shares), 1.0, places=4)
            self.assertAlmostEqual(
                epoch["retention_loss_share"],
                sum(epoch[f"retention_{name}_loss_share"] for name in sil_sft.STAGE_NAMES),
                places=4,
            )

    def test_selection_score_weights_each_source_by_its_own_baseline(self):
        self.assertEqual(sil_sft.selection_score(2.0, 2.0), 1.0)
        # put_into improves 10%, retention doubles: at a 0.2 share it loses.
        score = sil_sft.selection_score(
            0.9, 1.0, retention_val=2e-4, retention_base=1e-4, retention_fraction=0.2
        )
        self.assertAlmostEqual(score, 0.8 * 0.9 + 0.2 * 2.0)
        self.assertGreater(score, 1.0)


if __name__ == "__main__":
    unittest.main()
