"""Can you tell, from a checkpoint alone, whether SFT ever touched it?

Before this the answer was no. sil_sft stamps what it writes, but the stamp
lived only inside the file: not in the training log, not in TensorBoard, not in
the run directory. A 22-hour pick_up run was read as a loop iteration when its
launcher refuses resume checkpoints outright and cannot consume an SFT result.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "torch is required"
)
class ProvenanceTests(unittest.TestCase):
    def _write(self, directory, **extra):
        import torch

        payload = {
            "global_step": 6_026_063,
            "policy": {"log_std": torch.zeros(2)},
            "extra_state": {
                "approach_curriculum": {"pick_up": {"cap": 0.03}}
            },
            "vla_lora": {"a": torch.zeros(2)},
            "optimizer": {},
        }
        payload.update(extra)
        path = Path(directory) / "adapter.pt"
        torch.save(payload, path)
        return path

    def test_a_plain_rl_checkpoint_is_reported_as_untouched(self):
        import tempfile

        from tools.audit.checkpoint_provenance import describe, read_provenance

        with tempfile.TemporaryDirectory() as tmp:
            record = read_provenance(self._write(tmp))
        self.assertIsNone(record["sil_sft"])
        self.assertIn("NEVER been through sil_sft".lower(), describe(record).lower())

    def test_a_residual_only_stamp_says_the_adapter_was_untouched(self):
        """The distinction matters: that path copies LoRA over verbatim."""

        import tempfile

        from tools.audit.checkpoint_provenance import describe, read_provenance

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                sil_sft={
                    "dataset": "runs/iter1/dataset/demonstrations.npz",
                    "trained": "residual_only",
                    "epoch": 19,
                    "val_mse": 0.0187,
                },
            )
            record = read_provenance(path)
        text = describe(record)
        self.assertIn("residual only", text)
        self.assertIn("carried over untouched", text)

    def test_a_lora_stamp_reports_the_epoch_that_was_chosen(self):
        import tempfile

        from tools.audit.checkpoint_provenance import describe, read_provenance

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                sil_sft={
                    "dataset": "d.npz",
                    "trained": "residual+vla_lora",
                    "lora_best_epoch": 3,
                    "lora_epochs": 8,
                    "kl_coef": 0.1,
                },
            )
            record = read_provenance(path)
        text = describe(record)
        self.assertIn("residual+vla_lora", text)
        self.assertIn("epoch 3 of 8", text)

    def test_the_curriculum_caps_travel_with_the_report(self):
        """A resume keeps them; a warm start throws them away. That is the
        difference between continuing the ladder and starting it over."""

        import tempfile

        from tools.audit.checkpoint_provenance import read_provenance

        with tempfile.TemporaryDirectory() as tmp:
            record = read_provenance(self._write(tmp))
        self.assertEqual(record["approach_caps"], {"pick_up": 0.03})
        self.assertTrue(record["has_extra_state"])

    def test_a_warm_start_style_payload_reports_no_caps(self):
        import tempfile

        from tools.audit.checkpoint_provenance import read_provenance

        with tempfile.TemporaryDirectory() as tmp:
            record = read_provenance(self._write(tmp, extra_state={}))
        self.assertEqual(record["approach_caps"], {})
        self.assertFalse(record["has_extra_state"])


class TrainerProvenanceWiringTests(unittest.TestCase):
    def test_the_run_logs_provenance_on_both_entry_paths(self):
        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        source = inspect.getsource(mod.main)
        self.assertIn('_log_policy_provenance(\n                dist_ctx, checkpoint, "resume"', source)
        self.assertIn('"warm start"', source)

    def test_it_is_emitted_as_a_scalar_so_it_survives_in_the_event_file(self):
        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        source = inspect.getsource(mod.main)
        self.assertIn('"provenance/started_from_sft"', source)

    def test_it_is_not_summed_across_ranks(self):
        """It is a fact about the run, not a count. Two ranks must not make 2."""

        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        source = inspect.getsource(mod.main)
        merge = source[source.index('"provenance/started_from_sft"') - 4000 :]
        # It is merged into the same dict as approach_curriculum.metrics(),
        # which is added AFTER the collective -- verified by the caps logging
        # 0.03 rather than 0.06 on two ranks.
        self.assertIn("approach_curriculum.metrics()", merge)

    def test_a_broken_checkpoint_does_not_block_a_launch(self):
        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        source = inspect.getsource(mod._log_policy_provenance)
        self.assertIn("except Exception", source)
        self.assertIn("provenance unavailable", source)


if __name__ == "__main__":
    unittest.main()


class LoopClosureTests(unittest.TestCase):
    """The step that lets an accepted SFT result get back into RL.

    Until this launcher existed the loop could harvest, train and reach a
    verdict, and then had nowhere to put the answer: the catch_release launcher
    warm-starts weights only and refuses CHECKPOINT outright.
    """

    LAUNCHER = (
        ROOT
        / "scripts"
        / "train_cdpr_smolvla_catch_release_grpo_mjlab_dual_remote_resume.sh"
    )

    def test_the_resume_launcher_exists_and_takes_a_checkpoint(self):
        text = self.LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('export RLVLA_SMOLVLA_RESUME_CHECKPOINT="$CHECKPOINT"', text)
        self.assertIn("CHECKPOINT is required", text)

    def test_it_resumes_rather_than_warm_starts(self):
        """A warm start discards extra_state, where the earned caps live."""

        text = self.LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("RLVLA_SMOLVLA_WARMSTART_CHECKPOINT", text)
        self.assertIn("load_weights_only discards them", text)

    def test_it_defaults_to_the_phase_four_placement_config(self):
        text = self.LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("cdpr_smolvla_phase4_placement_loop.yaml", text)

    def test_it_prints_provenance_by_an_absolute_path(self):
        """It runs before the cd into REPO_ROOT, and || true hides a failure.

        A relative path here resolves against wherever the launcher was invoked
        from, so the one line that answers "did the demonstrations get used?"
        silently prints nothing.
        """

        text = self.LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            '"$REPO_ROOT/tools/audit/checkpoint_provenance.py"', text
        )
        index = text.index("checkpoint_provenance.py")
        self.assertLess(index, text.index('cd "$REPO_ROOT"'))

    def test_it_carries_the_run_naming_guard(self):
        text = self.LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("cdpr_compose_run_name", text)
        self.assertIn("cdpr_guard_run_dir", text)
