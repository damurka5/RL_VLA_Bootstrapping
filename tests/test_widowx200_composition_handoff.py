"""An interrupted old joint run cannot bypass final-prompt composition SFT."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.audit.sil_artifacts import atomic_savez
from tools.train import run_widowx200_pipeline as module
from tests.test_widowx200_pipeline_safety import pipeline, write_pair


class CompositionHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.p = pipeline(self.root / "production")
        self.stage = self.p.spec["stages"][-1]
        self.record = {"status": "pending"}
        self.initializer = self.root / "stage6_best.pt"
        self.initializer.write_bytes(b"stage6 policy")
        for stage in self.p.spec["stages"][:-1]:
            self.p.state["stages"][stage["name"]] = {"status": "done", "policy": str(self.initializer)}
        self.p.state["stages"][self.stage["name"]] = self.record
        self.p.state["current_policy"] = str(self.initializer)
        for skill in ("move_to", "pick_up", "put_into"):
            replay, frames = write_pair(self.root / skill)
            self.p.state["banks"][skill] = {"stage": f"best_{skill}", "checkpoint": f"best_{skill}.pt",
                "provenance": "policy", "replays": [str(replay)], "frames": [str(frames)], "mean_rate": .5,
                "cap": .2, "frame_selection": "scene_spread", "harvest_first_round": 100}
        self.commands = []
        self.root_patch = patch.object(module, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def run_child(self, command, **kwargs):
        self.commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if command[1].endswith("sil_sft.py"):
            (output / "sil_sft_adapter.pt").write_bytes(b"fitted student")
            (output / "sft_report.json").write_text(json.dumps({"selected": "residual_epoch_1"}))
        else:
            atomic_savez(output / "demonstrations.npz", state=np.zeros((1, 2)), prior=np.zeros((1, 1, 5)),
                         action=np.zeros((1, 1, 5)), episode_uid=np.array(["source/r0w0"]),
                         scene_uid=np.array(["source/r0s0"]),
                         teacher_instruction_text=np.array(["pick up apple"]), sft_validation=np.array([True]))
            (output / "dataset.json").write_text(json.dumps({"priors_stale": not command[1].endswith("sil_refresh_priors.py")}))
            if command[1].endswith("sil_refresh_priors.py"):
                (output / "refresh.json").write_text(json.dumps({"checkpoint": str(self.initializer)}))

    def preparation(self):
        with patch.object(module, "run_command", side_effect=self.run_child), \
             patch.object(self.p, "evaluate_all", return_value={"pick_up": {"rate": .1}}), \
             patch.object(self.p, "record", return_value={"put_into_plate": {"rate": .05}}), \
             patch.object(self.p, "write_config", return_value=self.root / "eval.yaml"):
            return self.p.prepare_composition(self.stage, self.record, self.initializer)

    def test_sft_relabels_and_refreshes_before_joint_rl_and_is_resumable(self):
        result = self.preparation()
        names = [Path(c[1]).name for c in self.commands]
        self.assertEqual(names, ["sil_record.py", "build_widowx200_composition_sft.py", "sil_refresh_priors.py", "sil_sft.py"])
        sft = self.commands[-1]
        self.assertIn("--frames", sft)
        self.assertEqual(sft[sft.index("--split-by") + 1], "provided")
        self.assertEqual(sft[sft.index("--sampler") + 1], "balanced")
        self.assertEqual(self.p.state["composition_sft"]["status"], "done")
        self.assertEqual(self.record["composition_sft_id"], self.p.state["composition_sft"]["id"])
        self.assertEqual(self.p.state["current_policy"], str(result))
        self.assertEqual(self.preparation(), result)
        self.assertEqual(len(self.commands), 4)
        # Already-correct joint resumption no longer depends on historical bank files.
        self.p.state["banks"].clear()
        self.assertEqual(self.preparation(), result)

    def test_joint_training_receives_sft_weights_instead_of_the_stage6_initializer(self):
        result = self.preparation()
        selected = self.root / "joint_best.pt"
        selected.write_bytes(b"joint result")
        with patch.object(self.p, "write_config", return_value=self.root / "joint.yaml"), \
             patch.object(self.p, "train", return_value=self.root / "rl") as train, \
             patch.object(module, "select_best_checkpoint", return_value={"best_checkpoint": str(selected),
                 "best_value": .1, "best_step": 100, "metric": "min_by_instruction_composed"}):
            self.p.run_stage(6, self.stage, self.record)
        self.assertEqual(train.call_args.args[2], result)
        self.assertEqual(self.record["init"], str(result))

    def test_old_joint_latest_is_preserved_and_cannot_bypass_sft(self):
        run = self.root / "runs" / f"{self.p.name}__{self.stage['name']}"
        (run / "rl").mkdir(parents=True)
        latest = run / "rl/latest.pt"
        latest.write_bytes(b"old joint optimizer/checkpoint")
        self.record.update(status="training", init=str(self.initializer))
        old = copy.deepcopy(self.record)
        with self.assertRaisesRegex(module.ChildFailed, "restart-joint-from-sft"):
            self.preparation()
        self.assertEqual(self.record, old)
        self.assertTrue(latest.is_file())
        self.p.args.restart_joint_from_sft = True
        result = self.preparation()
        archive = Path(self.p.state["joint_restarts"][0]["archive"])
        self.assertEqual((archive / "rl/latest.pt").read_bytes(), b"old joint optimizer/checkpoint")
        self.assertFalse(run.exists())
        self.assertEqual(self.record["status"], "pending")
        self.assertEqual(self.p.state["stages"]["s6_put_into_binary"]["status"], "done")
        # Reusing the same restart option must not archive corrected joint work.
        (run / "rl").mkdir(parents=True)
        (run / "rl/latest.pt").write_bytes(b"corrected joint")
        self.record["status"] = "training"
        self.assertEqual(self.preparation(), result)
        self.assertEqual((run / "rl/latest.pt").read_bytes(), b"corrected joint")
        self.assertEqual(len(self.p.state["joint_restarts"]), 1)

    def test_restart_reconciles_an_interruption_after_directory_rename(self):
        source = self.root / "runs/old_joint"
        archive = self.root / "runs/old_joint_saved"
        archive.mkdir(parents=True)
        (archive / "checkpoint.pt").write_bytes(b"preserved")
        self.record["status"] = "training"
        self.p.state["joint_restarts"] = [{"stage": self.stage["name"], "status": "pending",
            "run": str(source), "archive": str(archive), "previous_stage_record": dict(self.record)}]
        self.preparation()
        self.assertEqual(self.p.state["joint_restarts"][0]["status"], "archived")
        self.assertEqual((archive / "checkpoint.pt").read_bytes(), b"preserved")

    def test_no_sft_checkpoint_means_no_joint_handoff(self):
        original = self.run_child
        def no_adapter(command, **kwargs):
            original(command, **kwargs)
            if command[1].endswith("sil_sft.py"):
                (Path(command[command.index("--output") + 1]) / "sil_sft_adapter.pt").unlink()
        with patch.object(module, "run_command", side_effect=no_adapter), \
             self.assertRaisesRegex(module.ChildFailed, "no improved adapter"):
            self.p.prepare_composition(self.stage, self.record, self.initializer)
        self.assertNotEqual(self.p.state["composition_sft"]["status"], "done")
        self.assertEqual(self.p.state["current_policy"], str(self.initializer))

    def test_stop_before_joint_finishes_skills_without_starting_sft_or_training(self):
        self.p.args.stop_before_stage = self.stage["name"]
        with patch.object(self.p, "run_stage", side_effect=AssertionError("must stop before joint")), \
             patch.object(self.p, "final_report", side_effect=AssertionError("pipeline not finished")):
            self.p.run()
        self.assertNotIn("composition_sft", self.p.state)

    def test_missing_pickup_bank_is_recovered_from_the_better_dense_checkpoint(self):
        retained = self.p.state["banks"].pop("pick_up")
        for name, value in (("s2_pick_up_dense", .12109), ("s5_pick_up_binary", 0.0)):
            path = self.root / f"{name}.pt"
            path.write_bytes(b"teacher")
            self.p.state["stages"][name].update(best_value=value, best_checkpoint=str(path))
        def recover(stage, skill, policy, record):
            self.assertEqual(stage["name"], "s2_pick_up_dense")
            self.p.state["banks"][skill] = retained
        with patch.object(self.p, "harvest", side_effect=recover) as harvest:
            banks = self.p.composition_banks()
        harvest.assert_called_once()
        self.assertEqual(banks["pick_up"], retained)

    def test_production_composition_does_not_use_oracle_smoke_experience(self):
        self.p.state["banks"]["pick_up"]["provenance"] = "oracle_smoke_fixture"
        with self.assertRaisesRegex(module.ChildFailed, "No complete filmed success bank for pick_up"):
            self.p.composition_banks()

    def test_composition_requests_scene_spread_refilming_for_all_retained_skills(self):
        with patch.object(self.p, "refilm_bank") as refilm:
            self.p.composition_banks()
        self.assertEqual([call.args[0] for call in refilm.call_args_list], ["move_to", "pick_up", "put_into"])

    def test_interrupted_refresh_without_its_completion_metadata_is_repeated(self):
        original = self.run_child
        def interrupted(command, **kwargs):
            original(command, **kwargs)
            if command[1].endswith("sil_refresh_priors.py"):
                output = Path(command[command.index("--output") + 1])
                (output / "refresh.json").unlink()
                raise module.ChildFailed("interrupted after refreshed NPZ")
        with patch.object(module, "run_command", side_effect=interrupted), \
             self.assertRaisesRegex(module.ChildFailed, "interrupted"):
            self.p.prepare_composition(self.stage, self.record, self.initializer)
        self.preparation()
        self.assertEqual(sum(c[1].endswith("sil_refresh_priors.py") for c in self.commands), 2)
        self.assertEqual(sum(c[1].endswith("build_widowx200_composition_sft.py") for c in self.commands), 1)


if __name__ == "__main__":
    unittest.main()
