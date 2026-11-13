"""Only the last stage restarts, with phased rewards and a weights-only handoff."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools.train import continue_widowx200_stage7 as continuation
from scripts.preflight_widowx200_pipeline import merged_skill


class Stage7ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.patch_root = patch.object(continuation, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.pipeline = self.root / "runs" / "old_pipeline"
        self.pipeline.mkdir(parents=True)
        self.checkpoint = self.pipeline / "best.pt"
        self.checkpoint.write_bytes(b"fixture checkpoint; never loaded in CPU tests")
        folder = self.root / "configs" / "widowx200_pipeline"
        folder.mkdir(parents=True)
        self.source = folder / "original_stage7.yaml"
        self.config = {
            "project": {"name": "original", "output_root": "../../runs"},
            "embodiment": {"name": "widowx200_5dof_mjwarp"},
            "simulator": {"worlds_per_rank": 512, "fixed_scene_xml": "../../robots/scene.xml"},
            "task": {"instruction_types": sorted(continuation.INSTRUCTIONS), "metadata": {
                "staged_milestone_reward": False, "sparse_binary_reward": True,
                "placement_caught_object_fraction": .5, "put_bowl_xy_tolerance": .048,
            }},
            "training": {"rl": {"args": {
                "max_train_steps": 30000000, "resume_checkpoint": "old/latest.pt",
                "early_stop_metric": "validation/success_rate", "early_stop_min_steps": 5000000,
                "three_stage_sparse_credit": True, "split_credit_at_grasp": True,
            }}},
        }
        self.source.write_text(yaml.safe_dump(self.config))
        self.state = self.pipeline / "pipeline_state.json"
        self.state.write_text(json.dumps({"stages": {continuation.STAGE: {
            "status": "done", "config": str(self.source), "policy": str(self.checkpoint),
        }}}))

    def prepare(self, name="new_stage7", **kwargs):
        return continuation.prepare(self.pipeline, name, **kwargs)

    def test_handoff_preserves_the_source_and_uses_the_real_phased_reward(self):
        source_before, state_before = self.source.read_bytes(), self.state.read_bytes()
        payload = self.prepare()
        config = yaml.safe_load(Path(payload["config"]).read_text())
        self.assertEqual(Path(payload["config"]).parent, self.source.parent)
        self.assertEqual(config["simulator"], self.config["simulator"])
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(self.state.read_bytes(), state_before)
        self.assertEqual(payload["warmstart"], str(self.checkpoint.resolve()))
        self.assertEqual(payload["initialization"], "weights_only")
        self.assertEqual(payload["global_step_start"], 0)
        self.assertEqual(payload["max_train_steps"], 30000000)
        metadata = config["task"]["metadata"]
        self.assertEqual(metadata["put_bowl_xy_tolerance"], .048)
        self.assertEqual(metadata["placement_caught_object_fraction"], .5)
        self.assertEqual(metadata["open_grasp_hover_height"], .04)
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedCatchReleaseDenseReward, BatchedMoveToDistanceReward,
        )
        for reward in (BatchedCatchReleaseDenseReward.from_metadata(metadata),
                       BatchedMoveToDistanceReward.from_metadata(metadata)):
            self.assertTrue(reward.staged_milestones)
            self.assertEqual(reward.distance_reward_weight, 0.)
        args = config["training"]["rl"]["args"]
        self.assertNotIn("resume_checkpoint", args)
        self.assertFalse(args["three_stage_sparse_credit"])
        self.assertFalse(args["split_credit_at_grasp"])
        self.assertEqual(args["early_stop_metric"], "min_by_instruction_composed")
        self.assertGreater(args["composed_validation_episodes_per_instruction"], 0)
        self.assertIn("--skills", payload["commands"][0])
        self.assertIn("joint", payload["commands"][0])
        self.assertIn("rl_vla_bootstrapping.cli.train", payload["commands"][1])

    def test_refuses_overwrite_and_invalid_names_or_budget(self):
        for name, steps in (("../old_pipeline", None), ("bad/name", None), ("negative", -1)):
            with self.assertRaises(ValueError):
                self.prepare(name, steps=steps)
        payload = self.prepare()
        before = Path(payload["run_dir"], "continuation.json").read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(Path(payload["run_dir"], "continuation.json").read_bytes(), before)

    def test_checkpoint_override_and_new_budget(self):
        latest = self.pipeline / "latest.pt"
        latest.write_bytes(b"newer fixture")
        payload = self.prepare(checkpoint=latest, steps=10000000)
        self.assertEqual(payload["warmstart"], str(latest.resolve()))
        self.assertEqual(payload["max_train_steps"], 10000000)

    def test_environment_discards_the_old_resume_state(self):
        payload = self.prepare()
        with patch.dict(os.environ, {"RLVLA_SMOLVLA_RESUME_CHECKPOINT": "old/latest.pt",
                                    "RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT": "old/legacy.pt",
                                    "RLVLA_SMOLVLA_MJWARP_MAX_UPDATES": "2",
                                    "RLVLA_SMOLVLA_MAX_TRAIN_STEPS": "1",
                                    "RLVLA_TASK_METADATA_JSON": '{"staged_milestone_reward":false}'}):
            env = continuation.continuation_env(payload)
        self.assertNotIn("RLVLA_SMOLVLA_RESUME_CHECKPOINT", env)
        self.assertNotIn("RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT", env)
        self.assertNotIn("RLVLA_SMOLVLA_MJWARP_MAX_UPDATES", env)
        self.assertNotIn("RLVLA_SMOLVLA_MAX_TRAIN_STEPS", env)
        self.assertNotIn("RLVLA_TASK_METADATA_JSON", env)
        self.assertEqual(env["RLVLA_SMOLVLA_WARMSTART_CHECKPOINT"], payload["warmstart"])
        self.assertEqual(env["RLVLA_SMOLVLA_RESET_LOG_STD"], "1")
        self.assertEqual(env["RLVLA_EMBODIMENT"], "widowx200")

    def test_execute_runs_preflight_then_training(self):
        with patch.object(continuation, "run_logged") as child, contextlib.redirect_stdout(io.StringIO()):
            code = continuation.main(["--pipeline-dir", str(self.pipeline), "--run-name", "execute", "--execute"])
        self.assertEqual(code, 0)
        self.assertEqual(child.call_count, 2)
        self.assertIn("--skills", child.call_args_list[0].args[0])
        self.assertIn("rl_vla_bootstrapping.cli.train", child.call_args_list[1].args[0])

    def test_failed_preflight_prevents_training(self):
        failure = subprocess.CalledProcessError(3, ["preflight"])
        with patch.object(continuation, "run_logged", side_effect=failure) as child, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = continuation.main(["--pipeline-dir", str(self.pipeline), "--run-name", "fails", "--execute"])
        self.assertEqual(code, 3)
        self.assertEqual(child.call_count, 1)

    def test_preparation_does_not_launch_children(self):
        with patch.object(continuation, "run_logged") as child, contextlib.redirect_stdout(io.StringIO()):
            code = continuation.main(["--pipeline-dir", str(self.pipeline), "--run-name", "prepare_only"])
        self.assertEqual(code, 0)
        child.assert_not_called()

    def test_preflight_accepts_saved_stage_yaml_without_changing_it(self):
        raw = yaml.safe_load(self.source.read_text())
        merged = merged_skill(raw, "joint")
        self.assertEqual(merged, raw)
        merged["task"]["metadata"]["open_grasp_hover_height"] = .04
        self.assertNotIn("open_grasp_hover_height", raw["task"]["metadata"])


if __name__ == "__main__":
    unittest.main()
