"""Only the last stage restarts, with phased rewards and a weights-only handoff."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
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

    def test_public_models_ignore_expired_tokens_and_preserve_the_model_cache(self):
        payload = self.prepare()
        token = self.root / "hf_token"
        token.write_text("expired-fixture-token")
        with patch.dict(os.environ, {"HF_TOKEN": "expired-fixture-token",
                                    "HUGGING_FACE_HUB_TOKEN": "expired-fixture-token",
                                    "HF_TOKEN_PATH": str(token), "HF_HOME": str(self.root / "hf_home"),
                                    "HF_HUB_CACHE": str(self.root / "hf_cache"),
                                    "RLVLA_HF_PUBLIC_MODELS_ONLY": "1"}):
            env = continuation.continuation_env(payload)
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", env)
        self.assertEqual(env["HF_TOKEN_PATH"], os.devnull)
        self.assertEqual(env["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertEqual(env["HF_HOME"], str(self.root / "hf_home"))
        self.assertEqual(env["HF_HUB_CACHE"], str(self.root / "hf_cache"))
        self.assertEqual(token.read_text(), "expired-fixture-token")
        config = yaml.safe_load(Path(payload["config"]).read_text())
        self.assertEqual(config["remote"]["env_vars"]["HF_TOKEN_PATH"], os.devnull)

    def test_real_hub_client_cannot_read_the_expired_token_in_child_environment(self):
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.skipTest("huggingface_hub is not installed")
        payload = self.prepare()
        token = self.root / "token"
        token.write_text("expired-fixture-token")
        with patch.dict(os.environ, {"HF_TOKEN_PATH": str(token), "HF_TOKEN": "expired-fixture-token",
                                    "RLVLA_HF_PUBLIC_MODELS_ONLY": "1"}):
            env = continuation.continuation_env(payload)
        code = """
import os
from huggingface_hub import constants, get_token
from huggingface_hub.utils import build_hf_headers
assert constants.HF_TOKEN_PATH == os.devnull
assert get_token() is None
assert 'authorization' not in build_hf_headers()
"""
        subprocess.run([sys.executable, "-c", code], env=env, check=True,
                       capture_output=True, text=True)
        self.assertEqual(token.read_text(), "expired-fixture-token")

    def test_private_opt_in_and_cache_only_mode_are_honored(self):
        env = {"RLVLA_HF_PUBLIC_MODELS_ONLY": "0", "HF_TOKEN": "private-fixture-token",
               "HF_TOKEN_PATH": "/fixture/token", "RLVLA_HF_OFFLINE": "1"}
        continuation.configure_huggingface_env(env)
        self.assertEqual(env["HF_TOKEN"], "private-fixture-token")
        self.assertEqual(env["HF_TOKEN_PATH"], "/fixture/token")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        with self.assertRaises(ValueError):
            continuation.configure_huggingface_env({"RLVLA_HF_OFFLINE": "yes"})
        with patch.dict(os.environ, {"RLVLA_HF_PUBLIC_MODELS_ONLY": "0", "RLVLA_HF_OFFLINE": "1"}):
            saved = {"HF_TOKEN": "private-fixture-token"}
            continuation.configure_saved_huggingface_env(saved)
        self.assertEqual(saved["HF_TOKEN"], "private-fixture-token")
        self.assertEqual(saved["HF_HUB_OFFLINE"], "1")

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

    def test_startup_retry_reuses_the_config_initializer_budget_and_passed_preflight(self):
        payload = self.prepare()
        run = Path(payload["run_dir"])
        (run / "preflight").mkdir()
        (run / "preflight/preflight.json").write_text('{"passed": true}')
        with patch.object(continuation, "run_logged") as child, contextlib.redirect_stdout(io.StringIO()):
            code = continuation.main(["--retry-run", str(run), "--execute"])
        self.assertEqual(code, 0)
        self.assertEqual(child.call_count, 1)
        self.assertIn("rl_vla_bootstrapping.cli.train", child.call_args.args[0])
        repaired = json.loads((run / "continuation.json").read_text())
        self.assertEqual(repaired["config"], payload["config"])
        self.assertEqual(repaired["warmstart"], payload["warmstart"])
        self.assertEqual(repaired["max_train_steps"], payload["max_train_steps"])

    def test_startup_retry_repairs_old_manifests_and_saved_token_overrides(self):
        payload = self.prepare()
        run = Path(payload["run_dir"])
        payload.pop("config_sha256")  # The version that hit the server auth failure.
        (run / "continuation.json").write_text(json.dumps(payload))
        config = Path(payload["config"])
        raw = yaml.safe_load(config.read_text())
        raw["remote"]["env_vars"].update(HF_TOKEN="expired-fixture-token", HF_HUB_DISABLE_IMPLICIT_TOKEN="0")
        config.write_text(yaml.safe_dump(raw))
        repaired = continuation.retry_startup(run)
        saved = yaml.safe_load(config.read_text())
        self.assertNotIn("HF_TOKEN", saved["remote"]["env_vars"])
        self.assertEqual(saved["remote"]["env_vars"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertEqual(saved["task"], raw["task"])
        self.assertEqual(saved["training"], raw["training"])
        self.assertIn("config_sha256", repaired)

    def test_startup_retry_rechecks_failed_preflight_and_refuses_to_overwrite_training(self):
        payload = self.prepare()
        run = Path(payload["run_dir"])
        (run / "preflight").mkdir()
        (run / "preflight/preflight.json").write_text('{"passed": false}')
        with patch.object(continuation, "run_logged") as child, contextlib.redirect_stdout(io.StringIO()):
            code = continuation.main(["--retry-run", str(run), "--execute"])
        self.assertEqual(code, 0)
        self.assertEqual(child.call_count, 2)
        (run / "rl").mkdir()
        (run / "rl/latest.pt").write_bytes(b"must preserve actual training")
        with self.assertRaisesRegex(ValueError, "already has checkpoints"):
            continuation.retry_startup(run)

    def test_startup_retry_rejects_a_changed_prepared_config(self):
        payload = self.prepare()
        path = Path(payload["config"])
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "config changed"):
            continuation.retry_startup(Path(payload["run_dir"]))

    def test_preflight_accepts_saved_stage_yaml_without_changing_it(self):
        raw = yaml.safe_load(self.source.read_text())
        merged = merged_skill(raw, "joint")
        self.assertEqual(merged, raw)
        merged["task"]["metadata"]["open_grasp_hover_height"] = .04
        self.assertNotIn("open_grasp_hover_height", raw["task"]["metadata"])


if __name__ == "__main__":
    unittest.main()
