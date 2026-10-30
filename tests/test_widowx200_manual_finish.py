"""A user-selected skill checkpoint still goes through harvest and retention."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.train import run_widowx200_pipeline as module


class ManualFinishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.config = module.ROOT / "configs/examples/widowx200_smolvla_pipeline.yaml"
        self.directory = self.root / "pipeline"
        p = self.load()
        self.stage = p.spec["stages"][5]
        self.name = self.stage["name"]
        self.rl = self.root / "runs" / f"pipeline__{self.name}" / "rl"
        self.rl.mkdir(parents=True)
        self.best = self.checkpoint(1000000)
        self.latest = self.checkpoint(1764679)
        self.rows = [self.validation(1000000, .9, .9), self.validation(1764679, .75, .69)]
        self.write_validation()
        for stage in p.spec["stages"][:5]:
            p.state["stages"][stage["name"]] = {"status": "done", "policy": str(self.best)}
        p.state["stages"][self.name] = {"status": "training", "init": str(self.best)}
        p.state["current_policy"] = str(self.best)
        p.save_state()
        self.root_patch = patch.object(module, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def load(self, request=None, *extra):
        argv = ["--config", str(self.config), "--pipeline-dir", str(self.directory), *extra]
        if request is not None:
            argv += ["--finish-stage-at-checkpoint", request]
        return module.Pipeline(module.parse_args(argv))

    def checkpoint(self, step):
        path = self.rl / f"step_{step:07d}/smolvla_grpo_adapter.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"saved policy")
        return path

    def validation(self, step, plate, bowl):
        return {"global_step": step, "validation/fixed_start_distance": 1.0,
                "validation/start_distance_cap/put_into_plate": .18,
                "validation/start_distance_cap/put_into_bowl": .18,
                "validation/by_instruction/put_into_plate/success_rate": plate,
                "validation/by_instruction/put_into_bowl/success_rate": bowl}

    def write_validation(self):
        (self.rl / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def test_latest_skips_rl_but_harvests_and_checks_retention(self):
        p = self.load(f"{self.name}=latest")
        record = p.state["stages"][self.name]
        retained = self.root / "accepted_sil.pt"
        with patch.object(p, "train", side_effect=AssertionError("must not train stage 6")), \
             patch.object(p, "write_config", return_value=self.root / "stage.yaml"), \
             patch.object(p, "harvest") as harvest, \
             patch.object(p, "retention", return_value=retained) as retention:
            p.run_stage(5, self.stage, record)
        self.assertEqual(harvest.call_args.args[2], self.latest)
        self.assertEqual(retention.call_args.args[2], self.latest)
        self.assertEqual(record["best_checkpoint"], str(self.best))
        self.assertEqual(record["best_value"], .9)
        self.assertEqual(record["selected_checkpoint"], str(self.latest))
        self.assertAlmostEqual(record["selected_value"], .69)
        self.assertEqual(record["early_stop"]["reason"], "user_checkpoint")
        self.assertEqual(record["checkpoint_selection"], "user_checkpoint")
        self.assertEqual(record["status"], "done")
        self.assertEqual(p.state["current_policy"], str(retained))

    def test_interrupted_harvest_keeps_initial_choice_on_resume(self):
        p = self.load(f"{self.name}=latest")
        record = p.state["stages"][self.name]
        with patch.object(p, "write_config", return_value=self.root / "stage.yaml"), \
             patch.object(p, "harvest", side_effect=module.ChildFailed("interrupted")), \
             self.assertRaisesRegex(module.ChildFailed, "interrupted"):
            p.run_stage(5, self.stage, record)
        self.checkpoint(2000000)
        self.rows.append(self.validation(2000000, .99, .99))
        self.write_validation()
        p = self.load(f"{self.name}=latest")
        with patch.object(p, "write_config", return_value=self.root / "stage.yaml"), \
             patch.object(p, "train", side_effect=AssertionError("must not resume RL")), \
             patch.object(p, "harvest") as harvest, \
             patch.object(p, "retention", return_value=self.latest):
            p.run_stage(5, self.stage, p.state["stages"][self.name])
        self.assertEqual(harvest.call_args.args[2], self.latest)
        self.assertEqual(len(p.state["events"]), 2)  # one selection, one stage completion
        # Repeating the entire launcher after skill completion also preserves it.
        p = self.load(f"{self.name}=latest", "--stop-before-stage", "s7_joint_staged")
        with patch.object(p, "run_stage", side_effect=AssertionError("skills are complete")):
            p.run()

    def test_explicit_checkpoint_and_partial_later_validation(self):
        with (self.rl / "validation.jsonl").open("a") as handle:
            handle.write('{"global_step":')
        p = self.load(f"{self.name}={self.latest}")
        record = p.state["stages"][self.name]
        p.finish_skill_training(self.stage, record, str(self.latest))
        self.assertEqual(record["selected_step"], 1764679)
        self.assertEqual(record["best_step"], 1000000)

    def test_rejects_unvalidated_or_wrong_stage_checkpoint_without_state_change(self):
        unvalidated = self.checkpoint(2000000)
        wrong = self.root / "other/step_1764679/smolvla_grpo_adapter.pt"
        wrong.parent.mkdir(parents=True)
        wrong.write_bytes(b"other stage")
        for request in (str(unvalidated), str(wrong), str(self.rl / "latest.pt")):
            with self.subTest(request=request):
                p = self.load(f"{self.name}={request}")
                before = copy.deepcopy(p.state)
                with self.assertRaises(module.ChildFailed):
                    p.finish_skill_training(self.stage, p.state["stages"][self.name], request)
                self.assertEqual(p.state, before)

    def test_rejects_curriculum_distance_even_when_metric_is_high(self):
        self.rows[-1]["validation/start_distance_cap/put_into_bowl"] = .05
        self.write_validation()
        p = self.load(f"{self.name}=latest")
        with self.assertRaisesRegex(module.ChildFailed, "fixed-distance"):
            p.finish_skill_training(self.stage, p.state["stages"][self.name], "latest")

    def test_rejects_joint_unknown_and_already_finished_stage(self):
        for request in ("s7_joint_staged=latest", "typo=latest", "s1_move_to_dense=latest"):
            with self.subTest(request=request), self.assertRaises(module.ChildFailed):
                self.load(request)

    def test_rejects_changing_skill_after_joint_has_started(self):
        p = self.load()
        p.state["stages"]["s7_joint_staged"] = {"status": "training"}
        p.save_state()
        with self.assertRaisesRegex(module.ChildFailed, "later stage"):
            self.load(f"{self.name}=latest")


if __name__ == "__main__":
    unittest.main()
