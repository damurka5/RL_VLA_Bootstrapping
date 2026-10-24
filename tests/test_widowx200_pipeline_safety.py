"""Pipeline handoffs and interrupted SIL artifacts, without a GPU/model download."""

import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

import numpy as np

from tests.test_sil_record_reporting import _synthetic, _with_observations
from tools.audit import sil_record
from tools.audit.sil_artifacts import atomic_savez, completed_replay, mark_replay_complete, valid_npz
from tools.audit.sil_sft import load_frame_meta, materialize_frames, resolve_frame_rows
from tools.train.run_widowx200_pipeline import ChildFailed, Pipeline, parse_args, select_best_checkpoint


def pipeline(directory, *args):
    return Pipeline(parse_args(["--pipeline-dir", str(directory), *args]))


def write_pair(directory, stem="record_record_00", value=1):
    replay = directory / f"replay_{stem}.npz"
    frames = directory / f"frames_{stem}.npz"
    rec = _with_observations(_synthetic())
    rec.states.fill(value)
    rec.to_npz(replay)
    shape = (2, 1, 2, 3, 3)
    atomic_savez(frames, world_index=np.array([0]), decisions=np.int64(2),
                 overview=np.full(shape, value, np.uint8), wrist=np.full(shape, value, np.uint8))
    mark_replay_complete(replay, frames, frame_successes=1)
    return replay, frames


class BankIdentityTests(unittest.TestCase):
    def test_same_filenames_from_two_skills_join_their_own_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = [write_pair(root / skill / "replay", value=value)
                     for skill, value in (("move_to", 1), ("pick_up", 2))]
            output = root / "dataset"
            self.assertEqual(sil_record.main([
                "--mode", "dataset", "--inputs", *[str(p[0]) for p in pairs],
                "--require-frames", *[str(p[1]) for p in pairs], "--output", str(output),
            ]), 0)
            with np.load(output / "demonstrations.npz") as data:
                found, lookups = resolve_frame_rows(data["episode_uid"], data["decision_index"],
                                                   load_frame_meta([p[1] for p in pairs]))
                self.assertTrue(found.all())
                self.assertEqual(len(np.unique(data["episode_uid"])), 2)
                images, _ = materialize_frames(load_frame_meta([p[1] for p in pairs]), lookups,
                                               list(range(len(lookups))))
                np.testing.assert_array_equal(images[:, 0, 0, 0], data["state"][:, 0])

    def test_ambiguous_legacy_rows_cannot_silently_choose_a_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [write_pair(Path(tmp) / name) for name in ("a", "b")]
            meta = load_frame_meta([p[1] for p in pairs])
            self.assertEqual(len(meta), 2)
            found, _ = resolve_frame_rows(np.array(["replay/replay_record_record_00/r0w0"]),
                                          np.array([0]), meta)
            self.assertFalse(found.any())

    def test_frame_filter_runs_before_row_arrays_are_stacked(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay, frames = write_pair(Path(tmp))
            rec = sil_record._Recording.from_npz(replay)
            original_stack = np.stack

            def bounded_stack(values, *args, **kwargs):
                # Only world's 0 single successful decision has pictures;
                # the fixture's other two successful episodes are unfilmed.
                self.assertEqual(len(values), 1)
                return original_stack(values, *args, **kwargs)

            with patch.object(sil_record.np, "stack", side_effect=bounded_stack):
                dataset, _ = sil_record._build_dataset(
                    [rec], keys=["replay/replay_record_record_00"], frame_paths=[frames]
                )
            self.assertEqual(len(dataset["state"]), 1)


class InterruptedArtifactsTests(unittest.TestCase):
    def test_failed_write_preserves_the_previous_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            atomic_savez(path, state=np.array([42]))
            with patch("numpy.savez_compressed", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    atomic_savez(path, state=np.array([99]))
            self.assertTrue(valid_npz(path, ("state",)))
            with np.load(path) as data:
                self.assertEqual(data["state"][0], 42)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_only_a_complete_pair_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay, frames = write_pair(Path(tmp))
            self.assertIsNotNone(completed_replay(replay, frames))
            frames.unlink()
            self.assertIsNone(completed_replay(replay, frames))
            frames.write_bytes(b"truncated zip")
            self.assertIsNone(completed_replay(replay, frames))

    def test_harvest_skips_empty_rounds_and_rebuilds_missing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "record"
            source.mkdir()
            atomic_savez(source / "record_00.npz", success=np.zeros((1, 1), bool))
            atomic_savez(source / "record_01.npz", success=np.ones((1, 1), bool))
            destination = root / "replay"
            replay, frames = write_pair(destination, "record_record_01")
            frames.unlink()
            p = pipeline(root)

            def finish(command, **kwargs):
                self.assertIn("--controller-workspace-from-config", command)
                self.assertEqual(command[command.index("--actions") + 1], str(source / "record_01.npz"))
                write_pair(destination, "record_record_01")

            with patch("tools.train.run_widowx200_pipeline.run_command", side_effect=finish) as run:
                self.assertEqual(p.harvest_replays(root / "policy.pt", root / "eval.yaml", .2,
                                                   source, destination, "test"), ([str(replay)], [str(frames)]))
                p.harvest_replays(root / "policy.pt", root / "eval.yaml", .2, source, destination, "test")
                self.assertEqual(run.call_count, 1)

    def test_empty_production_harvest_preserves_the_previous_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pipeline(tmp)
            p.state["banks"]["move_to"] = {"mean_rate": .8, "checkpoint": "previous.pt"}
            stage = p.spec["stages"][0]
            record = {}
            with patch.object(p, "record", return_value={"move_to_object": {"rate": 0.0, "episodes": 512}}), \
                 patch.object(p, "write_config", return_value=Path(tmp) / "eval.yaml"), \
                 patch.object(p, "harvest_replays", return_value=([], [])):
                p.harvest(stage, "move_to", Path("current.pt"), record)
            self.assertTrue(record["empty_harvest"])
            self.assertEqual(p.state["banks"]["move_to"]["checkpoint"], "previous.pt")

    def test_smoke_uses_a_labelled_oracle_fixture_when_learning_has_no_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pipeline(tmp, "--smoke")
            with patch.object(p, "record", return_value={"move_to_object": {"rate": 0.0, "episodes": 64}}), \
                 patch.object(p, "write_config", return_value=Path(tmp) / "eval.yaml"), \
                 patch.object(p, "harvest_replays", side_effect=[([], []), (["r.npz"], ["f.npz"])]), \
                 patch("tools.train.run_widowx200_pipeline.run_command") as run:
                p.harvest(p.spec["stages"][0], "move_to", Path("policy.pt"), {})
            self.assertIn("oracle", run.call_args.args[0])
            self.assertEqual(p.state["banks"]["move_to"]["provenance"], "oracle_smoke_fixture")


class FixedValidationTests(unittest.TestCase):
    def test_stopping_ignores_easy_scores_and_preserves_fixed_distance_history(self):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import EarlyStopRule

        args = Namespace(early_stop_target=.7, validation_start_distance_caps=["pick_up=0.2"])
        rule = EarlyStopRule(args)
        self.assertIsNone(rule.update({"validation/success_rate": .99}, 1))
        self.assertEqual(rule.history, [])
        row = {"validation/success_rate": .8, "validation/fixed_start_distance": 1.0,
               "validation/start_distance_cap/pick_up": .2}
        self.assertIsNone(rule.update(row, 2))
        resumed = EarlyStopRule(args)
        resumed.load_state_dict(rule.state_dict())
        self.assertEqual(resumed.update(row, 3)["reason"], "target")
        changed = EarlyStopRule(Namespace(validation_start_distance_caps=["pick_up=0.3"]))
        changed.load_state_dict(rule.state_dict())
        self.assertEqual(changed.history, [])

    def test_binary_pickup_keeps_terminal_credit_for_approach(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pipeline(tmp)
            dense, binary = [p.stage_config(p.spec["stages"][i]) for i in (1, 4)]
            self.assertTrue(dense["training"]["rl"]["args"]["split_credit_at_grasp"])
            self.assertFalse(binary["training"]["rl"]["args"]["split_credit_at_grasp"])

    def test_all_stages_have_fixed_caps_including_each_joint_instruction(self):
        from rl_vla_bootstrapping.policy.early_stopping import parse_validation_caps
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID

        with tempfile.TemporaryDirectory() as tmp:
            p = pipeline(tmp)
            for stage in p.spec["stages"]:
                config = p.stage_config(stage)
                caps = parse_validation_caps(config["training"]["rl"]["args"]["validation_start_distance_caps"], INSTRUCTION_TO_ID)
                self.assertEqual(set(caps), {INSTRUCTION_TO_ID[n] for n in config["task"]["instruction_types"]})
            self.assertEqual(p.validation_caps("joint"), {"move_to_object": .30, "pick_up": .20,
                                                         "put_into_plate": .18, "put_into_bowl": .18})
            with self.assertRaises(ValueError):
                parse_validation_caps(["pick_up=nan"], INSTRUCTION_TO_ID)

    def test_curriculum_only_scores_cannot_win_a_fixed_distance_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"global_step": 250000, "validation/success_rate": .95},
                    {"global_step": 3000000, "validation/success_rate": .60,
                     "validation/fixed_start_distance": 1.0, "validation/start_distance_cap/pick_up": .20}]
            for row in rows:
                path = root / f"step_{row['global_step']:07d}" / "smolvla_grpo_adapter.pt"
                path.parent.mkdir()
                path.touch()
            (root / "validation.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            self.assertEqual(select_best_checkpoint(root, "validation/success_rate", validation_caps={"pick_up": .2})["best_step"], 3000000)
            with self.assertRaises(ChildFailed):
                select_best_checkpoint(root, "validation/success_rate", validation_caps={"pick_up": .3})

    def test_recorder_forwards_task_controller_bounds_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, config = root / "policy.pt", root / "eval.yaml"
            checkpoint.touch()
            config.touch()
            with patch.object(sil_record, "_build_world", side_effect=RuntimeError("stop before CUDA")) as build:
                with self.assertRaisesRegex(RuntimeError, "stop before CUDA"):
                    sil_record.main(["--mode", "record", "--checkpoint", str(checkpoint),
                                     "--config", str(config), "--controller-workspace-from-config",
                                     "--output", str(root / "out")])
            self.assertTrue(build.call_args.kwargs["controller_workspace_from_config"])


class MemoryCheckTests(unittest.TestCase):
    def test_stress_keeps_production_batches_and_forces_the_largest_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pipeline(tmp, "--memory-check")
            config = p.memory_check_config()
            args = config["training"]["rl"]["args"]
            self.assertEqual(config["simulator"]["worlds_per_rank"], 512)
            self.assertTrue(args["smolvla_compile_model"])
            self.assertEqual(args["smolvla_inference_microbatch_size"], 256)
            self.assertEqual(args["vla_microbatch_size"], 16)
            self.assertEqual(args["vla_update_max_records"], 128)
            self.assertEqual(config["task"]["metadata"]["curriculum_horizon_min"], 40)
            self.assertEqual(config["task"]["metadata"]["curriculum_horizon_max"], 40)
            self.assertEqual(args["grpo_max_groups_per_update"], 256)
            self.assertEqual(args["grpo_max_collection_seconds_per_update"], 0)

    def test_inherited_toy_sizes_cannot_weaken_production_or_memory_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pipeline(tmp)
            with patch.dict("os.environ", {"RLVLA_MJWARP_WORLDS_PER_RANK": "8",
                                            "RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE": "8"}):
                env = p.child_env()
            self.assertNotIn("RLVLA_MJWARP_WORLDS_PER_RANK", env)
            self.assertNotIn("RLVLA_SMOLVLA_INFERENCE_MICROBATCH_SIZE", env)

    def test_report_requires_full_refill_and_lora_on_every_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = pipeline(root, "--memory-check")
            rl = root / "rl"
            rl.mkdir()
            good = {"rounds_collected": 4, "vla_records": 128}
            for rank in (0, 1):
                (rl / f"memory_rank{rank}.jsonl").write_text("\n".join(json.dumps(good) for _ in range(2)))
            with patch.object(p, "train", return_value=rl), \
                 patch.object(p, "write_config", return_value=root / "memory.yaml"):
                p.run_memory_check()
                report = json.loads((root / "memory_report.json").read_text())
                self.assertEqual(len(report["ranks"]), 2)
                (rl / "memory_rank1.jsonl").write_text(json.dumps({"rounds_collected": 1, "vla_records": 16}))
                with self.assertRaises(ChildFailed):
                    p.run_memory_check()
