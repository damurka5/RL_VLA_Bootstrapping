"""Evidence export preserves denominators, partial runs, and policy identity."""

import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.audit.export_widowx200_pipeline_results import export


class ResultsExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pipeline = self.root / "pipeline"
        self.pipeline.mkdir()
        self.output = self.root / "snapshot"
        self.stage = "s5_pick_up_binary"
        self.rl = self.root / f"pipeline__{self.stage}" / "rl"
        self.rl.mkdir(parents=True)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self):
        self.state = {
            "git_commit_at_start": "original_training_commit",
            "stages": {self.stage: {
                "status": "done", "rl_dir": str(self.rl), "best_step": 500000,
                "metric": "validation/success_rate", "best_value": 0.0,
                "best_checkpoint": "rl_before_sil.pt", "policy": "accepted_sil.pt",
                "early_stop": {"reason": "plateau"},
                "retention_rates": {"pick_up": {"rate": 0.0, "successes": 0, "episodes": 1024}},
                "retention_rates_after_sil": {"pick_up": {"rate": .125, "successes": 128, "episodes": 1024}},
                "sil_cycles": [{"accepted": True, "candidate_rates": {
                    "pick_up": {"rate": .125, "successes": 128, "episodes": 1024}}}],
            }},
        }
        self.write(self.pipeline / "pipeline_state.json", self.state)
        self.write(self.rl / "config.json", {"max_train_steps": 8000000,
                                              "validation_start_distance_caps": ["pick_up=0.2"]})
        self.validation = {"global_step": 500000, "validation/success_rate": 0.0,
                           "validation/fixed_start_distance": 1.0,
                           "validation/start_distance_cap/pick_up": .2,
                           "validation/by_instruction/pick_up/episodes": 512,
                           "validation/by_instruction/pick_up/success_rate": 0.0,
                           "validation/by_object/apple/episodes": 128,
                           "validation/by_object/apple/success_rate": 0.0,
                           "validation_composed/by_instruction/put_into_plate/episodes": 256,
                           "validation_composed/by_instruction/put_into_plate/success_rate": .25}
        (self.rl / "validation.jsonl").write_text(json.dumps(self.validation) + "\n")
        (self.rl / "metrics.jsonl").write_text(json.dumps({"global_step": 2500000, "update_index": 300}) + "\n")

    def table(self, name):
        with (self.output / f"{name}.csv").open() as handle:
            return list(csv.DictReader(handle))

    def test_keeps_best_rl_and_accepted_sil_evaluations_separate(self):
        self.fixture()
        export(self.pipeline, self.output)
        row = next(row for row in self.table("stages") if row["stage"] == self.stage)
        self.assertEqual(row["best_rl_value"], "0.0")
        self.assertEqual(row["best_rl_checkpoint"], "rl_before_sil.pt")
        self.assertEqual(row["handed_off_policy"], "accepted_sil.pt")
        self.assertEqual(row["sil_cycles_accepted"], "1")
        self.assertEqual(row["last_step"], "2500000")
        retention = self.table("retention")
        self.assertEqual([r["successes"] for r in retention], ["0", "128", "128"])
        self.assertEqual(retention[-1]["accepted"], "True")

    def test_validation_slices_preserve_protocol_and_exact_episode_denominators(self):
        self.fixture()
        export(self.pipeline, self.output)
        rows = self.table("validation_slices")
        pickup = next(r for r in rows if r["label"] == "pick_up")
        self.assertEqual(pickup["episodes"], "512")
        self.assertEqual(pickup["start_distance_cap"], "0.2")
        self.assertEqual(pickup["selected_best_rl"], "True")
        composed = next(r for r in rows if r["leg"] == "validation_composed")
        self.assertEqual(composed["episodes"], "256")
        self.assertEqual(composed["success_rate"], "0.25")
        self.assertEqual(next(r for r in rows if r["slice"] == "object")["episodes"], "128")

    def test_manual_checkpoint_selection_does_not_replace_validation_best(self):
        self.fixture()
        self.state["stages"][self.stage].update(
            checkpoint_selection="user_checkpoint", selected_step=750000,
            selected_checkpoint="chosen_by_user.pt", selected_value=.1,
            early_stop={"reason": "user_checkpoint"})
        self.write(self.pipeline / "pipeline_state.json", self.state)
        export(self.pipeline, self.output)
        row = next(row for row in self.table("stages") if row["stage"] == self.stage)
        self.assertEqual(row["best_rl_checkpoint"], "rl_before_sil.pt")
        self.assertEqual(row["selected_rl_checkpoint"], "chosen_by_user.pt")
        self.assertEqual(row["handed_off_policy"], "accepted_sil.pt")
        self.assertEqual(row["checkpoint_selection"], "user_checkpoint")
        self.assertEqual(row["stop_reason"], "user_checkpoint")
        self.assertEqual(self.table("validation")[0]["selected_best_rl"], "True")
        self.assertEqual(self.table("validation")[0]["selected_rl"], "False")

    def test_failed_pooled_summary_does_not_double_count_or_hide_partial_request(self):
        self.fixture()
        folder = self.pipeline / "banks/pick_up/test/record"
        run = {"mode": "record", "run_01": {"by_instruction": {
            "pick_up": {"episodes": 512, "successes": 60, "source_success_rate": .11719}}}}
        self.write(folder / "summary_cuda1.json", run)
        self.write(folder / "summary.json", {
            "mode": "record", "sharded": True, "round_index": 0, "rounds": 2,
            "shard_summaries": ["summary_cuda0.json", "summary_cuda1.json"],
            "failed_shards": ["cuda:0 (exit 1)"], "by_instruction": {"pick_up": {"episodes": 512}},
        })
        export(self.pipeline, self.output)
        rows = self.table("record_rounds")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episodes"], "512")
        self.assertEqual(rows[0]["sharded_request_complete"], "False")

    def test_snapshot_tolerates_in_progress_jsonl_and_keeps_hashed_originals(self):
        self.fixture()
        path = self.rl / "validation.jsonl"
        path.write_text(path.read_text() + '{"global_step":')
        original_state = (self.pipeline / "pipeline_state.json").read_bytes()
        original_validation = path.read_bytes()
        archive = export(self.pipeline, self.output)
        self.assertEqual(len(self.table("validation")), 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(manifest["warnings"])
        entry = next(s for s in manifest["sources"] if s["source"] == str(path.resolve()))
        self.assertEqual(entry["sha256"], hashlib.sha256(original_validation).hexdigest())
        self.assertEqual((self.pipeline / "pipeline_state.json").read_bytes(), original_state)
        self.assertEqual(path.read_bytes(), original_validation)
        with zipfile.ZipFile(archive) as bundle:
            raw_path = "snapshot/" + entry["exported"]
            self.assertEqual(bundle.read(raw_path), original_validation)

    def test_bank_inventory_and_replay_survival_do_not_read_or_archive_arrays(self):
        self.fixture()
        folder = self.pipeline / "banks/pick_up/test/replay"
        folder.mkdir(parents=True)
        replay, frames = folder / "replay_test.npz", folder / "frames_test.npz"
        for path in (replay, frames, self.rl / "checkpoint.pt"):
            path.write_bytes(b"not a loadable array or model; exporting must not read it")
        self.state["banks"] = {"pick_up": {
            "stage": "s2_pick_up_dense", "checkpoint": "retained.pt", "cap": .2,
            "provenance": "policy", "replays": [str(replay)], "frames": [str(frames)],
            "rates": {"pick_up": {"rate": .125, "successes": 256, "episodes": 2048}},
        }}
        self.write(self.pipeline / "pipeline_state.json", self.state)
        self.write(replay.with_suffix(".complete.json"), {"frame_successes": 40})
        self.write(folder / "summary_test.json", {"replay": {"by_instruction": {
            "pick_up": {"recorded_successes": 64, "survived": 48, "survival_rate": .75}}}})
        archive = export(self.pipeline, self.output)
        bank = self.table("banks")[0]
        self.assertEqual(bank["source_harvest_episodes"], "2048")
        self.assertEqual(bank["replay_markers_present"], "1")
        self.assertEqual(bank["filmed_successes_marker_sum_all_instructions"], "40")
        self.assertEqual(self.table("replay_survival")[0]["survived"], "48")
        with zipfile.ZipFile(archive) as bundle:
            self.assertFalse(any(name.endswith((".npz", ".pt")) for name in bundle.namelist()))

    def test_existing_export_directory_is_never_overwritten(self):
        self.fixture()
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            export(self.pipeline, self.output)

    def test_composition_scores_and_archived_old_joint_remain_separate_evidence(self):
        self.fixture()
        old = self.root / "old_joint/rl"
        old.mkdir(parents=True)
        (old / "metrics.jsonl").write_text('{"global_step": 100}\n')
        self.state["joint_restarts"] = [{"archive": str(old.parent)}]
        self.state["composition_sft"] = {"id": "new_sft", "status": "done",
            "initializer": "stage6.pt", "policy": "sft.pt",
            "after_composed_rates": {"put_into_bowl": {"successes": 32, "episodes": 512, "rate": .0625}}}
        self.write(self.pipeline / "pipeline_state.json", self.state)
        archive = export(self.pipeline, self.output)
        report = (self.output / "report.md").read_text()
        self.assertIn("identifier: new_sft", report)
        self.assertIn("after | composed | put_into_bowl | 32/512", report)
        self.assertEqual(len(self.table("training")), 1)
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn("snapshot/raw/archived_joint/0/metrics.jsonl", bundle.namelist())


if __name__ == "__main__":
    unittest.main()
