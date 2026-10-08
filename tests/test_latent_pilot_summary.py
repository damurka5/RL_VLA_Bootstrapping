"""Artifact-based diagnosis must distinguish smoke, capped, resumed and failed runs."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.audit.summarize_latent_pilot import ARCHITECTURES, summarize


class PilotSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "latent_diag_candidate"
        (self.run / "rl").mkdir(parents=True)

    def write(self, name, obj):
        (self.run / name).write_text(json.dumps(obj) + "\n")

    def fixture(self, *, arm="candidate", cap=10, updates=10, selected=850000, start=0):
        common = dict(policy_architecture=ARCHITECTURES[arm],
                      action_likelihood="latent_gaussian_conditional_offset_v1",
                      max_train_steps=1000000, ppo_epochs=1)
        mode = "resume" if start else "legacy_conversion"
        self.write("launch_provenance.json", dict(common, arm=arm, init_mode=mode, max_updates=cap))
        protocol_name = "latent_pilot_protocol_resume.json" if start else "latent_pilot_protocol.json"
        self.write("rl/" + protocol_name, dict(common, mjwarp_max_updates=cap,
                   pilot_counters_at_start={"updates": start}))
        rows = []
        for index in range(1, updates + 1):
            step = selected * index // updates
            row = {"global_step": step, "update_index": start + index,
                   "pilot/updates": start + index, "pilot/selected_environment_actions": step,
                   "loss_policy_mean": -0.001, "approx_kl_mean": 0.005,
                   "clip_fraction_mean": 0.02, "gradient_norm_mean": 6,
                   "log_std_mean": -1.2, "policy/lora_max_abs_change": 0,
                   "non_finite_live_episode_rate": 0.01}
            if arm == "candidate":
                row.update({"correction/reference_max_abs_change": 0,
                            "correction/grad_norm_final_mean": 10,
                            "correction/grad_norm_hidden_mean": 0.023})
            rows.append(row)
        (self.run / "rl/metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.write("rl/validation.jsonl", {"global_step": selected, "validation/success_rate": 0.38})

    def test_old_smoke_is_not_a_ten_update_diagnostic(self):
        self.fixture(arm="control", cap=1, updates=1, selected=85406)
        report = summarize(self.run, 10)
        self.assertEqual(report["status"], "limit reached: update cap")
        self.assertFalse(report["exit_record"])
        self.assertTrue(any("expected 10 updates" in issue for issue in report["issues"]))
        self.assertFalse(any("reference" in issue for issue in report["issues"]))

    def test_completed_diagnostic_and_nonzero_hidden_gradient(self):
        self.fixture()
        self.write("launch_result.json", dict(train_exit_code=0, log_exit_code=0))
        report = summarize(self.run, 10)
        self.assertEqual(report["status"], "completed: update cap")
        self.assertEqual(report["issues"], [])

    def test_action_cap_can_end_diagnostic_before_ten_updates(self):
        self.fixture(updates=8, selected=1010000)
        report = summarize(self.run, 10)
        self.assertEqual(report["status"], "limit reached: selected-action cap")
        self.assertTrue(any("found 8" in issue for issue in report["issues"]))

    def test_resume_cap_counts_only_this_invocation(self):
        self.fixture(start=9, updates=10)
        report = summarize(self.run, 10)
        self.assertEqual(report["updates"], 10)
        self.assertEqual(report["issues"], [])

    def test_failed_process_overrides_limit_evidence(self):
        self.fixture()
        for train_code, log_code in ((1, 0), (0, 1)):
            with self.subTest(train=train_code, log=log_code):
                self.write("launch_result.json", dict(train_exit_code=train_code, log_exit_code=log_code))
                self.assertEqual(summarize(self.run)["status"], "launcher failed")

    def test_truncated_metrics_do_not_hide_completed_rows(self):
        self.fixture()
        with (self.run / "rl/metrics.jsonl").open("a") as stream:
            stream.write('{"global_step":')
        report = summarize(self.run)
        self.assertEqual(len(report["metrics"]), 10)
        self.assertTrue(any("metrics.jsonl:11" in issue for issue in report["issues"]))

    def test_bad_or_missing_metrics_are_flagged(self):
        self.fixture(updates=1)
        path = self.run / "rl/metrics.jsonl"
        row = json.loads(path.read_text())
        row["approx_kl_mean"] = float("nan")
        row["correction/reference_max_abs_change"] = 0.01
        del row["loss_policy_mean"]
        self.write("rl/metrics.jsonl", row)
        issues = "\n".join(summarize(self.run)["issues"])
        for message in ("non-finite metrics", "frozen weights changed", "missing: loss_policy_mean"):
            self.assertIn(message, issues)

    def test_empty_run_is_not_a_pass(self):
        report = summarize(self.run)
        self.assertIn("incomplete", report["status"])
        self.assertTrue(any("no completed update" in issue for issue in report["issues"]))

    def test_contact_force_nan_warns_without_failing_successful_training(self):
        self.fixture(cap=1, updates=1)
        self.write("launch_result.json", dict(train_exit_code=0, log_exit_code=0))
        row = json.loads((self.run / "rl/metrics.jsonl").read_text())
        row.update(left_pad_normal_force_mean_n=float("nan"),
                   right_pad_normal_force_mean_n=float("inf"))
        self.write("rl/metrics.jsonl", row)
        report = summarize(self.run)
        self.assertEqual(report["status"], "completed: update cap")
        self.assertEqual(report["issues"], [])
        self.assertIn("contact-force diagnostics", report["warnings"][0])
        tool = Path(__file__).resolve().parents[1] / "tools/audit/summarize_latent_pilot.py"
        result = subprocess.run([sys.executable, str(tool), str(self.run)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WARNING:", result.stdout)
        self.assertIn("not been repaired", result.stdout)

    def test_contact_warning_does_not_hide_optimizer_or_other_nan(self):
        self.fixture(cap=1, updates=1)
        row = json.loads((self.run / "rl/metrics.jsonl").read_text())
        row.update(left_pad_normal_force_mean_n=float("nan"),
                   gradient_norm_mean=float("nan"), candidate_reward_mean=float("nan"))
        self.write("rl/metrics.jsonl", row)
        report = summarize(self.run)
        self.assertTrue(report["warnings"])
        self.assertIn("gradient_norm_mean", "\n".join(report["issues"]))
        self.assertIn("candidate_reward_mean", "\n".join(report["issues"]))
        tool = Path(__file__).resolve().parents[1] / "tools/audit/summarize_latent_pilot.py"
        result = subprocess.run([sys.executable, str(tool), str(self.run)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)

    def test_protocol_mismatch_and_missing_final_validation(self):
        self.fixture()
        path = self.run / "rl/latent_pilot_protocol.json"
        protocol = json.loads(path.read_text())
        protocol["mjwarp_max_updates"] = 1
        protocol["policy_architecture"] = ARCHITECTURES["control"]
        self.write("rl/latent_pilot_protocol.json", protocol)
        self.write("rl/validation.jsonl", {"global_step": 0, "validation/success_rate": 0.38})
        issues = "\n".join(summarize(self.run)["issues"])
        for message in ("architecture disagree", "mismatch: max_updates", "no validation row at the final"):
            self.assertIn(message, issues)

    def test_zero_signal_stop_does_not_count_as_completion(self):
        self.fixture(updates=1, cap=1)
        row = json.loads((self.run / "rl/metrics.jsonl").read_text())
        row["three_stage/stopped_for_zero_signal"] = 1
        self.write("rl/metrics.jsonl", row)
        self.write("launch_result.json", dict(train_exit_code=0, log_exit_code=0))
        self.assertEqual(summarize(self.run)["status"], "stopped for zero signal")

    def test_cli_lists_and_reports_without_ml_dependencies(self):
        self.fixture()
        tool = Path(__file__).resolve().parents[1] / "tools/audit/summarize_latent_pilot.py"
        for args in (["--list", "--runs-root", self.run.parent], [self.run / "rl", "--expect-updates", "10"]):
            result = subprocess.run([sys.executable, str(tool), *map(str, args)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("arm=candidate updates=10/10", result.stdout)

    def test_cli_rejects_missing_paths_without_fabricating_run_status(self):
        tool = Path(__file__).resolve().parents[1] / "tools/audit/summarize_latent_pilot.py"
        for name, message in (("latent_diag_*", "Unexpanded run pattern"),
                              ("missing_run", "Run directory does not exist")):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, str(tool), str(self.run.parent / name)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stdout)
                self.assertNotIn("incomplete", result.stdout)
                self.assertNotIn("missing metrics", result.stdout)

    def test_cli_still_reports_existing_runs_with_an_unmatched_pattern(self):
        self.fixture()
        tool = Path(__file__).resolve().parents[1] / "tools/audit/summarize_latent_pilot.py"
        result = subprocess.run([sys.executable, str(tool), str(self.run.parent / "missing_*"),
                                 str(self.run)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unexpanded run pattern", result.stdout)
        self.assertIn("arm=candidate updates=10/10", result.stdout)

    def check_launcher_exit(self, exit_code):
        """Run the real shell launcher with only the GPU/conda boundary stubbed."""
        self.fixture()
        root = Path(self.tmp.name)
        (root / "runs").mkdir()
        destination = root / "runs" / self.run.name
        self.run.rename(destination)
        self.run = destination
        repo = Path(__file__).resolve().parents[1]
        binary = root / "bin"
        binary.mkdir()
        fake_conda = binary / "conda"
        fake_conda.write_text("#!" + sys.executable + "\n" + '''
import os, subprocess, sys
from pathlib import Path
args = sys.argv[sys.argv.index("python3") + 1:]
if args[:2] == ["-m", "rl_vla_bootstrapping.cli.train"]:
    assert os.environ["RLVLA_SMOLVLA_MJWARP_MAX_UPDATES"] == "10"
    print("mock training output", flush=True)
    raise SystemExit(int(os.environ["FAKE_TRAIN_EXIT"]))
if args[0] == "-":
    source = sys.stdin.read()
    if args[1].endswith("launch_result.json"):
        raise SystemExit(subprocess.run([sys.executable, *args], input=source, text=True).returncode)
    # GPU preflight/provenance are covered elsewhere; fixture artifacts remain.
    raise SystemExit(0)
args[0] = str(Path(os.environ["TEST_REPO"]) / args[0])
raise SystemExit(subprocess.run([sys.executable, *args]).returncode)
''')
        fake_conda.chmod(0o755)
        for name in ("config.yaml", "scenes.json", "init.pt"):
            (root / name).touch()
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                   TEST_REPO=str(repo), FAKE_TRAIN_EXIT=str(exit_code), REPO_ROOT=str(root),
                   CONFIG=str(root / "config.yaml"), SCENES=str(root / "scenes.json"),
                   LEGACY_INIT_CHECKPOINT=str(root / "init.pt"), RESUME_CHECKPOINT="",
                   WARMSTART_CHECKPOINT="", ARM="candidate", MAX_UPDATES="10",
                   MAX_TRAIN_STEPS="1000000", RUN_NAME=self.run.name, RUN_LABEL="",
                   RUN_PREFLIGHT="0", DRY_RUN="0", CUDA_VISIBLE_DEVICES="0,1")
        result = subprocess.run(["bash", str(repo / "scripts/train_cdpr_latent_correction_pilot_remote.sh")],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
        record = json.loads((self.run / "launch_result.json").read_text())
        self.assertEqual(record["train_exit_code"], exit_code)
        self.assertEqual(record["log_exit_code"], 0)
        self.assertIn("mock training output", (self.run / "train.log").read_text())
        self.assertIn(f"train_exit={exit_code} log_exit=0", result.stdout)
        return result.stdout

    def test_launcher_reports_success(self):
        self.assertIn("completed: update cap", self.check_launcher_exit(0))

    def test_launcher_reports_failure_and_preserves_exit_code(self):
        self.assertIn("launcher failed", self.check_launcher_exit(7))


if __name__ == "__main__":
    unittest.main()
