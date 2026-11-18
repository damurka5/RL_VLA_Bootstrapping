"""A grasp audit must expose rejected real gates without relabeling motion."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.audit import grasp_detector_trace as audit
from rl_vla_bootstrapping.policy import mjwarp_rank_local_collector as collector_module
from rl_vla_bootstrapping.simulation.cdpr_backend import CDPRFingerContactBatch
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import BatchedTaskState, INSTRUCTION_TO_ID
from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import FINGER_CONTACT_MAX_DISTANCE_M, MJLabMJWarpCDPRBackend


def fixture():
    steps, worlds = 4, 1
    xyz = np.zeros((steps, worlds, 2, 3), dtype=np.float32)
    xyz[:, 0, 0, 2] = [.17, .23, .24, .25]
    record = {
        "active": np.ones((steps, worlds), dtype=bool),
        "caught_target": np.array([[False], [True], [True], [True]]),
        "success": np.zeros((steps, worlds), dtype=bool), "object_xyz": xyz,
        "ee_xyz": xyz[:, :, 0].copy(), "target_slots": np.array([0]),
        "initial_target_xyz": np.array([[0., 0., .17]]),
        "reset_object_xyz": xyz[0].copy(), "reset_ee_xyz": xyz[0, :, 0].copy(),
        "gripper_opening": np.ones((steps, worlds)), "target_catalog_ids": np.array([0]),
        "round_index": np.array(1000), "instruction_ids": np.array([0]),
    }
    columns = {key: np.zeros((steps, worlds), dtype=np.float32) for key in audit.COLUMNS}
    for key in ("active", "eligible", "left_contact", "right_contact", "stable_pose", "release_open", "loaded_finger_contact"):
        columns[key][:] = 1
    columns["left_force_n"][:] = columns["right_force_n"][:] = .2
    columns["persistence_steps"][:, 0] = [1, 2, 3, 4]
    columns["physical_grasp"][:] = columns["scored_grasp"][:] = record["caught_target"]
    metadata = {"schema": 1, "min_force_n": .05, "persistence_steps": 2,
                "max_position_slip_m": .008, "max_orientation_slip_rad": .15,
                "contact_based_grasp": True, "catalogs": ["bell_pepper"], "instructions": ["put_into_bowl"]}
    return record, columns, metadata


def diagnose(record, columns, metadata):
    values = np.stack([columns[key] for key in audit.COLUMNS], axis=-1)
    return audit.diagnose(record, values, audit.COLUMNS, metadata, np.array([0]))


class GateAnalysisTests(unittest.TestCase):
    def test_loaded_raw_contact_rejected_by_distance_is_distinguished(self):
        record, columns, metadata = fixture()
        metadata["raw_contact_available"] = True
        record["caught_target"][:] = False
        columns["physical_grasp"][:] = columns["scored_grasp"][:] = 0
        columns["left_contact"][:] = columns["left_force_n"][:] = 0
        columns["persistence_steps"][:] = 0
        columns["left_distance_rejected_force_n"][:] = .2
        rows, _, reasons, _ = diagnose(record, columns, metadata)
        self.assertEqual(rows[0]["loaded_contacts_outside_distance_window_steps"], 4)
        self.assertEqual(reasons[-1, 0], "contact_distance_filter")
        self.assertEqual(rows[0]["detector_mismatch_steps"], 0)

    def test_correct_wide_grasp_has_no_detector_or_scoring_mismatch(self):
        record, columns, metadata = fixture()
        rows, _, _, suspicious = diagnose(record, columns, metadata)
        self.assertEqual(rows[0]["detector_mismatch_steps"], 0)
        self.assertEqual(rows[0]["scoring_mismatch_steps"], 0)
        self.assertEqual(rows[0]["wide_loaded_counted_steps"], 3)
        self.assertEqual(rows[0]["wide_loaded_unrecognized_steps"], 0)
        self.assertEqual(rows[0]["opening_v1_would_reject_steps"], 3)
        self.assertFalse(suspicious.any())

    def test_all_gates_passing_but_grasp_rejected_is_a_detector_error(self):
        record, columns, metadata = fixture()
        record["caught_target"][:] = False
        columns["physical_grasp"][:] = columns["scored_grasp"][:] = 0
        rows, _, _, suspicious = diagnose(record, columns, metadata)
        self.assertEqual(rows[0]["all_gates_passed_but_uncounted_steps"], 3)
        self.assertTrue(suspicious[-1, 0])

    def test_physical_grasp_rejected_only_by_predicate_is_scoring_error(self):
        record, columns, metadata = fixture()
        columns["scored_grasp"][:] = 0
        rows, _, _, _ = diagnose(record, columns, metadata)
        self.assertEqual(rows[0]["detector_mismatch_steps"], 0)
        self.assertEqual(rows[0]["scoring_mismatch_steps"], 3)
        self.assertEqual(rows[0]["grasped_but_not_scored_steps"], 3)

    def test_legacy_opening_veto_is_identified_as_the_old_contract(self):
        record, columns, metadata = fixture()
        metadata["contact_based_grasp"] = False
        record["caught_target"][:] = False
        columns["physical_grasp"][:] = columns["scored_grasp"][:] = 0
        rows, _, reasons, _ = diagnose(record, columns, metadata)
        self.assertEqual(rows[0]["detector_mismatch_steps"], 0)
        self.assertEqual(rows[0]["legacy_opening_veto_steps"], 3)
        self.assertEqual(reasons[-1, 0], "legacy_opening_veto")

    def test_motion_without_bilateral_force_is_only_a_suspicion_flag(self):
        record, columns, metadata = fixture()
        record["caught_target"][:] = False
        columns["physical_grasp"][:] = columns["scored_grasp"][:] = 0
        columns["left_force_n"][:] = 0
        columns["persistence_steps"][:] = 0
        before = record["success"].copy()
        rows, _, reasons, suspicious = diagnose(record, columns, metadata)
        self.assertTrue(suspicious[-1, 0])
        self.assertEqual(reasons[-1, 0], "force_below_threshold")
        self.assertEqual(rows[0]["detector_mismatch_steps"], 0)
        np.testing.assert_array_equal(record["success"], before)

    def test_nonfinite_force_is_reported_separately(self):
        record, columns, metadata = fixture()
        metadata["raw_contact_available"] = True
        record["caught_target"][:] = False
        columns["physical_grasp"][:] = columns["scored_grasp"][:] = 0
        columns["left_force_n"][:] = float("nan")
        columns["raw_left_force_n"][:] = float("nan")
        columns["persistence_steps"][:] = 0
        rows, _, reasons, _ = diagnose(record, columns, metadata)
        self.assertEqual(rows[0]["nonfinite_force_steps"], 4)
        self.assertEqual(rows[0]["nonfinite_raw_pad_force_steps"], 4)
        self.assertEqual(reasons[-1, 0], "nonfinite_force")

    def test_live_divergence_is_excluded_from_missed_grasp_evidence(self):
        record, columns, metadata = fixture()
        columns["active"][2:] = 0
        rows, _, _, suspicious = diagnose(record, columns, metadata)
        self.assertTrue(rows[0]["live_diverged"])
        self.assertEqual(rows[0]["audited_steps"], 0)
        self.assertFalse(suspicious.any())

    def test_seeded_persistence_is_not_reported_as_a_counter_bug(self):
        record, columns, metadata = fixture()
        record["caught_target"][:] = True
        columns["physical_grasp"][:] = columns["scored_grasp"][:] = 1
        columns["persistence_steps"][:, 0] = [3, 4, 5, 6]
        values = np.stack([columns[key] for key in audit.COLUMNS], axis=-1)
        rows, _, _, _ = audit.diagnose(record, values, audit.COLUMNS, metadata, np.array([2]))
        self.assertEqual(rows[0]["persistence_mismatch_steps"], 0)


class CaptureTests(unittest.TestCase):
    def test_raw_probe_finds_force_filtered_by_distance_without_resolving_contacts(self):
        calls = []
        backend = SimpleNamespace(
            torch=torch, _device="cpu", worlds_per_rank=2,
            _contact_geom=torch.tensor([[5, 1], [1, 6], [5, 3], [2, 4]]),
            _contact_worldid=torch.tensor([0, 0, 0, -1]), _contact_indices=torch.arange(4),
            _nacon=torch.tensor([4]), _contact_dist=torch.tensor([.003, .001, 0., 0.]),
            _slot_geom_ids=torch.tensor([[1, 2], [3, 4]]), _finger_pad_geom_ids_tensor=torch.tensor([5, 6]),
            _contact_forces=torch.tensor([[.2, 0, 0, 0, 0, 0], [.3, 0, 0, 0, 0, 0],
                                           [float("nan"), 0, 0, 0, 0, 0], [float("inf"), 0, 0, 0, 0, 0]]),
            wp=SimpleNamespace(ScopedDevice=lambda _: contextlib.nullcontext()),
            mjw=SimpleNamespace(contact_force=lambda *args: calls.append(1)), model=None, data=None,
            _contact_ids_wp=None, _contact_forces_wp=None, _left_pad_force=torch.zeros(2), _right_pad_force=torch.zeros(2),
            _left_pad_contact_count=torch.zeros(2, dtype=torch.int32), _right_pad_contact_count=torch.zeros(2, dtype=torch.int32))
        native = MJLabMJWarpCDPRBackend.finger_object_contact_metrics(backend, torch.tensor([0, 1]))
        raw = audit.raw_pad_metrics(backend, torch.tensor([0, 1]), FINGER_CONTACT_MAX_DISTANCE_M)
        self.assertEqual(native.left_normal_force[0].item(), 0.)
        self.assertAlmostEqual(raw["raw_left_force_n"][0].item(), .2)
        self.assertAlmostEqual(raw["left_distance_rejected_force_n"][0].item(), .2)
        self.assertAlmostEqual(raw["left_contact_distance_m"][0].item(), .003)
        self.assertEqual(raw["right_distance_rejected_force_n"][0].item(), 0.)
        self.assertTrue(torch.isinf(raw["left_contact_distance_m"][1]))
        self.assertEqual(len(calls), 1)

    def test_trace_and_round_recorder_share_the_same_physics_and_predicate_steps(self):
        from tools.audit.sil_record import _RoundRecorder
        identity = torch.tensor([[1., 0., 0., 0.]])
        current = [None]
        def observation(z):
            return SimpleNamespace(ee_position=torch.tensor([[0., 0., z]]), ee_quaternion=identity,
                                   object_positions=torch.tensor([[[0., 0., z], [.3, 0., .15]]]),
                                   object_quaternions=identity[:, None].repeat(1, 2, 1), gripper_opening=torch.ones(1))
        current[0] = observation(.17)
        calls = []
        def contact(slots):
            calls.append(1)
            return CDPRFingerContactBatch(torch.tensor([True]), torch.tensor([True]),
                                          torch.tensor([.2]), torch.tensor([.3]))
        def step(actions, active):
            current[0] = observation(.17 if len(calls) == 0 else .23)
            return current[0]
        backend = SimpleNamespace(finger_object_contact_metrics=contact, step=step,
                                  low_dim_observations=lambda: current[0], pop_nonfinite_world_events=lambda: 0,
                                  pop_nonfinite_world_report=lambda: (0, np.zeros(1, dtype=bool)),
                                  nonfinite_world_mask=lambda: torch.tensor([False]))
        collector = SimpleNamespace(torch=torch, backend=backend, _world_rows=torch.arange(1),
                                    actions_per_policy_decision=1, catch_release_dense_reward=None)
        collector._update_physical_grasp = MethodType(collector_module.RankLocalMJWarpGRPOCollector._update_physical_grasp, collector)
        state = BatchedTaskState(
            instruction_ids=torch.tensor([INSTRUCTION_TO_ID["pick_up"]]), target_slots=torch.tensor([0]),
            reference_slots=torch.tensor([1]), second_reference_slots=torch.tensor([-1]),
            initial_target_positions=torch.tensor([[0., 0., .17]]), ever_grasped=torch.tensor([False]),
            grasped=torch.tensor([False]), step_count=torch.tensor([0]), release_threshold=torch.tensor([.90]),
            support_surface_z=torch.tensor([.15]), target_rest_height=torch.tensor([.02]), contact_based_grasp=True)
        reset = SimpleNamespace(task_state=state, grasp_eligible=torch.tensor([True]),
                                bilateral_contact_steps=torch.tensor([0]), physical_grasp=torch.tensor([False]),
                                previous_relative_position=torch.zeros(1, 3), previous_relative_quaternion=identity.clone(),
                                target_rest_height=torch.tensor([.02]), horizons=torch.tensor([2]),
                                instructions=["pick up apple"], group_target_catalog_ids=torch.tensor([0]))
        collector.resetter = SimpleNamespace(reset=lambda **kwargs: reset)
        collector.trainer = SimpleNamespace(deterministic_action_chunks_tensor=lambda **kwargs: torch.zeros(1, 1, 5))
        def validate_round(**kwargs):
            current_reset = collector.resetter.reset()
            for _ in range(2):
                action = collector.trainer.deterministic_action_chunks_tensor(
                    states=torch.zeros(1, 6), priors=torch.zeros(1, 1, 5), action_count=1)
                low = backend.step(action[:, 0], torch.tensor([True]))
                _, grasp, diagnostics = collector._update_physical_grasp(current_reset, low, torch.tensor([True]))
                collector_module.evaluate_active_sparse_tasks(
                    state=state, ee_position=low.ee_position, object_positions=low.object_positions,
                    gripper_opening=low.gripper_opening, caught_target=grasp, active_mask=torch.tensor([True]),
                    loaded_finger_contact=diagnostics["loaded_finger_contact"], max_steps=100)
        collector.validate_round = validate_round
        world = SimpleNamespace(torch=torch, backend=backend, collector=collector,
                                layout=SimpleNamespace(group_size=1, worlds_per_rank=1))
        native_score = collector_module.evaluate_active_sparse_tasks
        with audit.GraspTraceCapture(world) as capture:
            with _RoundRecorder(world) as recorder:
                recording = recorder.run(round_index=1000)
        self.assertEqual(len(calls), 2)
        self.assertEqual(recording.caught_target[:, 0].tolist(), [False, True])
        self.assertEqual(len(capture.rows), len(recording.active))
        self.assertIs(collector_module.evaluate_active_sparse_tasks, native_score)

    def test_capture_calls_the_real_detector_once_and_restores_all_hooks(self):
        contacts = CDPRFingerContactBatch(torch.tensor([True]), torch.tensor([True]),
                                          torch.tensor([.2]), torch.tensor([.3]))
        calls = []
        def contact(slots):
            calls.append(1)
            return contacts
        backend = SimpleNamespace(finger_object_contact_metrics=contact)
        collector = SimpleNamespace(torch=torch, backend=backend, _world_rows=torch.arange(1))
        collector._update_physical_grasp = MethodType(collector_module.RankLocalMJWarpGRPOCollector._update_physical_grasp, collector)
        world = SimpleNamespace(torch=torch, backend=backend, collector=collector, layout=SimpleNamespace(group_size=1))
        state = BatchedTaskState(
            instruction_ids=torch.tensor([INSTRUCTION_TO_ID["pick_up"]]), target_slots=torch.tensor([0]),
            reference_slots=torch.tensor([1]), second_reference_slots=torch.tensor([-1]),
            initial_target_positions=torch.tensor([[0., 0., .17]]), ever_grasped=torch.tensor([False]),
            grasped=torch.tensor([False]), step_count=torch.tensor([0]), release_threshold=torch.tensor([.90]),
            support_surface_z=torch.tensor([.15]), target_rest_height=torch.tensor([.02]), contact_based_grasp=True)
        reset = SimpleNamespace(task_state=state, grasp_eligible=torch.tensor([True]),
                                bilateral_contact_steps=torch.tensor([0]), physical_grasp=torch.tensor([False]),
                                previous_relative_position=torch.zeros(1, 3), previous_relative_quaternion=torch.tensor([[1., 0., 0., 0.]]),
                                target_rest_height=torch.tensor([.02]))
        low = SimpleNamespace(ee_position=torch.tensor([[0., 0., .23]]), ee_quaternion=torch.tensor([[1., 0., 0., 0.]]),
                              object_positions=torch.tensor([[[0., 0., .23], [.3, 0., .15]]]),
                              object_quaternions=torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]]), gripper_opening=torch.ones(1))
        original_score = collector_module.evaluate_active_sparse_tasks
        original_grasp = collector._update_physical_grasp
        with audit.GraspTraceCapture(world) as capture:
            for _ in range(3):
                _, grasp, diagnostics = collector._update_physical_grasp(reset, low, torch.tensor([True]))
                collector_module.evaluate_active_sparse_tasks(
                    state=state, ee_position=low.ee_position, object_positions=low.object_positions,
                    gripper_opening=low.gripper_opening, caught_target=grasp, active_mask=torch.tensor([True]),
                    loaded_finger_contact=diagnostics["loaded_finger_contact"], max_steps=100)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(capture.rows), 3)
        self.assertIs(collector_module.evaluate_active_sparse_tasks, original_score)
        self.assertIs(collector._update_physical_grasp, original_grasp)
        self.assertIs(backend.finger_object_contact_metrics, contact)
        contacts.left_normal_force.fill_(9.)
        self.assertAlmostEqual(capture.rows[0][0, audit.COLUMNS.index("left_force_n")], .2, places=6)

    def test_hooks_restore_after_a_failed_scoring_step(self):
        def contact(slots):
            return None
        def grasp(*args):
            return None
        backend = SimpleNamespace(finger_object_contact_metrics=contact)
        collector = SimpleNamespace(_update_physical_grasp=grasp)
        world = SimpleNamespace(torch=torch, backend=backend, collector=collector, layout=SimpleNamespace(group_size=1))
        native = collector_module.evaluate_active_sparse_tasks
        with self.assertRaisesRegex(RuntimeError, "no detector call"):
            with audit.GraspTraceCapture(world):
                # Native scoring need not be run to exercise an unwind on failure.
                raise RuntimeError("no detector call")
        self.assertIs(collector_module.evaluate_active_sparse_tasks, native)
        self.assertIs(backend.finger_object_contact_metrics, contact)


class ReportTests(unittest.TestCase):
    def test_pair_fingerprint_and_reports_roundtrip_without_changing_verdicts(self):
        record, columns, metadata = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "record_1000.npz"
            np.savez_compressed(path, **record)
            capture = audit.GraspTraceCapture(None)
            capture.rows = list(np.stack([columns[key] for key in audit.COLUMNS], axis=-1))
            capture.initial_persistence = np.array([0])
            capture.metadata = metadata
            rec = SimpleNamespace(active=record["active"], caught_target=record["caught_target"])
            capture.save(root / "grasp_trace_1000.npz", rec, path)
            with contextlib.redirect_stdout(io.StringIO()):
                report = audit.analyze_files(str(root / "record_*.npz"), root / "audit")
            self.assertEqual(report["by_object"][0]["wide_loaded_counted_steps"], 3)
            self.assertTrue((root / "audit/episodes.csv").is_file())
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                audit.analyze_files(str(path), root / "changed")

    def test_gpu_command_has_trace_mode_and_anonymous_hf_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "rl").mkdir()
            (run / "rl/latest.pt").write_bytes(b"fixture; never loaded")
            (run / "continuation.json").write_text(json.dumps({"source_pipeline": "/saved/pipeline"}))
            with patch.dict(audit.os.environ, {"HF_TOKEN": "expired-fixture", "RLVLA_HF_PUBLIC_MODELS_ONLY": "1"}), \
                    patch.object(audit.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
                result = audit.main(["--run-dir", str(run), "--output", str(run / "audit")])
            self.assertEqual(result, 0)
            self.assertIn("--grasp-audit", child.call_args.args[0])
            self.assertEqual(child.call_args.kwargs["env"]["RLVLA_EMBODIMENT"], "widowx200")
            self.assertNotIn("HF_TOKEN", child.call_args.kwargs["env"])

    def test_overlay_uses_previous_chunk_endpoint_and_labels_terminal_as_later(self):
        from PIL import ImageDraw
        record, columns, metadata = fixture()
        capture = SimpleNamespace(values=np.stack([columns[key] for key in audit.COLUMNS], axis=-1), metadata=metadata)
        rec = SimpleNamespace(actions_per_decision=1, gripper_opening=record["gripper_opening"], episode_success=[False])
        lines = []
        with patch.object(ImageDraw, "Draw", return_value=SimpleNamespace(text=lambda xy, text, **kwargs: lines.append(text))):
            frames = audit.annotate_frames([np.zeros((240, 640, 3), dtype=np.uint8)] * 5, rec, capture, 0)
        self.assertEqual(frames[0].shape, (336, 640, 3))
        self.assertTrue(any("After env step 1: physical=0" in line for line in lines))
        self.assertTrue(any("After env step 2: physical=1" in line for line in lines))
        self.assertTrue(any("POST-ROLLOUT VIEW" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
