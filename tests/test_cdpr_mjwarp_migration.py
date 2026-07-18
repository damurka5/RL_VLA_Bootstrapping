from __future__ import annotations

import ast
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    _SHELL_ACTION_HIGH,
    _SHELL_ACTION_LOW,
    _SHELL_COUNTS,
    _SHELL_HORIZON_HIGH,
    _SHELL_HORIZON_LOW,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import (
    EqualDDPSchedule,
    RankLocalGroupLayout,
    aggregated_global_step,
    deterministic_candidate_seeds,
    deterministic_group_seeds,
    numpy_group_advantages,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    ACTIVE_INSTRUCTION_TYPES,
)
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    CDPRRenderBatch,
    SimulatorDependencyError,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    INACTIVE_CATALOG_ID,
    PRIMITIVE_NAMES,
    build_variant_arrays,
    catalog_id,
)
from rl_vla_bootstrapping.simulation.mjwarp_compat import (
    PINNED_CUDA_RUNTIME,
    PINNED_MJLAB_VERSION,
    PINNED_MUJOCO_VERSION,
    PINNED_MJWARP_VERSION,
    PINNED_TORCH_VERSION,
    PINNED_WARP_VERSION,
    inspect_cdpr_mjcf,
)
from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import (
    _mjcf_tree_sha256,
)
from robots.cdpr.cdpr_dataset.cdpr_reverse_shells import (
    SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS,
    _SMOLVLA_COMPLEX_SHELL_COUNTS,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "examples"
    / "cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml"
)
XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr_mjwarp_smoke.xml"


class CDPRMJWarpMigrationTests(unittest.TestCase):
    def test_rank_local_reset_shell_constants_match_cpu_reference(self):
        self.assertEqual(
            _SHELL_COUNTS,
            tuple(
                _SMOLVLA_COMPLEX_SHELL_COUNTS[name]
                for name in ACTIVE_INSTRUCTION_TYPES
            ),
        )
        self.assertEqual(
            tuple(zip(_SHELL_HORIZON_LOW, _SHELL_HORIZON_HIGH)),
            SMOLVLA_COMPLEX_POLICY_DECISION_BOUNDS,
        )
        cpu = load_project_config(
            ROOT
            / "configs"
            / "examples"
            / "cdpr_smolvla_complex_reverse_frontier_grpo.yaml"
        )
        action_bounds = cpu.task.metadata[
            "reverse_frontier_action_step_bounds"
        ]
        self.assertEqual(
            tuple(zip(_SHELL_ACTION_LOW, _SHELL_ACTION_HIGH)),
            tuple(tuple(int(value) for value in bounds) for bounds in action_bounds),
        )

    def test_mjlab_config_builds_separate_two_rank_plan(self):
        config = load_project_config(CONFIG)
        plan = BootstrapPipeline(config).build_stage_plans(
            ROOT / "runs" / "mjwarp_unit", ["rl"]
        )[0]
        command = plan.command
        self.assertEqual(config.simulator.backend, "mjlab_mjwarp")
        self.assertEqual(config.simulator.worlds_per_rank, 16)
        self.assertEqual(config.simulator.groups_per_rank, 2)
        self.assertEqual(command[command.index("--nproc-per-node") + 1], "2")
        self.assertIn(
            "rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr", command
        )
        self.assertEqual(command[command.index("--worlds-per-rank") + 1], "16")
        self.assertEqual(command[command.index("--groups-per-rank") + 1], "2")
        self.assertEqual(command[command.index("--grpo-group-size") + 1], "8")
        self.assertEqual(command[command.index("--hold-steps") + 1], "6")
        self.assertEqual(
            command[command.index("--action-step-gripper") + 1], "0.05"
        )
        self.assertIn("--no-lock-non-commanded-axes", command)
        self.assertEqual(
            command[
                command.index("--lock-non-commanded-axes-threshold") + 1
            ],
            "0.05",
        )
        self.assertEqual(
            command[
                command.index("--reverse-frontier-validation-episodes") + 1
            ],
            "50",
        )
        self.assertEqual(
            command[
                command.index("--reverse-frontier-demotion-success") + 1
            ],
            "-1.0",
        )
        self.assertEqual(
            Path(command[command.index("--mjwarp-xml-path") + 1]), XML
        )
        cpu_config = load_project_config(
            ROOT
            / "configs"
            / "examples"
            / "cdpr_smolvla_complex_reverse_frontier_grpo.yaml"
        )
        self.assertEqual(
            config.embodiment.action_adapter.controller_limits,
            cpu_config.embodiment.action_adapter.controller_limits,
        )

    def test_legacy_checkpoint_environment_override_emits_boolean_flag(self):
        config = load_project_config(CONFIG)
        with mock.patch.dict(
            os.environ,
            {"RLVLA_SMOLVLA_ALLOW_LEGACY_SIMULATOR_CHECKPOINT": "0"},
            clear=False,
        ):
            command = BootstrapPipeline(config).build_stage_plans(
                ROOT / "runs" / "mjwarp_boolean_unit", ["rl"]
            )[0].command
        self.assertIn("--no-allow-legacy-simulator-checkpoint", command)
        self.assertNotIn("0", command)

    def test_existing_cpu_config_remains_on_cpu_backend(self):
        config = load_project_config(
            ROOT
            / "configs"
            / "examples"
            / "cdpr_smolvla_complex_reverse_frontier_grpo.yaml"
        )
        self.assertEqual(config.simulator.backend, "mujoco_cpu")
        self.assertEqual(config.simulator.worlds_per_rank, 1)

    def test_complete_group_layout_is_contiguous_and_rank_local(self):
        layout = RankLocalGroupLayout(
            worlds_per_rank=64, groups_per_rank=8, group_size=8
        )
        layout.validate()
        np.testing.assert_array_equal(
            layout.candidate_indices[3], np.arange(24, 32)
        )
        np.testing.assert_array_equal(
            layout.base_world_indices, np.arange(0, 64, 8)
        )
        layout.assert_no_cross_rank_group(rank=1, world_size=2)
        with self.assertRaisesRegex(ValueError, "complete groups"):
            RankLocalGroupLayout(
                worlds_per_rank=63, groups_per_rank=8, group_size=8
            ).validate()

    def test_rank_and_candidate_seed_streams_are_distinct_and_deterministic(self):
        rank0 = deterministic_group_seeds(
            base_seed=17, rank=0, update_index=4, groups_per_rank=8
        )
        rank1 = deterministic_group_seeds(
            base_seed=17, rank=1, update_index=4, groups_per_rank=8
        )
        self.assertEqual(len(np.unique(rank0)), 8)
        self.assertTrue(set(rank0).isdisjoint(set(rank1)))
        candidates = deterministic_candidate_seeds(rank0, group_size=8)
        self.assertEqual(candidates.shape, (8, 8))
        self.assertEqual(len(np.unique(candidates)), 64)
        np.testing.assert_array_equal(
            rank0,
            deterministic_group_seeds(
                base_seed=17, rank=0, update_index=4, groups_per_rank=8
            ),
        )

    def test_group_advantages_are_computed_only_within_each_group(self):
        outcomes = np.array(
            [[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=np.float32,
        )
        advantages = numpy_group_advantages(outcomes, normalize=True)
        self.assertAlmostEqual(float(advantages[0].mean()), 0.0, places=6)
        np.testing.assert_allclose(advantages[1], 0.0)
        self.assertTrue(np.all(advantages[0, :4] < 0.0))
        self.assertTrue(np.all(advantages[0, 4:] > 0.0))

    def test_equal_ddp_schedule_has_fixed_backward_count(self):
        schedule = EqualDDPSchedule(
            records_per_minibatch=512,
            ppo_epochs=4,
            global_max_records=1025,
        )
        self.assertEqual(schedule.minibatches_per_epoch, 3)
        self.assertEqual(schedule.padded_records_per_rank, 1536)
        self.assertEqual(schedule.backward_collectives, 12)
        self.assertEqual(
            aggregated_global_step(
                prior_global_step=100,
                local_selected_environment_actions=64,
                world_size=2,
            ),
            228,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "the two-rank Gloo smoke requires PyTorch",
    )
    def test_two_rank_gloo_schedule_and_zero_record_backward(self):
        with tempfile.TemporaryDirectory(prefix="cdpr_ddp_smoke_") as tmp:
            output = Path(tmp) / "report.json"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                master_port = listener.getsockname()[1]
            environment = os.environ.copy()
            environment.setdefault("OMP_NUM_THREADS", "1")
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (str(ROOT), environment.get("PYTHONPATH", "")),
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--master-addr=127.0.0.1",
                    f"--master-port={master_port}",
                    "--nproc-per-node=2",
                    str(ROOT / "tests" / "_rank_local_ddp_smoke.py"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["world_size"], 2)
        self.assertEqual(report["global_max_records"], 1025)
        self.assertEqual(report["padded_records_per_rank"], 1536)
        self.assertEqual(report["backward_collectives"], 12)
        self.assertEqual(report["minimum_backward_calls"], 12)
        self.assertEqual(report["maximum_backward_calls"], 12)
        self.assertEqual(report["max_parameter_mismatch"], 0.0)

    def test_backend_config_preserves_seven_physics_substeps(self):
        config = CDPRBackendConfig(
            backend="mjlab_mjwarp",
            worlds_per_rank=16,
            groups_per_rank=2,
            grpo_group_size=8,
            hold_steps=6,
            lock_non_commanded_axes=False,
            xml_path=XML,
        )
        config.validate()
        self.assertEqual(config.physics_substeps, 7)
        self.assertFalse(config.lock_non_commanded_axes)
        with self.assertRaisesRegex(ValueError, "groups_per_rank"):
            CDPRBackendConfig(
                backend="mjlab_mjwarp",
                worlds_per_rank=8,
                groups_per_rank=2,
                grpo_group_size=8,
                xml_path=XML,
            ).validate()

    def test_fixed_mjcf_has_required_cdpr_features_and_two_cameras(self):
        report = inspect_cdpr_mjcf(XML)
        self.assertTrue(report.parse_ok)
        self.assertFalse(report.compatible)
        self.assertFalse(report.put_model_ok)
        self.assertEqual(report.unsupported, [])
        self.assertTrue(all(report.required_features.values()), report.as_dict())
        self.assertEqual(report.counts["spatial_tendons"], 4)
        self.assertGreaterEqual(report.counts["tendon_wrap_geoms"], 8)
        self.assertEqual(report.counts["cameras"], 2)
        self.assertEqual(len(_mjcf_tree_sha256(XML)), 64)

    def test_fixed_mjcf_compiles_in_reference_mujoco_when_available(self):
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("mujoco is not installed")
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(XML))
        self.assertEqual(model.ncam, 2)
        self.assertEqual(model.ntendon, 4)
        self.assertGreaterEqual(model.neq, 1)
        self.assertEqual(model.nu, 6)
        self.assertEqual(model.nq, 46)
        for name in ("overview", "ee_camera"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, name
                ),
                0,
            )

    def test_named_overview_camera_matches_legacy_free_camera_orientation(self):
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("mujoco is not installed")
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=10_000)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, 0.10]
        camera.distance = 1.5
        camera.azimuth = 90
        camera.elevation = -30
        mujoco.mjv_updateScene(
            model,
            data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        legacy_forward = np.asarray(scene.camera[0].forward).copy()
        legacy_up = np.asarray(scene.camera[0].up).copy()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        mujoco.mjv_updateScene(
            model,
            data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        np.testing.assert_allclose(scene.camera[0].forward, legacy_forward, atol=1e-6)
        np.testing.assert_allclose(scene.camera[0].up, legacy_up, atol=1e-6)

    def test_four_object_slots_keep_fixed_primitive_topology(self):
        ids = np.array(
            [
                [0, 1, 5, 6],
                [4, INACTIVE_CATALOG_ID, 2, 3],
            ],
            dtype=np.int32,
        )
        arrays = build_variant_arrays(ids)
        self.assertEqual(
            arrays["geom_size"].shape, (2, 4, len(PRIMITIVE_NAMES), 3)
        )
        self.assertEqual(arrays["body_mass"].shape, (2, 4))
        self.assertEqual(arrays["catalog_ids"].shape, (2, 4))
        self.assertAlmostEqual(float(arrays["body_mass"][1, 1]), 1.0e-4)
        self.assertEqual(catalog_id("apple"), 0)
        self.assertEqual(catalog_id("bowl"), len(ACTIVE_CDPR_CATALOGS) - 1)

    def test_camera_aux_slot_is_exact_wrist_object(self):
        overview = object()
        wrist = object()
        batch = CDPRRenderBatch(overview=overview, wrist=wrist)
        self.assertIs(batch.aux, wrist)

    def test_new_collector_has_no_per_group_object_collective(self):
        paths = (
            ROOT
            / "rl_vla_bootstrapping"
            / "policy"
            / "mjwarp_rank_local_collector.py",
            ROOT
            / "rl_vla_bootstrapping"
            / "policy"
            / "smolvla_grpo_mjwarp_cdpr.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
            self.assertNotIn("all_gather_object", source)
            self.assertNotIn("_dist_all_gather_object", source)

    def test_remote_parity_reports_sparse_success_outputs(self):
        source = (
            ROOT / "scripts" / "validate_cdpr_mjwarp_parity.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"sparse_success_output"', source)
        self.assertIn('"sparse_success":', source)

    def test_curriculum_keeps_validation_quota_and_canonical_state(self):
        source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "policy"
            / "mjwarp_rank_local_collector.py"
        ).read_text(encoding="utf-8")
        self.assertIn("validation_rollouts_per_shell", source)
        self.assertIn("pending_success_sum", source)
        self.assertIn("last_promoted_update", source)
        self.assertIn("dist.broadcast(canonical, src=0)", source)
        self.assertIn("success_rate <= float(self.demotion_success)", source)

    def test_pinned_object_hot_path_preserves_unpinned_free_bodies(self):
        collector_path = (
            ROOT
            / "rl_vla_bootstrapping"
            / "policy"
            / "mjwarp_rank_local_collector.py"
        )
        backend_path = (
            ROOT
            / "rl_vla_bootstrapping"
            / "simulation"
            / "mjlab_mjwarp_backend.py"
        )
        collector_source = collector_path.read_text(encoding="utf-8")
        backend_source = backend_path.read_text(encoding="utf-8")
        self.assertIn("finger_object_contact_mask(target_slots)", collector_source)
        self.assertIn("set_target_body_positions(", collector_source)
        self.assertNotIn(
            "self.backend.set_free_body_poses(\n"
            "            self.backend.object_body_ids,\n"
            "            positions,",
            collector_source,
        )
        self.assertIn("_write_configured_pinned_poses()", backend_source)
        self.assertIn("self._pinned_mask, torch.zeros_like(current), current", backend_source)

    def test_throughput_accounting_preserves_one_selected_candidate_per_group(self):
        source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "policy"
            / "mjwarp_rank_local_collector.py"
        ).read_text(encoding="utf-8")
        self.assertIn("sampled_actions += step_active.sum()", source)
        self.assertIn("action_counts_by_group.gather(", source)
        self.assertIn('"trajectory_work_amplification"', source)

    def test_production_disables_cuda_timing_barriers_but_benchmark_enables_them(self):
        trainer_source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "policy"
            / "smolvla_grpo_mjwarp_cdpr.py"
        ).read_text(encoding="utf-8")
        benchmark_source = (
            ROOT / "scripts" / "benchmark_cdpr_mjlab_grpo.py"
        ).read_text(encoding="utf-8")
        smoke_source = (
            ROOT / "scripts" / "smoke_cdpr_mjlab_two_gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("profile=bool(args.mjwarp_profile_timers)", trainer_source)
        self.assertIn('"--mjwarp-profile-timers"', benchmark_source)
        self.assertIn('"--no-lock-non-commanded-axes"', benchmark_source)
        self.assertIn("--no-lock-non-commanded-axes", smoke_source)

    def test_pinned_stack_is_exact_and_cuda_12_8(self):
        self.assertEqual(PINNED_MJLAB_VERSION, "1.5.0")
        self.assertEqual(PINNED_MUJOCO_VERSION, "3.10.0")
        self.assertEqual(PINNED_MJWARP_VERSION, "3.10.0.1")
        self.assertEqual(PINNED_WARP_VERSION, "1.14.0")
        self.assertEqual(PINNED_TORCH_VERSION, "2.7.1")
        self.assertEqual(PINNED_CUDA_RUNTIME, "12.8")

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is None,
        "dependency-error assertion is for hosts without the pinned CUDA stack",
    )
    def test_explicit_mjwarp_selection_fails_with_dependency_error(self):
        with self.assertRaises(SimulatorDependencyError):
            create_cdpr_backend(
                CDPRBackendConfig(
                    backend="mjlab_mjwarp",
                    worlds_per_rank=8,
                    groups_per_rank=1,
                    grpo_group_size=8,
                    xml_path=XML,
                )
            )

    def test_xml_declares_overview_and_wrist_frame_sensors(self):
        roots = []
        root = ET.parse(XML).getroot()
        roots.append(root)
        for include in root.findall(".//include"):
            roots.append(ET.parse(XML.parent / include.get("file")).getroot())
        sensors = {
            (node.tag, node.get("objtype"), node.get("objname"))
            for tree in roots
            for node in tree.findall(".//sensor/*")
        }
        self.assertTrue(
            {
                ("framepos", "camera", "overview"),
                ("framequat", "camera", "overview"),
                ("framepos", "camera", "ee_camera"),
                ("framequat", "camera", "ee_camera"),
            }.issubset(sensors)
        )
        backend_source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "simulation"
            / "mjlab_mjwarp_backend.py"
        ).read_text(encoding="utf-8")
        smoke_source = (
            ROOT / "scripts" / "smoke_cdpr_mjlab_backend.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"sensordata"', backend_source)
        self.assertIn('"finger_equality_active"', smoke_source)
        self.assertIn('"spatial_tendons_evolve_under_control"', smoke_source)
        self.assertIn('"contact_step_generated_contacts"', smoke_source)


if __name__ == "__main__":
    unittest.main()
