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
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    RankLocalMJWarpGRPOCollector,
    ValidationRound,
    _SHELL_ACTION_HIGH,
    _SHELL_ACTION_LOW,
    _SHELL_COUNTS,
    _SHELL_HORIZON_HIGH,
    _SHELL_HORIZON_LOW,
    resolve_mjwarp_catalog_ids,
    resolve_mjwarp_instruction_ids,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import (
    EqualDDPSchedule,
    RankLocalGroupLayout,
    aggregated_global_step,
    deterministic_candidate_seeds,
    deterministic_group_seeds,
    numpy_group_advantages,
)
from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
    _end_to_end_time_metrics,
    _log_tensorboard_metrics,
    _make_mjwarp_progress_bar,
    _synchronize_validation_rounds,
    _update_mjwarp_progress_bar,
    _validation_due as _mjwarp_validation_due,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    ACTIVE_INSTRUCTION_TYPES,
    BatchedMoveToDistanceReward,
)
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    CDPRFingerContactBatch,
    CDPRRenderBatch,
    SimulatorDependencyError,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    COLLISION_GEOM_SLOT_NAMES,
    GEOM_SLOT_NAMES,
    INACTIVE_CATALOG_ID,
    OBJECT_VARIANTS,
    _spec_body,
    catalog_id,
    compile_catalog_variant_models,
    object_assets_sha256,
    slot_geom_name,
    validate_object_assets,
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
    _calibrate_host_cdpr,
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
SCRATCH_MOVE_TO_CONFIG = (
    ROOT
    / "configs"
    / "examples"
    / "cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml"
)
XML = ROOT / "robots" / "cdpr" / "cdpr_mujoco" / "cdpr_mjwarp_smoke.xml"


class CDPRMJWarpMigrationTests(unittest.TestCase):
    def test_scratch_move_to_config_is_step_zero_two_rank_distance_grpo(self):
        config = load_project_config(SCRATCH_MOVE_TO_CONFIG)
        plan = BootstrapPipeline(config).build_stage_plans(
            ROOT / "runs" / "mjwarp_move_to_unit", ["rl"]
        )[0]
        command = plan.command
        expected_objects = (
            "robocasa_apple",
            "robocasa_banana",
            "robocasa_tomato",
            "robocasa_orange",
            "robocasa_potato",
            "robocasa_mug",
            "robocasa_plate",
            "robocasa_bowl",
        )
        self.assertEqual(config.task.instruction_types, ("move_to_object",))
        self.assertEqual(config.task.target_objects, expected_objects)
        self.assertEqual(config.task.metadata["reward_mode"], "dense")
        self.assertEqual(
            command[command.index("--nproc-per-node") + 1], "2"
        )
        self.assertEqual(
            command[command.index("--max-train-steps") + 1], "2000000"
        )
        self.assertEqual(command[command.index("--hidden-dim") + 1], "1024")
        self.assertEqual(
            command[command.index("--smolvla-compile-mode") + 1],
            "max-autotune-no-cudagraphs",
        )
        self.assertEqual(
            command[command.index("--mjwarp-profile-updates") + 1], "4"
        )
        self.assertEqual(
            command[command.index("--validation-every-steps") + 1],
            "200000",
        )
        self.assertEqual(
            command[
                command.index("--validation-episodes-per-instruction") + 1
            ],
            "1024",
        )
        self.assertEqual(
            command[command.index("--validation-seed") + 1],
            "1000000",
        )
        self.assertEqual(
            command[command.index("--save-every-steps") + 1],
            "200000",
        )
        self.assertNotIn("--resume-checkpoint", command)
        allowed_index = command.index("--allowed-objects") + 1
        self.assertEqual(
            tuple(command[allowed_index : allowed_index + len(expected_objects)]),
            expected_objects,
        )
        self.assertEqual(
            tuple(
                ACTIVE_CDPR_CATALOGS[index]
                for index in resolve_mjwarp_catalog_ids(expected_objects)
            ),
            expected_objects,
        )
        self.assertEqual(resolve_mjwarp_instruction_ids(("move_to_object",)), (0,))
        self.assertEqual(
            tuple(OBJECT_VARIANTS[name].label for name in expected_objects),
            (
                "apple",
                "banana",
                "tomato",
                "orange",
                "potato",
                "mug",
                "plate",
                "bowl",
            ),
        )

    def test_tensorized_move_to_reward_reads_existing_distance_parameters(self):
        reward = BatchedMoveToDistanceReward.from_metadata(
            {
                "move_to_object_xy_window_low": 0.0,
                "move_to_object_xy_window_high": 0.02,
                "move_to_object_xy_reward_scale": 0.08,
                "move_to_object_distance_reward_weight": 1.0,
                "distance_reward_exponent": 2.0,
            }
        )
        self.assertEqual(reward.xy_window_low, 0.0)
        self.assertEqual(reward.xy_window_high, 0.02)
        self.assertEqual(reward.xy_reward_scale, 0.08)
        self.assertEqual(reward.distance_reward_weight, 1.0)
        self.assertEqual(reward.distance_reward_exponent, 2.0)

    def test_scratch_launcher_rejects_resume_state(self):
        source = (
            ROOT
            / "scripts"
            / "train_cdpr_smolvla_move_to_grpo_mjlab_dual_remote.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("MAX_TRAIN_STEPS=\"${MAX_TRAIN_STEPS:-2000000}\"", source)
        self.assertIn("unset RLVLA_SMOLVLA_RESUME_CHECKPOINT", source)
        self.assertIn("Scratch training refuses CHECKPOINT", source)
        self.assertIn("configure_huggingface_public_models", source)
        self.assertIn("huggingface_public_models_preflight", source)
        self.assertIn(
            'printf \'tensorboard_dir=%s\\n\' "$RUN_DIR/rl/tensorboard"',
            source,
        )

    def test_mjwarp_validation_crosses_cadence_and_filters_tensorboard_values(
        self,
    ):
        args = SimpleNamespace(
            validation_every_steps=200_000,
            validation_episodes_per_instruction=1024,
        )
        self.assertFalse(
            _mjwarp_validation_due(
                args,
                global_step=199_999,
                last_validation_step=0,
            )
        )
        self.assertTrue(
            _mjwarp_validation_due(
                args,
                global_step=200_064,
                last_validation_step=0,
            )
        )
        self.assertFalse(
            _mjwarp_validation_due(
                args,
                global_step=399_999,
                last_validation_step=200_064,
            )
        )
        self.assertTrue(
            _mjwarp_validation_due(
                args,
                global_step=400_128,
                last_validation_step=200_064,
            )
        )

        writer = mock.Mock()
        _log_tensorboard_metrics(
            writer,
            {
                "loss": 1.25,
                "profile/dominant_stage": "smolvla_inference",
                "not_finite": float("nan"),
            },
            200_064,
        )
        writer.add_scalar.assert_called_once_with(
            "loss", 1.25, 200_064
        )
        writer.flush.assert_called_once_with()

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "validation aggregation requires PyTorch",
    )
    def test_mjwarp_validation_reports_each_object(self):
        import torch

        catalog_count = len(ACTIVE_CDPR_CATALOGS)
        successes = torch.tensor(
            [[False, True]] * catalog_count,
            dtype=torch.bool,
        )
        validation_round = ValidationRound(
            candidate_rewards=torch.full(
                (catalog_count, 2), 0.75, dtype=torch.float32
            ),
            candidate_success=successes,
            final_xy_distance=torch.full(
                (catalog_count, 2), 0.03, dtype=torch.float32
            ),
            group_target_catalog_ids=torch.arange(
                catalog_count, dtype=torch.int64
            ),
            group_shell_ids=torch.zeros(
                (catalog_count,), dtype=torch.int64
            ),
            metrics={
                "validation/time_s": 2.0,
                "validation/environment_actions": 64.0,
            },
        )
        metrics = _synchronize_validation_rounds(
            [validation_round],
            device=torch.device("cpu"),
        )
        self.assertEqual(
            metrics["validation/episodes"],
            float(catalog_count * 2),
        )
        self.assertEqual(metrics["validation/success_rate"], 0.5)
        self.assertAlmostEqual(metrics["validation/reward_mean"], 0.75)
        for catalog_name in ACTIVE_CDPR_CATALOGS:
            label = OBJECT_VARIANTS[catalog_name].label.replace(" ", "_")
            self.assertEqual(
                metrics[f"validation/by_object/{label}/episodes"],
                2.0,
            )
            self.assertEqual(
                metrics[f"validation/by_object/{label}/success_rate"],
                0.5,
            )

    def test_mjwarp_progress_uses_end_to_end_time_and_global_selected_steps(
        self,
    ):
        time_metrics = _end_to_end_time_metrics(
            start_step=0,
            global_step=100,
            max_train_steps=1_000,
            elapsed_seconds=2.0,
        )
        self.assertEqual(
            time_metrics[
                "training/end_to_end_selected_actions_per_second"
            ],
            50.0,
        )
        self.assertEqual(
            time_metrics["training/estimated_remaining_time_s"],
            18.0,
        )
        self.assertEqual(
            time_metrics["training/estimated_total_time_s"],
            20.0,
        )

        progress = mock.Mock()
        displayed = _update_mjwarp_progress_bar(
            progress,
            previous_display_step=9_984,
            global_step=10_471,
            max_train_steps=2_000_000,
            update_index=21,
            metrics={
                "sampled_actions_per_second_global": 1246.1,
                "selected_actions_per_second_global": 155.6,
                "candidate_successes": 167.0,
                "candidate_worlds": 1024.0,
                "informative_records": 3948.0,
            },
        )
        self.assertEqual(displayed, 10_471)
        progress.update.assert_called_once_with(487)
        postfix = progress.set_postfix.call_args.args[0]
        self.assertEqual(postfix["update"], 21)
        self.assertEqual(postfix["sampled/s"], "1246.1")
        self.assertEqual(postfix["rollout-selected/s"], "155.6")
        self.assertEqual(postfix["success"], "167/1024")
        self.assertEqual(postfix["records"], "3948")
        self.assertFalse(progress.set_postfix.call_args.kwargs["refresh"])

    def test_mjwarp_progress_bar_is_enabled_through_remote_tee(self):
        args = SimpleNamespace(
            progress=True,
            max_train_steps=2_000_000,
            progress_refresh_seconds=10.0,
        )
        fake_bar = mock.Mock()
        with mock.patch(
            "rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr.tqdm",
            return_value=fake_bar,
        ) as tqdm_mock:
            created = _make_mjwarp_progress_bar(
                args=args,
                is_main=True,
                start_step=10_471,
            )
        self.assertIs(created, fake_bar)
        kwargs = tqdm_mock.call_args.kwargs
        self.assertEqual(kwargs["total"], 2_000_000)
        self.assertEqual(kwargs["initial"], 10_471)
        self.assertFalse(kwargs["disable"])
        self.assertEqual(kwargs["unit"], " selected-step")

    def test_mjspec_body_lookup_supports_current_and_legacy_apis(self):
        body = SimpleNamespace(name="mjwarp_object_slot_0")
        modern = SimpleNamespace(
            body=lambda name: body if name == body.name else None
        )
        legacy = SimpleNamespace(
            find_body=lambda name: body if name == body.name else None
        )
        plural_only = SimpleNamespace(bodies=[body])

        self.assertIs(_spec_body(modern, body.name), body)
        self.assertIs(_spec_body(legacy, body.name), body)
        self.assertIs(_spec_body(plural_only, body.name), body)
        with self.assertRaises(KeyError):
            _spec_body(SimpleNamespace(bodies=[]), body.name)

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
        self.assertEqual(config.simulator.njmax, 1024)
        self.assertEqual(command[command.index("--nproc-per-node") + 1], "2")
        self.assertIn(
            "rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr", command
        )
        self.assertEqual(command[command.index("--worlds-per-rank") + 1], "16")
        self.assertEqual(command[command.index("--groups-per-rank") + 1], "2")
        self.assertEqual(command[command.index("--mjwarp-njmax") + 1], "1024")
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
        self.assertEqual(config.njmax, 1024)
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
        self.assertEqual(report.counts["object_visual_mesh_slots"], 4)
        self.assertEqual(
            report.counts["object_collision_primitive_slots"], 44
        )
        self.assertGreaterEqual(report.counts["mesh_assets"], 8)
        self.assertGreaterEqual(report.counts["texture_assets"], 7)
        self.assertEqual(len(_mjcf_tree_sha256(XML)), 64)

    def test_mjwarp_mjcf_uses_compatible_ccd_and_safe_reset_poses(self):
        root = ET.parse(XML).getroot()
        flag = root.find("./option/flag")
        self.assertIsNotNone(flag)
        self.assertEqual(flag.get("multiccd"), "disable")
        self.assertEqual(flag.get("nativeccd"), "disable")

        slots = ET.parse(
            XML.parent / "cdpr_mjwarp_object_slots.xml"
        ).getroot()
        positions = {
            tuple(float(value) for value in body.get("pos").split())
            for body in slots.findall("./worldbody/body")
            if str(body.get("name") or "").startswith("mjwarp_object_slot_")
        }
        self.assertEqual(len(positions), 4)
        self.assertTrue(
            all(
                abs(x) >= 4.0 and abs(y) >= 4.0 and z >= 4.0
                for x, y, z in positions
            )
        )

        backend_source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "simulation"
            / "mjlab_mjwarp_backend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.host_model.opt.timestep = 0.002", backend_source)
        self.assertNotIn(
            "self.host_model.opt.timestep = 1.0 / 60.0", backend_source
        )

    def test_fixed_mjcf_compiles_in_reference_mujoco_when_available(self):
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("mujoco is not installed")
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(XML))
        self.assertAlmostEqual(float(model.opt.timestep), 0.002)
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

    def test_preload_calibration_preserves_model_reference_pose(self):
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("mujoco is not installed")
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(XML))
        qpos0 = np.asarray(model.qpos0).copy()
        calibration = _calibrate_host_cdpr(mujoco, model)
        np.testing.assert_array_equal(model.qpos0, qpos0)

        data = mujoco.MjData(model)
        data.qpos[:] = calibration["base_qpos"]
        data.qvel[:] = calibration["base_qvel"]
        data.ctrl[:] = calibration["base_ctrl"]
        mujoco.mj_forward(model, data)
        tendon_ids = np.asarray(calibration["tendon_ids"], dtype=np.int64)
        np.testing.assert_allclose(
            data.ten_length[tendon_ids],
            model.tendon_range[tendon_ids, 1],
            atol=5.0e-5,
        )
        self.assertTrue(np.isfinite(data.qpos).all())

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
        overview_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        np.testing.assert_allclose(
            model.cam_pos[overview_id],
            np.array([0.0, -1.0825, 0.725]),
            atol=1e-7,
        )
        self.assertAlmostEqual(
            float(
                np.linalg.norm(
                    model.cam_pos[overview_id]
                    - np.array([0.0, 0.0, 0.10])
                )
            ),
            1.25,
            places=4,
        )

    def test_four_object_slots_keep_fixed_real_mesh_topology(self):
        assets = validate_object_assets(XML)
        self.assertEqual(len(assets), 50)
        self.assertTrue(all(path.is_file() for path in assets))
        self.assertEqual(len(object_assets_sha256(XML)), 64)
        self.assertEqual(
            {
                catalog: Path(variant.asset_directory).name
                for catalog, variant in OBJECT_VARIANTS.items()
            },
            {
                "robocasa_apple": "apple_20",
                "robocasa_banana": "banana_19",
                "robocasa_carrot": "carrot_1",
                "robocasa_bell_pepper": "bell_pepper_0",
                "robocasa_tomato": "tomato_8",
                "robocasa_orange": "orange_4",
                "robocasa_potato": "potato_3",
                "robocasa_mug": "mug_1",
                "robocasa_plate": "plate_12",
                "robocasa_bowl": "bowl_1",
            },
        )
        self.assertTrue(
            all(
                "visual/image0.png" in variant.asset_files
                for variant in OBJECT_VARIANTS.values()
            )
        )
        total_faces = 0
        for path in assets:
            if path.suffix == ".obj":
                faces = sum(
                    line.startswith("f ")
                    for line in path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                )
                self.assertLessEqual(faces, 6_000)
                total_faces += faces
        self.assertLessEqual(total_faces, 20_000)
        self.assertEqual(catalog_id("apple"), 0)
        self.assertEqual(
            catalog_id("bell_pepper"),
            ACTIVE_CDPR_CATALOGS.index("robocasa_bell_pepper"),
        )
        self.assertEqual(catalog_id("bowl"), len(ACTIVE_CDPR_CATALOGS) - 1)
        self.assertTrue(
            all(name.startswith("robocasa_") for name in ACTIVE_CDPR_CATALOGS)
        )
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("mujoco is not installed")
        import mujoco

        variants = compile_catalog_variant_models(mujoco, XML)
        for catalog, model in variants.items():
            expected_collision_count = len(
                OBJECT_VARIANTS[catalog].primitives
            )
            for slot in range(4):
                dataids = []
                active_colliders = 0
                for mesh_slot in GEOM_SLOT_NAMES:
                    geom_id = mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        slot_geom_name(slot, mesh_slot),
                    )
                    self.assertGreaterEqual(geom_id, 0)
                    dataids.append(int(model.geom_dataid[geom_id]))
                    if mesh_slot in COLLISION_GEOM_SLOT_NAMES:
                        if float(model.geom_size[geom_id, 0]) > 1.0e-3:
                            active_colliders += 1
                        else:
                            self.assertGreater(
                                float(model.geom_pos[geom_id, 2]), 0.0
                            )
                self.assertGreaterEqual(dataids[0], 0)
                self.assertTrue(all(value == -1 for value in dataids[1:]))
                self.assertEqual(active_colliders, expected_collision_count)
                body_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"mjwarp_object_slot_{slot}",
                )
                self.assertGreater(float(model.body_pos[body_id, 2]), 0.0)
                self.assertGreater(
                    float(np.linalg.norm(model.body_pos[body_id, :2])),
                    1.0,
                )

    def test_policy_rgb_excludes_collision_geom_group(self):
        backend_source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "simulation"
            / "mjlab_mjwarp_backend.py"
        ).read_text(encoding="utf-8")
        renderer_source = (
            ROOT / "scripts" / "render_cdpr_mjlab_camera_videos.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "enabled_geom_groups=[0, 1, 2, 4]", backend_source
        )
        self.assertNotIn(
            "enabled_geom_groups=[0, 1, 2, 3]", backend_source
        )
        self.assertIn(
            "self.scene_option.geomgroup[:3] = 1", renderer_source
        )

    def test_inactive_primitives_do_not_penetrate_infinite_floor(self):
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("mujoco is not installed")
        import mujoco

        variants = compile_catalog_variant_models(mujoco, XML)
        xy_positions = (
            (-0.12, -0.08),
            (0.12, -0.08),
            (-0.12, 0.10),
            (0.12, 0.10),
        )
        for catalog, model in variants.items():
            data = mujoco.MjData(model)
            for slot, xy in enumerate(xy_positions):
                joint_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"mjwarp_object_slot_{slot}_free",
                )
                qadr = int(model.jnt_qposadr[joint_id])
                data.qpos[qadr : qadr + 3] = (
                    xy[0],
                    xy[1],
                    0.15 + OBJECT_VARIANTS[catalog].rest_height,
                )
                data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)

            mujoco.mj_forward(model, data)
            contact_distances = np.asarray(
                [data.contact[index].dist for index in range(data.ncon)]
            )
            self.assertTrue(np.isfinite(contact_distances).all())
            if contact_distances.size:
                self.assertGreaterEqual(
                    float(contact_distances.min()), -0.05
                )

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

    def test_physical_grasp_hot_path_never_pins_free_bodies(self):
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
        self.assertIn(
            "finger_object_contact_metrics(target_slots)", collector_source
        )
        self.assertIn("contacts.bilateral_contact", collector_source)
        self.assertIn("left_normal_force", collector_source)
        self.assertIn("relative_position_slip", collector_source)
        self.assertIn("physically_lifted", collector_source)
        self.assertNotIn("centered_close_fallback", collector_source)
        self.assertNotIn("set_target_body_positions(", collector_source)
        self.assertNotIn("configure_pinned_objects(", collector_source)
        self.assertNotIn("_write_configured_pinned_poses", backend_source)
        self.assertNotIn("_pinned_mask", backend_source)
        self.assertIn("self.model.geom_dataid", backend_source)
        self.assertIn("self.mjw.contact_force(", backend_source)

    def test_checkpoint_compatibility_includes_real_object_asset_hash(self):
        source = (
            ROOT
            / "rl_vla_bootstrapping"
            / "policy"
            / "smolvla_grpo_finetune_cdpr.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"object_assets_sha256"', source)
        self.assertIn('"object_geometry"', source)

    def test_gpu_video_probe_has_no_cpu_simulator_fallback(self):
        source = (
            ROOT
            / "scripts"
            / "render_cdpr_mjwarp_physical_grasp_videos.py"
        ).read_text(encoding="utf-8")
        self.assertIn('backend="mjlab_mjwarp"', source)
        self.assertIn("finger_object_contact_metrics", source)
        self.assertIn('"cpu_contact_fallback": False', source)
        self.assertNotIn("mujoco_cpu", source)
        self.assertNotIn("MujocoCPUReferenceBackend", source)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "physical grasp predicate fixture requires PyTorch",
    )
    def test_physical_grasp_requires_persistent_bilateral_force_and_lift(self):
        import torch

        contact = CDPRFingerContactBatch(
            left_contact=torch.tensor([True]),
            right_contact=torch.tensor([True]),
            left_normal_force=torch.tensor([0.2]),
            right_normal_force=torch.tensor([0.2]),
        )
        backend = SimpleNamespace(
            finger_object_contact_metrics=lambda target_slots: contact
        )
        collector = object.__new__(RankLocalMJWarpGRPOCollector)
        collector.torch = torch
        collector.device = torch.device("cpu")
        collector.layout = RankLocalGroupLayout(
            worlds_per_rank=1, groups_per_rank=1, group_size=1
        )
        collector._world_rows = torch.arange(1, dtype=torch.int64)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        low_dim = SimpleNamespace(
            ee_position=torch.tensor([[0.0, 0.0, 0.30]]),
            ee_quaternion=identity.clone(),
            gripper_opening=torch.tensor([0.5]),
            object_positions=torch.tensor([[[0.0, 0.0, 0.25]]]),
            object_quaternions=identity[:, None, :].clone(),
        )
        task_state = SimpleNamespace(
            target_slots=torch.tensor([0]),
            support_surface_z=torch.tensor([0.15]),
            release_threshold=torch.tensor([0.55]),
            ever_grasped=torch.tensor([False]),
        )
        reset = SimpleNamespace(
            task_state=task_state,
            grasp_eligible=torch.tensor([True]),
            bilateral_contact_steps=torch.zeros(1, dtype=torch.int64),
            previous_relative_position=torch.tensor([[0.0, 0.0, -0.05]]),
            previous_relative_quaternion=identity.clone(),
            target_rest_height=torch.tensor([0.04]),
            physical_grasp=torch.tensor([False]),
        )
        collector.backend = backend
        active = torch.tensor([True])

        _, first, _ = collector._update_physical_grasp(
            reset, low_dim, active
        )
        self.assertFalse(bool(first.item()))
        _, second, diagnostics = collector._update_physical_grasp(
            reset, low_dim, active
        )
        self.assertTrue(bool(second.item()))
        self.assertTrue(bool(diagnostics["physically_lifted"].item()))

        task_state.ever_grasped.fill_(True)
        low_dim.gripper_opening.fill_(0.8)
        _, released, diagnostics = collector._update_physical_grasp(
            reset, low_dim, active
        )
        self.assertFalse(bool(released.item()))
        self.assertTrue(bool(diagnostics["physical_release"].item()))

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
        self.assertIn('"max-autotune-no-cudagraphs"', benchmark_source)
        self.assertIn('environment.pop("HF_TOKEN", None)', benchmark_source)
        self.assertIn(
            'environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"',
            benchmark_source,
        )
        self.assertIn("--no-lock-non-commanded-axes", smoke_source)
        self.assertIn("configure_huggingface_public_models", smoke_source)
        self.assertIn("huggingface_public_models_preflight", smoke_source)

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
        self.assertIn('"controller_reaches_xyz_targets"', smoke_source)
        self.assertIn(
            '"training_put_into_bowl_succeeds"', smoke_source
        )
        self.assertIn(
            '"training_put_on_plate_succeeds"', smoke_source
        )


if __name__ == "__main__":
    unittest.main()
