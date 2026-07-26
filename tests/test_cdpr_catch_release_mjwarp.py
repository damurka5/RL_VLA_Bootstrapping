from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "examples"
    / "cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml"
)
PICK_UP_CONFIG = (
    ROOT
    / "configs"
    / "examples"
    / "cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml"
)
MOVE_TO_CONFIG = (
    ROOT
    / "configs"
    / "examples"
    / "cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml"
)
STAGED_CONFIGS = (MOVE_TO_CONFIG, PICK_UP_CONFIG, CONFIG)


class CDPRCatchReleaseConfigTests(unittest.TestCase):
    def test_catch_release_profile_uses_random_workspace_and_warmstart(self):
        config = load_project_config(CONFIG)
        plan = BootstrapPipeline(config).build_stage_plans(
            ROOT / "runs" / "catch_release_unit", ["rl"]
        )[0]
        command = plan.command
        self.assertEqual(
            config.task.instruction_types,
            ("put_into_plate", "put_into_bowl", "pick_up"),
        )
        self.assertTrue(
            config.task.metadata["random_workspace_gripper_start"]
        )
        self.assertTrue(
            config.task.metadata["placement_start_with_caught_object"]
        )
        self.assertEqual(config.simulator.worlds_per_rank, 512)
        self.assertEqual(config.simulator.groups_per_rank, 64)
        self.assertEqual(
            config.task.metadata["ee_workspace_x_bounds"], [-0.28, 0.28]
        )
        # Weights-only handoff: the phase must not pin a resume checkpoint, or
        # it would also restore the previous phase's curriculum caps/optimizer.
        self.assertNotIn("--resume-checkpoint", command)
        self.assertEqual(
            command[command.index("--max-train-steps") + 1], "15000000"
        )
        self.assertEqual(
            command[command.index("--complex-training-approach") + 1],
            "none",
        )
        self.assertEqual(
            config.training.rl.algorithm, "smolvla_residual_grpo_mjwarp"
        )
        self.assertNotIn(
            "reverse_frontier_profile", config.task.metadata
        )

    def test_every_staged_config_keeps_the_move_to_stability_fixes(self):
        """Each phase hands weights to the next through a strict load.

        The residual architecture and the entropy/log-std settings that stopped
        move-to from diffusing into target-independent homing have to be
        identical across all three phases: an architecture mismatch fails the
        load outright, and a hyperparameter regression silently reintroduces a
        failure that already cost a run.
        """

        for path in STAGED_CONFIGS:
            config = load_project_config(path)
            args = config.training.rl.args
            with self.subTest(config=config.project.name):
                self.assertTrue(args["residual_vision_features"])
                self.assertEqual(args["residual_vision_dim"], 512)
                self.assertTrue(args["train_vla_lora"])
                self.assertEqual(args["lora_rank"], 16)
                self.assertTrue(args["lora_include_mlp"])
                self.assertEqual(args["state_dim"], 6)
                self.assertFalse(args["residual_relative_target"])
                self.assertEqual(args["residual_scale"], 1.0)
                self.assertEqual(args["max_log_std"], -0.3)
                self.assertEqual(args["entropy_coef"], 0.0005)
                # 512 sat on the A40 limit and OOM'd Warp once the LoRA
                # backward was added.
                self.assertEqual(
                    args["smolvla_inference_microbatch_size"], 256
                )

    def test_grasp_phases_run_a_success_gated_approach_curriculum(self):
        for path in (PICK_UP_CONFIG, CONFIG):
            metadata = load_project_config(path).task.metadata
            with self.subTest(config=path.name):
                self.assertTrue(
                    metadata[
                        "random_workspace_start_distance_curriculum_enabled"
                    ]
                )
                self.assertEqual(
                    metadata["random_workspace_start_distance_initial"], 0.03
                )
                self.assertTrue(
                    metadata["curriculum_horizon_coupling_enabled"]
                )
                # Longer than move-to's 8: a grasp still has to close the
                # gripper and lift after arriving at the hover point.
                self.assertGreaterEqual(
                    metadata["curriculum_horizon_min"], 16
                )
                # Scenes start distractor-free and unlock more objects later.
                self.assertEqual(metadata["min_scene_objects"], 1)
                self.assertTrue(metadata["scene_object_curriculum_steps"])

    def test_grasp_phases_share_one_dense_shaping_curve(self):
        for path in (PICK_UP_CONFIG, CONFIG):
            metadata = load_project_config(path).task.metadata
            with self.subTest(config=path.name):
                self.assertEqual(
                    metadata["catch_release_distance_reward_scale"], 0.08
                )
                self.assertEqual(
                    metadata["catch_release_distance_reward_weight"], 1.0
                )
                self.assertEqual(metadata["pick_distance_window"], 0.02)
                self.assertEqual(
                    metadata["catch_release_fine_distance_reward_weight"], 0.5
                )
                self.assertEqual(
                    metadata["catch_release_fine_distance_reward_scale"], 0.01
                )
        # Placement shaping must use the same window as pick_up, NOT the plate
        # and bowl success radii, which flatten the term over the whole endgame.
        placement = load_project_config(CONFIG).task.metadata
        self.assertEqual(
            placement["placement_distance_window"],
            placement["pick_distance_window"],
        )
        self.assertNotEqual(
            placement["placement_distance_window"],
            placement["put_plate_xy_tolerance"],
        )

    def test_launcher_warm_starts_weights_only_at_microbatch_256(self):
        for name in (
            "train_cdpr_smolvla_pick_up_grpo_mjlab_dual_remote.sh",
            "train_cdpr_smolvla_catch_release_grpo_mjlab_dual_remote.sh",
        ):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(launcher=name):
                self.assertIn(
                    'WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-}"', source
                )
                self.assertIn(
                    'export RLVLA_SMOLVLA_WARMSTART_CHECKPOINT="$WARMSTART_CHECKPOINT"',
                    source,
                )
                # A resume would restore the previous phase's curriculum caps.
                self.assertIn("unset RLVLA_SMOLVLA_RESUME_CHECKPOINT", source)
                self.assertIn(
                    'SMOLVLA_MICROBATCH_SIZE="${SMOLVLA_MICROBATCH_SIZE:-256}"',
                    source,
                )
                self.assertIn(
                    'WORLDS_PER_RANK="${WORLDS_PER_RANK:-512}"', source
                )

    def test_staged_launcher_chains_three_phases_by_discovered_checkpoint(self):
        source = (
            ROOT
            / "scripts"
            / "train_cdpr_smolvla_move_to_then_catch_release_grpo_mjlab_dual_remote.sh"
        ).read_text(encoding="utf-8")
        phase_1 = source.rindex('bash "$MOVE_TO_LAUNCHER"')
        move_to_discovery = source.index('MOVE_TO_CHECKPOINT="$(latest_checkpoint')
        pick_up_discovery = source.index('PICK_UP_CHECKPOINT="$(latest_checkpoint')
        self.assertLess(phase_1, move_to_discovery)
        self.assertLess(move_to_discovery, pick_up_discovery)
        self.assertIn(
            "curriculum=per_instruction_success_gated lchol=disabled "
            "handoff=weights_only",
            source,
        )
        for config_var in (
            "$MOVE_TO_CONFIG",
            "$PICK_UP_CONFIG",
            "$CATCH_RELEASE_CONFIG",
        ):
            self.assertIn(
                f'assert_plain_grpo_config "{config_var}"', source
            )
        # The handoff guard: a config missing the vision residual would fail the
        # strict weight load only after the phase had already started.
        self.assertIn("residual_vision_features:[[:space:]]+true", source)

    def test_all_staged_configs_are_plain_grpo_without_shells_or_lchol(self):
        for path in STAGED_CONFIGS:
            config = load_project_config(path)
            with self.subTest(config=config.project.name):
                self.assertEqual(
                    config.training.rl.algorithm,
                    "smolvla_residual_grpo_mjwarp",
                )
                self.assertEqual(
                    config.training.rl.args["complex_training_approach"],
                    "none",
                )
                self.assertNotIn(
                    "reverse_frontier_profile", config.task.metadata
                )
                self.assertNotIn(
                    "resume_checkpoint", config.training.rl.args
                )

    def test_pick_up_phase_trains_only_graspable_objects(self):
        config = load_project_config(PICK_UP_CONFIG)
        self.assertEqual(config.task.instruction_types, ("pick_up",))
        # Plate and bowl have fitted_gripper_opening 0.0 and cannot be lifted,
        # so they must not appear even as distractors in the pick-up phase.
        for key in (
            "target_object_pool",
            "scene_object_pool",
            "distractor_object_pool",
        ):
            pool = config.task.metadata[key]
            with self.subTest(pool=key):
                self.assertNotIn("robocasa_plate", pool)
                self.assertNotIn("robocasa_bowl", pool)
        self.assertFalse(
            config.task.metadata["placement_start_with_caught_object"]
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required for tensorized catch/release rewards",
)
class CDPRCatchReleaseRewardTests(unittest.TestCase):
    @staticmethod
    def _fake_backend(torch):
        class FakeBackend:
            def __init__(self) -> None:
                self.torch = torch
                self.device = torch.device("cpu")
                self.object_body_ids = torch.arange(4, dtype=torch.int64)
                self.object_positions = None
                self.object_quaternions = None
                self.ee_positions = None
                self.ee_yaw = None

            def reset_worlds(self, _worlds):
                return None

            def set_object_catalogs(self, _catalogs):
                return None

            def set_free_body_poses(
                self, _body_ids, positions, quaternions
            ):
                self.object_positions = positions.clone()
                self.object_quaternions = quaternions.clone()

            def set_end_effector_poses(self, positions, yaw):
                self.ee_positions = positions.clone()
                self.ee_yaw = yaw.clone()

            def set_gripper_openings(self, _openings):
                return None

            def set_visual_variants(self, *_args):
                return None

            def broadcast_group_state(self, _base_worlds):
                return None

            def low_dim_observations(self):
                ee_quaternion = torch.zeros(
                    (self.ee_positions.shape[0], 4), dtype=torch.float32
                )
                ee_quaternion[:, 0] = torch.cos(0.5 * self.ee_yaw)
                ee_quaternion[:, 3] = torch.sin(0.5 * self.ee_yaw)
                return SimpleNamespace(
                    ee_position=self.ee_positions,
                    ee_quaternion=ee_quaternion,
                    object_positions=self.object_positions,
                    object_quaternions=self.object_quaternions,
                )

        return FakeBackend()

    def test_random_workspace_resets_keep_placement_caught_and_pick_on_desk(
        self,
    ):
        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
            RankLocalCurriculum,
        )
        from rl_vla_bootstrapping.policy.rank_local_grpo import (
            RankLocalGroupLayout,
        )

        layout = RankLocalGroupLayout(
            worlds_per_rank=8, groups_per_rank=1, group_size=8
        )
        metadata = {
            "random_workspace_gripper_start": True,
            "placement_start_with_caught_object": True,
            "random_workspace_min_goal_xy_distance": 0.10,
            "random_workspace_horizon_low": 21,
            "random_workspace_horizon_high": 32,
            "ee_workspace_x_bounds": [-0.28, 0.28],
            "ee_workspace_y_bounds": [-0.28, 0.28],
            "ee_workspace_z_bounds": [0.30, 0.48],
        }
        objects = (
            "robocasa_apple",
            "robocasa_banana",
            "robocasa_plate",
            "robocasa_bowl",
        )
        for instruction in (
            "put_into_plate",
            "put_into_bowl",
            "pick_up",
        ):
            with self.subTest(instruction=instruction):
                backend = self._fake_backend(torch)
                resetter = BatchedReverseFrontierResetter(
                    backend=backend,
                    layout=layout,
                    curriculum=RankLocalCurriculum(device=backend.device),
                    rank=0,
                    base_seed=17,
                    instruction_types=(instruction,),
                    allowed_objects=objects,
                    task_metadata=metadata,
                )
                reset = resetter.reset(update_index=0, round_index=0)
                ee = backend.ee_positions
                target = backend.object_positions[:, 0]
                if instruction == "pick_up":
                    self.assertFalse(
                        bool(reset.task_state.ever_grasped.any().item())
                    )
                    self.assertTrue(
                        torch.allclose(
                            target[:, 2],
                            reset.task_state.support_surface_z
                            + reset.task_state.target_rest_height,
                        )
                    )
                    self.assertTrue(
                        reset.instructions[0].startswith("pick up ")
                    )
                    goal = target
                else:
                    self.assertTrue(
                        bool(reset.task_state.ever_grasped.all().item())
                    )
                    self.assertTrue(
                        torch.allclose(
                            target[:, 2], ee[:, 2] - 0.08, atol=1.0e-6
                        )
                    )
                    reference = backend.object_positions[:, 1]
                    goal = reference
                xy_distance = torch.linalg.vector_norm(
                    ee[:, :2] - goal[:, :2], dim=-1
                )
                self.assertTrue(bool((xy_distance >= 0.10).all().item()))
                self.assertTrue(bool((reset.horizons >= 21).all().item()))

    def test_release_radius_wrong_drop_and_grasp_lift_rewards(self):
        import torch

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        objects = torch.zeros((4, 4, 3), dtype=torch.float32)
        # Correct plate release.
        objects[0, 0] = torch.tensor([0.08, 0.00, 0.18])
        objects[0, 1] = torch.tensor([0.00, 0.00, 0.16])
        # Wrong bowl drop, outside the 0.057 m radius and settled on support.
        objects[1, 0] = torch.tensor([0.10, 0.00, 0.18])
        objects[1, 1] = torch.tensor([0.00, 0.00, 0.176])
        # Grasped but not lifted.
        objects[2, 0] = torch.tensor([0.00, 0.00, 0.18])
        # Grasped and lifted 0.06 m.
        objects[3, 0] = torch.tensor([0.00, 0.00, 0.24])

        initial = objects[:, 0].clone()
        initial[0] = torch.tensor([0.20, 0.00, 0.32])
        initial[1] = torch.tensor([0.20, 0.00, 0.32])
        initial[3, 2] = 0.18
        instruction_ids = torch.tensor(
            [
                INSTRUCTION_TO_ID["put_into_plate"],
                INSTRUCTION_TO_ID["put_into_bowl"],
                INSTRUCTION_TO_ID["pick_up"],
                INSTRUCTION_TO_ID["pick_up"],
            ],
            dtype=torch.int64,
        )
        state = BatchedTaskState(
            instruction_ids=instruction_ids,
            target_slots=torch.zeros((4,), dtype=torch.int64),
            reference_slots=torch.tensor([1, 1, -1, -1]),
            second_reference_slots=torch.full((4,), -1, dtype=torch.int64),
            initial_target_positions=initial,
            ever_grasped=torch.tensor([True, True, False, False]),
            grasped=torch.tensor([False, False, False, False]),
            step_count=torch.zeros((4,), dtype=torch.int64),
            release_threshold=torch.full((4,), 0.55),
            support_surface_z=torch.full((4,), 0.15),
            target_rest_height=torch.full((4,), 0.03),
        )
        ee = torch.tensor(
            [
                [0.08, 0.00, 0.26],
                [0.10, 0.00, 0.26],
                [0.00, 0.00, 0.26],
                [0.00, 0.00, 0.32],
            ],
            dtype=torch.float32,
        )
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.tensor([0.90, 0.90, 0.50, 0.50]),
            caught_target=torch.tensor([False, False, True, True]),
            active_mask=torch.ones((4,), dtype=torch.bool),
            max_steps=128,
            catch_release_dense_reward=BatchedCatchReleaseDenseReward(),
        )

        self.assertEqual(
            result.success.tolist(), [True, False, False, True]
        )
        self.assertEqual(
            result.terminated.tolist(), [True, True, False, True]
        )
        self.assertTrue(
            bool(result.diagnostics["wrong_place_drop"][1].item())
        )
        # World 0 sits 0.08 m from the plate centre. Success still uses the real
        # 0.091 m plate radius, but the SHAPING window is 0.02 m, so the distance
        # term is 1/(1+((0.08-0.02)/0.08)^2) = 0.64 rather than a flat 1.0.
        # Decoupling the two is the whole point: with the window tied to the
        # success radius every candidate inside the plate scored identically.
        self.assertAlmostEqual(float(result.rewards[0].item()), 2.64, places=5)
        self.assertAlmostEqual(float(result.rewards[2].item()), 2.0, places=5)
        self.assertAlmostEqual(float(result.rewards[3].item()), 5.0, places=5)

    def test_placement_shaping_is_not_flat_across_the_receptacle(self):
        """Placement and pick-up must share one non-degenerate shaping curve.

        The regression this guards: taking the shaping window from the success
        radius (plate 0.091 m) made the reward >= 0.988 anywhere inside 10 cm.
        Eight GRPO candidates then scored within ~0.003 of each other, and the
        group normalization divided that by a near-zero std -- amplifying pure
        rollout noise into full-magnitude advantages.
        """

        import torch

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedCatchReleaseDenseReward,
            _fine_distance_reward,
            inverse_polynomial_distance_reward,
        )

        config = BatchedCatchReleaseDenseReward.from_metadata(
            {
                "catch_release_distance_reward_scale": 0.08,
                "catch_release_distance_reward_weight": 1.0,
                "placement_distance_window": 0.02,
                "pick_distance_window": 0.02,
                "catch_release_fine_distance_reward_weight": 0.5,
                "catch_release_fine_distance_reward_scale": 0.01,
            }
        )
        self.assertNotEqual(
            config.placement_distance_window, config.plate_radius
        )

        def shaping(distance, window):
            values = torch.as_tensor(distance, dtype=torch.float32)
            return inverse_polynomial_distance_reward(
                values,
                window_high=torch.full_like(values, float(window)),
                scale=config.distance_reward_scale,
                weight=config.distance_reward_weight,
                exponent=config.distance_reward_exponent,
            ) + _fine_distance_reward(values, config=config)

        distances = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
        placement = shaping(distances, config.placement_distance_window)
        pick = shaping(distances, config.pick_distance_window)
        # Identical curve for placement and pick-up: one reward scale across
        # every instruction the phase trains.
        self.assertTrue(torch.allclose(placement, pick))
        # Strictly decreasing, so every centimetre of progress is scored --
        # including the last one, where the coarse term alone is flat.
        for nearer, farther in zip(placement[:-1], placement[1:]):
            self.assertGreater(float(nearer.item()), float(farther.item()))
        # And the spread across a realistic group is large enough that the
        # group-normalized advantage tracks real progress, not noise.
        self.assertGreater(float(placement.std().item()), 0.1)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required for the per-instruction approach curriculum",
)
class CDPRPerInstructionCurriculumTests(unittest.TestCase):
    """An easy instruction must not widen a hard one's start distance."""

    METADATA = {
        "random_workspace_start_distance_curriculum_enabled": True,
        "random_workspace_start_distance_initial": 0.03,
        "random_workspace_start_distance_final": 0.34,
        "random_workspace_start_distance_increment": 0.02,
        "random_workspace_start_distance_promote_pass_rate": 0.03,
        "random_workspace_start_distance_demote_pass_rate": 0.01,
        "random_workspace_start_distance_cooldown_updates": 1,
        "random_workspace_start_distance_pass_rate_ema_decay": 0.0,
    }

    # A gate with a realistic band, and an EMA that actually averages, for the
    # promote/hold/demote behaviour. METADATA above uses decay 0.0 so its own
    # tests can step the cap in a fixed number of observations.
    GATED_METADATA = {
        **METADATA,
        "random_workspace_start_distance_promote_pass_rate": 0.30,
        "random_workspace_start_distance_demote_pass_rate": 0.12,
        "random_workspace_start_distance_cooldown_updates": 1,
        "random_workspace_start_distance_pass_rate_ema_decay": 0.9,
    }

    def _curriculum(self, names, metadata=None):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            PerInstructionApproachCurriculum,
        )

        return PerInstructionApproachCurriculum(
            metadata if metadata is not None else self.METADATA,
            instruction_types=names,
        )

    def _gated(self):
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
        )

        curriculum = self._curriculum(
            ("move_to_object",), metadata=self.GATED_METADATA
        )

        def cap():
            return curriculum.caps_by_instruction_id()[
                INSTRUCTION_TO_ID["move_to_object"]
            ]

        def feed(rate, updates):
            for _ in range(updates):
                curriculum.observe({"move_to_object": rate})

        return curriculum, cap, feed

    def test_a_pass_rate_inside_the_band_holds_the_cap(self):
        """The gate must be able to say no.

        Regression: promote was 0.03 while the realized pass rate never fell
        below ~0.06, so the gate was open on every update and the cap advanced
        on its cooldown alone -- 0.03 -> 0.19 m in 350k steps regardless of what
        the policy was doing.
        """

        _, cap, feed = self._gated()
        feed(0.20, 50)
        self.assertAlmostEqual(cap(), 0.03, places=6)

    def test_the_cap_retreats_when_the_policy_stops_keeping_up(self):
        _, cap, feed = self._gated()
        feed(0.90, 10)
        promoted = cap()
        self.assertAlmostEqual(promoted, 0.11, places=6)
        feed(0.02, 20)
        self.assertLess(cap(), promoted)
        # The initial cap is the floor; demotion never goes below it.
        self.assertAlmostEqual(cap(), 0.03, places=6)

    def test_a_new_cap_is_judged_on_its_own_pass_rate(self):
        """The EMA is re-seeded on a cap change, not carried across it.

        Carried over, the average right after a promotion is still dominated by
        the easier level's higher rate, so the next decision repeats the last
        one and the cap ratchets away from the policy under its own momentum.
        """

        _, cap, feed = self._gated()
        feed(0.90, 4)
        self.assertAlmostEqual(cap(), 0.05, places=6)
        # Two updates at a rate under the demote floor. Re-seeded, the average
        # is 0.05 and the cap steps straight back; carrying the 0.31 it held at
        # the previous cap would leave it parked at 0.05.
        feed(0.05, 2)
        self.assertAlmostEqual(cap(), 0.03, places=6)

    def test_an_object_unlock_restarts_the_cap(self):
        """A distractor unlock must hand the cap back to the initial value.

        The cap was earned on single-object scenes, where the instruction is
        irrelevant -- there is nothing to disambiguate, so the policy only ever
        learned object-agnostic servoing. Carrying that cap across the unlock
        asks for target selection AND the far starts in the same instant, which
        is how the 5.7M-step attempt lost its grounding (cosine 0.20 -> 0.05).
        """

        curriculum, cap, feed = self._gated()
        feed(0.90, 10)
        self.assertAlmostEqual(cap(), 0.11, places=6)

        self.assertEqual(curriculum.restart(), ("move_to_object",))
        self.assertAlmostEqual(cap(), 0.03, places=6)

        # The restart also clears the pass-rate history, so the next promotion
        # has to be earned on the new scene rather than inherited from the old
        # level's average.
        metrics = curriculum.metrics()
        self.assertEqual(
            metrics["curriculum/approach_pass_rate_ema/move_to_object"], 0.0
        )
        feed(0.90, 1)
        self.assertAlmostEqual(cap(), 0.03, places=6)

    def test_restarting_an_unpromoted_curriculum_reports_no_change(self):
        """So the unlock does not log a restart that did not happen."""

        curriculum, cap, _ = self._gated()
        self.assertAlmostEqual(cap(), 0.03, places=6)
        self.assertEqual(curriculum.restart(), ())

    def test_object_unlock_is_detected_by_step_not_by_edge(self):
        """Resuming past a threshold must not read as a fresh unlock.

        The training loop compares the current object count against the one it
        started at, so a run that resumes at 9M -- already past the 8M unlock --
        keeps the cap it earned instead of restarting on its first update.
        """

        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            _apply_scene_object_curriculum,
        )

        class StubResetter:
            scene_object_bounds = (1, 2)

            def __init__(self) -> None:
                self.scene_object_range = (1, 1)

            def set_scene_object_range(self, low, high):
                low = min(2, max(1, int(low)))
                self.scene_object_range = (low, min(2, max(low, int(high))))

        steps = (8_000_000,)

        def count_at(step):
            resetter = StubResetter()
            return _apply_scene_object_curriculum(
                resetter, curriculum_steps=steps, global_step=step
            )[1]

        self.assertEqual(count_at(0), 1)
        self.assertEqual(count_at(7_999_999), 1)
        self.assertEqual(count_at(8_000_000), 2)
        # A resume at 9M seeds previous_scene_object_max at 2, so the loop's
        # `range[1] > previous` check is false and the cap survives.
        self.assertEqual(count_at(9_000_000), 2)

    def test_a_passing_instruction_does_not_promote_a_failing_one(self):
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
        )

        curriculum = self._curriculum(
            ("put_into_plate", "put_into_bowl", "pick_up")
        )
        for _ in range(6):
            curriculum.observe(
                {
                    "put_into_plate": 0.50,
                    "put_into_bowl": 0.50,
                    "pick_up": 0.0,
                }
            )
        caps = curriculum.caps_by_instruction_id()
        plate_cap = caps[INSTRUCTION_TO_ID["put_into_plate"]]
        pick_cap = caps[INSTRUCTION_TO_ID["pick_up"]]
        self.assertGreater(plate_cap, 0.03)
        # pick_up never passed, so it keeps the close starts it still needs.
        self.assertAlmostEqual(pick_cap, 0.03, places=6)

    def test_an_instruction_absent_this_update_is_not_scored_as_zero(self):
        """Instruction sampling is random per group, so a task can be missing.

        Feeding it a zero would ratchet its EMA down and eventually demote a cap
        the policy had genuinely earned.
        """

        curriculum = self._curriculum(("put_into_plate", "pick_up"))
        for _ in range(6):
            curriculum.observe({"pick_up": 0.50})
        before = dict(curriculum.caps_by_instruction_id())
        for _ in range(6):
            curriculum.observe({"put_into_plate": 0.50})
        after = curriculum.caps_by_instruction_id()
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
        )

        pick_id = INSTRUCTION_TO_ID["pick_up"]
        self.assertEqual(after[pick_id], before[pick_id])

    def test_state_round_trips_and_accepts_legacy_single_curriculum(self):
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
        )

        curriculum = self._curriculum(("put_into_plate", "pick_up"))
        for _ in range(4):
            curriculum.observe({"put_into_plate": 0.5, "pick_up": 0.0})
        restored = self._curriculum(("put_into_plate", "pick_up"))
        restored.load_state_dict(curriculum.state_dict())
        self.assertEqual(
            restored.caps_by_instruction_id(),
            curriculum.caps_by_instruction_id(),
        )
        # A checkpoint written before the split stored one flat entry; replay it
        # into every instruction instead of silently resetting to the initial.
        legacy = self._curriculum(("put_into_plate", "pick_up"))
        legacy.load_state_dict(
            {"cap": 0.11, "pass_rate_ema": 0.2, "cooldown": 0}
        )
        caps = legacy.caps_by_instruction_id()
        self.assertAlmostEqual(
            caps[INSTRUCTION_TO_ID["pick_up"]], 0.11, places=6
        )

    def test_resetter_applies_each_instruction_its_own_start_cap(self):
        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
            RankLocalCurriculum,
        )
        from rl_vla_bootstrapping.policy.rank_local_grpo import (
            RankLocalGroupLayout,
        )
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
        )

        layout = RankLocalGroupLayout(
            worlds_per_rank=64, groups_per_rank=8, group_size=8
        )
        metadata = {
            "random_workspace_gripper_start": True,
            "placement_start_with_caught_object": True,
            "random_workspace_min_goal_xy_distance": 0.10,
            "ee_workspace_x_bounds": [-0.28, 0.28],
            "ee_workspace_y_bounds": [-0.28, 0.28],
            "ee_workspace_z_bounds": [0.29, 0.40],
            "curriculum_horizon_coupling_enabled": True,
            "curriculum_horizon_min": 16,
            "curriculum_horizon_max": 32,
            "random_workspace_start_distance_initial": 0.03,
            "random_workspace_start_distance_final": 0.34,
        }
        objects = (
            "robocasa_apple",
            "robocasa_banana",
            "robocasa_plate",
            "robocasa_bowl",
        )
        # pick_up stays pinned at the 3 cm foothold while placement runs wide.
        caps = {
            INSTRUCTION_TO_ID["pick_up"]: 0.03,
            INSTRUCTION_TO_ID["put_into_plate"]: 0.30,
        }
        observed = {"pick_up": [], "put_into_plate": []}
        for update_index in range(12):
            backend = CDPRCatchReleaseRewardTests._fake_backend(torch)
            resetter = BatchedReverseFrontierResetter(
                backend=backend,
                layout=layout,
                curriculum=RankLocalCurriculum(device=backend.device),
                rank=0,
                base_seed=7,
                instruction_types=("pick_up", "put_into_plate"),
                allowed_objects=objects,
                task_metadata=metadata,
            )
            resetter.set_random_start_max_goal_distance(caps)
            reset = resetter.reset(update_index=update_index, round_index=0)
            ee = backend.ee_positions
            instruction_ids = reset.task_state.instruction_ids
            for name in observed:
                mask = instruction_ids == INSTRUCTION_TO_ID[name]
                if not bool(mask.any().item()):
                    continue
                # pick_up shapes toward the target, placement toward the
                # receptacle in slot 1.
                slot = 0 if name == "pick_up" else 1
                goal = backend.object_positions[:, slot]
                distance = torch.linalg.vector_norm(
                    ee[:, :2] - goal[:, :2], dim=-1
                )
                observed[name].extend(distance[mask].tolist())

        self.assertTrue(observed["pick_up"])
        self.assertTrue(observed["put_into_plate"])
        self.assertLessEqual(max(observed["pick_up"]), 0.03 + 1.0e-4)
        # Placement is genuinely sampling farther starts, i.e. the caps are
        # applied per instruction rather than collapsing to a single value.
        self.assertGreater(max(observed["put_into_plate"]), 0.05)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required for the approach-curriculum start geometry",
)
class CDPRApproachCurriculumGeometryTests(unittest.TestCase):
    """The cap has to bound the distance the reward actually measures."""

    OBJECTS = (
        "robocasa_apple",
        "robocasa_banana",
        "robocasa_tomato",
        "robocasa_orange",
    )

    def _move_to_starts(self, *, cap, include_z, scene_objects):
        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
            RankLocalCurriculum,
        )
        from rl_vla_bootstrapping.policy.rank_local_grpo import (
            RankLocalGroupLayout,
        )

        layout = RankLocalGroupLayout(
            worlds_per_rank=64, groups_per_rank=8, group_size=8
        )
        metadata = {
            "random_workspace_gripper_start": True,
            "ee_workspace_x_bounds": [-0.24, 0.24],
            "ee_workspace_y_bounds": [-0.24, 0.24],
            "ee_workspace_z_bounds": [0.27, 0.52],
            "random_workspace_min_goal_xy_distance": 0.12,
            "min_scene_objects": scene_objects,
            "max_scene_objects": scene_objects,
            "move_to_object_approach_z": 0.27,
            "curriculum_cap_includes_z": include_z,
        }
        planar: list[float] = []
        spatial: list[float] = []
        for update_index in range(12):
            backend = CDPRCatchReleaseRewardTests._fake_backend(torch)
            resetter = BatchedReverseFrontierResetter(
                backend=backend,
                layout=layout,
                curriculum=RankLocalCurriculum(device=backend.device),
                rank=0,
                base_seed=5,
                instruction_types=("move_to_object",),
                allowed_objects=self.OBJECTS,
                task_metadata=metadata,
            )
            resetter.set_random_start_max_goal_distance(cap)
            reset = resetter.reset(update_index=update_index, round_index=0)
            ee = backend.ee_positions
            target_slots = reset.task_state.target_slots
            rows = torch.arange(ee.shape[0], dtype=torch.int64)
            target = backend.object_positions[rows, target_slots]
            hover = target.clone()
            hover[:, 2] = 0.27
            planar.extend(
                torch.linalg.vector_norm(
                    ee[:, :2] - target[:, :2], dim=-1
                ).tolist()
            )
            spatial.extend(
                torch.linalg.vector_norm(ee - hover, dim=-1).tolist()
            )
        return planar, spatial

    def test_cap_measures_the_named_slot_not_slot_zero(self):
        """Move-to names a RANDOM active slot, so the cap must follow it.

        The named catalog is swapped into target_slot_group, but the curriculum
        used to measure against slot 0. With more than one object in the scene
        that pulled the start close to the wrong object entirely.
        """

        planar, _ = self._move_to_starts(
            cap=0.03, include_z=False, scene_objects=3
        )
        self.assertLessEqual(max(planar), 0.03 + 1.0e-3)

    def test_three_dimensional_cap_bounds_the_reward_distance(self):
        _, without = self._move_to_starts(
            cap=0.03, include_z=False, scene_objects=3
        )
        _, with_z = self._move_to_starts(
            cap=0.03, include_z=True, scene_objects=3
        )
        # XY-only: the Z spread alone puts most starts far outside the cap.
        self.assertGreater(max(without), 0.10)
        self.assertLessEqual(max(with_z), 0.03 + 1.0e-3)

    def test_wide_caps_keep_the_height_randomization(self):
        """Descent must still be learned once the cap is wide.

        The point of the fix is the early foothold, not removing the Z spread:
        at full reach the start distribution should be essentially unchanged.
        """

        _, with_z = self._move_to_starts(
            cap=0.34, include_z=True, scene_objects=3
        )
        spread = max(with_z) - min(with_z)
        self.assertGreater(spread, 0.15)

    def test_staged_configs_bound_the_cap_in_three_dimensions(self):
        for path in STAGED_CONFIGS:
            metadata = load_project_config(path).task.metadata
            with self.subTest(config=path.name):
                self.assertTrue(metadata["curriculum_cap_includes_z"])


if __name__ == "__main__":
    unittest.main()
