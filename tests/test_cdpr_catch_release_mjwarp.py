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

    def test_every_staged_config_actually_parses(self):
        """Every value a staged config emits must be one the parser accepts.

        The configs are rendered into a torchrun command and handed to
        smolvla_grpo_mjwarp_cdpr, so an unknown flag or a value outside an
        argparse `choices` list is not caught anywhere until the launcher exits
        2 -- after both ranks have spun up on a booked box. That is how
        `--residual-vision-pooling dual_random` shipped: the runtime's own
        validation was updated and the parser's choices list was not, and no
        test compared the two.

        Parsing the generated command is the general guard, not a check for
        that one flag: it validates every argument of every phase against the
        parser that will actually receive them.
        """

        from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            parse_args,
        )

        for path in STAGED_CONFIGS:
            config = load_project_config(path)
            with self.subTest(config=config.project.name):
                plan = BootstrapPipeline(config).build_stage_plans(
                    ROOT / "runs" / "parse_unit", ["rl"]
                )[0]
                command = list(plan.command)
                # Drop the launcher prefix: everything up to and including the
                # `-m <module>` pair is torchrun's, not the trainer's.
                module_index = command.index("-m")
                argv = command[module_index + 2 :]
                self.assertTrue(argv, "no trainer arguments were generated")
                try:
                    parse_args(argv)
                except SystemExit as error:  # argparse exits on a bad value
                    self.fail(
                        f"{config.project.name} generates arguments the "
                        f"trainer's parser rejects (exit {error.code}); run "
                        "the command by hand to see which."
                    )

    def test_consecutive_phases_can_load_each_others_residual(self):
        """The handoff contract, now that the width is allowed to change.

        Each phase warm-starts from the previous one's adapter through a strict
        state-dict load, so the residual's input width has to either MATCH or be
        reachable by the one documented expansion: adding a second vision
        pooling doubles the vision block, and the loader zero-fills the appended
        columns (RLVLA_SMOLVLA_EXPAND_VISION_COLUMNS). Anything else is a load
        that fails on the first step of a booked run.

        Pinning every phase to one number, as this used to, cannot express that
        -- and pinning nothing would let phase 2 silently stop being loadable
        from phase 1, which is exactly what changing pick_up alone did.
        """

        widths = []
        for path in STAGED_CONFIGS:
            config = load_project_config(path)
            args = config.training.rl.args
            widths.append(
                (
                    config.project.name,
                    int(args["residual_vision_dim"]),
                    str(args.get("residual_vision_pooling", "flat_random")),
                )
            )
        for (earlier_name, earlier, _), (later_name, later, later_pool) in zip(
            widths, widths[1:]
        ):
            with self.subTest(handoff=f"{earlier_name} -> {later_name}"):
                if later == earlier:
                    continue
                self.assertEqual(
                    later_pool,
                    "dual_random",
                    msg=(
                        f"{later_name} widens the residual to {later} without "
                        "dual_random; there is no loader path for that."
                    ),
                )
                self.assertEqual(
                    later,
                    2 * earlier,
                    msg=(
                        f"{later_name} takes {later} vision inputs against "
                        f"{earlier_name}'s {earlier}; the expansion appends "
                        "exactly one more pooling of the same width."
                    ),
                )

    def test_every_staged_config_keeps_the_move_to_stability_fixes(self):
        """Each phase hands weights to the next through a strict load.

        The residual architecture has to be identical across all three phases or
        the load fails outright. The stability settings are held too, but as
        bounds rather than equalities: max_log_std is a hard ceiling every phase
        must respect, while entropy_coef is phase-dependent because the two
        failure modes are opposite -- move-to diffuses without a ceiling, the
        grasp phases collapse without a floor. Both have already cost a run.
        """

        for path in STAGED_CONFIGS:
            config = load_project_config(path)
            args = config.training.rl.args
            with self.subTest(config=config.project.name):
                self.assertTrue(args["residual_vision_features"])
                # Width is checked across phases below, not pinned here: the
                # residual gained a second vision pooling and the two grasp
                # phases carry it while move_to still predates it.
                self.assertIn(args["residual_vision_dim"], (512, 1024))
                pooling = args.get("residual_vision_pooling", "flat_random")
                if pooling == "dual_random":
                    self.assertEqual(args["residual_vision_dim"] % 2, 0)
                self.assertTrue(args["train_vla_lora"])
                self.assertEqual(args["lora_rank"], 16)
                self.assertTrue(args["lora_include_mlp"])
                self.assertEqual(args["state_dim"], 6)
                self.assertFalse(args["residual_relative_target"])
                self.assertEqual(args["residual_scale"], 1.0)
                # max_log_std is the hard anti-diffusion guard and must hold in
                # every phase: inside the band the policy was productive in, not
                # merely below the old -0.3 ceiling that never bound.
                self.assertLessEqual(args["max_log_std"], -1.10)
                # entropy_coef is a BAND, not a fixed value, because the two
                # failure modes are opposite and phase-dependent. 0.002 diffused
                # move-to; 0.0 collapsed pick_up (entropy_mean fell monotonically
                # -0.404 -> -0.754 while pass rate, grasp rate and reward all
                # regressed from a 2.0M peak). move-to runs 0.0 because its
                # log_std drifts up on its own; the grasp phases run a small
                # positive floor. Anything at or above 0.0005 is the value that
                # was measured to keep diffusing.
                self.assertGreaterEqual(args["entropy_coef"], 0.0)
                self.assertLess(args["entropy_coef"], 0.0005)
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
                self.assertEqual(metadata["min_scene_objects"], 1)

    def test_every_approach_gate_can_actually_say_no(self):
        """No config may carry the degenerate promote/demote pair.

        0.03/0.01 was fitted while a resetter bug pinned the measured pass rate
        under 0.045. Once the cap reached the simulator the real range was
        0.06-0.41, so promote was true on every update, demote was unreachable,
        and the cap advanced on its cooldown alone -- 0.03 -> 0.19 m in 350k
        steps. The gate has to sit inside the range the metric actually takes,
        and the cooldown has to be long enough for the EMA to reflect the level
        it is judging.
        """

        for path in STAGED_CONFIGS:
            metadata = load_project_config(path).task.metadata
            if not metadata.get(
                "random_workspace_start_distance_curriculum_enabled"
            ):
                continue
            promote = metadata[
                "random_workspace_start_distance_promote_pass_rate"
            ]
            demote = metadata[
                "random_workspace_start_distance_demote_pass_rate"
            ]
            with self.subTest(config=path.name):
                # A pass rate this low is the noise floor, not mastery.
                self.assertGreaterEqual(promote, 0.15)
                self.assertGreater(promote, demote)
                # Demote must be reachable: a floor near zero means a level the
                # policy cannot do is never given back.
                self.assertGreaterEqual(demote, 0.05)
                self.assertGreaterEqual(
                    metadata[
                        "random_workspace_start_distance_cooldown_updates"
                    ],
                    10,
                )

    def test_pick_up_holds_a_hard_horizon_ceiling(self):
        """No pick_up episode may exceed 26 policy decisions, at any cap.

        Each decision runs a batched SmolVLA forward over every world, so the
        horizon is the dominant per-update cost. Two paths can produce an
        episode: the curriculum-coupled horizon, which interpolates up to
        curriculum_horizon_max, and the uncapped fallback sampled from
        [random_workspace_horizon_low, high] -- which is what validation gets,
        because validation never receives a cap. Both have to respect the
        ceiling or the run measures the policy over a longer episode than it
        trains on.
        """

        metadata = load_project_config(PICK_UP_CONFIG).task.metadata
        ceiling = 26
        self.assertLessEqual(metadata["curriculum_horizon_max"], ceiling)
        self.assertLessEqual(metadata["random_workspace_horizon_high"], ceiling)
        self.assertLessEqual(metadata["random_workspace_horizon_low"], ceiling)
        self.assertLessEqual(
            metadata["curriculum_horizon_min"],
            metadata["curriculum_horizon_max"],
        )

    def test_pick_up_spawns_inside_the_inherited_approach_range(self):
        """The final cap must not exceed what move-to actually reached.

        move-to plateaued with its cap at 0.23 m and a pass rate of 0.21 after
        9M steps, below its own promote gate. pick_up inherits exactly that
        approach ability, so a final cap beyond it asks the policy to extend its
        approach range and learn a grasp at the same time. The value also
        anchors the horizon interpolation, whose ceiling is reached at the final
        cap.
        """

        metadata = load_project_config(PICK_UP_CONFIG).task.metadata
        self.assertLessEqual(
            metadata["random_workspace_start_distance_final"], 0.23
        )
        self.assertGreater(
            metadata["random_workspace_start_distance_final"],
            metadata["random_workspace_start_distance_initial"],
        )

    def test_object_unlocks_are_one_per_run(self):
        """Each unlock restarts the start-distance curriculum (eeffbc0).

        Two thresholds in one run means two full re-climbs, and neither stage
        gets the steps to reach the final cap.
        """

        for path in STAGED_CONFIGS:
            metadata = load_project_config(path).task.metadata
            steps = metadata.get("scene_object_curriculum_steps") or []
            with self.subTest(config=path.name):
                self.assertLessEqual(len(steps), 1)
                span = (
                    metadata["max_scene_objects"]
                    - metadata["min_scene_objects"]
                )
                self.assertLessEqual(span, 1)

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
                self.gripper_openings = None

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

            def set_gripper_openings(self, openings):
                self.gripper_openings = openings.clone()

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
                    # A held object sits exactly one finger-pad offset below the
                    # end-effector. Asserted against the offset the resetter
                    # actually resolved, not a literal: this used to hard-code
                    # 0.08, which is the ee_platform offset, and so pinned a
                    # reset that spawned the "held" object 7.25 cm below the pads
                    # holding it -- in free space, falling on env step 1.
                    self.assertTrue(
                        torch.allclose(
                            target[:, 2],
                            ee[:, 2] - float(resetter.pick_grasp_height_offset),
                            atol=1.0e-6,
                        )
                    )
                    reference = backend.object_positions[:, 1]
                    goal = reference
                xy_distance = torch.linalg.vector_norm(
                    ee[:, :2] - goal[:, :2], dim=-1
                )
                self.assertTrue(bool((xy_distance >= 0.10).all().item()))
                self.assertTrue(bool((reset.horizons >= 21).all().item()))

    def test_a_caught_start_seeds_the_counter_that_reproduces_its_grasp(self):
        """`physical_grasp=True` and a zero persistence counter contradict.

        `_update_physical_grasp` recomputes `physical_grasp` from
        `bilateral_contact_steps >= _GRASP_PERSISTENCE_STEPS`, so a reset that
        seeds the flag but not the counter hands the detector a state it will
        deny for the next two env steps. On a placement episode
        `target_has_settled` is already true at the carried hover height, and
        that denial is enough for `wrong_place_settled` to terminate the episode
        on env step 1 of 64 with both pads loaded. Measured 2026-08-09.

        Asserted as agreement between the two fields rather than against a
        literal, so raising `_GRASP_PERSISTENCE_STEPS` cannot silently
        reintroduce the gap.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            _GRASP_PERSISTENCE_STEPS,
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
            "ee_workspace_z_bounds": [0.19, 0.32],
        }
        objects = (
            "robocasa_apple",
            "robocasa_banana",
            "robocasa_plate",
            "robocasa_bowl",
        )
        for instruction in ("put_into_plate", "put_into_bowl", "pick_up"):
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
                caught = reset.physical_grasp.to(dtype=torch.bool)
                warmed = reset.bilateral_contact_steps >= int(
                    _GRASP_PERSISTENCE_STEPS
                )
                self.assertTrue(bool(torch.equal(warmed, caught)))
                # And nothing is warmed that was not seeded as holding: the
                # counter is a floor for a real grasp, never a head start for a
                # world that has to earn one.
                self.assertTrue(
                    bool(
                        (reset.bilateral_contact_steps[~caught] == 0)
                        .all()
                        .item()
                    )
                )
                if instruction == "pick_up":
                    self.assertFalse(bool(caught.any().item()))
                else:
                    self.assertTrue(bool(caught.all().item()))

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
        # The two pick_up worlds hold an object, so their end-effector height is
        # object_z + the MEASURED pad offset 0.0075 m -- not the 0.08 m
        # ee_platform offset this fixture used to carry, which placed a "caught"
        # object 8 cm below the pads that were supposedly holding it. The
        # expected rewards below are unchanged: they encode the ladder, and the
        # ladder is reached from the geometrically consistent pose.
        ee = torch.tensor(
            [
                [0.08, 0.00, 0.26],
                [0.10, 0.00, 0.26],
                [0.00, 0.00, 0.1875],
                [0.00, 0.00, 0.2475],
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

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "log_std clamp requires PyTorch",
    )
    def test_the_action_distribution_cannot_diffuse(self):
        """max_log_std must actually bound the sampled std, not just record it.

        The 16M-step run died of diffusion, not of any curriculum problem: with
        entropy_coef pushing log_std up and nothing pushing it down, log_std rose
        for 12M steps straight from -1.227 to -0.895 while the action->target
        cosine fell 0.25 -> 0.14 and validation went from a 7.3% peak to 0.7%.
        The -0.3 ceiling never bound. This pins the ceiling INSIDE the band the
        policy was productive in, and pins that the clamp is applied on the path
        that produces actions rather than only stored on the module.
        """

        import torch
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            SmolVLAGRPOPolicy,
        )

        policy = SmolVLAGRPOPolicy(
            state_dim=6,
            chunk_size=8,
            action_dim=5,
            hidden_dim=32,
            residual_scale=1.0,
            init_log_std=-1.2,
            min_log_std=-5.0,
            max_log_std=-1.10,
        )
        # Drive the raw parameter far past the ceiling, the way 12M steps of a
        # net-positive entropy bonus did.
        with torch.no_grad():
            policy.log_std.fill_(0.5)
        clamped = policy.clamped_log_std()
        self.assertTrue(bool((clamped <= -1.10 + 1e-6).all().item()))
        # And the floor still holds in the other direction.
        with torch.no_grad():
            policy.log_std.fill_(-9.0)
        self.assertTrue(
            bool((policy.clamped_log_std() >= -5.0 - 1e-6).all().item())
        )

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


class CDPRGripperGeometryTests(unittest.TestCase):
    """The reward's grasp point must be somewhere the gripper can actually grasp.

    This is the test that was missing. pick_up ran 10M GPU-hours-worth of steps
    with pick_grasp_height_offset=0.08 against a real pad offset of 0.0075 m: the
    reward's optimum was a pose 7.25 cm above the object, the policy converged to
    within 1.2-1.5 cm of it, and the grasp rate DECAYED from 0.068 to 0.056 as
    convergence removed the erratic excursions that had been producing accidental
    contacts. Terminal successes went 8/1024 -> 0/1024. Nothing failed loudly.
    """

    TABLE_Z = 0.15
    GRASP_CONFIGS = (PICK_UP_CONFIG, CONFIG)

    def _geometry(self):
        from rl_vla_bootstrapping.simulation.cdpr_gripper_geometry import (
            load_cdpr_gripper_geometry,
        )

        config = load_project_config(PICK_UP_CONFIG)
        xml = config.resolve_path(config.embodiment.xml_path)
        self.assertIsNotNone(xml, "pick_up config must resolve an MJCF path")
        return load_cdpr_gripper_geometry(xml), xml

    def test_the_measured_pad_offset_is_not_the_platform_offset(self):
        """Pin the number the configs are calibrated against.

        ee_platform sits 0.08 m ABOVE ee_base and the pad 0.0875 m below that, so
        the pads are 0.0075 m BELOW the body ee_position tracks. Confusing the two
        is the whole bug, so if the model changes this test says so.
        """

        geometry, _ = self._geometry()
        self.assertAlmostEqual(geometry.pad_center_offset, -0.0075, places=6)
        self.assertAlmostEqual(geometry.grasp_height_offset, 0.0075, places=6)
        self.assertAlmostEqual(geometry.pad_half_height, 0.0175, places=6)
        self.assertAlmostEqual(geometry.finger_tip_offset, -0.0390, places=6)
        # The fingers reach well below the pads, which is why a grasp height that
        # looks harmless can still drive the tips through the desk.
        self.assertLess(
            geometry.finger_tip_offset, geometry.pad_span[0]
        )

    def test_grasp_configs_target_a_reachable_grasp_point(self):
        from rl_vla_bootstrapping.simulation.cdpr_gripper_geometry import (
            assert_grasp_offset_matches_model,
        )

        for path in self.GRASP_CONFIGS:
            config = load_project_config(path)
            metadata = config.task.metadata
            if "pick_grasp_height_offset" not in metadata:
                continue
            xml = config.resolve_path(config.embodiment.xml_path)
            with self.subTest(config=path.name):
                # Raises with the arithmetic if the offset leaves the pad span.
                assert_grasp_offset_matches_model(
                    metadata["pick_grasp_height_offset"],
                    xml_path=xml,
                    label=f"{path.name}:pick_grasp_height_offset",
                )

    def test_the_controller_floor_lets_the_pads_reach_every_object(self):
        """A floor above the grasp height makes the task impossible, silently."""

        from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
            OBJECT_VARIANTS,
        )

        geometry, _ = self._geometry()
        for path in self.GRASP_CONFIGS:
            config = load_project_config(path)
            metadata = config.task.metadata
            args = config.training.rl.args
            floor = args.get("controller_workspace_z_bounds")
            with self.subTest(config=path.name):
                self.assertIsNotNone(
                    floor,
                    "a grasp phase must set controller_workspace_z_bounds; the "
                    "0.25 default puts the pads above every object",
                )
                floor_z = float(floor[0])
                offset = float(metadata["pick_grasp_height_offset"])
                for name in metadata["target_object_pool"]:
                    variant = OBJECT_VARIANTS[name]
                    if variant.fitted_gripper_opening <= 0.0:
                        continue  # not liftable; excluded from grasp phases
                    center = self.TABLE_Z + variant.rest_height
                    with self.subTest(object=name):
                        self.assertTrue(
                            geometry.can_reach(
                                center, controller_floor=floor_z
                            ),
                            f"{name}: pads cannot reach a centre at "
                            f"{center:.4f} m from floor {floor_z:.4f} m",
                        )
                        # And the reward's own target must be reachable, not just
                        # the loosest grasp height.
                        self.assertGreaterEqual(
                            center + offset,
                            floor_z,
                            f"{name}: reward target {center + offset:.4f} m is "
                            f"below the controller floor {floor_z:.4f} m",
                        )

    def test_the_grasp_height_keeps_desk_contact_shallow(self):
        """Grasping small objects puts the finger tips into the desk, shallowly.

        This is a bound, not a prohibition. The fingers reach 0.039 m below
        ee_base while the pads are only 0.0075 m below it, so grasping anything
        resting on the desk necessarily drives the tips near or just past the
        surface -- for the shortest objects, 3-4 mm past. The oracle run in
        runs/cdpr_task_reference_episodes shows that is harmless: 3/3 pick_up
        successes at 5.70-5.72 with finite rewards throughout. What is NOT
        harmless is deep interpenetration, which has previously diverged MJWarp
        into NaN rewards, so this pins the depth rather than requiring zero.
        """

        MAX_DEPTH = 0.010

        from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
            OBJECT_VARIANTS,
        )

        geometry, _ = self._geometry()
        config = load_project_config(PICK_UP_CONFIG)
        metadata = config.task.metadata
        offset = float(metadata["pick_grasp_height_offset"])
        for name in metadata["target_object_pool"]:
            variant = OBJECT_VARIANTS[name]
            if variant.fitted_gripper_opening <= 0.0:
                continue
            center = self.TABLE_Z + variant.rest_height
            ee = center + offset
            with self.subTest(object=name):
                depth = self.TABLE_Z - geometry.finger_tip_height(ee)
                self.assertLessEqual(
                    depth,
                    MAX_DEPTH,
                    f"{name}: finger tips reach "
                    f"{geometry.finger_tip_height(ee):.4f} m, {depth * 1000:.1f} "
                    f"mm into the {self.TABLE_Z:.2f} m desk",
                )
                # The object centre still has to sit inside the pads.
                low, high = geometry.pad_span
                self.assertLessEqual(ee + low, center)
                self.assertGreaterEqual(ee + high, center)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") is not None,
        "near-plane check requires MuJoCo",
    )
    def test_the_wrist_camera_can_see_what_it_is_grasping(self):
        """The render near plane must be closer than the grasp working distance.

        znear and zfar are FRACTIONS of model.stat.extent, and extent here is
        ~14.7 m because a 5x5 m floor geom dominates it. At MuJoCo's default
        znear=0.01 the near plane sat at 0.147 m while the wrist camera works
        centimetres from the object: rasterized renders showed the desk vanish
        and the distant floor show through as a blank white field at every
        end-effector height at or below 0.22 m -- the entire grasp phase.

        Only the OpenGL rasterizer clipped; the mjwarp ray tracer used for
        training did not. But the rasterizer is what the reference-episode
        script renders with, and that script is how grasp geometry gets
        verified, so a blank wrist view there hides exactly what it exists to
        show.
        """

        import mujoco

        config = load_project_config(PICK_UP_CONFIG)
        xml = config.resolve_path(config.embodiment.xml_path)
        model = mujoco.MjModel.from_xml_path(str(xml))
        near = float(model.vis.map.znear) * float(model.stat.extent)

        # The wrist camera sits 0.045 m above ee_base, and a grasp puts ee_base
        # about 0.0075 m above the object centre -- so roughly 0.05 m of working
        # distance, less for anything nearer than the object's centre.
        self.assertLess(
            near,
            0.010,
            f"near plane {near:.4f} m (znear={model.vis.map.znear} x extent="
            f"{model.stat.extent:.2f} m) clips everything the wrist camera "
            "needs to see during a grasp",
        )
        # Keep the depth range sane once znear is small.
        far = float(model.vis.map.zfar) * float(model.stat.extent)
        self.assertLess(far / near, 5.0e4, "depth range too wide for precision")
        self.assertGreater(far, 6.0, "zfar must still cover the 5 m floor")

    def test_a_wrong_offset_is_rejected_with_the_arithmetic(self):
        """The guard has to fail on the value that actually shipped."""

        from rl_vla_bootstrapping.simulation.cdpr_gripper_geometry import (
            assert_grasp_offset_matches_model,
        )

        _, xml = self._geometry()
        with self.assertRaises(ValueError) as caught:
            assert_grasp_offset_matches_model(0.08, xml_path=xml)
        message = str(caught.exception)
        self.assertIn("0.0800", message)
        self.assertIn("cannot touch", message)


class CDPRPostGraspDiagnosticTests(unittest.TestCase):
    """The metrics that say WHY grasps do not become lifts.

    Across two pick_up runs the grasp->lift conversion sat at 0.31-0.33 and
    never moved while the grasp rate itself climbed. physical_grasp_rate and
    physical_lift_rate report that fact and cannot explain it, so these summarize
    what each world did after it first held a real grasp.
    """

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "post-grasp metrics operate on tensors",
    )
    def test_it_averages_only_over_worlds_that_grasped(self):
        """Counting never-grasped worlds as zero would track the grasp rate.

        The whole point is to describe the behaviour of the worlds that DID
        grasp, independently of how many of them there were.
        """

        import torch
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            post_grasp_metrics,
        )

        # Two grasped at steps 10 and 30, rising 0.02 and 0.04; two never did.
        first = torch.tensor([10, -1, 30, -1], dtype=torch.int64)
        at = torch.tensor([0.20, 0.0, 0.19, 0.0], dtype=torch.float32)
        peak = torch.tensor([0.22, 0.0, 0.23, 0.0], dtype=torch.float32)
        out = post_grasp_metrics(first, at, peak)

        self.assertEqual(out["post_grasp_worlds"], 2.0)
        self.assertAlmostEqual(out["post_grasp_first_env_step_mean"], 20.0, 5)
        self.assertAlmostEqual(out["post_grasp_rise_mean_m"], 0.03, 5)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "post-grasp metrics operate on tensors",
    )
    def test_no_grasp_reports_zero_worlds_rather_than_a_number(self):
        """A 0.0 mean must be distinguishable from a measured 0.0."""

        import torch
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            post_grasp_metrics,
        )

        n = 4
        out = post_grasp_metrics(
            torch.full((n,), -1, dtype=torch.int64),
            torch.zeros(n, dtype=torch.float32),
            torch.zeros(n, dtype=torch.float32),
        )
        self.assertEqual(out["post_grasp_worlds"], 0.0)
        self.assertEqual(out["post_grasp_first_env_step_mean"], 0.0)
        self.assertEqual(out["post_grasp_rise_mean_m"], 0.0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "post-grasp metrics operate on tensors",
    )
    def test_it_separates_the_three_failure_shapes(self):
        """Each hypothesis has to produce a distinguishable signature."""

        import torch
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            post_grasp_metrics,
        )

        horizon = 64

        # Grasps late: no steps left to lift.
        late = post_grasp_metrics(
            torch.tensor([60, 58], dtype=torch.int64),
            torch.tensor([0.19, 0.19], dtype=torch.float32),
            torch.tensor([0.19, 0.19], dtype=torch.float32),
        )
        self.assertGreater(late["post_grasp_first_env_step_mean"], 0.8 * horizon)
        self.assertAlmostEqual(late["post_grasp_rise_mean_m"], 0.0, 5)

        # Grasps early and never commands up.
        never_up = post_grasp_metrics(
            torch.tensor([12, 15], dtype=torch.int64),
            torch.tensor([0.19, 0.19], dtype=torch.float32),
            torch.tensor([0.191, 0.190], dtype=torch.float32),
        )
        self.assertLess(never_up["post_grasp_first_env_step_mean"], 0.3 * horizon)
        self.assertLess(never_up["post_grasp_rise_mean_m"], 0.005)

        # Lifts well clear of the 0.05 m success height, so a low lift rate then
        # means it settled back before the terminal step.
        lifts = post_grasp_metrics(
            torch.tensor([12, 15], dtype=torch.int64),
            torch.tensor([0.19, 0.19], dtype=torch.float32),
            torch.tensor([0.26, 0.25], dtype=torch.float32),
        )
        self.assertGreater(lifts["post_grasp_rise_mean_m"], 0.05)

        # The three shapes are mutually distinguishable on these two numbers.
        self.assertNotAlmostEqual(
            late["post_grasp_first_env_step_mean"],
            never_up["post_grasp_first_env_step_mean"],
        )
        self.assertNotAlmostEqual(
            never_up["post_grasp_rise_mean_m"], lifts["post_grasp_rise_mean_m"]
        )

    def test_the_metric_names_survive_the_rank_reduction(self):
        """Sums and means are reduced differently at the update boundary.

        Every metric is all-reduced with SUM, then a suffix rule divides some
        keys back down by world size. The two means must match that rule or they
        come out doubled; the world count must NOT, because a global count is
        what makes an all-zero mean readable.
        """

        divided = ("_time_s", "_mean", "_max", "_std", "_rate")
        names = {
            "post_grasp_first_env_step_mean": True,
            "post_grasp_rise_mean_m": True,
            "post_grasp_worlds": False,
        }
        for name, should_divide in names.items():
            with self.subTest(metric=name):
                matched = name.endswith(divided) or "_mean_" in name
                self.assertEqual(matched, should_divide)


class CDPRLiftIncentiveTests(unittest.TestCase):
    """Attempting a lift must never be worse than holding still.

    The GRPO return is the last active step's reward. While the grasp bonus was
    gated on state.grasped, a lift that failed cost the whole 1.0 grasp credit,
    so trying only paid off above P(success|grasp) = 0.322. Measured over three
    pick_up runs the ratio started at 0.319 -- essentially AT break-even -- and
    then decayed to 0.132 as the policy learned not to try, with the post-grasp
    rise falling from 18 mm to 6.9 mm against a 50 mm success height.
    """

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def _rewards(self, *, grasped, ever_grasped, object_z, initial_z):
        import torch
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        n = len(grasped)
        objects = torch.zeros((n, 4, 3), dtype=torch.float32)
        for i, z in enumerate(object_z):
            objects[i, 0] = torch.tensor([0.0, 0.0, float(z)])
        initial = objects[:, 0].clone()
        for i, z in enumerate(initial_z):
            initial[i, 2] = float(z)
        state = BatchedTaskState(
            instruction_ids=torch.full(
                (n,), INSTRUCTION_TO_ID["pick_up"], dtype=torch.int64
            ),
            target_slots=torch.zeros((n,), dtype=torch.int64),
            reference_slots=torch.full((n,), -1, dtype=torch.int64),
            second_reference_slots=torch.full((n,), -1, dtype=torch.int64),
            initial_target_positions=initial,
            ever_grasped=torch.tensor(ever_grasped, dtype=torch.bool),
            grasped=torch.zeros((n,), dtype=torch.bool),
            step_count=torch.zeros((n,), dtype=torch.int64),
            release_threshold=torch.full((n,), 0.55),
            support_surface_z=torch.full((n,), 0.15),
            target_rest_height=torch.full((n,), 0.03),
        )
        # End-effector one pad offset above whatever it is holding.
        ee = torch.zeros((n, 3), dtype=torch.float32)
        for i, z in enumerate(object_z):
            ee[i] = torch.tensor([0.0, 0.0, float(z) + 0.0075])
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.full((n,), 0.50),
            caught_target=torch.tensor(grasped, dtype=torch.bool),
            active_mask=torch.ones((n,), dtype=torch.bool),
            max_steps=128,
            catch_release_dense_reward=BatchedCatchReleaseDenseReward(),
        )
        return [float(v) for v in result.rewards]

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_a_failed_lift_keeps_the_grasp_credit(self):
        """Dropping must not erase what the policy already achieved."""

        rest = 0.18
        # 0: never grasped, sitting at the object.
        # 1: grasped once, dropped it, object back at rest.
        # 2: still holding at rest height.
        rewards = self._rewards(
            grasped=[False, False, True],
            ever_grasped=[False, True, True],
            object_z=[rest, rest, rest],
            initial_z=[rest, rest, rest],
        )
        never, dropped, holding = rewards
        self.assertGreater(
            dropped,
            never,
            "a world that achieved a grasp and lost it must still score above "
            "one that never grasped",
        )
        # The whole point: the gap between holding and having dropped is now the
        # lift term alone, not the lift term plus the entire grasp bonus.
        self.assertLess(
            holding - dropped,
            0.5,
            f"dropping still costs {holding - dropped:.3f}; attempting a lift "
            "remains a gamble against holding still",
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_a_lift_that_settles_keeps_its_credit(self):
        """The GRPO return is the terminal reward, so a transient lift paid nothing.

        With the lift term read off the INSTANT height, a policy that raised the
        object 40 mm at step 40 and let it settle by step 128 scored exactly the
        same as one that never moved. Measured consequence over 4.2M steps:
        physical_grasp_rate climbed 0.380 -> 0.447 while post_grasp_rise_mean_m
        fell 19 mm -> 10 mm and success fell 0.108 -> 0.042. The policy had found
        the "close the gripper and hold still" rung and every transient lift it
        produced was scored as though it had not happened.
        """

        import torch
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        rest = 0.18
        offset = 0.0075
        config = BatchedCatchReleaseDenseReward()

        def episode(heights, grasped_flags):
            """Step one world through a height trajectory; return final reward."""

            state = BatchedTaskState(
                instruction_ids=torch.full(
                    (1,), INSTRUCTION_TO_ID["pick_up"], dtype=torch.int64
                ),
                target_slots=torch.zeros((1,), dtype=torch.int64),
                reference_slots=torch.full((1,), -1, dtype=torch.int64),
                second_reference_slots=torch.full((1,), -1, dtype=torch.int64),
                initial_target_positions=torch.tensor([[0.0, 0.0, rest]]),
                ever_grasped=torch.zeros((1,), dtype=torch.bool),
                grasped=torch.zeros((1,), dtype=torch.bool),
                step_count=torch.zeros((1,), dtype=torch.int64),
                release_threshold=torch.full((1,), 0.55),
                support_surface_z=torch.full((1,), 0.15),
                target_rest_height=torch.full((1,), 0.03),
                peak_lift=torch.zeros((1,)),
            )
            reward = 0.0
            for height, held in zip(heights, grasped_flags):
                objects = torch.zeros((1, 4, 3))
                objects[0, 0] = torch.tensor([0.0, 0.0, height])
                ee = torch.tensor([[0.0, 0.0, height + offset]])
                result = evaluate_active_sparse_tasks(
                    state=state,
                    ee_position=ee,
                    object_positions=objects,
                    gripper_opening=torch.full((1,), 0.50),
                    caught_target=torch.tensor([held]),
                    active_mask=torch.ones((1,), dtype=torch.bool),
                    max_steps=1024,
                    catch_release_dense_reward=config,
                )
                reward = float(result.rewards[0].item())
            return reward

        # Never moves: grasps and holds at rest for the whole episode.
        held_still = episode([rest] * 6, [True] * 6)
        # Raises 40 mm, loses the grasp, object settles back to rest.
        lifted_and_settled = episode(
            [rest, rest + 0.02, rest + 0.04, rest + 0.02, rest, rest, ],
            [True, True, True, False, False, False],
        )
        self.assertGreater(
            lifted_and_settled,
            held_still,
            "a 40 mm lift that settled must still beat never having tried; "
            "otherwise the policy is correct to hold still, which is exactly "
            "what four runs converged on",
        )
        # And the credit is the PEAK, not the final height: 40 mm of a 50 mm
        # scale at weight 1.0 is 0.8 of the lift term.
        self.assertAlmostEqual(lifted_and_settled - held_still, 0.8, places=5)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_a_grasp_that_never_lifts_earns_no_grasp_bonus(self):
        """Pressing the object into the desk must stop paying.

        Grasp quality and lift are anti-correlated in training -- corr(grasp
        rate, prelifted rise) = -0.786 over 451 updates, with pad force rising
        6.4 -> 8.0 N and slip FALLING 3.86 -> 2.97 mm as the lift died
        35 mm -> 15 mm. Pressing down maximizes pad force, bilateral contact and
        pose stability, all of which the reward and the physical-grasp detector
        pay for, while making the object unliftable. Gating the grasp bonus on a
        real lift removes that payoff.
        """

        import torch
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        rest, offset = 0.18, 0.0075
        config = BatchedCatchReleaseDenseReward.from_metadata(
            {"pick_grasp_bonus": 0.5, "pick_grasp_bonus_min_lift": 0.005}
        )

        def reward_at(object_z, held):
            state = BatchedTaskState(
                instruction_ids=torch.full(
                    (1,), INSTRUCTION_TO_ID["pick_up"], dtype=torch.int64
                ),
                target_slots=torch.zeros((1,), dtype=torch.int64),
                reference_slots=torch.full((1,), -1, dtype=torch.int64),
                second_reference_slots=torch.full((1,), -1, dtype=torch.int64),
                initial_target_positions=torch.tensor([[0.0, 0.0, rest]]),
                ever_grasped=torch.zeros((1,), dtype=torch.bool),
                grasped=torch.zeros((1,), dtype=torch.bool),
                step_count=torch.zeros((1,), dtype=torch.int64),
                release_threshold=torch.full((1,), 0.55),
                support_surface_z=torch.full((1,), 0.15),
                target_rest_height=torch.full((1,), 0.03),
                peak_lift=torch.zeros((1,)),
            )
            objects = torch.zeros((1, 4, 3))
            objects[0, 0] = torch.tensor([0.0, 0.0, object_z])
            result = evaluate_active_sparse_tasks(
                state=state,
                ee_position=torch.tensor([[0.0, 0.0, object_z + offset]]),
                object_positions=objects,
                gripper_opening=torch.full((1,), 0.50),
                caught_target=torch.tensor([held]),
                active_mask=torch.ones((1,), dtype=torch.bool),
                max_steps=1024,
                catch_release_dense_reward=config,
                bilateral_contact=torch.tensor([True]),
            )
            return float(result.rewards[0].item()), result

        # A latched grasp at rest height -- the pressing pose -- earns contact
        # but NOT the grasp bonus.
        pinned, _ = reward_at(rest, True)
        # 2 mm up: still under the 5 mm gate.
        barely, _ = reward_at(rest + 0.002, True)
        # 10 mm up: qualifies.
        lifted, _ = reward_at(rest + 0.010, True)

        # Under the gate, the only thing that changed is the lift term itself:
        # 2 mm of a 50 mm scale at weight 1.0 = 0.04. No grasp bonus.
        lift_only = (
            float(config.pick_lift_reward_weight)
            * 0.002
            / float(config.pick_lift_reward_scale)
        )
        self.assertAlmostEqual(barely - pinned, lift_only, places=4)
        # Clearing it hands over the whole bonus on top of the lift term.
        expected = float(config.pick_grasp_bonus) + (
            float(config.pick_lift_reward_weight)
            * 0.008
            / float(config.pick_lift_reward_scale)
        )
        self.assertAlmostEqual(lifted - barely, expected, places=4)
        self.assertGreater(
            lifted - barely,
            float(config.pick_grasp_bonus),
            "clearing the gate must hand over the whole grasp bonus",
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_the_lift_gate_is_a_ratchet_not_an_instantaneous_test(self):
        """Once a grasp has lifted, dropping must not revoke the bonus."""

        import torch
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        rest, offset = 0.18, 0.0075
        config = BatchedCatchReleaseDenseReward.from_metadata(
            {"pick_grasp_bonus": 0.5, "pick_grasp_bonus_min_lift": 0.005}
        )
        state = BatchedTaskState(
            instruction_ids=torch.full(
                (1,), INSTRUCTION_TO_ID["pick_up"], dtype=torch.int64
            ),
            target_slots=torch.zeros((1,), dtype=torch.int64),
            reference_slots=torch.full((1,), -1, dtype=torch.int64),
            second_reference_slots=torch.full((1,), -1, dtype=torch.int64),
            initial_target_positions=torch.tensor([[0.0, 0.0, rest]]),
            ever_grasped=torch.zeros((1,), dtype=torch.bool),
            grasped=torch.zeros((1,), dtype=torch.bool),
            step_count=torch.zeros((1,), dtype=torch.int64),
            release_threshold=torch.full((1,), 0.55),
            support_surface_z=torch.full((1,), 0.15),
            target_rest_height=torch.full((1,), 0.03),
            peak_lift=torch.zeros((1,)),
        )

        def step(object_z, held):
            objects = torch.zeros((1, 4, 3))
            objects[0, 0] = torch.tensor([0.0, 0.0, object_z])
            return evaluate_active_sparse_tasks(
                state=state,
                ee_position=torch.tensor([[0.0, 0.0, object_z + offset]]),
                object_positions=objects,
                gripper_opening=torch.full((1,), 0.50),
                caught_target=torch.tensor([held]),
                active_mask=torch.ones((1,), dtype=torch.bool),
                max_steps=1024,
                catch_release_dense_reward=config,
                bilateral_contact=torch.tensor([held]),
            )

        step(rest, True)
        step(rest + 0.020, True)          # qualifies the grasp
        after_drop = step(rest, False)    # loses it; object back on the desk
        self.assertGreaterEqual(float(state.peak_lift[0].item()), 0.019)
        # ever_grasped is true and the lift ratchet is above the gate, so the
        # bonus survives the drop -- otherwise attempting a lift would again be
        # priced as losing the grasp, which is the trap e787c80 removed.
        self.assertGreater(float(after_drop.rewards[0].item()), 1.5)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_zero_min_lift_reproduces_the_ever_grasped_bonus(self):
        """The knob defaults off and must be inert when it is."""

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedCatchReleaseDenseReward,
        )

        self.assertEqual(
            BatchedCatchReleaseDenseReward().pick_grasp_bonus_min_lift, 0.0
        )
        self.assertEqual(
            BatchedCatchReleaseDenseReward.from_metadata(
                {}
            ).pick_grasp_bonus_min_lift,
            0.0,
        )
        # Negative values are clamped rather than inverting the comparison.
        self.assertEqual(
            BatchedCatchReleaseDenseReward.from_metadata(
                {"pick_grasp_bonus_min_lift": -0.01}
            ).pick_grasp_bonus_min_lift,
            0.0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_peak_lift_cannot_be_earned_without_holding_the_object(self):
        """Batting the object upward must not pay -- only lifting it does."""

        import torch
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        rest = 0.18
        state = BatchedTaskState(
            instruction_ids=torch.full(
                (1,), INSTRUCTION_TO_ID["pick_up"], dtype=torch.int64
            ),
            target_slots=torch.zeros((1,), dtype=torch.int64),
            reference_slots=torch.full((1,), -1, dtype=torch.int64),
            second_reference_slots=torch.full((1,), -1, dtype=torch.int64),
            initial_target_positions=torch.tensor([[0.0, 0.0, rest]]),
            ever_grasped=torch.zeros((1,), dtype=torch.bool),
            grasped=torch.zeros((1,), dtype=torch.bool),
            step_count=torch.zeros((1,), dtype=torch.int64),
            release_threshold=torch.full((1,), 0.55),
            support_surface_z=torch.full((1,), 0.15),
            target_rest_height=torch.full((1,), 0.03),
            peak_lift=torch.zeros((1,)),
        )
        objects = torch.zeros((1, 4, 3))
        # Object 6 cm up -- well past the success height -- but NOT held.
        objects[0, 0] = torch.tensor([0.0, 0.0, rest + 0.06])
        evaluate_active_sparse_tasks(
            state=state,
            ee_position=torch.tensor([[0.0, 0.0, rest + 0.0675]]),
            object_positions=objects,
            gripper_opening=torch.full((1,), 0.50),
            caught_target=torch.tensor([False]),
            active_mask=torch.ones((1,), dtype=torch.bool),
            max_steps=1024,
            catch_release_dense_reward=BatchedCatchReleaseDenseReward(),
        )
        self.assertEqual(float(state.peak_lift[0].item()), 0.0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_omitting_peak_lift_reproduces_the_instantaneous_term(self):
        """Callers that predate the ratchet must be unaffected."""

        import torch
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedCatchReleaseDenseReward,
        )

        config = BatchedCatchReleaseDenseReward()
        rest = 0.18
        # _rewards() below constructs the state WITHOUT peak_lift.
        holding = self._rewards(
            grasped=[True], ever_grasped=[True],
            object_z=[rest + 0.04], initial_z=[rest],
        )[0]
        dropped = self._rewards(
            grasped=[False], ever_grasped=[True],
            object_z=[rest], initial_z=[rest],
        )[0]
        # Without the ratchet the settled world keeps no lift credit, which is
        # the pre-change behaviour this fallback has to preserve exactly.
        self.assertAlmostEqual(
            holding - dropped,
            0.8 * float(config.pick_lift_reward_weight),
            places=5,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "reward assembly operates on tensors",
    )
    def test_trying_beats_holding_at_the_measured_success_rate(self):
        """Break-even must sit below the rate the policy actually achieves.

        With P(success|grasp) measured at 0.123 and falling, the old 0.322
        break-even made not-trying correct. The check is that the expected value
        of attempting now exceeds holding at that same measured rate.
        """

        rest = 0.18
        holding = self._rewards(
            grasped=[True], ever_grasped=[True],
            object_z=[rest], initial_z=[rest],
        )[0]
        lifted = self._rewards(
            grasped=[True], ever_grasped=[True],
            object_z=[rest + 0.06], initial_z=[rest],
        )[0]
        dropped = self._rewards(
            grasped=[False], ever_grasped=[True],
            object_z=[rest], initial_z=[rest],
        )[0]

        self.assertGreater(lifted, holding)
        break_even = (holding - dropped) / (lifted - dropped)
        self.assertLess(
            break_even,
            0.123,
            f"break-even P(success|grasp) is {break_even:.3f}, at or above the "
            "measured success rate, so the policy is still better off not "
            "trying to lift",
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "the resetter operates on tensors",
)
class CDPRPreliftedPickUpStartTests(unittest.TestCase):
    """pick_up starts that begin already holding the object.

    Three multi-million-step runs plateaued at a ~0.30 grasp rate while the lift
    decayed -- post_grasp_rise_mean_m fell from 18 mm to 7-10 mm against a 50 mm
    success height, with the first grasp landing at env step ~27 of 64. The
    entropy floor and the ever_grasped ratchet each slowed that decay and
    neither stopped it, so a fraction of groups now skips the discovery problem
    entirely and gets dense signal on the lift from env step 0.
    """

    _OBJECTS = (
        "robocasa_apple",
        "robocasa_banana",
        "robocasa_plate",
        "robocasa_bowl",
    )

    @staticmethod
    def _metadata(fraction, **overrides):
        metadata = {
            "random_workspace_gripper_start": True,
            "placement_start_with_caught_object": False,
            "random_workspace_min_goal_xy_distance": 0.10,
            "random_workspace_horizon_low": 26,
            "random_workspace_horizon_high": 26,
            "ee_workspace_x_bounds": [-0.24, 0.24],
            "ee_workspace_y_bounds": [-0.24, 0.24],
            "ee_workspace_z_bounds": [0.19, 0.30],
            "min_scene_objects": 1,
            "max_scene_objects": 2,
            "pick_grasp_height_offset": 0.0075,
        }
        if fraction is not None:
            metadata["pick_up_prelifted_group_fraction"] = fraction
        metadata.update(overrides)
        return metadata

    def _resetter(
        self, torch, *, groups, fraction, instruction="pick_up", **overrides
    ):
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
            RankLocalCurriculum,
        )
        from rl_vla_bootstrapping.policy.rank_local_grpo import (
            RankLocalGroupLayout,
        )

        backend = CDPRCatchReleaseRewardTests._fake_backend(torch)
        resetter = BatchedReverseFrontierResetter(
            backend=backend,
            layout=RankLocalGroupLayout(
                worlds_per_rank=groups * 8, groups_per_rank=groups, group_size=8
            ),
            curriculum=RankLocalCurriculum(device=backend.device),
            rank=0,
            base_seed=17,
            instruction_types=(instruction,),
            allowed_objects=self._OBJECTS,
            task_metadata=self._metadata(fraction, **overrides),
        )
        return backend, resetter

    # -- aligned starts: at the grasp point, gripper OPEN ------------------

    def _aligned(self, torch, *, groups=64, aligned=0.5, prelifted=0.0):
        backend, resetter = self._resetter(
            torch,
            groups=groups,
            fraction=prelifted,
            pick_up_aligned_group_fraction=aligned,
        )
        reset = resetter.reset(update_index=0, round_index=0)
        return backend, reset

    def test_an_aligned_start_is_not_holding_anything(self):
        """The whole point of the stage: the closing action is NOT handed over.

        If aligned starts were folded into the caught path they would arrive
        with the gripper shut and the grasp seeded, and the stage would train
        the lift a second time instead of the descend-and-close it exists for.
        """

        import torch

        _backend, reset = self._aligned(torch)
        aligned = reset.aligned.to(dtype=torch.bool)
        self.assertTrue(bool(aligned.any().item()), "no aligned starts drawn")
        for name, flat in (
            ("ever_grasped", reset.task_state.ever_grasped),
            ("grasped", reset.task_state.grasped),
            ("physical_grasp", reset.physical_grasp),
            ("prelifted", reset.prelifted),
        ):
            self.assertFalse(
                bool(flat.to(dtype=torch.bool)[aligned].any().item()),
                f"aligned starts have {name} set",
            )

    def test_an_aligned_start_has_the_gripper_open(self):
        import torch

        backend, reset = self._aligned(torch)
        aligned = reset.aligned.to(dtype=torch.bool)
        openings = backend.gripper_openings[aligned]
        self.assertTrue(
            bool((openings > 0.99).all().item()),
            f"aligned starts are not open: {openings.min().item()}",
        )

    def test_an_aligned_start_is_placed_at_the_grasp_point(self):
        """Object XY, one centimetre above the grasp height."""

        import torch

        backend, reset = self._aligned(torch)
        aligned = reset.aligned.to(dtype=torch.bool)
        rows = torch.arange(backend.object_positions.shape[0])
        target = backend.object_positions[rows, reset.task_state.target_slots]
        ee = backend.ee_positions
        xy_error = torch.linalg.vector_norm(
            (ee - target)[:, :2][aligned], dim=-1
        )
        self.assertLess(float(xy_error.max()), 1.0e-5)
        z_error = (ee[:, 2] - target[:, 2])[aligned] - (0.0075 + 0.01)
        self.assertLess(float(z_error.abs().max()), 1.0e-5)

    def test_aligned_and_pregrasped_starts_do_not_overlap(self):
        """They partition the groups; overlap would shrink one silently."""

        import torch

        _backend, reset = self._aligned(torch, aligned=0.5, prelifted=0.5)
        aligned = reset.aligned.to(dtype=torch.bool)
        prelifted = reset.prelifted.to(dtype=torch.bool)
        self.assertFalse(bool((aligned & prelifted).any().item()))
        self.assertTrue(bool(aligned.any().item()))
        self.assertTrue(bool(prelifted.any().item()))

    def test_every_candidate_in_a_group_shares_the_aligned_stage(self):
        """Same reason as the pre-grasped stage: GRPO normalizes within a group."""

        import torch

        groups = 64
        _backend, reset = self._aligned(torch, groups=groups)
        by_group = reset.aligned.reshape(groups, 8)
        mixed = by_group.any(dim=1) & ~by_group.all(dim=1)
        self.assertFalse(bool(mixed.any().item()))
        self.assertTrue(bool(by_group[:, 0].any().item()))
        self.assertFalse(bool(by_group[:, 0].all().item()))

    def test_a_zero_fraction_leaves_the_starts_byte_identical(self):
        """The knob must not perturb the RNG stream when it is off.

        Every start this resetter produces comes off one generator, so drawing
        for a disabled stage would silently change every other run's starts --
        the reason the pre-grasped draw is guarded the same way.
        """

        import torch

        backend_off, resetter_off = self._resetter(
            torch, groups=16, fraction=0.5
        )
        reset_off = resetter_off.reset(update_index=0, round_index=0)
        ee_off = backend_off.ee_positions.clone()

        backend_zero, resetter_zero = self._resetter(
            torch,
            groups=16,
            fraction=0.5,
            pick_up_aligned_group_fraction=0.0,
        )
        reset_zero = resetter_zero.reset(update_index=0, round_index=0)
        self.assertTrue(
            torch.equal(backend_zero.ee_positions, ee_off),
            "an explicit 0.0 changed the starts",
        )
        self.assertTrue(
            torch.equal(reset_zero.prelifted, reset_off.prelifted)
        )
        self.assertFalse(bool(reset_zero.aligned.any().item()))

    def test_validation_gets_no_aligned_starts(self):
        """allow_prelifted=False means the FULL task, every stage off."""

        import torch

        _backend, resetter = self._resetter(
            torch,
            groups=16,
            fraction=0.5,
            pick_up_aligned_group_fraction=0.5,
        )
        reset = resetter.reset(
            update_index=0, round_index=0, allow_prelifted=False
        )
        self.assertFalse(bool(reset.aligned.any().item()))
        self.assertFalse(bool(reset.prelifted.any().item()))

    def test_every_candidate_in_a_group_shares_the_stage(self):
        """A mixed group would make GRPO score the spawn, not the actions.

        The advantage is normalized WITHIN a group of eight. If some candidates
        started pre-grasped and others had to earn the grasp, the ones handed it
        would carry a large positive advantage for having been lucky, and the
        update would be silently corrupted rather than obviously broken.
        """

        import torch

        groups = 64
        backend, resetter = self._resetter(torch, groups=groups, fraction=0.5)
        reset = resetter.reset(update_index=0, round_index=0)

        for name, flat in (
            ("prelifted", reset.prelifted),
            ("ever_grasped", reset.task_state.ever_grasped),
            ("grasped", reset.task_state.grasped),
            ("physical_grasp", reset.physical_grasp),
        ):
            by_group = flat.reshape(groups, 8)
            mixed = by_group.any(dim=1) & ~by_group.all(dim=1)
            self.assertFalse(
                bool(mixed.any().item()),
                f"{name} differs across candidates of the same GRPO group",
            )
        # And the stage is really varying across groups, so the check above is
        # not passing because every group happened to land on one side.
        by_group = reset.prelifted.reshape(groups, 8)[:, 0]
        self.assertTrue(bool(by_group.any().item()))
        self.assertFalse(bool(by_group.all().item()))
        # The object poses each group starts from must be group-uniform too --
        # the pre-grasped path rewrites the end-effector, so it is the one that
        # could break this.
        ee_by_group = backend.ee_positions.reshape(groups, 8, 3)
        self.assertTrue(
            torch.allclose(ee_by_group, ee_by_group[:, :1, :].expand_as(ee_by_group))
        )

    def test_a_pregrasped_start_is_the_object_at_rest_in_a_closed_gripper(self):
        """Rest height, one pad offset below the pads, fingers closed.

        Three ways this silently degrades into something else:

        * reusing the placement path's random_caught_position would put the
          object below wherever the end-effector already was -- for pick_up,
          floating in mid-air above the desk;
        * an end-effector offset other than the measured 0.0075 m pad offset
          would spawn the object away from the pads that are supposedly holding
          it, and it would fall on env step 1;
        * an open gripper leaves state.grasped false, which requires
          gripper_opening <= 0.94, so the world would not be pre-grasped at all.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            _FITTED_GRIPPER,
        )

        groups = 32
        backend, resetter = self._resetter(torch, groups=groups, fraction=0.5)
        reset = resetter.reset(update_index=0, round_index=0)

        prelifted = reset.prelifted
        self.assertTrue(bool(prelifted.any().item()))
        target = backend.object_positions[:, 0]
        ee = backend.ee_positions
        offset = float(resetter.pick_grasp_height_offset)

        rest_z = (
            reset.task_state.support_surface_z
            + reset.task_state.target_rest_height
        )
        self.assertTrue(
            torch.allclose(
                target[prelifted, 2], rest_z[prelifted], atol=1.0e-6
            ),
            "the pre-grasped object must start on the desk at its rest height",
        )
        self.assertTrue(
            torch.allclose(
                ee[prelifted], target[prelifted] + torch.tensor(
                    [0.0, 0.0, offset]
                ), atol=1.0e-6
            ),
            "the pads must straddle the object, one pad offset above its centre",
        )

        fitted = torch.tensor(
            _FITTED_GRIPPER, dtype=torch.float32
        ).index_select(0, reset.group_target_catalog_ids)
        expected_opening = (fitted - (0.001 / 0.03)).clamp(
            0.0, 1.0
        ).repeat_interleave(8)
        openings = backend.gripper_openings
        self.assertTrue(
            torch.allclose(
                openings[prelifted], expected_opening[prelifted], atol=1.0e-6
            )
        )
        self.assertTrue(
            bool((openings[prelifted] <= 0.94).all().item()),
            "state.grasped requires gripper_opening <= 0.94",
        )
        self.assertTrue(bool(reset.task_state.grasped[prelifted].all().item()))
        self.assertTrue(
            bool(reset.task_state.ever_grasped[prelifted].all().item())
        )
        self.assertTrue(bool(reset.physical_grasp[prelifted].all().item()))
        # Untouched groups still start empty-handed at the full task.
        normal = ~prelifted
        self.assertTrue(bool(normal.any().item()))
        self.assertFalse(bool(reset.task_state.grasped[normal].any().item()))
        self.assertTrue(
            torch.allclose(target[normal, 2], rest_z[normal], atol=1.0e-6)
        )

    def test_the_lift_is_still_measured_against_the_desk(self):
        """initial_target_positions must be the REST position.

        pick_success is `grasped & (target_z - initial_target_z >= 0.05)`. If the
        reset wrote a raised position into initial_target_positions, the
        requirement would silently become "5 cm above wherever the reset put it"
        -- or, if it wrote a position below rest, success would come free.
        """

        import torch

        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedCatchReleaseDenseReward,
            evaluate_active_sparse_tasks,
        )

        groups = 32
        offset = 0.0075
        lift_height = 0.05

        def outcome(raise_by):
            backend, resetter = self._resetter(
                torch, groups=groups, fraction=0.5
            )
            reset = resetter.reset(update_index=0, round_index=0)
            prelifted = reset.prelifted
            rest_z = (
                reset.task_state.support_surface_z
                + reset.task_state.target_rest_height
            )
            self.assertTrue(
                torch.allclose(
                    reset.task_state.initial_target_positions[prelifted, 2],
                    rest_z[prelifted],
                    atol=1.0e-6,
                ),
                "the lift baseline is not the rest height",
            )
            objects = backend.object_positions.clone()
            objects[:, 0, 2] += raise_by
            ee = backend.ee_positions.clone()
            ee[:, 2] += raise_by
            result = evaluate_active_sparse_tasks(
                state=reset.task_state,
                ee_position=ee,
                object_positions=objects,
                gripper_opening=backend.gripper_openings.clone(),
                caught_target=prelifted,
                active_mask=prelifted,
                max_steps=128,
                catch_release_dense_reward=BatchedCatchReleaseDenseReward(),
            )
            return prelifted, result

        prelifted, raised = outcome(lift_height + 0.001)
        self.assertTrue(
            bool(raised.success[prelifted].all().item()),
            "a 5 cm raise from a pre-grasped start must satisfy pick_success",
        )
        _, short = outcome(lift_height - 0.001)
        self.assertFalse(
            bool(short.success[prelifted].any().item()),
            "a raise under the success height must not succeed -- the baseline "
            "has drifted below the rest position",
        )
        # And the pads really are on the object at that pre-grasped pose, so the
        # dense term is at its maximum rather than penalizing the start.
        self.assertTrue(
            bool(
                (
                    raised.diagnostics["pick_grasp_distance"][prelifted]
                    <= offset * 1.01
                ).all().item()
            )
        )

    def test_zero_fraction_reproduces_the_previous_resets_exactly(self):
        """The knob defaults off and must be inert when it is.

        Not merely "no pre-grasped groups": the sampler must not consume a draw
        either, or every downstream random quantity in the reset shifts and a run
        that set the fraction to 0 would still differ from one that predates the
        knob.
        """

        import torch

        groups = 16
        absent_backend, absent = self._resetter(
            torch, groups=groups, fraction=None
        )
        zero_backend, zero = self._resetter(torch, groups=groups, fraction=0.0)

        self.assertEqual(absent.pick_up_prelifted_group_fraction, 0.0)
        absent_reset = absent.reset(update_index=3, round_index=1)
        zero_reset = zero.reset(update_index=3, round_index=1)

        self.assertFalse(bool(zero_reset.prelifted.any().item()))
        self.assertFalse(bool(zero_reset.task_state.grasped.any().item()))
        for name, left, right in (
            ("objects", absent_backend.object_positions, zero_backend.object_positions),
            ("quaternions", absent_backend.object_quaternions, zero_backend.object_quaternions),
            ("ee", absent_backend.ee_positions, zero_backend.ee_positions),
            ("yaw", absent_backend.ee_yaw, zero_backend.ee_yaw),
            ("openings", absent_backend.gripper_openings, zero_backend.gripper_openings),
        ):
            with self.subTest(field=name):
                self.assertTrue(torch.equal(left, right))
        self.assertTrue(
            torch.equal(absent_reset.horizons, zero_reset.horizons)
        )

    def test_the_realized_fraction_matches_the_configured_one(self):
        import torch

        groups = 128
        rounds = 8
        for fraction in (0.25, 0.5):
            with self.subTest(fraction=fraction):
                _, resetter = self._resetter(
                    torch, groups=groups, fraction=fraction
                )
                prelifted = 0
                for round_index in range(rounds):
                    reset = resetter.reset(
                        update_index=0, round_index=round_index
                    )
                    # Count GROUPS, since the stage is a per-group draw.
                    prelifted += int(
                        reset.prelifted.reshape(groups, 8)[:, 0].sum().item()
                    )
                realized = prelifted / float(groups * rounds)
                self.assertAlmostEqual(realized, fraction, delta=0.05)

    def test_only_pick_up_and_only_training_gets_a_pregrasped_start(self):
        """move_to has nothing to hold, and validation must run the full task.

        Letting the stage into validation would make the held-out success rate --
        the number that says whether this intervention worked -- move with the
        knob instead of with the policy.
        """

        import torch

        _, move_to = self._resetter(
            torch, groups=16, fraction=1.0, instruction="move_to_object"
        )
        self.assertFalse(
            bool(
                move_to.reset(update_index=0, round_index=0)
                .prelifted.any()
                .item()
            )
        )

        _, pick_up = self._resetter(torch, groups=16, fraction=1.0)
        self.assertTrue(
            bool(
                pick_up.reset(update_index=0, round_index=0)
                .prelifted.all()
                .item()
            )
        )
        validation = pick_up.reset(
            update_index=0, round_index=0, allow_prelifted=False
        )
        self.assertFalse(bool(validation.prelifted.any().item()))
        self.assertFalse(bool(validation.task_state.grasped.any().item()))

    def test_post_grasp_metrics_keep_the_two_populations_apart(self):
        """A pre-grasped world grasps at env step 0 by construction.

        Folding those into post_grasp_first_env_step_mean would drag it toward 0
        as the pre-grasped fraction rose, and it would stop meaning "how late the
        policy earns its grasp" -- which is the comparison the metric exists to
        support across runs.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            post_grasp_metrics,
        )

        first = torch.tensor([0, 0, 20, 30], dtype=torch.int64)
        at = torch.tensor([0.1875, 0.1875, 0.19, 0.19], dtype=torch.float32)
        peak = torch.tensor([0.2475, 0.2275, 0.20, 0.21], dtype=torch.float32)
        prelifted = torch.tensor([True, True, False, False])

        out = post_grasp_metrics(first, at, peak, prelifted)
        self.assertEqual(out["post_grasp_worlds"], 2.0)
        self.assertAlmostEqual(out["post_grasp_first_env_step_mean"], 25.0, 5)
        self.assertAlmostEqual(out["post_grasp_rise_mean_m"], 0.015, 5)
        self.assertEqual(out["post_grasp_worlds_prelifted"], 2.0)
        self.assertAlmostEqual(
            out["post_grasp_first_env_step_mean_prelifted"], 0.0, 5
        )
        self.assertAlmostEqual(out["post_grasp_rise_mean_m_prelifted"], 0.05, 5)

        # Omitting the mask reproduces the pre-existing three numbers over every
        # grasped world, so the metric predates and survives this stage.
        legacy = post_grasp_metrics(first, at, peak)
        self.assertEqual(legacy["post_grasp_worlds"], 4.0)
        self.assertAlmostEqual(
            legacy["post_grasp_first_env_step_mean"], 12.5, 5
        )
        self.assertEqual(legacy["post_grasp_worlds_prelifted"], 0.0)

    def test_the_approach_gate_is_measured_without_pregrasped_groups(self):
        """The curriculum cap must not promote on episodes that never approach.

        The approach curriculum widens the start-distance cap once an
        instruction's pass rate crosses 0.30. A pre-grasped pick_up start begins
        with the object already in the gripper, performs no approach at all, and
        succeeds far more often -- 0.32 against 0.12 measured over a 400k-step
        probe. Feeding those to the gate inflated the observed rate by ~40%
        relative (0.169 against 0.121) and pushed the cap toward promoting on
        the strength of a different task.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            instruction_outcome_counts,
        )

        # 4 groups x 8 candidates. Groups 0-1 are pre-grasped and mostly
        # succeed; groups 2-3 approach normally and mostly do not.
        successes = torch.zeros((4, 8), dtype=torch.bool)
        successes[0, :6] = True
        successes[1, :6] = True
        successes[2, :1] = True
        successes[3, :1] = True
        task_ids = torch.zeros((4,), dtype=torch.int64)
        prelifted = torch.tensor([True, True, False, False])

        counts = instruction_outcome_counts(
            successes, task_ids, {"pick_up": 0}, prelifted
        )
        # Whole-run outcome keeps counting everything.
        self.assertEqual(counts["instruction_successes/pick_up"], 14.0)
        self.assertEqual(counts["instruction_worlds/pick_up"], 32.0)
        # The gate sees only the two normal-start groups.
        self.assertEqual(
            counts["instruction_successes_normal_start/pick_up"], 2.0
        )
        self.assertEqual(counts["instruction_worlds_normal_start/pick_up"], 16.0)

        contaminated = (
            counts["instruction_successes/pick_up"]
            / counts["instruction_worlds/pick_up"]
        )
        gated = (
            counts["instruction_successes_normal_start/pick_up"]
            / counts["instruction_worlds_normal_start/pick_up"]
        )
        self.assertAlmostEqual(contaminated, 0.4375, places=6)
        self.assertAlmostEqual(gated, 0.125, places=6)
        self.assertLess(
            gated,
            contaminated,
            "the pre-grasped groups must not be able to raise the gate",
        )

    def test_outcome_counts_are_unchanged_when_nothing_is_pregrasped(self):
        """With the stage off the two pairs must be identical.

        Both when the mask is absent (a collector round that predates the
        field) and when it is present and all-false.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            instruction_outcome_counts,
        )

        successes = torch.tensor(
            [[True, False], [False, False], [True, True]], dtype=torch.bool
        )
        task_ids = torch.tensor([0, 1, 0], dtype=torch.int64)
        ids = {"pick_up": 0, "move_to_object": 1}
        for label, mask in (
            ("absent", None),
            ("all false", torch.zeros((3,), dtype=torch.bool)),
        ):
            with self.subTest(mask=label):
                counts = instruction_outcome_counts(
                    successes, task_ids, ids, mask
                )
                for name in ids:
                    self.assertEqual(
                        counts[f"instruction_successes_normal_start/{name}"],
                        counts[f"instruction_successes/{name}"],
                    )
                    self.assertEqual(
                        counts[f"instruction_worlds_normal_start/{name}"],
                        counts[f"instruction_worlds/{name}"],
                    )
        # And the per-instruction split itself is right.
        counts = instruction_outcome_counts(successes, task_ids, ids, None)
        self.assertEqual(counts["instruction_successes/pick_up"], 3.0)
        self.assertEqual(counts["instruction_worlds/pick_up"], 4.0)
        self.assertEqual(counts["instruction_successes/move_to_object"], 0.0)
        self.assertEqual(counts["instruction_worlds/move_to_object"], 2.0)

    def test_rounds_carry_the_pregrasped_mask_through_concatenation(self):
        """The gate is computed after rounds are merged, so the mask must survive."""

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            CollectorRound,
            concatenate_collector_rounds,
        )

        def make(prelifted, ever_grasped=None):
            return CollectorRound(
                records={"advantage": torch.zeros((2,))},
                loss_mask=torch.ones((2,)),
                candidate_rewards=torch.zeros((2, 8)),
                candidate_success=torch.zeros((2, 8), dtype=torch.bool),
                group_instruction_ids=torch.zeros((2,), dtype=torch.int64),
                group_shell_ids=torch.zeros((2,), dtype=torch.int64),
                metrics={"rollout_time_s": 1.0},
                group_prelifted=prelifted,
                candidate_ever_grasped=ever_grasped,
            )

        first = torch.tensor([True, False])
        second = torch.tensor([False, True])
        merged = concatenate_collector_rounds([make(first), make(second)])
        self.assertEqual(len(merged), 10)
        self.assertTrue(
            torch.equal(merged[7], torch.tensor([True, False, False, True]))
        )
        # A round without the mask disables the split rather than guessing,
        # so the gate falls back to counting every world.
        partial = concatenate_collector_rounds([make(first), make(None)])
        self.assertIsNone(partial[7])

    def test_the_gate_mask_excludes_aligned_starts_as_well_as_pregrasped(self):
        """An aligned start performs no approach either.

        It begins at the grasp point with the gripper open, so scoring the
        approach curriculum on it promotes the cap on episodes that skipped the
        thing the cap measures -- the same reason pre-grasped starts were
        excluded. It is a SEPARATE mask because it is not holding anything, so
        the post-grasp diagnostics and the episode-offset gate must keep reading
        group_prelifted.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            CollectorRound,
            concatenate_collector_rounds,
        )

        def make(prelifted, skips):
            return CollectorRound(
                records={"advantage": torch.zeros((2,))},
                loss_mask=torch.ones((2,)),
                candidate_rewards=torch.zeros((2, 8)),
                candidate_success=torch.zeros((2, 8), dtype=torch.bool),
                group_instruction_ids=torch.zeros((2,), dtype=torch.int64),
                group_shell_ids=torch.zeros((2,), dtype=torch.int64),
                metrics={"rollout_time_s": 1.0},
                group_prelifted=prelifted,
                group_skips_approach=skips,
            )

        prelifted = torch.tensor([True, False])
        skips = torch.tensor([True, True])  # group 1 is aligned, not holding
        merged = concatenate_collector_rounds([make(prelifted, skips)])
        self.assertTrue(torch.equal(merged[7], prelifted))
        self.assertTrue(torch.equal(merged[9], skips))
        # The two must not be the same object, or one knob moves both.
        self.assertFalse(torch.equal(merged[7], merged[9]))

    def test_the_gate_mask_falls_back_to_prelifted_when_absent(self):
        """Checkpoints and rounds predating the aligned stage still work."""

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            CollectorRound,
            concatenate_collector_rounds,
        )

        prelifted = torch.tensor([True, False])
        item = CollectorRound(
            records={"advantage": torch.zeros((2,))},
            loss_mask=torch.ones((2,)),
            candidate_rewards=torch.zeros((2, 8)),
            candidate_success=torch.zeros((2, 8), dtype=torch.bool),
            group_instruction_ids=torch.zeros((2,), dtype=torch.int64),
            group_shell_ids=torch.zeros((2,), dtype=torch.int64),
            metrics={"rollout_time_s": 1.0},
            group_prelifted=prelifted,
        )
        merged = concatenate_collector_rounds([item])
        self.assertTrue(torch.equal(merged[9], prelifted))

    def test_rounds_carry_the_grasp_outcome_through_concatenation(self):
        """The approach cap promotes on this, so it has to survive the merge.

        The gate reads the grasp rate rather than full-task success: success
        also requires the lift, whose conversion is governed by the rollout
        budget left after the grasp lands, so gating the approach on it made the
        cap wait on a skill the approach cannot influence.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            CollectorRound,
            concatenate_collector_rounds,
        )

        def make(ever_grasped):
            return CollectorRound(
                records={"advantage": torch.zeros((2,))},
                loss_mask=torch.ones((2,)),
                candidate_rewards=torch.zeros((2, 8)),
                candidate_success=torch.zeros((2, 8), dtype=torch.bool),
                group_instruction_ids=torch.zeros((2,), dtype=torch.int64),
                group_shell_ids=torch.zeros((2,), dtype=torch.int64),
                metrics={"rollout_time_s": 1.0},
                group_prelifted=torch.zeros((2,), dtype=torch.bool),
                candidate_ever_grasped=ever_grasped,
            )

        first = torch.zeros((2, 8), dtype=torch.bool)
        first[0] = True
        second = torch.zeros((2, 8), dtype=torch.bool)
        merged = concatenate_collector_rounds([make(first), make(second)])
        self.assertEqual(tuple(merged[8].shape), (4, 8))
        self.assertEqual(int(merged[8].sum().item()), 8)
        # Absent on any round disables it rather than guessing, and the gate
        # then skips the instruction instead of feeding the curriculum a zero
        # it would ratchet down on.
        partial = concatenate_collector_rounds([make(first), make(None)])
        self.assertIsNone(partial[8])

    def test_the_grasp_gate_counts_only_normal_start_groups(self):
        """A pre-grasped world grasps at env step 0 by construction.

        Counting it would make the approach cap promote on the pre-grasped
        fraction knob rather than on anything the policy learned.
        """

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            instruction_outcome_counts,
        )

        successes = torch.zeros((2, 4), dtype=torch.bool)
        ever_grasped = torch.ones((2, 4), dtype=torch.bool)
        counts = instruction_outcome_counts(
            successes,
            torch.zeros((2,), dtype=torch.int64),
            {"pick_up": 0},
            torch.tensor([True, False]),
            ever_grasped,
        )
        # Group 0 is pre-grasped and excluded; only group 1's four candidates
        # count, and all four grasped.
        self.assertEqual(counts["instruction_worlds_normal_start/pick_up"], 4.0)
        self.assertEqual(counts["instruction_grasps_normal_start/pick_up"], 4.0)

    def test_the_grasp_key_is_absent_when_no_outcome_is_supplied(self):
        """Absence has to be visible, or the gate divides by a missing count."""

        import torch

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            instruction_outcome_counts,
        )

        counts = instruction_outcome_counts(
            torch.zeros((2, 4), dtype=torch.bool),
            torch.zeros((2,), dtype=torch.int64),
            {"pick_up": 0},
            torch.zeros((2,), dtype=torch.bool),
        )
        self.assertNotIn("instruction_grasps_normal_start/pick_up", counts)

    def test_the_new_metric_names_survive_the_rank_reduction(self):
        """Means are divided back down at the update boundary; counts are not."""

        divided = ("_time_s", "_mean", "_max", "_std", "_rate")
        names = {
            "post_grasp_first_env_step_mean_prelifted": True,
            "post_grasp_rise_mean_m_prelifted": True,
            "post_grasp_worlds_prelifted": False,
            "prelifted_start_rate": True,
            # Counts. A success count divided by the world size would read as a
            # rate and quietly understate the split by a factor of 1024.
            "successes_prelifted": False,
            "worlds_prelifted": False,
            "successes_normal_start": False,
            "worlds_normal_start": False,
            "degenerate_reward_groups": False,
            "degenerate_reward_groups_prelifted": False,
            # Means across ranks.
            "group_reward_std_mean": True,
            "group_reward_std_mean_prelifted": True,
            "group_reward_std_mean_normal_start": True,
            # Counts again: these two drive the approach curriculum's gate, and
            # dividing them by the world size would leave the rate unchanged but
            # the logged counts wrong by 1024x.
            "instruction_successes_normal_start/pick_up": False,
            "instruction_worlds_normal_start/pick_up": False,
            # Residual telemetry: two means and a rate, all divided.
            "residual_action_norm_mean": True,
            "prior_action_norm_mean": True,
            "residual_target_cosine_mean": True,
            "residual_target_alignment_rate": True,
            "object_press_depth_mean_m": True,
            "object_press_depth_mean_m_prelifted": True,
        }
        for name, should_divide in names.items():
            with self.subTest(metric=name):
                matched = name.endswith(divided) or "_mean_" in name
                self.assertEqual(matched, should_divide)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "the log_std projection operates on tensors",
)
class LogStdProjectionTests(unittest.TestCase):
    """clamp() has no gradient outside its bounds, so a drifted entry is dead.

    Dumping the actor after a 3.6M-step run found all twenty TRAINED log_std
    entries outside the [-1.45, -1.10] band -- fifteen at the ceiling, five at
    the floor, none inside -- with raw values from -3.51 to -0.2990, and eight
    sitting at -0.2996, the move-to ceiling of -0.30 that 7d99c3e replaced.
    That is why log_std_mean was EXACTLY -1.193750 on every update of five
    consecutive runs: a fixed pattern of saturated bounds, not a frozen
    parameter, and it meant entropy_coef could not do anything at all.
    """

    @staticmethod
    def _policy(torch, **kwargs):
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            SmolVLAGRPOPolicy,
        )

        return SmolVLAGRPOPolicy(
            state_dim=6, chunk_size=8, action_dim=5, hidden_dim=16,
            residual_scale=1.0, init_log_std=-1.2,
            min_log_std=-1.45, max_log_std=-1.10, **kwargs
        )

    def test_clamp_alone_kills_the_gradient_outside_the_band(self):
        """The bug, pinned so it cannot come back by a different route."""

        import torch

        policy = self._policy(torch)
        with torch.no_grad():
            policy.log_std.fill_(-3.0)          # drifted far below the floor
        policy.clamped_log_std().sum().backward()
        self.assertAlmostEqual(
            float(policy.log_std.grad.abs().sum().item()), 0.0, places=8,
            msg="an entry outside the clamp receives no gradient, so nothing "
                "can ever bring it back -- which is the whole failure",
        )

    def test_projection_returns_the_parameter_to_the_band(self):
        import torch

        policy = self._policy(torch)
        with torch.no_grad():
            policy.log_std[0].fill_(-3.5124)    # observed raw values
            policy.log_std[1].fill_(-0.2996)
        policy.project_log_std()
        raw = policy.log_std.detach()
        self.assertTrue(bool((raw >= -1.45 - 1e-6).all().item()))
        self.assertTrue(bool((raw <= -1.10 + 1e-6).all().item()))
        # And once back on the boundary the gradient is live again.
        policy.log_std.grad = None
        policy.clamped_log_std().sum().backward()
        self.assertGreater(float(policy.log_std.grad.abs().sum().item()), 0.0)

    def test_the_saturated_fraction_reproduces_the_observed_constant(self):
        """The diagnostic must flag the exact state the run was in."""

        import torch

        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            _log_std_saturated_fraction,
        )

        policy = self._policy(torch)
        observed = torch.tensor([
            [-3.5124, -0.7563, -0.2996, -0.2996, -1.0996],
            [-2.8856, -0.7427, -0.2996, -0.2995, -0.5588],
            [-2.9693, -1.0999, -0.2995, -0.4795, -0.6278],
            [-2.4415, -0.2996, -0.2999, -0.2990, -2.8820],
            [-1.2] * 5, [-1.2] * 5, [-1.2] * 5, [-1.2] * 5,
        ])
        with torch.no_grad():
            policy.log_std.copy_(observed)
        # Every trained entry saturated; rows 4-7 hold init and are never used.
        self.assertAlmostEqual(
            _log_std_saturated_fraction(policy), 0.5, places=6
        )
        # And this is the number that was logged, to six decimals.
        self.assertAlmostEqual(
            float(policy.clamped_log_std().mean().item()), -1.193750, places=6
        )
        policy.project_log_std()
        self.assertAlmostEqual(
            _log_std_saturated_fraction(policy), 0.5, places=6,
            msg="projection puts entries ON the boundary, which still counts "
                "as saturated -- it restores the gradient, it does not hide it",
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "the curricula are imported from the trainer module",
)
class StartDistanceLadderTests(unittest.TestCase):
    """A single increment makes the first promotion the largest relative jump.

    0.03 -> 0.05 is +67%, and the measured pass rate fell 0.400 -> 0.180 on the
    update after it, then held ~0.21 for 9M steps across two runs without ever
    recovering to the 0.30 gate.
    """

    LADDER = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]

    def _curriculum(self, **overrides):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            ApproachDistanceCurriculum,
        )

        metadata = {
            "random_workspace_start_distance_curriculum_enabled": True,
            "random_workspace_start_distance_ladder": self.LADDER,
            "random_workspace_start_distance_promote_pass_rate": 0.30,
            "random_workspace_start_distance_demote_pass_rate": 0.12,
            "random_workspace_start_distance_cooldown_updates": 1,
            "random_workspace_start_distance_pass_rate_ema_decay": 0.0,
        }
        metadata.update(overrides)
        return ApproachDistanceCurriculum(metadata)

    def test_promotion_walks_the_ladder_one_rung_at_a_time(self):
        curriculum = self._curriculum()
        self.assertAlmostEqual(curriculum.current_cap(), 0.03)
        seen = [curriculum.current_cap()]
        for _ in range(30):
            curriculum.observe(0.9)          # always passing
            if curriculum.current_cap() != seen[-1]:
                seen.append(curriculum.current_cap())
        for value in seen:
            self.assertIn(round(value, 6), [round(v, 6) for v in self.LADDER])
        self.assertAlmostEqual(seen[1], 0.04, places=6)
        self.assertAlmostEqual(seen[-1], 0.20, places=6)
        # The first step is +33%, not the +67% a flat 0.02 increment gave.
        self.assertAlmostEqual((seen[1] - seen[0]) / seen[0], 1.0 / 3.0, places=3)

    def test_demotion_walks_back_down_the_same_rungs(self):
        curriculum = self._curriculum()
        for _ in range(12):
            curriculum.observe(0.9)
        top = curriculum.current_cap()
        self.assertGreater(top, 0.03)
        for _ in range(30):
            curriculum.observe(0.0)          # always failing
        self.assertAlmostEqual(curriculum.current_cap(), 0.03, places=6)

    def test_a_resumed_cap_snaps_onto_a_rung(self):
        """A cap earned under the old flat increment must not keep its spacing."""

        curriculum = self._curriculum()
        curriculum.load_state_dict({"cap": 0.07, "pass_rate_ema": 0.2})
        self.assertIn(
            round(curriculum.current_cap(), 6),
            [round(v, 6) for v in self.LADDER],
        )
        self.assertAlmostEqual(curriculum.current_cap(), 0.06, places=6)

    def test_without_a_ladder_the_flat_increment_is_unchanged(self):
        curriculum = self._curriculum(
            random_workspace_start_distance_ladder=None,
            random_workspace_start_distance_initial=0.03,
            random_workspace_start_distance_final=0.20,
            random_workspace_start_distance_increment=0.02,
        )
        self.assertEqual(curriculum.ladder, ())
        curriculum.observe(0.9)
        self.assertAlmostEqual(curriculum.current_cap(), 0.05, places=6)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "the curricula are imported from the trainer module",
)
class PreliftedStageCurriculumTests(unittest.TestCase):
    """Practice should follow the deficit, not stay where it was set.

    At a fixed 0.5 the pre-grasped stage spent half of every batch on a
    sub-task measuring 0.79-0.82 while the full task sat at 0.21.
    """

    def _curriculum(self, **overrides):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            PreliftedStageCurriculum,
        )

        metadata = {
            "pick_up_prelifted_curriculum_enabled": True,
            "pick_up_prelifted_group_fraction": 0.5,
            "pick_up_prelifted_fraction_min": 0.15,
            "pick_up_prelifted_fraction_step": 0.05,
            "pick_up_prelifted_reduce_success": 0.70,
            "pick_up_prelifted_restore_success": 0.45,
            "pick_up_prelifted_cooldown_updates": 1,
            "pick_up_prelifted_success_ema_decay": 0.0,
        }
        metadata.update(overrides)
        return PreliftedStageCurriculum(metadata)

    def test_a_learned_lift_shrinks_the_stage_to_its_floor(self):
        curriculum = self._curriculum()
        self.assertAlmostEqual(curriculum.current_fraction(), 0.5)
        for _ in range(40):
            curriculum.observe(0.82)         # the measured plateau
        self.assertAlmostEqual(curriculum.current_fraction(), 0.15, places=6)

    def test_a_decaying_lift_brings_the_stage_back(self):
        curriculum = self._curriculum()
        for _ in range(40):
            curriculum.observe(0.82)
        self.assertAlmostEqual(curriculum.current_fraction(), 0.15, places=6)
        for _ in range(40):
            curriculum.observe(0.20)         # the lift collapses again
        self.assertAlmostEqual(curriculum.current_fraction(), 0.5, places=6)

    def test_the_band_between_the_thresholds_holds_steady(self):
        curriculum = self._curriculum()
        for _ in range(40):
            curriculum.observe(0.55)         # between restore and reduce
        self.assertAlmostEqual(curriculum.current_fraction(), 0.5, places=6)

    def test_disabled_by_default_and_inert(self):
        curriculum = self._curriculum(
            pick_up_prelifted_curriculum_enabled=False
        )
        for _ in range(40):
            curriculum.observe(0.95)
        self.assertAlmostEqual(curriculum.current_fraction(), 0.5, places=6)

    def test_no_prelifted_worlds_is_not_an_observation(self):
        """A batch with no pre-grasped groups must not read as zero success."""

        curriculum = self._curriculum()
        for _ in range(40):
            curriculum.observe(None)
        self.assertAlmostEqual(curriculum.current_fraction(), 0.5, places=6)

    def test_state_survives_a_resume(self):
        curriculum = self._curriculum()
        for _ in range(40):
            curriculum.observe(0.82)
        state = curriculum.state_dict()
        resumed = self._curriculum()
        resumed.load_state_dict(state)
        self.assertAlmostEqual(
            resumed.current_fraction(), curriculum.current_fraction(), places=6
        )
