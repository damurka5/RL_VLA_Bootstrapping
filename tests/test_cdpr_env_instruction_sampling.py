from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv, SceneSpec


class EnvInstructionSamplingTests(unittest.TestCase):
    def test_uniform_cycle_sampling_covers_each_instruction_once_per_cycle(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.instruction_types = ("move_up", "move_down", "move_left")
        env.instruction_sampling = "uniform_cycle"
        env._instruction_cycle = []
        env.np_random = np.random.default_rng(3)

        first_cycle = [env._sample_instruction_type() for _ in range(3)]
        second_cycle = [env._sample_instruction_type() for _ in range(3)]

        self.assertEqual(set(first_cycle), {"move_up", "move_down", "move_left"})
        self.assertEqual(set(second_cycle), {"move_up", "move_down", "move_left"})

    def test_requested_instruction_type_bypasses_cycle(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.instruction_types = ("move_up", "move_down")
        env.instruction_sampling = "uniform_cycle"
        env._instruction_cycle = ["move_down"]
        env.np_random = np.random.default_rng(0)

        selected = env._sample_instruction_type(options={"instruction_type": "move_up"})

        self.assertEqual(selected, "move_up")
        self.assertEqual(env._instruction_cycle, ["move_down"])

    def test_episode_yaw_randomization_samples_within_bounds(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.randomize_ee_yaw = True
        env.ee_yaw_bounds = (-0.25, 0.25)
        env.np_random = np.random.default_rng(4)

        yaw = env._sample_episode_ee_yaw()

        self.assertGreaterEqual(yaw, -0.25)
        self.assertLessEqual(yaw, 0.25)

    def test_set_ee_yaw_updates_sim_qpos_immediately(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        data = type("Data", (), {"qpos": np.zeros(3, dtype=np.float32), "qvel": np.ones(3, dtype=np.float32)})()
        model = type("Model", (), {"jnt_dofadr": np.array([1], dtype=np.int32)})()
        calls: list[float] = []
        env.sim = type(
            "Sim",
            (),
            {
                "yaw_min": -1.0,
                "yaw_max": 1.0,
                "jnt_yaw_qadr": 2,
                "jnt_yaw": 0,
                "data": data,
                "model": model,
                "set_yaw": lambda self, yaw: calls.append(float(yaw)),
            },
        )()
        env._yaw = 0.0

        env._set_ee_yaw(0.42)

        self.assertEqual(calls, [0.42])
        self.assertAlmostEqual(float(data.qpos[2]), 0.42, places=6)
        self.assertAlmostEqual(float(data.qvel[1]), 0.0, places=6)
        self.assertAlmostEqual(env._yaw, 0.42, places=6)

    def test_direct_actuator_episodes_start_from_nontrivial_opposite_state(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {"direct_yaw_start_angle": 0.15}
        gripper_calls: list[float] = []
        yaw_calls: list[float] = []
        env._force_gripper_opening = lambda value: gripper_calls.append(float(value))
        env._set_ee_yaw = lambda value: yaw_calls.append(float(value))

        for instruction_type in (
            "open_gripper",
            "close_gripper",
            "rotate_gripper_clockwise",
            "rotate_gripper_counterclockwise",
        ):
            env._instruction_spec = type(
                "Instruction",
                (),
                {"instruction_type": instruction_type},
            )()
            env._initialize_direct_actuator_episode_state()

        self.assertEqual(gripper_calls, [0.0, 1.0])
        self.assertEqual(yaw_calls, [0.15, 0.15])

    def test_instruction_text_can_infer_type_and_object_bindings(self):
        from robots.cdpr.cdpr_dataset.rl_cdpr_env import (
            _infer_instruction_object_options,
            _infer_instruction_type_from_text,
        )

        self.assertEqual(_infer_instruction_type_from_text("put apple into plate"), "put_into_plate")
        self.assertEqual(_infer_instruction_type_from_text("push apple forward"), "push_forward")
        self.assertEqual(_infer_instruction_type_from_text("pick apple"), "pick_up")
        self.assertEqual(_infer_instruction_type_from_text("take apple"), "pick_up")
        self.assertEqual(_infer_instruction_type_from_text("catch apple"), "catch_object")
        self.assertEqual(_infer_instruction_type_from_text("free apple"), "free_object")
        self.assertEqual(_infer_instruction_type_from_text("rotate apple clockwise"), "rotate_clockwise")
        self.assertEqual(
            _infer_instruction_type_from_text("rotate apple counterclockwise"),
            "rotate_counterclockwise",
        )
        self.assertEqual(_infer_instruction_type_from_text("open the gripper"), "open_gripper")
        self.assertEqual(_infer_instruction_type_from_text("close the gripper"), "close_gripper")
        self.assertEqual(
            _infer_instruction_type_from_text("rotate the gripper clockwise"),
            "rotate_gripper_clockwise",
        )
        self.assertEqual(
            _infer_instruction_type_from_text("rotate the gripper counterclockwise"),
            "rotate_gripper_counterclockwise",
        )
        self.assertEqual(_infer_instruction_type_from_text("move apple in front of pear"), "move_in_front_of_object")
        self.assertEqual(
            _infer_instruction_object_options(
                "put apple into plate",
                candidate_catalogs=["ycb_apple", "plate", "ycb_baseball"],
            ),
            {"target_object": "ycb_apple", "reference_object": "plate"},
        )

    def test_sample_scene_can_filter_required_objects(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.scenes = [
            SceneSpec(name="desk_a", objects=("ycb_apple", "ycb_mug")),
            SceneSpec(name="desk_b", objects=("ycb_apple", "ycb_plate")),
        ]
        env.np_random = np.random.default_rng(0)

        scene = env._sample_scene(options={"required_objects": ["ycb_plate"]})

        self.assertEqual(scene.name, "desk_b")

    def test_move_to_object_can_reduce_scene_to_single_target_object(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.scenes = [
            SceneSpec(name="desk", objects=("ycb_apple", "ycb_mug"), target_object="ycb_apple"),
        ]
        env._task_metadata = {"move_to_object_single_object_scene": True}
        env.target_object_pool = ("ycb_pear",)
        env.scene_object_pool = ("plate",)
        env.allowed_objects = ("ycb_apple", "ycb_mug", "ycb_pear", "plate")
        env.np_random = np.random.default_rng(0)

        scene = env._sample_scene(options={"instruction_type": "move_to_object", "target_object": "ycb_pear"})

        self.assertEqual(scene.name, "desk")
        self.assertEqual(scene.objects, ("ycb_pear",))
        self.assertEqual(scene.target_object, "ycb_pear")

    def test_sample_scene_filters_put_instruction_to_container_scene(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.scenes = [
            SceneSpec(name="desk", objects=("ycb_apple", "ycb_pear")),
            SceneSpec(name="desk", objects=("ycb_apple", "plate")),
        ]
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple", "ycb_pear"],
            "container_object_pool": ["plate", "bowl"],
        }
        env.np_random = np.random.default_rng(0)

        scene = env._sample_scene(options={"scene": "desk", "instruction_type": "put_into_plate"})

        self.assertEqual(scene.objects, ("ycb_apple", "plate"))

    def test_sample_scene_augments_put_instruction_with_container(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.scenes = [
            SceneSpec(name="desk", objects=("ycb_apple", "ycb_pear")),
        ]
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple", "ycb_pear"],
            "container_object_pool": ["plate", "bowl"],
        }
        env.np_random = np.random.default_rng(0)

        scene = env._sample_scene(options={"instruction_type": "put_into_plate"})

        self.assertEqual(scene.objects, ("ycb_apple", "ycb_pear", "plate"))

    def test_sample_scene_augments_relation_instruction_with_catchable_target(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.scenes = [
            SceneSpec(name="desk", objects=("plate",)),
        ]
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple", "ycb_pear"],
            "container_object_pool": ["plate", "bowl"],
        }
        env.allowed_objects = ("ycb_apple", "ycb_pear", "plate", "bowl")
        env.np_random = np.random.default_rng(0)

        scene = env._sample_scene(options={"scene": "desk", "instruction_type": "move_right_of_object"})

        self.assertIn("plate", scene.objects)
        self.assertTrue({"ycb_apple", "ycb_pear"}.intersection(scene.objects))
        self.assertTrue(env._scene_supports_instruction_type(scene, "move_right_of_object"))

        env._catalog_to_body = {name: f"{name}_body" for name in scene.objects}
        target_catalog, _target_body, reference_catalog, _reference_body, _second_catalog, _second_body = (
            env._select_instruction_objects(scene, instruction_type="move_right_of_object")
        )
        self.assertIn(target_catalog, {"ycb_apple", "ycb_pear"})
        self.assertEqual(reference_catalog, "plate")

    def test_sample_scene_augments_between_instruction_to_three_objects(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.scenes = [
            SceneSpec(name="desk", objects=("plate",)),
        ]
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple"],
            "container_object_pool": ["plate", "bowl"],
            "distractor_object_pool": ["bowl"],
        }
        env.allowed_objects = ("ycb_apple", "plate", "bowl")
        env.np_random = np.random.default_rng(0)

        scene = env._sample_scene(options={"instruction_type": "move_between_objects"})

        self.assertEqual(len(set(scene.objects)), 3)
        self.assertIn("ycb_apple", scene.objects)
        self.assertTrue(env._scene_supports_instruction_type(scene, "move_between_objects"))

    def test_instruction_curriculum_filters_allowed_candidates_by_episode(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.instruction_types = ("grab_object", "push_left", "put_into_plate")
        env._task_metadata = {
            "instruction_curriculum": [
                {"until_episode": 2, "instruction_types": ["grab_object"]},
                {"until_episode": 4, "instruction_types": ["grab_object", "push_left"]},
                {"instruction_types": ["grab_object", "push_left", "put_into_plate"]},
            ],
        }

        env._episode_index = 0
        self.assertEqual(env._allowed_instruction_candidates(), ("grab_object",))

        env._episode_index = 3
        self.assertEqual(env._allowed_instruction_candidates(), ("grab_object", "push_left"))

        env._episode_index = 4
        self.assertEqual(
            env._allowed_instruction_candidates(),
            ("grab_object", "push_left", "put_into_plate"),
        )

    def test_put_instruction_uses_catchable_target_and_container_reference(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple", "ycb_pear"],
            "container_object_pool": ["plate", "bowl"],
        }
        env._catalog_to_body = {
            "ycb_apple": "apple_body",
            "plate": "plate_body",
            "bowl": "bowl_body",
        }
        env.np_random = np.random.default_rng(0)
        scene = SceneSpec(name="desk", objects=("ycb_apple", "plate", "bowl"))

        target_catalog, _target_body, reference_catalog, _reference_body, _second_catalog, _second_body = (
            env._select_instruction_objects(scene, instruction_type="put_into_plate")
        )

        self.assertEqual(target_catalog, "ycb_apple")
        self.assertIn(reference_catalog, {"plate", "bowl"})

    def test_relation_instruction_uses_catchable_target(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple"],
            "container_object_pool": ["plate", "bowl"],
        }
        env._catalog_to_body = {
            "ycb_apple": "apple_body",
            "plate": "plate_body",
            "bowl": "bowl_body",
        }
        env.np_random = np.random.default_rng(1)
        scene = SceneSpec(name="desk", objects=("plate", "ycb_apple", "bowl"))

        target_catalog, _target_body, reference_catalog, _reference_body, _second_catalog, _second_body = (
            env._select_instruction_objects(scene, instruction_type="put_in_front_of_object")
        )

        self.assertEqual(target_catalog, "ycb_apple")
        self.assertIn(reference_catalog, {"plate", "bowl"})

    def test_resolved_scene_support_requires_loaded_catchable_body(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "catchable_object_pool": ["ycb_apple"],
            "container_object_pool": ["plate", "bowl"],
        }
        scene = SceneSpec(name="desk", objects=("plate", "ycb_apple"))

        env._catalog_to_body = {"plate": "plate_body"}
        self.assertFalse(env._resolved_scene_supports_instruction_type(scene, "put_into_plate"))

        env._catalog_to_body = {"plate": "plate_body", "ycb_apple": "p1_ycb_apple"}
        self.assertTrue(env._resolved_scene_supports_instruction_type(scene, "put_into_plate"))

    def test_ycb_logical_body_aliases_include_stripped_name(self):
        from robots.cdpr.cdpr_dataset.synthetic_tasks import _logical_body_aliases

        self.assertEqual(_logical_body_aliases("ycb_apple"), ("ycb_apple", "apple"))
        self.assertEqual(_logical_body_aliases("plate"), ("plate",))

    def test_caught_object_start_spawns_target_at_end_effector_for_relation_task(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "caught_object_start_probability": 1.0,
            "caught_object_start_object_offset": [0.0, 0.0, -0.04],
            "caught_object_start_xy_jitter": 0.0,
            "caught_object_start_z_jitter": 0.0,
        }
        env.np_random = np.random.default_rng(0)
        env._support_surface_z = 0.15
        env._target_body_name = "apple_body"
        env._target_catalog_name = "ycb_apple"
        env.sim = object()
        env._get_ee_position = lambda: np.array([0.10, -0.05, 0.42], dtype=np.float32)
        env._force_gripper_opening = lambda target: None
        placed: dict[str, np.ndarray] = {}

        def _set_body_position(body_name, xyz):
            placed[str(body_name)] = np.asarray(xyz, dtype=np.float32).copy()
            return True

        env._set_body_position = _set_body_position
        env._reset_caught_object_start_state()

        spawned = env._maybe_spawn_target_caught_at_ee(instruction_type="move_between_objects")

        self.assertTrue(spawned)
        self.assertTrue(env._caught_object_start_active)
        self.assertEqual(env._caught_object_start_body, "apple_body")
        np.testing.assert_allclose(placed["apple_body"], np.array([0.10, -0.05, 0.38], dtype=np.float32))
        np.testing.assert_allclose(
            env._caught_object_start_ee_offset,
            np.array([0.0, 0.0, -0.04], dtype=np.float32),
            atol=1e-6,
        )

    def test_caught_object_start_pose_follows_closed_gripper_until_release(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "caught_object_start_pin_object": True,
            "caught_object_start_release_opening_threshold": 0.10,
        }
        env._caught_object_start_active = True
        env._caught_object_start_body = "apple_body"
        env._caught_object_start_catalog = "ycb_apple"
        env._caught_object_start_position = np.array([0.10, -0.05, 0.38], dtype=np.float32)
        env._caught_object_start_ee_offset = np.array([0.0, 0.0, -0.04], dtype=np.float32)
        env._get_ee_position = lambda: np.array([0.20, 0.10, 0.44], dtype=np.float32)
        env._get_gripper_target = lambda: 0.0
        env._get_gripper_opening = lambda: 0.0
        placed: dict[str, np.ndarray] = {}

        def _set_body_position(body_name, xyz):
            placed[str(body_name)] = np.asarray(xyz, dtype=np.float32).copy()
            return True

        env._set_body_position = _set_body_position

        self.assertTrue(env._maintain_caught_object_start_pose())
        np.testing.assert_allclose(placed["apple_body"], np.array([0.20, 0.10, 0.40], dtype=np.float32))

        env._get_gripper_target = lambda: 0.25
        self.assertFalse(env._maintain_caught_object_start_pose())
        self.assertFalse(env._caught_object_start_active)

    def test_caught_object_start_physical_mode_does_not_pin_body_pose(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "caught_object_start_pin_object": False,
            "caught_object_start_release_opening_threshold": 0.10,
        }
        env._caught_object_start_active = True
        env._caught_object_start_body = "apple_body"
        env._caught_object_start_catalog = "ycb_apple"
        env._caught_object_start_position = np.array([0.10, -0.05, 0.38], dtype=np.float32)
        env._caught_object_start_ee_offset = np.array([0.0, 0.0, -0.04], dtype=np.float32)
        env._caught_object_start_hold_offset = np.array([0.0, 0.0, -0.04], dtype=np.float32)
        env._get_gripper_target = lambda: 0.0
        env._get_gripper_opening = lambda: 0.0
        env._get_body_position = lambda body_name: np.array([0.12, -0.02, 0.36], dtype=np.float32)
        calls: list[tuple[str, np.ndarray]] = []
        env._set_body_position = lambda body_name, xyz: calls.append((str(body_name), np.asarray(xyz))) or True

        self.assertTrue(env._maintain_caught_object_start_pose())
        self.assertEqual(calls, [])
        np.testing.assert_allclose(
            env._caught_object_start_position,
            np.array([0.12, -0.02, 0.36], dtype=np.float32),
        )

    def test_caught_object_start_held_opening_counts_as_closed_until_release_margin(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "caught_object_start_release_opening_threshold": 0.01,
            "caught_object_start_release_opening_margin": 0.08,
        }
        env._caught_object_start_active = True
        env._caught_object_start_gripper_opening = 0.42
        env.sim = type("Sim", (), {"gripper_min": 0.0, "gripper_max": 1.0})()

        self.assertAlmostEqual(env._caught_object_start_release_opening_threshold(), 0.50, places=6)
        self.assertTrue(env._is_gripper_closed(0.49))
        self.assertFalse(env._is_gripper_closed(0.51))

    def test_caught_object_start_uses_measured_ycb_openings(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {}
        env._target_body_name = "p2_ycb_baseball"
        env._target_catalog_name = "ycb_baseball"
        env._inverse_catalog_to_body = {}

        self.assertAlmostEqual(
            env._caught_object_start_gripper_opening_for_body("p2_ycb_baseball"),
            0.8337,
            places=4,
        )

    def test_body_width_along_axis_uses_mesh_vertices(self):
        try:
            import mujoco as mj
        except Exception as exc:
            self.skipTest(f"MuJoCo is not available: {exc}")

        model = mj.MjModel.from_xml_string(
            """
            <mujoco>
              <asset>
                <mesh name="wedge"
                  vertex="-0.03 0 0 0.02 0 0 0.02 0.02 0 -0.03 0.02 0 0 0.01 0.02"
                  face="0 1 4 1 2 4 2 3 4 3 0 4 0 3 2 0 2 1"/>
              </asset>
              <worldbody>
                <body name="object" pos="0.1 0 0">
                  <geom name="object_geom" type="mesh" mesh="wedge"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env.sim = type("Sim", (), {"model": model, "data": data})()

        width = env._body_width_along_axis("object", np.array([1.0, 0.0, 0.0], dtype=np.float32))

        self.assertIsNotNone(width)
        self.assertAlmostEqual(float(width), 0.05, places=6)

    def test_caught_object_start_does_not_apply_to_grab_task_by_default(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {"caught_object_start_probability": 1.0}
        env.np_random = np.random.default_rng(0)
        env._target_body_name = "apple_body"

        self.assertFalse(env._should_spawn_target_caught_at_ee(instruction_type="grab_object"))

    def test_release_and_rotate_force_caught_start_by_default(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {}
        env.np_random = np.random.default_rng(0)
        env._target_body_name = "apple_body"

        self.assertTrue(env._should_spawn_target_caught_at_ee(instruction_type="release_object"))
        self.assertTrue(env._should_spawn_target_caught_at_ee(instruction_type="rotate_clockwise"))

    def test_empty_configured_caught_start_lists_disable_default_forced_starts(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "caught_object_start_probability": 1.0,
            "caught_object_start_instruction_types": [],
            "force_caught_object_start_instruction_types": [],
        }
        env.np_random = np.random.default_rng(0)
        env._target_body_name = "apple_body"

        self.assertFalse(env._should_spawn_target_caught_at_ee(instruction_type="release_object"))
        self.assertFalse(env._should_spawn_target_caught_at_ee(instruction_type="move_between_objects"))

    def test_pick_grab_catch_and_grip_spawn_target_at_gripper_by_default(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {}
        env.np_random = np.random.default_rng(0)
        env._target_body_name = "apple_body"

        self.assertTrue(env._should_spawn_target_at_gripper(instruction_type="pick_up"))
        self.assertTrue(env._should_spawn_target_at_gripper(instruction_type="grab_object"))
        self.assertTrue(env._should_spawn_target_at_gripper(instruction_type="catch_object"))
        self.assertTrue(env._should_spawn_target_at_gripper(instruction_type="grip_object"))
        self.assertFalse(env._should_spawn_target_at_gripper(instruction_type="move_to_object"))

    def test_target_at_gripper_instruction_can_start_near_table_surface(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "target_at_gripper_start_instruction_types": ["pick_up", "grab_object"],
            "target_at_gripper_start_ee_z": 0.24,
        }
        env.defaults = {"ee_start": [0.0, 0.0, 0.40]}
        env.ee_start_z = None
        env.randomize_ee_start = False
        env.ee_start_x_bounds = (-1.0, 1.0)
        env.ee_start_y_bounds = (-1.0, 1.0)
        env._ee_min_z = float("-inf")

        pick_start = env._sample_episode_ee_start(options={"instruction_type": "pick_up"})
        release_start = env._sample_episode_ee_start(options={"instruction_type": "release_object"})

        self.assertAlmostEqual(float(pick_start[2]), 0.24, places=6)
        self.assertAlmostEqual(float(release_start[2]), 0.40, places=6)

    def test_empty_configured_target_at_gripper_list_disables_default_catch_start(self):
        env = CDPRLanguageRLEnv.__new__(CDPRLanguageRLEnv)
        env._task_metadata = {
            "target_at_gripper_start_probability": 1.0,
            "target_at_gripper_start_instruction_types": [],
        }
        env.np_random = np.random.default_rng(0)
        env._target_body_name = "apple_body"

        self.assertFalse(env._should_spawn_target_at_gripper(instruction_type="catch_object"))


if __name__ == "__main__":
    unittest.main()
