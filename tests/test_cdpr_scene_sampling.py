from __future__ import annotations

import unittest

from robots.cdpr.cdpr_dataset.rl_cdpr_env import (
    SceneSpec,
    _configure_scene_sampling,
    _resolve_object_spawn_config,
)


class SceneSamplingTests(unittest.TestCase):
    def test_resolve_object_spawn_config_defaults_match_previous_central_window(self):
        config = _resolve_object_spawn_config({}, support_surface_z=0.15)

        self.assertEqual(config["xy_bounds"], ((-0.20, 0.20), (-0.20, 0.20), 0.15))
        self.assertEqual(config["min_gap"], 0.02)
        self.assertEqual(config["min_ee_dist"], 0.10)
        self.assertEqual(config["avoid_xy_center"], (0.0, 0.0))
        self.assertEqual(config["avoid_xy_radius"], 0.0)

    def test_resolve_object_spawn_config_honors_wide_off_center_metadata(self):
        metadata = {
            "goal_center_xy": [0.05, -0.03],
            "object_spawn_x_bounds": [-0.30, 0.32],
            "object_spawn_y_bounds": [-0.28, 0.31],
            "object_spawn_center_exclusion_radius": 0.17,
            "object_spawn_min_gap": 0.04,
            "object_spawn_min_ee_dist": 0.12,
            "object_spawn_max_tries": 320,
            "object_spawn_support_clearance": 0.006,
        }

        config = _resolve_object_spawn_config(metadata, support_surface_z=0.18)

        self.assertEqual(config["xy_bounds"], ((-0.30, 0.32), (-0.28, 0.31), 0.18))
        self.assertEqual(config["min_gap"], 0.04)
        self.assertEqual(config["min_ee_dist"], 0.12)
        self.assertEqual(config["max_tries"], 320)
        self.assertEqual(config["support_clearance"], 0.006)
        self.assertEqual(config["avoid_xy_center"], (0.05, -0.03))
        self.assertEqual(config["avoid_xy_radius"], 0.17)

    def test_configure_scene_sampling_builds_scene_variants_from_single_pool(self):
        base_scenes = [SceneSpec(name="desk", objects=("ycb_apple",))]
        metadata = {
            "scene_object_pool": ["ycb_apple", "plate", "ycb_spoon", "ycb_lemon"],
            "min_scene_objects": 1,
            "max_scene_objects": 3,
            "scene_variant_count": 12,
        }

        scenes, allowed, targets, distractors = _configure_scene_sampling(
            base_scenes=base_scenes,
            allowed_objects=("ycb_apple",),
            task_metadata=metadata,
            seed=5,
        )

        self.assertEqual(allowed, ("ycb_apple", "plate", "ycb_spoon", "ycb_lemon"))
        self.assertEqual(targets, ())
        self.assertEqual(distractors, ())
        self.assertGreaterEqual(len(scenes), 1)
        for scene in scenes:
            self.assertEqual(scene.name, "desk")
            self.assertIsNone(scene.target_object)
            self.assertGreaterEqual(len(scene.objects), 1)
            self.assertLessEqual(len(scene.objects), 3)
            self.assertEqual(len(scene.objects), len(set(scene.objects)))
            for name in scene.objects:
                self.assertIn(name, allowed)

    def test_configure_scene_sampling_builds_target_plus_distractors(self):
        base_scenes = [
            SceneSpec(name="desk", objects=("ycb_apple",)),
            SceneSpec(name="desk", objects=("ycb_apple", "ycb_peach")),
        ]
        metadata = {
            "target_object_pool": ["ycb_apple", "ycb_pear", "ycb_peach"],
            "distractor_object_pool": ["milk", "ketchup", "ycb_banana"],
            "min_scene_objects": 2,
            "max_scene_objects": 4,
            "scene_variant_count": 12,
        }

        scenes, allowed, targets, distractors = _configure_scene_sampling(
            base_scenes=base_scenes,
            allowed_objects=("ycb_apple",),
            task_metadata=metadata,
            seed=7,
        )

        self.assertEqual(targets, ("ycb_apple", "ycb_pear", "ycb_peach"))
        self.assertEqual(distractors, ("milk", "ketchup", "ycb_banana"))
        self.assertEqual(allowed, ("ycb_apple", "ycb_pear", "ycb_peach", "milk", "ketchup", "ycb_banana"))
        self.assertGreaterEqual(len(scenes), 3)
        for scene in scenes:
            self.assertEqual(scene.name, "desk")
            self.assertIsNotNone(scene.target_object)
            self.assertIn(scene.target_object, scene.objects)
            self.assertIn(scene.target_object, targets)
            self.assertGreaterEqual(len(scene.objects), 2)
            self.assertLessEqual(len(scene.objects), 4)
            self.assertEqual(len(scene.objects), len(set(scene.objects)))

    def test_configure_scene_sampling_can_require_container_reference(self):
        base_scenes = [SceneSpec(name="desk", objects=("ycb_apple",))]
        metadata = {
            "target_object_pool": ["ycb_apple", "ycb_pear"],
            "distractor_object_pool": ["ycb_peach"],
            "container_object_pool": ["plate", "bowl"],
            "min_scene_objects": 3,
            "max_scene_objects": 3,
            "scene_variant_count": 8,
        }

        scenes, allowed, targets, distractors = _configure_scene_sampling(
            base_scenes=base_scenes,
            allowed_objects=("ycb_apple",),
            task_metadata=metadata,
            seed=11,
        )

        self.assertIn("plate", allowed)
        self.assertIn("bowl", allowed)
        self.assertEqual(targets, ("ycb_apple", "ycb_pear"))
        self.assertEqual(distractors, ("ycb_peach",))
        for scene in scenes:
            self.assertTrue({"plate", "bowl"}.intersection(scene.objects))
            self.assertIn(scene.target_object, scene.objects)

    def test_configure_scene_sampling_uses_target_pool_as_distractors_when_not_provided(self):
        base_scenes = [SceneSpec(name="desk", objects=("ycb_apple",))]
        metadata = {
            "target_object_pool": ["ycb_apple", "ycb_pear", "ycb_peach"],
            "min_scene_objects": 1,
            "max_scene_objects": 3,
            "scene_variant_count": 12,
        }

        scenes, allowed, targets, distractors = _configure_scene_sampling(
            base_scenes=base_scenes,
            allowed_objects=("ycb_apple",),
            task_metadata=metadata,
            seed=13,
        )

        self.assertEqual(targets, ("ycb_apple", "ycb_pear", "ycb_peach"))
        self.assertEqual(distractors, ("ycb_apple", "ycb_pear", "ycb_peach"))
        self.assertEqual(allowed, ("ycb_apple", "ycb_pear", "ycb_peach"))
        self.assertTrue(any(len(scene.objects) > 1 for scene in scenes))
        for scene in scenes:
            self.assertIn(scene.target_object, targets)
            self.assertIn(scene.target_object, scene.objects)
            self.assertGreaterEqual(len(scene.objects), 1)
            self.assertLessEqual(len(scene.objects), 3)

    def test_configure_scene_sampling_falls_back_to_allowed_objects_without_metadata(self):
        base_scenes = [SceneSpec(name="desk", objects=("ycb_apple", "ycb_pear"))]

        scenes, allowed, targets, distractors = _configure_scene_sampling(
            base_scenes=base_scenes,
            allowed_objects=("ycb_apple", "ycb_pear"),
            task_metadata={},
            seed=0,
        )

        self.assertEqual(scenes, base_scenes)
        self.assertEqual(allowed, ("ycb_apple", "ycb_pear"))
        self.assertEqual(targets, ("ycb_apple", "ycb_pear"))
        self.assertEqual(distractors, ())


if __name__ == "__main__":
    unittest.main()
