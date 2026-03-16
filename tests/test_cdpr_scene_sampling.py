from __future__ import annotations

import unittest

from robots.cdpr.cdpr_dataset.rl_cdpr_env import SceneSpec, _configure_scene_sampling


class SceneSamplingTests(unittest.TestCase):
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
