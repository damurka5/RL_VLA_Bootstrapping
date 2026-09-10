"""Full-task scene generation: the bounds are rejected against, not clamped.

Why each property is here.

A CENTRE-DISTANCE RULE IS NOT AN OVERLAP TEST. "The object is 0.10 m from the
bowl" says nothing about whether an apple is resting on the bowl's rim. The
generator compares HULLS, using rotation-invariant circumradii taken from the
catalog primitives, because the object's yaw is sampled.

THE HULL BOUND MUST BE TIGHT ENOUGH TO BE USABLE. Bounding a cylinder by its
axis-aligned box overstates the plate by 41% -- 0.129 m against 0.091 -- and
that alone makes a plate scene unplaceable inside the camera envelope. Round
shapes are bounded as round.

CLAMPING IS THE FAILURE MODE. The phase-4 container reset samples a distance and
then clamps the object into the workspace, so the realized distance is smaller
than the sampled one and the logged cap is not the cap that ran. Here a
proposal that misses its bounds is resampled, and the realized distances are
stored and re-derived on load.

THE OBJECT MUST START OUTSIDE THE GOAL. 92.6% of composed plate episodes under
the old reset began within the 0.091 m success radius, so the metric moved with
the reset rather than with the policy. Zero scenes may do that here.

SPLITS ARE KEYED TO A STABLE IDENTITY. A scene's split follows from a hash of
its realized geometry, so regenerating a manifest at a different count does not
move scenes across the line that separates teacher selection from the final
test.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from rl_vla_bootstrapping.simulation.cdpr_composition_scenes import (
    DEFAULT_SPLIT_WEIGHTS,
    SPLIT_NAMES,
    SceneGeometryConfig,
    SceneRejection,
    assign_split,
    catalog_xy_radius,
    generate_scenes,
    manifest_payload,
    read_manifest,
    scene_counts,
    validate_scene,
    write_manifest,
)


class HullRadiusTests(unittest.TestCase):
    def test_a_cylinder_is_bounded_as_a_cylinder(self):
        """The plate is 0.091 m of radius, not hypot(0.091, 0.091)."""

        self.assertAlmostEqual(catalog_xy_radius("robocasa_plate"), 0.091, places=6)

    def test_a_sphere_is_its_radius(self):
        self.assertAlmostEqual(catalog_xy_radius("robocasa_apple"), 0.0345, places=6)

    def test_the_bowl_walls_are_included(self):
        """Wider than the 0.057 success radius, which is a rim, not a hull."""

        radius = catalog_xy_radius("robocasa_bowl")
        self.assertGreater(radius, 0.057)
        # hypot(0.055 + 0.005, 0.052) = 0.0794 -- the far corner of a wall box.
        self.assertAlmostEqual(radius, math.hypot(0.060, 0.052), places=4)

    def test_a_rotated_capsule_follows_its_axis(self):
        """The potato's capsule lies along Y; a Z-axis assumption misses it."""

        radius = catalog_xy_radius("robocasa_potato")
        self.assertAlmostEqual(radius, 0.025 + 0.029, places=6)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = SceneGeometryConfig()
        self.scenes = generate_scenes(count=48, seed=11, config=self.config)

    def test_every_generated_scene_passes_its_own_validation(self):
        for scene in self.scenes:
            validate_scene(scene, self.config)

    def test_no_object_starts_inside_its_destination_region(self):
        for scene in self.scenes:
            self.assertGreater(
                scene.transport_xy_distance,
                scene.destination_success_radius + self.config.transport_margin
                - 1e-9,
            )

    def test_the_approach_is_outside_the_reaching_success_region(self):
        for scene in self.scenes:
            low, high = self.config.approach_xy_bounds
            self.assertGreaterEqual(scene.approach_xy_distance, low - 1e-9)
            self.assertLessEqual(scene.approach_xy_distance, high + 1e-9)

    def test_the_gripper_starts_open_and_clear(self):
        for scene in self.scenes:
            self.assertEqual(scene.gripper_opening, 1.0)
            self.assertGreaterEqual(
                scene.approach_xyz_distance,
                self.config.min_ee_target_xyz_distance - 1e-9,
            )

    def test_hulls_never_overlap(self):
        for scene in self.scenes:
            objects = scene.objects
            for first in range(len(objects)):
                for second in range(first + 1, len(objects)):
                    a, b = objects[first], objects[second]
                    gap = math.dist(a.xy, b.xy) - (
                        catalog_xy_radius(a.catalog) + catalog_xy_radius(b.catalog)
                    )
                    self.assertGreaterEqual(
                        gap, self.config.min_object_gap - 1e-9
                    )

    def test_every_object_is_inside_the_camera_envelope(self):
        for scene in self.scenes:
            for entry in scene.objects:
                reach = math.hypot(*entry.xy) + catalog_xy_radius(entry.catalog)
                self.assertLessEqual(
                    reach, self.config.camera_half_extent + 1e-9
                )

    def test_unused_slots_are_explicitly_disabled(self):
        for scene in self.scenes:
            ids = scene.catalog_ids()
            self.assertEqual(len(ids), 4)
            self.assertNotEqual(ids[0], -1)
            self.assertNotEqual(ids[1], -1)
            self.assertEqual(ids[2], -1)
            self.assertEqual(ids[3], -1)

    def test_destinations_and_objects_are_balanced(self):
        counts = scene_counts(self.scenes)
        self.assertEqual(counts["by_destination"]["plate"], 24)
        self.assertEqual(counts["by_destination"]["bowl"], 24)
        self.assertEqual(len(counts["by_target"]), 4)
        self.assertEqual(set(counts["by_target"].values()), {12})

    def test_generation_is_reproducible_from_the_seed(self):
        again = generate_scenes(count=48, seed=11, config=self.config)
        self.assertEqual(
            [scene.scene_uid for scene in again],
            [scene.scene_uid for scene in self.scenes],
        )

    def test_a_scene_identity_follows_the_geometry_not_a_counter(self):
        first = self.scenes[0]
        moved = replace(
            first, ee_xyz=(first.ee_xyz[0] + 0.01, *first.ee_xyz[1:])
        )
        self.assertEqual(moved.scene_index, first.scene_index)
        # The uid is a function of the geometry, so the moved copy keeps the
        # ORIGINAL uid only because `replace` does not recompute it -- which is
        # exactly why a consumer must validate rather than trust the label.
        with self.assertRaises(SceneRejection):
            validate_scene(moved, self.config)


class SplitTests(unittest.TestCase):
    def test_splits_are_disjoint_and_cover_every_scene(self):
        scenes = generate_scenes(count=64, seed=5)
        by_split: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
        for scene in scenes:
            by_split[scene.split].add(scene.scene_uid)
        total = sum(len(values) for values in by_split.values())
        self.assertEqual(total, len({scene.scene_uid for scene in scenes}))
        for first in SPLIT_NAMES:
            for second in SPLIT_NAMES:
                if first < second:
                    self.assertFalse(by_split[first] & by_split[second])

    def test_a_scene_keeps_its_split_when_the_manifest_grows(self):
        small = generate_scenes(count=16, seed=3)
        large = generate_scenes(count=64, seed=3)
        large_by_uid = {scene.scene_uid: scene.split for scene in large}
        for scene in small:
            if scene.scene_uid in large_by_uid:
                self.assertEqual(scene.split, large_by_uid[scene.scene_uid])

    def test_the_split_is_a_function_of_the_identity_alone(self):
        first = assign_split("scene_abc", DEFAULT_SPLIT_WEIGHTS, "salt")
        second = assign_split("scene_abc", DEFAULT_SPLIT_WEIGHTS, "salt")
        self.assertEqual(first, second)
        self.assertIn(first, SPLIT_NAMES)


class ManifestTests(unittest.TestCase):
    def test_a_manifest_round_trips_and_is_revalidated_on_load(self):
        import tempfile

        config = SceneGeometryConfig()
        scenes = generate_scenes(count=24, seed=17, config=config)
        payload = manifest_payload(
            scenes,
            config=config,
            seed=17,
            split_weights=DEFAULT_SPLIT_WEIGHTS,
            split_salt="salt",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/scenes.json"
            write_manifest(path, payload)
            restored, restored_payload = read_manifest(path)
        self.assertEqual(len(restored), len(scenes))
        self.assertEqual(
            [scene.scene_uid for scene in restored],
            [scene.scene_uid for scene in scenes],
        )
        self.assertEqual(
            restored_payload["manifest_sha256"], payload["manifest_sha256"]
        )

    def test_a_hand_edited_manifest_is_refused_on_load(self):
        import json
        import tempfile

        config = SceneGeometryConfig()
        scenes = generate_scenes(count=8, seed=23, config=config)
        payload = manifest_payload(
            scenes,
            config=config,
            seed=23,
            split_weights=DEFAULT_SPLIT_WEIGHTS,
            split_salt="salt",
        )
        # Someone widens the plate's success radius in the stored geometry to
        # make demonstrations easier to obtain. The scenes still carry the
        # radius they were generated against, so the load must object.
        payload["geometry"]["destination_success_radius"]["plate"] = 0.2
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/scenes.json"
            write_manifest(path, payload)
            with self.assertRaises(SceneRejection):
                read_manifest(path)


class RejectionTests(unittest.TestCase):
    def test_impossible_bounds_are_reported_rather_than_relaxed(self):
        config = SceneGeometryConfig(
            transport_margin=0.30, transport_xy_max=0.18
        )
        with self.assertRaises(SceneRejection) as caught:
            generate_scenes(count=1, seed=1, config=config)
        self.assertIn("margin", str(caught.exception))

    def test_an_unreachable_approach_band_is_reported(self):
        config = SceneGeometryConfig(
            approach_xy_bounds=(0.60, 0.70), max_attempts_per_scene=32
        )
        with self.assertRaises(SceneRejection):
            generate_scenes(count=1, seed=1, config=config)

    def test_a_non_graspable_target_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            SceneGeometryConfig(target_catalogs=("robocasa_plate",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
