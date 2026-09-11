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
    projected_grasp_xy_offset,
    read_manifest,
    scene_counts,
    scene_object_quaternion,
    validate_scene,
    write_manifest,
)

# The calibrated fixed pickup yaw the screens ran at. Every scene test that
# exercises the clearance filter uses this rather than a round number, because
# the filter's whole point is that the manifest and the collector agree on ONE
# measured angle.
CALIBRATED_PICKUP_YAW = -7.936838537690663e-15


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


class PickupClearanceFilterTests(unittest.TestCase):
    """The manifest may refuse a presentation the open fingers cannot bracket.

    The gripper's yaw is pinned for pickup while the object's is sampled, so an
    elongated catalog shows a different width to the closing axis in every
    scene. Roughly half of potato draws have NEGATIVE centring slack: no reach,
    however accurate, can be followed by a grasp, and the screens spent chains
    on them. The filter is OFF by default because the pickup yaw is a
    calibration rather than a property of the scene.
    """

    def test_it_is_off_by_default_so_old_manifests_still_validate(self):
        """A manifest written before this field existed must keep loading.

        read_manifest reconstructs SceneGeometryConfig from the stored payload,
        so a new field that defaulted to ON would retroactively reject every
        scene set already on disk.
        """

        self.assertIsNone(SceneGeometryConfig().clearance_pickup_yaw)
        scenes = generate_scenes(count=48, seed=11)
        blocked = [
            scene
            for scene in scenes
            if projected_grasp_xy_offset(
                scene.target_catalog,
                scene_object_quaternion(scene.target.yaw),
                CALIBRATED_PICKUP_YAW,
            ) <= 0.0
        ]
        # Not an assertion about the exact count -- an assertion that the
        # unfiltered generator really does emit these, which is what makes the
        # filter worth having.
        self.assertTrue(blocked)
        for scene in scenes:
            validate_scene(scene, SceneGeometryConfig())

    def test_every_accepted_target_is_bracketable_at_the_calibrated_yaw(self):
        config = SceneGeometryConfig(clearance_pickup_yaw=CALIBRATED_PICKUP_YAW)
        for scene in generate_scenes(count=96, seed=12, config=config):
            slack = projected_grasp_xy_offset(
                scene.target_catalog,
                scene_object_quaternion(scene.target.yaw),
                CALIBRATED_PICKUP_YAW,
                margin=config.grasp_xy_margin,
            )
            self.assertGreater(slack, 0.0, scene.scene_uid)

    def test_an_elongated_object_is_resampled_rather_than_dropped(self):
        """The census must not change. Balance is by cycling over catalogs, so
        a rejected potato yaw is redrawn; losing the stratum would trade one
        silent bias for another."""

        unfiltered = scene_counts(generate_scenes(count=96, seed=13))
        filtered = scene_counts(
            generate_scenes(
                count=96,
                seed=13,
                config=SceneGeometryConfig(
                    clearance_pickup_yaw=CALIBRATED_PICKUP_YAW
                ),
            )
        )
        self.assertEqual(unfiltered["by_target"], filtered["by_target"])
        self.assertEqual(unfiltered["by_destination"], filtered["by_destination"])
        self.assertIn("robocasa_potato", filtered["by_target"])

    def test_only_the_elongated_catalog_is_ever_rejected(self):
        """Spheres are yaw-invariant; if the filter touched them the projection
        would be wrong rather than the objects ungraspable."""

        config = SceneGeometryConfig(clearance_pickup_yaw=CALIBRATED_PICKUP_YAW)
        rejected = []
        for scene in generate_scenes(count=96, seed=14):
            try:
                validate_scene(scene, config)
            except SceneRejection:
                rejected.append(scene.target_catalog)
        self.assertTrue(rejected)
        self.assertEqual(set(rejected), {"robocasa_potato"})

    def test_the_filter_survives_a_manifest_round_trip(self):
        """The yaw a set was filtered against travels with it. A consumer
        running a different calibration is running an unfiltered manifest and
        has to be able to see that."""

        import tempfile

        config = SceneGeometryConfig(clearance_pickup_yaw=CALIBRATED_PICKUP_YAW)
        scenes = generate_scenes(count=32, seed=15, config=config)
        payload = manifest_payload(
            scenes,
            config=config,
            seed=15,
            split_weights=DEFAULT_SPLIT_WEIGHTS,
            split_salt="test",
        )
        self.assertAlmostEqual(
            payload["geometry"]["clearance_pickup_yaw"],
            CALIBRATED_PICKUP_YAW,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/scenes.json"
            write_manifest(path, payload)
            loaded, _ = read_manifest(path)
        self.assertEqual(len(loaded), len(scenes))

    def test_a_wider_margin_than_the_collector_would_admit_impossible_scenes(self):
        """grasp_xy_margin has to match the collector's readiness margin. A
        scene accepted against a wider margin hands the same impossible
        presentation back through the other door."""

        broadside = math.pi / 2.0
        strict = projected_grasp_xy_offset(
            "robocasa_potato",
            scene_object_quaternion(broadside),
            0.0,
            margin=0.003,
        )
        loose = projected_grasp_xy_offset(
            "robocasa_potato",
            scene_object_quaternion(broadside),
            0.0,
            margin=0.0,
        )
        self.assertLess(strict, loose)
        self.assertAlmostEqual(loose - strict, 0.003, places=9)


class SceneObjectQuaternionTests(unittest.TestCase):
    def test_it_matches_what_the_full_task_reset_writes(self):
        """The filter predicts feasibility at the COMMANDED orientation. If this
        helper and FullTaskSceneResetter ever disagree, the manifest would be
        filtering scenes that are not the ones the collector runs."""

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            _scene_slot_yaws,
        )

        scene = generate_scenes(count=1, seed=16)[0]
        yaws = _scene_slot_yaws(scene)
        self.assertAlmostEqual(yaws[0], scene.target.yaw, places=12)
        # The resetter builds (cos(yaw/2), 0, 0, sin(yaw/2)) per slot.
        for yaw in yaws:
            quaternion = scene_object_quaternion(yaw)
            self.assertAlmostEqual(quaternion[0], math.cos(0.5 * yaw), places=12)
            self.assertEqual(quaternion[1], 0.0)
            self.assertEqual(quaternion[2], 0.0)
            self.assertAlmostEqual(quaternion[3], math.sin(0.5 * yaw), places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
