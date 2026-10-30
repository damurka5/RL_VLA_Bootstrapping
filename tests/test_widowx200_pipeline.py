"""The WidowX-200 training pipeline: catalog switch, stage configs, stop rule,
staged reward, and the driver's checkpoint selection.

The MJWarp backend itself needs Warp and is exercised by
scripts/preflight_widowx200_pipeline.py (CUDA, or Warp's CPU device with
RLVLA_MJWARP_ALLOW_CPU=1). Everything here is CPU-only and fast.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG = ROOT / "configs" / "examples" / "widowx200_smolvla_pipeline.yaml"

try:  # pragma: no cover - availability, not behaviour
    import torch

    _TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH = False

try:  # pragma: no cover
    import yaml  # noqa: F401

    _YAML = True
except Exception:  # pragma: no cover
    _YAML = False

requires_torch = unittest.skipUnless(_TORCH, "torch is not installed.")
requires_yaml = unittest.skipUnless(_YAML, "PyYAML is not installed.")


def _catalog_probe(embodiment: str) -> dict:
    """Import the catalog in a fresh interpreter: it sizes itself at import."""

    code = (
        "import json\n"
        "from rl_vla_bootstrapping.simulation import cdpr_object_catalog as c\n"
        "v = c.OBJECT_VARIANTS\n"
        "print(json.dumps({'embodiment': c.ACTIVE_EMBODIMENT,"
        " 'squeeze': c.CAUGHT_START_GRIP_SQUEEZE,"
        " 'apple_radius': v['robocasa_apple'].primitives[0].size[0],"
        " 'apple_rest': v['robocasa_apple'].rest_height,"
        " 'apple_fitted': v['robocasa_apple'].fitted_gripper_opening,"
        " 'apple_mesh_scale': v['robocasa_apple'].visual_mesh_scale,"
        " 'plate_radius': v['robocasa_plate'].primitives[0].size[0],"
        " 'bowl_wall_x': v['robocasa_bowl'].primitives[1].pos[0],"
        " 'mug_mass': v['robocasa_mug'].mass,"
        " 'graspable_fitted': {k: v[k].fitted_gripper_opening for k in"
        "   ('robocasa_apple','robocasa_tomato','robocasa_orange','robocasa_potato','robocasa_bell_pepper')}}))\n"
    )
    env = dict(os.environ)
    env["RLVLA_EMBODIMENT"] = embodiment
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    output = subprocess.check_output(
        [sys.executable, "-c", code], cwd=ROOT, env=env, text=True, stderr=subprocess.DEVNULL
    )
    return json.loads(output.strip().splitlines()[-1])


class CatalogSwitchTest(unittest.TestCase):
    def test_cdpr_catalog_is_unchanged_by_default(self) -> None:
        probe = _catalog_probe("cdpr")
        self.assertEqual(probe["embodiment"], "cdpr")
        self.assertAlmostEqual(probe["squeeze"], 0.04)
        self.assertAlmostEqual(probe["apple_radius"], 0.0345)
        self.assertAlmostEqual(probe["apple_rest"], 0.03456)
        self.assertAlmostEqual(probe["apple_fitted"], 0.632)
        self.assertAlmostEqual(probe["apple_mesh_scale"], 1.0)

    def test_widowx_catalog_is_scaled_and_measured(self) -> None:
        probe = _catalog_probe("widowx200")
        self.assertEqual(probe["embodiment"], "widowx200")
        # ~42 mm minor width: inside the 30-56 mm band the gripper lifts, with
        # ~5 mm of pad clearance per side at the 52 mm full opening.
        self.assertAlmostEqual(2.0 * probe["apple_radius"], 0.0345 * 1.22, places=3)
        self.assertAlmostEqual(probe["apple_rest"], 0.03456 * 0.61)
        self.assertAlmostEqual(probe["apple_mesh_scale"], 0.61)
        self.assertAlmostEqual(probe["plate_radius"], 0.091 * 0.8)
        self.assertAlmostEqual(probe["bowl_wall_x"], 0.055 * 0.85)
        self.assertGreaterEqual(probe["mug_mass"], 0.04)
        self.assertAlmostEqual(probe["squeeze"], 0.20)
        for name, fitted in probe["graspable_fitted"].items():
            # Measured, never left at the 0.0 placeholder.
            self.assertGreater(fitted, 0.75, name)
            self.assertLess(fitted, 0.9, name)

    def test_unknown_embodiment_is_refused(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            _catalog_probe("ur5")


@requires_yaml
class PipelineConfigTest(unittest.TestCase):
    def _pipeline(self, *extra: str):
        from tools.train.run_widowx200_pipeline import Pipeline, parse_args

        directory = tempfile.mkdtemp(prefix="widowx200_pipeline_test_")
        return Pipeline(parse_args(["--config", str(PIPELINE_CONFIG), "--pipeline-dir", directory, "--dry-run", *extra]))

    def test_every_stage_config_parses_with_the_trainer(self) -> None:
        from rl_vla_bootstrapping.core.commands import append_cli_arg
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        pipeline = self._pipeline()
        stages = pipeline.spec["stages"]
        self.assertEqual(
            [s["name"] for s in stages],
            [
                "s1_move_to_dense",
                "s2_pick_up_dense",
                "s3_put_into_dense",
                "s4_move_to_binary",
                "s5_pick_up_binary",
                "s6_put_into_binary",
                "s7_joint_staged",
            ],
        )
        for stage in stages:
            config = pipeline.stage_config(stage)
            args = config["training"]["rl"]["args"]
            argv = [
                "--simulator-backend", "mjlab_mjwarp",
                "--mjwarp-xml-path", str(ROOT / "robots/widowx200/widowx200_mujoco/wx200_mjwarp_scene.xml"),
                "--worlds-per-rank", "512", "--groups-per-rank", "64",
                "--instruction-types", *config["task"]["instruction_types"],
            ]
            for key, value in args.items():
                append_cli_arg(argv, key, value)
            parsed = parse_args(argv)
            self.assertEqual(parsed.hold_steps, 24, stage["name"])
            self.assertEqual(parsed.max_train_steps, int(stage["budget_steps"]))
            self.assertEqual(parsed.early_stop_metric, stage["early_stop"]["metric"])
            # Identical residual architecture in every stage: the weights-only
            # handoff is a strict load.
            self.assertTrue(parsed.residual_vision_features)
            self.assertEqual(parsed.residual_vision_dim, 512)
            self.assertTrue(parsed.train_vla_lora)
            self.assertFalse(parsed.train_vla_vision_lora)

    def test_phases_set_the_reward_mode(self) -> None:
        pipeline = self._pipeline()
        by_name = {s["name"]: pipeline.stage_config(s) for s in pipeline.spec["stages"]}
        meta = {name: cfg["task"]["metadata"] for name, cfg in by_name.items()}
        self.assertFalse(meta["s2_pick_up_dense"].get("sparse_binary_reward", False))
        self.assertTrue(meta["s5_pick_up_binary"]["sparse_binary_reward"])
        self.assertTrue(meta["s7_joint_staged"]["staged_milestone_reward"])
        self.assertEqual(
            by_name["s7_joint_staged"]["task"]["instruction_types"],
            ["move_to_object", "pick_up", "put_into_plate", "put_into_bowl"],
        )
        # Grasp stages may go down to the pads' desk clearance; move_to may not.
        self.assertEqual(by_name["s1_move_to_dense"]["training"]["rl"]["args"]["controller_workspace_z_bounds"], [0.20, 0.27])
        self.assertEqual(by_name["s2_pick_up_dense"]["training"]["rl"]["args"]["controller_workspace_z_bounds"], [0.168, 0.27])
        for name, values in meta.items():
            self.assertEqual(values["pick_grasp_height_offset"], 0.0, name)
            self.assertEqual(len(values["scene_object_cells"]), 6, name)

    def test_generated_configs_load_and_resolve_paths(self) -> None:
        from rl_vla_bootstrapping.core.config import load_project_config

        pipeline = self._pipeline()
        stage = pipeline.spec["stages"][1]
        path = pipeline.write_config("unit_test_" + stage["name"], pipeline.stage_config(stage))
        try:
            config = load_project_config(path)
            xml = config.resolve_path(config.simulator.fixed_scene_xml)
            self.assertTrue(xml.is_file(), xml)
            self.assertEqual(xml.name, "wx200_mjwarp_scene.xml")
            self.assertEqual(tuple(config.task.instruction_types), ("pick_up",))
        finally:
            path.unlink(missing_ok=True)

    def test_perturbed_spawns_stay_inside_the_arms_reach(self) -> None:
        """Cells plus the worst-case scene shift and jitter, not the cells alone.

        The layout tests check the nominal lattice; the pipeline spawns up to
        `sqrt(2) * (shift + jitter) / 2` off it. At an inner radius of 0.16 that
        put 1.3% of evaluated targets inside the waist keep-out.
        """

        from robots.widowx200.widowx200_mujoco import workspace as W

        layout = W.DEFAULT_LAYOUT
        pipeline = self._pipeline()
        stages = [pipeline.stage_config(s) for s in pipeline.spec["stages"]]
        for config in stages:
            meta = config["task"]["metadata"]
            cells = [tuple(cell) for cell in meta["scene_object_cells"]]
            self.assertEqual(
                sorted(cells),
                sorted(tuple(round(v, 4) for v in cell) for cell in W.sector_lattice()),
                "config cells drifted from sector_lattice(DEFAULT_LAYOUT)",
            )
            spread = math.sqrt(2.0) * 0.5 * (
                float(meta["scene_object_scene_shift_m"]) + float(meta["scene_object_jitter_m"])
            )
            # Highest point a task commands above an object: the move_to hover
            # the reward pulls toward, the lift, and the release over a
            # receptacle. (The move_to window top, 0.245, is 1 mm past the 2 cm
            # reach margin for the outermost perturbed spawn -- inside the
            # margin, not past the arm -- and success needs only some z in it.)
            top = max(
                float(meta["move_to_object_approach_z"]),
                layout.base_position[2] + 0.03 + float(meta["pick_lift_success_height"]),
                layout.base_position[2] + 0.02 + float(meta["put_plate_release_height"]),
            )
            for x, y in cells:
                radius = math.hypot(x - layout.base_position[0], y - layout.base_position[1])
                self.assertGreaterEqual(radius - spread, layout.min_reach_radius, (x, y))
                self.assertLessEqual(radius + spread, W.usable_radius(top), (x, y))

    def test_eval_config_turns_training_aids_off(self) -> None:
        pipeline = self._pipeline()
        pick = pipeline.eval_config("pick_up")["task"]["metadata"]
        self.assertEqual(pick["pick_up_prelifted_group_fraction"], 0.0)
        self.assertEqual(pick["pick_up_aligned_group_fraction"], 0.0)
        caught = pipeline.eval_config("put_into")["task"]["metadata"]
        composed = pipeline.eval_config("put_into", composed=True)["task"]["metadata"]
        self.assertEqual(caught["placement_caught_object_fraction"], 1.0)
        self.assertEqual(composed["placement_caught_object_fraction"], 0.0)
        self.assertTrue(composed["placement_uncaught_object_at_cell"])


class DriverHelpersTest(unittest.TestCase):
    def test_deep_merge_replaces_lists_and_merges_maps(self) -> None:
        from tools.train.run_widowx200_pipeline import deep_merge

        base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
        merged = deep_merge(base, {"a": {"c": [3], "e": 2}})
        self.assertEqual(merged, {"a": {"b": 1, "c": [3], "e": 2}, "d": 1})
        self.assertEqual(base["a"]["c"], [1, 2])

    def test_best_checkpoint_follows_the_stage_metric(self) -> None:
        from tools.train.run_widowx200_pipeline import select_best_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            rl = Path(directory)
            rows = [
                (250000, {"validation/by_instruction/put_into_plate/success_rate": 0.5,
                          "validation/by_instruction/put_into_bowl/success_rate": 0.1}),
                (500000, {"validation/by_instruction/put_into_plate/success_rate": 0.4,
                          "validation/by_instruction/put_into_bowl/success_rate": 0.3}),
                (750000, {"validation/by_instruction/put_into_plate/success_rate": 0.9,
                          "validation/by_instruction/put_into_bowl/success_rate": 0.2}),
            ]
            with (rl / "validation.jsonl").open("w") as stream:
                for step, metrics in rows:
                    stream.write(json.dumps({"global_step": step, **metrics}) + "\n")
                    folder = rl / f"step_{step:07d}"
                    folder.mkdir()
                    (folder / "smolvla_grpo_adapter.pt").write_bytes(b"x")
            best = select_best_checkpoint(rl, "min_by_instruction")
            self.assertEqual(best["best_step"], 500000)
            self.assertAlmostEqual(best["best_value"], 0.3)


class EarlyStopTest(unittest.TestCase):
    def _rule(self, **kwargs):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import EarlyStopRule

        defaults = dict(
            early_stop_metric="validation/success_rate",
            early_stop_target=0.0,
            early_stop_patience=2,
            early_stop_plateau_validations=0,
            early_stop_min_delta=0.02,
            early_stop_min_steps=0,
        )
        defaults.update(kwargs)
        return EarlyStopRule(Namespace(**defaults))

    @requires_torch
    def test_disabled_by_default(self) -> None:
        rule = self._rule()
        self.assertIsNone(rule.update({"validation/success_rate": 1.0}, 10))

    @requires_torch
    def test_target_needs_consecutive_validations(self) -> None:
        rule = self._rule(early_stop_target=0.7)
        self.assertIsNone(rule.update({"validation/success_rate": 0.75}, 1))
        self.assertIsNone(rule.update({"validation/success_rate": 0.60}, 2))
        self.assertIsNone(rule.update({"validation/success_rate": 0.71}, 3))
        decision = rule.update({"validation/success_rate": 0.72}, 4)
        self.assertEqual(decision["reason"], "target")

    @requires_torch
    def test_plateau_and_min_steps(self) -> None:
        rule = self._rule(early_stop_plateau_validations=3, early_stop_min_steps=5)
        values = [0.10, 0.30, 0.31, 0.30, 0.32, 0.31]
        decisions = [rule.update({"validation/success_rate": v}, step) for step, v in enumerate(values, start=1)]
        self.assertIsNone(decisions[3])  # below min_steps
        self.assertEqual(decisions[5]["reason"], "plateau")

    def test_composed_metric_prefers_the_composed_leg(self) -> None:
        from rl_vla_bootstrapping.policy.early_stopping import early_stop_metric_value

        metrics = {
            "validation/by_instruction/move_to_object/success_rate": 0.8,
            "validation/by_instruction/put_into_plate/success_rate": 0.9,
            "validation_composed/by_instruction/put_into_plate/success_rate": 0.2,
        }
        self.assertAlmostEqual(early_stop_metric_value(metrics, "min_by_instruction"), 0.8)
        self.assertAlmostEqual(early_stop_metric_value(metrics, "min_by_instruction_composed"), 0.2)


@requires_torch
class StagedMilestoneRewardTest(unittest.TestCase):
    def test_fraction_of_ordered_milestones(self) -> None:
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            INSTRUCTION_TO_ID,
            BatchedCatchReleaseDenseReward,
            BatchedMoveToDistanceReward,
            BatchedTaskState,
            evaluate_active_sparse_tasks,
        )

        metadata = {
            "staged_milestone_reward": True,
            "put_plate_xy_tolerance": 0.073,
            "pick_grasp_height_offset": 0.0,
            "move_to_object_require_z_window": False,
        }
        move = BatchedMoveToDistanceReward.from_metadata(metadata)
        place = BatchedCatchReleaseDenseReward.from_metadata(metadata)
        self.assertTrue(move.staged_milestones and place.staged_milestones)
        self.assertEqual(place.distance_reward_weight, 0.0)

        # world 0: put_into_plate, grasped and lifted 6 cm, held over the plate
        # world 1: pick_up, gripper 1 cm from the object, nothing grasped
        # world 2: move_to, 4 cm away (near, not there)
        ids = torch.tensor([INSTRUCTION_TO_ID["put_into_plate"], INSTRUCTION_TO_ID["pick_up"], INSTRUCTION_TO_ID["move_to_object"]])
        objects = torch.zeros((3, 4, 3))
        objects[0, 0] = torch.tensor([0.0, 0.0, 0.23])  # held object, lifted
        objects[0, 1] = torch.tensor([0.01, 0.0, 0.158])  # plate under it
        objects[1, 0] = torch.tensor([0.1, 0.1, 0.17])
        objects[2, 0] = torch.tensor([0.2, 0.0, 0.17])
        ee = torch.tensor([[0.0, 0.0, 0.23], [0.1, 0.1, 0.18], [0.24, 0.0, 0.23]])
        state = BatchedTaskState(
            instruction_ids=ids,
            target_slots=torch.zeros(3, dtype=torch.int64),
            reference_slots=torch.tensor([1, -1, -1]),
            second_reference_slots=torch.full((3,), -1),
            initial_target_positions=torch.tensor([[0.2, 0.0, 0.17], [0.1, 0.1, 0.17], [0.2, 0.0, 0.17]]),
            ever_grasped=torch.tensor([False, False, False]),
            grasped=torch.tensor([False, False, False]),
            step_count=torch.zeros(3, dtype=torch.int64),
            release_threshold=torch.full((3,), 0.73),
            support_surface_z=torch.full((3,), 0.1502),
            target_rest_height=torch.full((3,), 0.017),
            peak_lift=torch.zeros(3),
            release_clearance=torch.full((3,), float("nan")),
        )
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee,
            object_positions=objects,
            gripper_opening=torch.tensor([0.5, 1.0, 1.0]),
            caught_target=torch.tensor([True, False, False]),
            active_mask=torch.ones(3, dtype=torch.bool),
            max_steps=100,
            move_to_distance_reward=move,
            catch_release_dense_reward=place,
        )
        rewards = result.rewards.tolist()
        # approach + grasp + lift + carry, not placed: 4/5
        self.assertAlmostEqual(rewards[0], 0.8, places=5)
        # approach only: 1/3
        self.assertAlmostEqual(rewards[1], 1.0 / 3.0, places=5)
        # near only: 1/2
        self.assertAlmostEqual(rewards[2], 0.5, places=5)
        self.assertFalse(bool(result.success.any()))
        # Latched: moving away keeps the credit.
        ee_far = ee.clone()
        ee_far[1, :2] += 0.2
        again = evaluate_active_sparse_tasks(
            state=state,
            ee_position=ee_far,
            object_positions=objects,
            gripper_opening=torch.tensor([0.5, 1.0, 1.0]),
            caught_target=torch.tensor([True, False, False]),
            active_mask=torch.ones(3, dtype=torch.bool),
            max_steps=100,
            move_to_distance_reward=move,
            catch_release_dense_reward=place,
        )
        self.assertAlmostEqual(again.rewards.tolist()[1], 1.0 / 3.0, places=5)


@requires_torch
class ReachProjectionTest(unittest.TestCase):
    def test_points_move_radially_into_the_annulus(self) -> None:
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import BatchedReverseFrontierResetter

        holder = type("Holder", (), {})()
        holder.torch = torch
        holder.reach_annulus = (0.0, 0.24, 0.16, 0.35)
        xy = torch.tensor([[0.0, 0.0], [0.0, 0.20], [0.40, -0.20], [0.1, 0.0]])
        out = BatchedReverseFrontierResetter._project_into_reach(holder, xy)
        radius = (out - torch.tensor([0.0, 0.24])).norm(dim=-1)
        self.assertTrue(bool(((radius >= 0.16 - 1e-6) & (radius <= 0.35 + 1e-6)).all()))
        self.assertTrue(torch.allclose(out[0], xy[0]))  # 0.24 from the base: inside
        self.assertTrue(torch.allclose(out[3], xy[3]))
        # The bearing survives the projection.
        before = xy[2] - torch.tensor([0.0, 0.24])
        after = out[2] - torch.tensor([0.0, 0.24])
        self.assertAlmostEqual(math.atan2(float(before[1]), float(before[0])), math.atan2(float(after[1]), float(after[0])), places=5)


if __name__ == "__main__":
    unittest.main()
