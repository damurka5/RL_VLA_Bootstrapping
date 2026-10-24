"""The batched oracle's phase machine, driven against a kinematic stand-in.

The premise of the whole oracle-to-bank path is that the scripted chain can
complete a composed put_into: approach, close, lift, traverse, descend,
release. If it stalls in a phase it produces a demonstration of hovering, and
if it skips one it produces a demonstration missing a motion. Neither raises,
and neither is visible until an SFT has been trained on the result.

The plant here is deliberately crude -- a first-order response with the
measured ~0.5 realised gain, plus a kinematic grasp. It is not a physics check;
it exists to advance the state machine so the chain can be observed end to end.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tools.audit.sil_oracle import BatchedOracle

GRASP_OFFSET = 0.0075
STEP_XYZ = 0.015
STEP_GRIPPER = 0.05
FITTED = 0.35
PLANT_GAIN = 0.5


def _oracle(instruction: str, *, starts_grasped: bool) -> BatchedOracle:
    return BatchedOracle(
        instruction_types=[instruction],
        starts_grasped=[starts_grasped],
        instruction_texts=[f"put apple into {instruction.rsplit('_', 1)[-1]}"],
        target_slots=[0],
        reference_slots=[1],
        target_catalogs=["robocasa_apple"],
        fitted_openings=[FITTED],
        release_openings=[0.60],
        grasp_height_offset=GRASP_OFFSET,
        release_heights=[0.10],
        support_surface_z=[0.15],
        lift_success_height=0.05,
        action_step_xyz=STEP_XYZ,
        action_step_gripper=STEP_GRIPPER,
    )


def _rollout(oracle: BatchedOracle, *, steps: int, held: bool = False):
    """Advance the chain against a first-order plant. Returns phases seen."""

    ee = np.array([[0.0, 0.0, 0.32]])
    objects = np.zeros((1, 4, 3))
    objects[0, 0] = [0.10, 0.05, 0.185]          # the target on the desk
    objects[0, 1] = [0.16, 0.09, 0.150]          # the receptacle
    if held:
        objects[0, 0] = ee[0] - np.array([0.0, 0.0, GRASP_OFFSET])
    commanded = np.array([0.0 if held else 1.0])
    grasped = bool(held)
    previous = ee.copy()
    seen: list[str] = []
    for _ in range(steps):
        seen.append(oracle.phase_names()[0])
        action = oracle.actions(
            ee=ee,
            ee_velocity=ee - previous,
            ee_yaw=np.zeros(1),
            measured_gripper=commanded.copy(),
            commanded_gripper=commanded,
            object_positions=objects,
            physical_grasp=np.array([grasped]),
            initial_target_z=np.array([0.185]),
        )
        previous = ee.copy()
        ee = ee + action[:, :3] * STEP_XYZ * PLANT_GAIN
        commanded = np.clip(commanded + action[:, 4] * STEP_GRIPPER, 0.0, 1.0)
        close_enough = (
            float(np.linalg.norm(ee[0] - (objects[0, 0] + [0, 0, GRASP_OFFSET])))
            < 0.02
        )
        if not grasped and close_enough and commanded[0] <= FITTED:
            grasped = True
        if grasped and commanded[0] > 0.55:
            grasped = False                       # released
        if grasped:
            objects[0, 0] = ee[0] - np.array([0.0, 0.0, GRASP_OFFSET])
    return seen


class ComposedChainTests(unittest.TestCase):
    def test_the_composed_chain_is_the_grasp_then_the_carry(self) -> None:
        names = [p.name for p in _oracle("put_into_bowl", starts_grasped=False).phases[0]]
        self.assertEqual(
            names,
            [
                "align_above_object",
                "descend_to_grasp_point",
                "close_fingers",
                "raise_to_transit_height",
                "traverse_to_receptacle",
                "descend_to_release_height",
                "open_gripper_release",
                "settle_in_receptacle",
            ],
        )

    def test_a_caught_start_skips_the_grasp(self) -> None:
        names = [p.name for p in _oracle("put_into_plate", starts_grasped=True).phases[0]]
        self.assertNotIn("close_fingers", names)
        self.assertEqual(names[0], "raise_to_transit_height")

    def test_the_chain_advances_past_the_grasp_and_reaches_the_release(self) -> None:
        # The whole premise: a script can do what no policy here can.
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        seen = _rollout(oracle, steps=400)
        for phase in ("align_above_object", "descend_to_grasp_point",
                      "close_fingers", "raise_to_transit_height",
                      "traverse_to_receptacle"):
            self.assertIn(phase, seen, f"never reached {phase}")

    def test_no_phase_swallows_the_whole_episode(self) -> None:
        # A phase whose done() never fires yields a demonstration of hovering.
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        seen = _rollout(oracle, steps=400)
        longest = max(seen.count(name) for name in set(seen))
        self.assertLess(longest, 380, f"one phase held {longest}/400 steps")

    def test_phases_only_move_forward(self) -> None:
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        _rollout(oracle, steps=200)
        chain = [p.name for p in oracle.phases[0]]
        self.assertLessEqual(int(oracle.phase_index[0]), len(chain) - 1)
        self.assertGreater(int(oracle.phase_index[0]), 0)


class BatchBehaviourTests(unittest.TestCase):
    def test_an_unsupported_instruction_gets_a_zero_action(self) -> None:
        # Relation tasks have no oracle. Keep their batch slots intact.
        oracle = BatchedOracle(
            instruction_types=["move_left_of_object", "put_into_bowl"],
            starts_grasped=[False, False],
            instruction_texts=["move to apple", "put apple into bowl"],
            target_slots=[0, 0],
            reference_slots=[-1, 1],
            target_catalogs=["robocasa_apple"] * 2,
            fitted_openings=[FITTED] * 2,
            release_openings=[0.60] * 2,
            grasp_height_offset=GRASP_OFFSET,
            release_heights=[0.10] * 2,
            support_surface_z=[0.15] * 2,
            lift_success_height=0.05,
            action_step_xyz=STEP_XYZ,
            action_step_gripper=STEP_GRIPPER,
        )
        self.assertEqual(oracle.supported.tolist(), [False, True])
        objects = np.zeros((2, 4, 3))
        objects[:, 0] = [0.10, 0.05, 0.185]
        objects[:, 1] = [0.16, 0.09, 0.150]
        action = oracle.actions(
            ee=np.array([[0.0, 0.0, 0.30]] * 2),
            ee_velocity=np.zeros((2, 3)),
            ee_yaw=np.zeros(2),
            measured_gripper=np.ones(2),
            commanded_gripper=np.ones(2),
            object_positions=objects,
            physical_grasp=np.zeros(2, dtype=bool),
            initial_target_z=np.full(2, 0.185),
        )
        self.assertEqual(action.shape, (2, 5))
        self.assertTrue(bool((action[0] == 0).all()))
        self.assertTrue(bool(np.abs(action[1]).sum() > 0))

    def test_move_to_tracks_the_named_slot_at_the_configured_hover_height(self):
        oracle = BatchedOracle(
            instruction_types=["move_to_object"], starts_grasped=[False],
            instruction_texts=["move to apple"], target_slots=[2], reference_slots=[-1],
            target_catalogs=["robocasa_apple"], fitted_openings=[FITTED],
            release_openings=[.95], grasp_height_offset=0.0, release_heights=[.07],
            support_surface_z=[.1502], lift_success_height=.05,
            action_step_xyz=STEP_XYZ, action_step_gripper=STEP_GRIPPER,
            move_to_approach_z=.23, workspace_z_bounds=(.20, .27),
        )
        self.assertEqual(oracle.supported.tolist(), [True])
        objects = np.zeros((1, 4, 3))
        objects[0, 2] = [.10, -.05, .17]
        ee = np.array([[0.0, 0.0, .25]])
        previous = ee.copy()
        for _ in range(96):
            action = oracle.actions(
                ee=ee, ee_velocity=ee-previous, ee_yaw=np.zeros(1),
                measured_gripper=np.ones(1), commanded_gripper=np.ones(1),
                object_positions=objects, physical_grasp=np.zeros(1, bool), initial_target_z=np.array([.17]),
            )
            previous = ee.copy()
            ee += action[:, :3] * STEP_XYZ * PLANT_GAIN
        np.testing.assert_allclose(ee[0], [.10, -.05, .23], atol=.002)

    def test_placement_advances_to_release_under_widowx_height_limit(self):
        oracle = BatchedOracle(
            instruction_types=["put_into_bowl"], starts_grasped=[True],
            instruction_texts=["put apple into bowl"], target_slots=[0], reference_slots=[1],
            target_catalogs=["robocasa_apple"], fitted_openings=[.7], release_openings=[.95],
            grasp_height_offset=0.0, release_heights=[.07], support_surface_z=[.1502],
            lift_success_height=.05, action_step_xyz=STEP_XYZ, action_step_gripper=STEP_GRIPPER,
            workspace_z_bounds=(.168, .27),
        )
        ee = np.array([[0.0, 0.0, .22]])
        previous = ee.copy()
        objects = np.zeros((1, 4, 3))
        objects[0, 1] = [.10, .02, .159]
        commanded = np.array([.5])
        seen = []
        for _ in range(128):
            objects[0, 0] = ee[0]
            action = oracle.actions(
                ee=ee, ee_velocity=ee-previous, ee_yaw=np.zeros(1),
                measured_gripper=commanded.copy(), commanded_gripper=commanded,
                object_positions=objects, physical_grasp=np.ones(1, bool), initial_target_z=np.array([.17]),
            )
            seen.extend(oracle.phase_names())
            previous = ee.copy()
            ee += action[:, :3] * STEP_XYZ * PLANT_GAIN
            ee[:, 2] = ee[:, 2].clip(.168, .27)
            commanded = np.clip(commanded + action[:, 4] * STEP_GRIPPER, 0, 1)
        self.assertEqual(oracle.contexts[0].frozen["transit_z"], .27)
        self.assertIn("open_gripper_release", seen)
        self.assertIn("settle_in_receptacle", seen)

    def test_move_to_reaches_each_widowx_cell_with_the_native_controller(self):
        try:
            import mujoco as mj
        except ImportError:
            self.skipTest("MuJoCo is not installed")
        from robots.widowx200.widowx200_mujoco import workspace as workspace
        from robots.widowx200.widowx200_mujoco.controller import (
            WidowX200ControlSpec, WidowX200MountPose, WidowX200TaskSpaceController,
        )
        from tools.train.run_widowx200_pipeline import load_yaml

        root = Path(__file__).resolve().parents[1]
        config = load_yaml(root / "configs/examples/widowx200_smolvla_pipeline.yaml")
        metadata = config["task"]["metadata"]
        model = mj.MjModel.from_xml_path(str(root / "robots/widowx200/widowx200_mujoco/wx200_scene.xml"))
        model.opt.timestep = .002
        layout = workspace.DEFAULT_LAYOUT
        workspace.mount_widowx200(model, layout, mujoco=mj)
        spec = WidowX200ControlSpec(
            mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
            workspace_x=layout.workspace_x, workspace_y=layout.workspace_y,
            workspace_z=(.20, .27), min_reach_radius=layout.min_reach_radius,
        )
        for xy in metadata["scene_object_cells"]:
            with self.subTest(cell=xy):
                data = mj.MjData(model)
                controller = WidowX200TaskSpaceController(model, data, spec=spec, mujoco=mj)
                controller.reset_to_pose([xy[0] + .08, xy[1] + .04, .25])
                oracle = BatchedOracle(
                    instruction_types=["move_to_object"], starts_grasped=[False],
                    instruction_texts=["move to apple"], target_slots=[0], reference_slots=[-1],
                    target_catalogs=["robocasa_apple"], fitted_openings=[.7], release_openings=[.95],
                    grasp_height_offset=0, release_heights=[.07], support_surface_z=[.1502],
                    lift_success_height=.05, action_step_xyz=spec.action_step_xyz,
                    action_step_gripper=spec.action_step_gripper,
                    move_to_approach_z=metadata["move_to_object_approach_z"], workspace_z_bounds=spec.workspace_z,
                )
                objects = np.zeros((1, 4, 3))
                objects[0, 0] = [*xy, .17]
                previous = controller.get_end_effector_position()
                success = False
                for _ in range(4 * metadata["random_workspace_horizon_high"]):
                    ee = controller.get_end_effector_position()
                    action = oracle.actions(
                        ee=ee[None], ee_velocity=(ee - previous)[None], ee_yaw=np.array([controller.get_yaw()]),
                        measured_gripper=np.array([controller.get_gripper_opening()]),
                        commanded_gripper=np.array([controller.get_gripper_target()]),
                        object_positions=objects, physical_grasp=np.zeros(1, bool), initial_target_z=np.array([.17]),
                    )
                    previous = ee
                    controller.apply_normalized_action(action[0])
                    measured = controller.get_end_effector_position()
                    if (np.linalg.norm(measured[:2] - xy) <= metadata["move_to_object_xy_tolerance"]
                            and metadata["move_to_object_z_window_low"] <= measured[2] <= metadata["move_to_object_z_window_high"]):
                        success = True
                        break
                self.assertTrue(success, f"oracle missed cell {xy}; final EE {measured}")

    def test_actions_stay_inside_the_normalised_range(self) -> None:
        oracle = _oracle("put_into_bowl", starts_grasped=False)
        ee = np.array([[0.4, -0.4, 0.5]])          # deliberately far
        objects = np.zeros((1, 4, 3))
        objects[0, 0] = [-0.2, 0.2, 0.185]
        objects[0, 1] = [0.16, 0.09, 0.150]
        action = oracle.actions(
            ee=ee, ee_velocity=np.zeros((1, 3)), ee_yaw=np.zeros(1),
            measured_gripper=np.ones(1), commanded_gripper=np.ones(1),
            object_positions=objects, physical_grasp=np.zeros(1, dtype=bool),
            initial_target_z=np.array([0.185]),
        )
        self.assertTrue(bool((action >= -1.0).all() and (action <= 1.0).all()))

    def test_worlds_hold_independent_phase_state(self) -> None:
        oracle = BatchedOracle(
            instruction_types=["put_into_bowl", "put_into_bowl"],
            starts_grasped=[False, True],
            instruction_texts=["put apple into bowl"] * 2,
            target_slots=[0, 0], reference_slots=[1, 1],
            target_catalogs=["robocasa_apple"] * 2,
            fitted_openings=[FITTED] * 2, release_openings=[0.60] * 2,
            grasp_height_offset=GRASP_OFFSET, release_heights=[0.10] * 2,
            support_surface_z=[0.15] * 2, lift_success_height=0.05,
            action_step_xyz=STEP_XYZ, action_step_gripper=STEP_GRIPPER,
        )
        self.assertEqual(
            oracle.phase_names(),
            ["align_above_object", "raise_to_transit_height"],
        )


if __name__ == "__main__":
    unittest.main()
