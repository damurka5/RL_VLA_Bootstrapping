"""Wide loaded grips earn credit; loaded fingers cannot count as release."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    RankLocalMJWarpGRPOCollector, ThreeStageMilestones, advance_three_stage_milestones,
)
from rl_vla_bootstrapping.simulation.cdpr_backend import CDPRFingerContactBatch
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    BatchedTaskState, BatchedCatchReleaseDenseReward, INSTRUCTION_TO_ID,
    evaluate_active_sparse_tasks,
)
from tools.audit.sil_record import _Recording, _episode_rows
from tools.audit.placement_failure_decomposition import _Thresholds, _episode_terms


def task(instruction="put_into_bowl", worlds=1, contact=True):
    return BatchedTaskState(
        instruction_ids=torch.full((worlds,), INSTRUCTION_TO_ID[instruction]),
        target_slots=torch.zeros(worlds, dtype=torch.int64),
        reference_slots=torch.ones(worlds, dtype=torch.int64),
        second_reference_slots=torch.full((worlds,), -1, dtype=torch.int64),
        initial_target_positions=torch.tensor([[0.,0.,.17]]).repeat(worlds,1),
        ever_grasped=torch.zeros(worlds,dtype=torch.bool), grasped=torch.zeros(worlds,dtype=torch.bool),
        step_count=torch.zeros(worlds,dtype=torch.int64), release_threshold=torch.full((worlds,),.90),
        support_surface_z=torch.full((worlds,),.15), target_rest_height=torch.full((worlds,),.02),
        peak_lift=torch.zeros(worlds),release_clearance=torch.full((worlds,),float("nan")),
        contact_based_grasp=contact,
    )


def score(state, *, z=.23, caught=True, loaded=True, opening=.99, active=True, reference_x=.3):
    objects=torch.tensor([[[0.,0.,z],[reference_x,0.,.15]]])
    result=evaluate_active_sparse_tasks(
        state=state,ee_position=objects[:,0],object_positions=objects,
        gripper_opening=torch.tensor([opening]),caught_target=torch.tensor([caught]),
        active_mask=torch.tensor([active]),max_steps=100,
        catch_release_dense_reward=BatchedCatchReleaseDenseReward.from_metadata({"staged_milestone_reward":True}),
        bilateral_contact=torch.tensor([caught]),loaded_finger_contact=torch.tensor([loaded]),
    )
    return result, objects


class ContactGraspTests(unittest.TestCase):
    def test_wide_physical_grasp_earns_pickup_and_staged_lift(self):
        state=task("pick_up")
        result,_=score(state)
        self.assertTrue(result.success.item())
        self.assertTrue(state.grasped.item())
        self.assertAlmostEqual(state.peak_lift.item(),.06,places=6)
        state=task()
        result,_=score(state)
        self.assertEqual(state.milestones[0,:3].tolist(),[True,True,True])
        self.assertFalse(result.diagnostics['released'].item())

    def test_wide_loaded_grip_over_bowl_cannot_pass_as_placement(self):
        state=task()
        result,_=score(state,z=.17,reference_x=0)
        self.assertFalse(result.success.item())
        self.assertFalse(result.diagnostics['released'].item())
        self.assertTrue(torch.isnan(state.release_clearance).item())
        # Even after the bilateral grasp disappears, ONE loaded pad blocks release.
        result,_=score(state,z=.17,reference_x=0,caught=False,loaded=True)
        self.assertFalse(result.success.item())
        self.assertTrue(torch.isnan(state.release_clearance).item())
        result,_=score(state,z=.17,reference_x=0,caught=False,loaded=False)
        self.assertTrue(result.success.item())
        self.assertAlmostEqual(state.release_clearance.item(),.02,places=6)

    def test_three_stage_credit_tracks_a_wide_grasp_lift_and_real_release(self):
        state = task()
        milestones = ThreeStageMilestones.zeros(torch, 1, "cpu")
        for z, caught, loaded in ((.23, True, True), (.17, True, True), (.17, False, False)):
            result, objects = score(state, z=z, caught=caught, loaded=loaded, reference_x=0)
            milestones = advance_three_stage_milestones(
                milestones, reset=SimpleNamespace(task_state=state),
                low_dim=SimpleNamespace(object_positions=objects, ee_position=objects[:, 0],
                                        gripper_opening=torch.tensor([.99])),
                result=result, physical_grasp=torch.tensor([caught]),
                gripper_command=torch.tensor([1. if not caught else -1.]),
                previous_opening=torch.tensor([.99]), active_mask=torch.tensor([True]),
            )
            self.assertTrue(milestones.picked_up.item())
            self.assertEqual(milestones.placed.item(), not loaded)
            self.assertEqual(state.milestones[0, 4].item(), not loaded)
        self.assertTrue(milestones.outcome.strict.item())
        self.assertFalse(milestones.carry_slip.item())

    def test_lift_without_grasp_cannot_earn_pickup_or_placement(self):
        state=task("pick_up")
        result,_=score(state,caught=False,loaded=False)
        self.assertFalse(result.success.item())
        self.assertFalse(state.ever_grasped.item())
        self.assertEqual(state.peak_lift.item(),0)
        state=task()
        result,_=score(state,z=.17,reference_x=0,caught=False,loaded=False)
        self.assertFalse(result.success.item())

    def test_closed_dropped_object_is_not_an_intentional_release(self):
        state=task();score(state)
        result,_=score(state,z=.17,reference_x=0,caught=False,loaded=False,opening=.80)
        self.assertFalse(result.success.item())
        self.assertFalse(result.diagnostics['released'].item())

    def test_inactive_world_cannot_add_grasp_history_or_release_clearance(self):
        state=task()
        result,_=score(state,active=False)
        self.assertFalse(state.ever_grasped.item())
        self.assertFalse(result.success.item())
        self.assertTrue(torch.isnan(state.release_clearance).item())

    def test_legacy_callers_keep_their_opening_filter(self):
        state=task("pick_up",contact=False)
        result,_=score(state)
        self.assertFalse(result.success.item())
        self.assertFalse(state.grasped.item())

    def test_physical_detector_requires_both_loaded_stable_persistent_pads(self):
        worlds=7
        left=torch.ones(worlds,dtype=torch.bool);left[1]=False
        force=torch.ones(worlds);force[2]=.01
        contacts=CDPRFingerContactBatch(left,torch.ones_like(left),force,torch.ones(worlds))
        state=task(worlds=worlds)
        reset=SimpleNamespace(task_state=state,grasp_eligible=torch.tensor([True]*6+[False]),
            bilateral_contact_steps=torch.zeros(worlds,dtype=torch.int64),
            previous_relative_position=torch.zeros(worlds,3),
            previous_relative_quaternion=torch.tensor([[1.,0.,0.,0.]]).repeat(worlds,1),
            target_rest_height=torch.full((worlds,),.02),physical_grasp=torch.zeros(worlds,dtype=torch.bool))
        low=SimpleNamespace(ee_position=torch.zeros(worlds,3),
            ee_quaternion=torch.tensor([[1.,0.,0.,0.]]).repeat(worlds,1),
            object_positions=torch.zeros(worlds,2,3),
            object_quaternions=torch.tensor([[[1.,0.,0.,0.]]]).repeat(worlds,2,1),
            gripper_opening=torch.ones(worlds))
        collector=SimpleNamespace(torch=torch,_world_rows=torch.arange(worlds),
            backend=SimpleNamespace(finger_object_contact_metrics=lambda _:contacts))
        active=torch.tensor([True]*5+[False,True])
        for index in range(2):
            low.object_positions[3,0,0]+=.02  # >8 mm relative slip every step
            yaw=.4*(index+1)
            low.object_quaternions[4,0]=torch.tensor([np.cos(yaw/2),0,0,np.sin(yaw/2)])
            _,grasp,_=RankLocalMJWarpGRPOCollector._update_physical_grasp(collector,reset,low,active)
            if index==0:
                self.assertFalse(grasp.any().item())
        self.assertEqual(grasp.tolist(),[True,False,False,False,False,False,False])

    def test_recording_and_decomposition_preserve_contact_v2_semantics(self):
        state=task();success=[];objects=[]
        for z,caught,loaded in ((.23,True,True),(.17,False,True),(.17,False,False)):
            result,obj=score(state,z=z,caught=caught,loaded=loaded,reference_x=0)
            success.append(result.success.numpy().copy());objects.append(obj.numpy())
        recording=_Recording(
            actions=np.zeros((3,1,5)),active=np.ones((3,1),dtype=bool),success=np.stack(success),
            terminated=np.stack(success),caught_target=np.array([[True],[False],[False]]),
            ee_xyz=np.zeros((3,1,3)),gripper_opening=np.full((3,1),.99),object_xyz=np.stack(objects),
            instruction_ids=np.array([INSTRUCTION_TO_ID['put_into_bowl']]),target_slots=np.array([0]),
            reference_slots=np.array([1]),second_reference_slots=np.array([-1]),horizons=np.array([3]),
            initial_target_xyz=np.array([[0.,0.,.17]]),support_surface_z=np.array([.15]),release_threshold=np.array([.90]),
            target_rest_height=np.array([.02]),physical_grasp_at_reset=np.array([False]),instructions=np.array(['put apple into bowl']),
            actions_per_decision=1,round_index=0,diverged_worlds=0,pick_lift_success_height=.05,
            contact_based_grasp=True,loaded_finger_contact=np.array([[True],[True],[False]]),
        )
        self.assertEqual(recording.release_mask()[:,0].tolist(),[False,False,True])
        self.assertAlmostEqual(_episode_rows(recording)[0]['peak_held_lift_m'],.06,places=5)
        rows=_episode_terms(recording,_Thresholds({}))
        self.assertTrue(rows[0]['recomputed_success'])
        self.assertEqual(rows[0]['recorded_success'],rows[0]['recomputed_success'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.npz';recording.to_npz(path)
            restored=_Recording.from_npz(path)
            self.assertTrue(restored.contact_based_grasp)
            np.testing.assert_array_equal(restored.release_mask(),recording.release_mask())
            # Actual legacy archives do not contain the new scoring fields.
            with np.load(path) as data:
                fields={k:data[k] for k in data.files if k not in ('contact_based_grasp','loaded_finger_contact')}
            np.savez(path,**fields);legacy=_Recording.from_npz(path)
            self.assertFalse(legacy.contact_based_grasp)
            self.assertFalse(legacy.held_mask.any())


if __name__=='__main__':
    unittest.main()
