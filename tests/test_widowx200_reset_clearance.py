"""Caught carry starts must clear bowl walls without moving pickup starts."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResetClearanceTests(unittest.TestCase):
    def test_sampler_preserves_other_starts_and_removes_rim_intersections(self):
        # A fresh process selects the arm catalog before any CDPR imports.
        code = r'''
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco as mj
from scripts.preflight_widowx200_pipeline import merged_skill, load_yaml, ladder_points
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import BatchedReverseFrontierResetter, RankLocalCurriculum
from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID
from tools.audit import audit_widowx200_object_scale as a

class CaptureBackend:
    torch = torch
    device = torch.device('cpu')
    worlds_per_rank = 512
    object_body_ids = (0, 1, 2, 3)
    def reset_worlds(self, indices): pass
    def set_object_catalogs(self, catalogs): self.catalogs = catalogs.clone()
    def set_free_body_poses(self, body_ids, positions, quaternions):
        self.positions, self.quaternions = positions.clone(), quaternions.clone()
    def set_end_effector_poses(self, positions, yaws):
        self.ee, self.yaw = positions.clone(), yaws.clone()
    def set_gripper_openings(self, openings): self.openings = openings.clone()
    def set_visual_variants(self, *args): pass
    def broadcast_group_state(self, indices): pass
    def low_dim_observations(self):
        q = torch.zeros((512, 4))
        q[:, 0], q[:, 3] = torch.cos(self.yaw / 2), torch.sin(self.yaw / 2)
        return SimpleNamespace(ee_position=self.ee, ee_quaternion=q,
                               object_positions=self.positions, object_quaternions=self.quaternions)

raw = load_yaml(Path('configs/examples/widowx200_smolvla_pipeline.yaml'))
models = {}
def sample(config, caps, round_index, corrected):
    backend = CaptureBackend()
    metadata = dict(config['task']['metadata'])
    if not corrected:
        metadata.pop('placement_caught_object_min_height', None)
    resetter = BatchedReverseFrontierResetter(
        backend=backend, layout=RankLocalGroupLayout(512, 64, 8),
        curriculum=RankLocalCurriculum(device='cpu'), rank=0, base_seed=12345,
        instruction_types=config['task']['instruction_types'],
        allowed_objects=metadata['scene_object_pool'], task_metadata=metadata)
    resetter.set_random_start_max_goal_distance({INSTRUCTION_TO_ID[k]:v for k,v in caps.items()})
    reset = resetter.reset(update_index=0, round_index=round_index, allow_prelifted=True)
    return backend, reset

def violating_contacts(backend, config):
    count = 0
    for world in range(0, 512, 8):
        ids = backend.catalogs[world].tolist()
        names = tuple(a.catalog.ACTIVE_CDPR_CATALOGS[max(0, i)] for i in ids)
        if names not in models:
            models[names] = a.mixed_model(list(names))
        model = models[names]
        data = mj.MjData(model)
        controller = a.WidowX200TaskSpaceController(model, data, spec=a.control_spec(config), mujoco=mj)
        controller.reset_to_pose(backend.ee[world].numpy(), yaw=float(backend.yaw[world]),
                                 gripper=float(backend.openings[world]))
        slot_for_geom = {}
        for slot in range(4):
            body = a.body_id(model, slot)
            address = int(model.jnt_qposadr[model.body_jntadr[body]])
            data.qpos[address:address+3] = backend.positions[world, slot].numpy()
            data.qpos[address+3:address+7] = backend.quaternions[world, slot].numpy()
            for geom in range(model.ngeom):
                if model.geom_bodyid[geom] == body:
                    slot_for_geom[geom] = slot
        mj.mj_forward(model, data)
        for index in range(data.ncon):
            contact = data.contact[index]
            slots = [slot_for_geom.get(int(g), -1) for g in contact.geom]
            count += int(min(slots) >= 0 and slots[0] != slots[1] and contact.dist < -0.003)
    return count

changed = old_contacts = new_contacts = 0
for skill in ('move_to', 'pick_up', 'put_into', 'joint'):
    config = merged_skill(raw, skill)
    for point, caps in enumerate(ladder_points(config['task']['metadata'], config['task']['instruction_types'])):
        for round_index in range(2):
            old, old_reset = sample(config, caps, round_index, False)
            new, new_reset = sample(config, caps, round_index, True)
            caught_placement = new_reset.physical_grasp & (
                (new_reset.task_state.instruction_ids == INSTRUCTION_TO_ID['put_into_plate']) |
                (new_reset.task_state.instruction_ids == INSTRUCTION_TO_ID['put_into_bowl']))
            for attr in ('catalogs', 'quaternions', 'yaw', 'openings'):
                assert torch.equal(getattr(old, attr), getattr(new, attr)), (skill, attr)
            assert torch.equal(old.positions[:, 1:], new.positions[:, 1:])
            assert torch.equal(old.positions[~caught_placement], new.positions[~caught_placement])
            assert torch.equal(old.ee[~caught_placement], new.ee[~caught_placement])
            assert torch.equal(old.ee[:, :2], new.ee[:, :2])
            assert torch.equal(old_reset.horizons, new_reset.horizons)
            if caught_placement.any():
                assert (new.positions[caught_placement, 0, 2] >= 0.215 - 1e-7).all()
            moved = (old.ee != new.ee).any(dim=1)
            changed += int(moved.sum())
            if moved.any():
                goal = new_reset.curriculum_goal_xyz[moved]
                assert ((new.ee[moved] - goal).norm(dim=-1) <=
                        (old.ee[moved] - goal).norm(dim=-1) + 1e-6).all()
            if skill == 'put_into' and point == 1:
                old_contacts += violating_contacts(old, config)
                new_contacts += violating_contacts(new, config)
assert changed > 0
assert old_contacts > 0, 'Legacy sampler should reproduce an actual bowl-rim intersection'
assert new_contacts == 0, new_contacts
print(f'changed worlds={changed}; middle-rung initial contacts deeper than 3 mm: {old_contacts} -> {new_contacts}')
'''
        env = dict(os.environ, RLVLA_EMBODIMENT="widowx200", MUJOCO_GL="glfw", PYTHONPATH=str(ROOT))
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                text=True, capture_output=True)
        if "No module named 'mujoco'" in result.stderr or "requires the curated RoboCasa" in result.stderr:
            self.skipTest("MuJoCo and the staged catalog are required")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
