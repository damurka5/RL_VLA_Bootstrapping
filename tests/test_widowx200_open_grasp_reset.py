"""Open starts clear real colliders; the repair uses actual per-world contacts."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rl_vla_bootstrapping.simulation.widowx200_mjwarp_backend import WidowX200MJWarpBackend

ROOT = Path(__file__).resolve().parents[1]


class DeviceContactMatchingTests(unittest.TestCase):
    def backend(self, pairs, distances, worlds, nacon=None):
        count = len(distances)
        return SimpleNamespace(
            torch=torch, _device="cpu", worlds_per_rank=2,
            host_model=SimpleNamespace(geom_contype=np.array([0, 1, 1, 1, 1, 2, 8])),
            _slot_geom_ids=torch.tensor([[1, 2], [3, 4]]),
            _contact_geom=torch.tensor(pairs).reshape(-1, 2),
            _contact_dist=torch.tensor(distances, dtype=torch.float32),
            _contact_indices=torch.arange(count), _nacon=torch.tensor([count if nacon is None else nacon]),
            _contact_worldid=torch.tensor(worlds, dtype=torch.int64),
        )

    def test_matches_each_world_target_in_both_orders_and_includes_arm(self):
        backend = self.backend([[1, 5], [6, 4], [3, 5], [1, 3]],
                               [-.004, -.008, -.2, -.3], [0, 1, 0, 0])
        depth = WidowX200MJWarpBackend.grasp_start_penetration(backend, torch.tensor([0, 1]))
        torch.testing.assert_close(depth, torch.tensor([.004, .008]))

    def test_ignores_unused_entries_invalid_worlds_and_positive_distances(self):
        backend = self.backend([[1, 5]] * 4, [-.1, -.2, .001, -.4], [-1, 2, 0, 0], nacon=3)
        depth = WidowX200MJWarpBackend.grasp_start_penetration(backend, [0, 1])
        self.assertEqual(depth.tolist(), [0., 0.])

    def test_empty_contact_buffer_and_slot_count_validation(self):
        backend = self.backend([], [], [])
        self.assertEqual(WidowX200MJWarpBackend.grasp_start_penetration(backend, [0, 1]).tolist(), [0., 0.])
        with self.assertRaises(ValueError):
            WidowX200MJWarpBackend.grasp_start_penetration(backend, [0])


class OpenStartGeometryTests(unittest.TestCase):
    def test_real_pads_are_visible_and_clear_of_long_objects_at_all_yaws(self):
        code = r'''
from tools.audit import audit_widowx200_object_scale as a
import numpy as np
import mujoco as mj
config = a.eval_config(a.pipeline_view(a.DEFAULT_CONFIG), "put_into", False)
spec = a.control_spec(config)
for name in ("robocasa_bell_pepper", "robocasa_potato"):
    model = a.mixed_model([name])
    data = mj.MjData(model)
    controller = a.WidowX200TaskSpaceController(model, data, spec=spec, mujoco=mj)
    for side in ("left", "right"):
        pad = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"{side}_finger_pad")
        visual = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"{side}_finger_pad_visual")
        assert min(pad, visual) >= 0
        assert model.geom_group[visual] == 1
        assert model.geom_contype[visual] == model.geom_conaffinity[visual] == 0
        assert model.geom_bodyid[visual] == model.geom_bodyid[pad]
        np.testing.assert_allclose(model.geom_pos[visual], model.geom_pos[pad])
        np.testing.assert_allclose(model.geom_size[visual], model.geom_size[pad])
    for slot in range(1, 4):
        address = model.jnt_qposadr[model.body_jntadr[a.body_id(model, slot)]]
        data.qpos[address:address+3] = [2+slot, 2, 1]
    body = a.body_id(model, 0)
    address = model.jnt_qposadr[model.body_jntadr[body]]
    position = np.array([.1, 0, .15+a.catalog.OBJECT_VARIANTS[name].rest_height])
    overlaps = {}
    for hover in (.01, .04):
        controller.reset_to_pose(position+[0, 0, hover], yaw=0, gripper=1)
        depths = []
        for yaw in np.linspace(0, np.pi, 36, endpoint=False):
            data.qpos[address:address+3] = position
            data.qpos[address+3:address+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            mj.mj_forward(model, data)
            depths.append(max((max(0., -float(c.dist)) for c in data.contact
                if ((model.geom_bodyid[c.geom[0]] == body and model.geom_contype[c.geom[1]] & 10)
                 or (model.geom_bodyid[c.geom[1]] == body and model.geom_contype[c.geom[0]] & 10))), default=0.))
        overlaps[hover] = max(depths)
    assert overlaps[.01] > .001, (name, "did not reproduce old overlap", overlaps)
    assert overlaps[.04] <= .001, (name, overlaps)
    print(name, overlaps)
'''
        self.run_isolated(code, require_mujoco=True)

    def test_device_guard_moves_the_entire_blocked_group_and_fails_if_unclearable(self):
        code = r'''
from pathlib import Path
import torch
from scripts.preflight_widowx200_pipeline import merged_skill, load_yaml
from tools.audit.start_distance_probe import build_recording_backend
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import BatchedReverseFrontierResetter, RankLocalCurriculum
from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout

config = merged_skill(load_yaml(Path("configs/examples/widowx200_smolvla_pipeline.yaml")), "put_into")
metadata = dict(config["task"]["metadata"], placement_caught_object_fraction=0.)
def sample(mode):
    backend = build_recording_backend(torch)
    backend.embodiment = "widowx200"
    starts = []
    def depth(slots):
        starts.append(backend.ee_positions.clone())
        result = torch.zeros(16)
        if mode == "always" or (mode == "once" and len(starts) == 1):
            result[3] = .012  # only one copy collides; move its whole GRPO group
        return result
    backend.grasp_start_penetration = depth
    resetter = BatchedReverseFrontierResetter(
        backend=backend, layout=RankLocalGroupLayout(16, 2, 8),
        curriculum=RankLocalCurriculum(device="cpu"), rank=0, base_seed=12345,
        instruction_types=config["task"]["instruction_types"],
        allowed_objects=metadata["scene_object_pool"], task_metadata=metadata)
    reset = resetter.reset(update_index=0, round_index=0, allow_prelifted=True)
    assert reset.task_state.contact_based_grasp
    return backend, starts
normal, _ = sample("clear")
repaired, starts = sample("once")
assert len(starts) == 2
torch.testing.assert_close(repaired.ee_positions[:8, 2], normal.ee_positions[:8, 2]+.01)
torch.testing.assert_close(repaired.ee_positions[:, :2], normal.ee_positions[:, :2])
torch.testing.assert_close(repaired.ee_positions[8:], normal.ee_positions[8:])
torch.testing.assert_close(repaired.object_positions, normal.object_positions)
torch.testing.assert_close(repaired.object_quaternions, normal.object_quaternions)
try:
    sample("always")
except RuntimeError as error:
    assert "collision-free" in str(error)
else:
    raise AssertionError("Cannot silently train on an intersecting open start")
'''
        self.run_isolated(code)

    def run_isolated(self, code, require_mujoco=False):
        env = dict(os.environ, RLVLA_EMBODIMENT="widowx200", PYTHONPATH=str(ROOT))
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                text=True, capture_output=True)
        if require_mujoco and ("No module named 'mujoco'" in result.stderr
                               or "requires the curated RoboCasa" in result.stderr):
            self.skipTest("MuJoCo and the staged catalog are required")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
