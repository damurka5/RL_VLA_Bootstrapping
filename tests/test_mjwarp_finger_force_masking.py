"""Irrelevant non-finite contacts must not erase an otherwise valid pad load."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace
import unittest

import numpy as np  # Initialize the local NumPy runtime before Torch.
import torch

from rl_vla_bootstrapping.simulation.mjlab_mjwarp_backend import MJLabMJWarpCDPRBackend


class FingerForceMaskingTests(unittest.TestCase):
    def contacts(self, pairs, normal_forces, worlds, *, nacon=None, distances=None):
        count = len(pairs)
        forces = torch.zeros(count, 6)
        forces[:, 0] = torch.tensor(normal_forces)
        backend = SimpleNamespace(
            torch=torch, _device="cpu", worlds_per_rank=2,
            _contact_geom=torch.tensor(pairs).reshape(-1, 2),
            _contact_worldid=torch.tensor(worlds, dtype=torch.int64),
            _contact_indices=torch.arange(count),
            _nacon=torch.tensor([count if nacon is None else nacon]),
            _contact_dist=torch.tensor(distances if distances is not None else [0.] * count),
            _slot_geom_ids=torch.tensor([[1, 2], [3, 4]]),
            _finger_pad_geom_ids_tensor=torch.tensor([5, 6]),
            wp=SimpleNamespace(ScopedDevice=lambda _: contextlib.nullcontext()),
            mjw=SimpleNamespace(contact_force=lambda *args: None),
            model=None, data=None, _contact_ids_wp=None, _contact_forces_wp=None,
            _contact_forces=forces, _left_pad_force=torch.zeros(2), _right_pad_force=torch.zeros(2),
            _left_pad_contact_count=torch.zeros(2, dtype=torch.int32),
            _right_pad_contact_count=torch.zeros(2, dtype=torch.int32),
        )
        return MJLabMJWarpCDPRBackend.finger_object_contact_metrics(backend, [0, 1])

    def test_unrelated_contacts_cannot_contaminate_real_bilateral_grasps(self):
        result = self.contacts(
            [[5, 1], [1, 6], [3, 5], [6, 4], [5, 3], [2, 4], [1, 6]],
            [2., 3., 4., 5., float("nan"), float("inf"), float("nan")],
            [0, 0, 1, 1, 0, 0, 1],
        )
        torch.testing.assert_close(result.left_normal_force, torch.tensor([2., 4.]))
        torch.testing.assert_close(result.right_normal_force, torch.tensor([3., 5.]))
        self.assertEqual(result.bilateral_contact.tolist(), [True, True])

    def test_unused_invalid_world_and_separated_contacts_are_excluded(self):
        result = self.contacts(
            [[5, 1]] * 5, [1., float("nan"), float("inf"), float("nan"), float("inf")],
            [0, -1, 2, 0, 0], nacon=4, distances=[0., 0., 0., .003, 0.],
        )
        self.assertEqual(result.left_normal_force.tolist(), [1., 0.])
        self.assertEqual(result.right_normal_force.tolist(), [0., 0.])
        self.assertEqual(result.bilateral_contact.tolist(), [False, False])

    def test_real_nonfinite_pad_load_stays_visible_without_poisoning_the_other_pad(self):
        result = self.contacts([[5, 1], [1, 6]], [float("nan"), 2.], [0, 0])
        self.assertTrue(torch.isnan(result.left_normal_force[0]))
        self.assertEqual(result.right_normal_force.tolist(), [2., 0.])
        self.assertEqual(result.bilateral_contact.tolist(), [True, False])

    def test_finite_pad_loads_keep_the_solver_sign_and_sum_contract(self):
        result = self.contacts([[5, 1], [2, 5], [6, 1]], [-1., 2., -3.], [0, 0, 0])
        self.assertEqual(result.left_normal_force.tolist(), [3., 0.])
        self.assertEqual(result.right_normal_force.tolist(), [3., 0.])


if __name__ == "__main__":
    unittest.main()
