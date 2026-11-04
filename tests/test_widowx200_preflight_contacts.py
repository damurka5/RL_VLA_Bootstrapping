"""A valid object inside a bowl can have centres less than 3 cm apart."""

import types
import unittest
import numpy as np  # Initialize the local NumPy OpenMP runtime before Torch.
import torch

from scripts.preflight_widowx200_pipeline import max_object_penetration


class ObjectPenetrationTests(unittest.TestCase):
    def backend(self, pairs, distances, *, nacon=None, worlds=None):
        pairs = torch.tensor(pairs, dtype=torch.int32).reshape(-1, 2)
        count = len(distances)
        return types.SimpleNamespace(
            torch=torch, device="cpu", worlds_per_rank=2,
            host_model=types.SimpleNamespace(ngeom=10),
            _slot_geom_ids=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 0]]),
            _contact_geom=pairs,
            _contact_dist=torch.tensor(distances),
            _contact_indices=torch.arange(count),
            _nacon=torch.tensor([count if nacon is None else nacon]),
            _contact_worldid=torch.tensor([0] * count if worlds is None else worlds),
        )

    def test_touching_object_and_bowl_are_not_penetrating(self):
        backend = self.backend([[1, 3]], [0.0002])
        self.assertEqual(max_object_penetration(backend), 0.0)

    def test_detects_real_overlap_between_distinct_objects(self):
        backend = self.backend([[1, 3], [2, 4]], [-0.004, -0.001])
        self.assertAlmostEqual(max_object_penetration(backend), 0.004, places=6)

    def test_ignores_same_body_unused_and_invalid_world_contacts(self):
        backend = self.backend([[1, 2], [1, 3], [1, 3], [1, 3]],
                               [-0.1, -0.2, -0.3, -0.4],
                               nacon=3, worlds=[0, -1, 2, 0])
        self.assertEqual(max_object_penetration(backend), 0.0)

    def test_no_contacts(self):
        self.assertEqual(max_object_penetration(self.backend([], [])), 0.0)

    def test_finger_contact_is_not_object_object_penetration(self):
        backend = self.backend([[1, 8]], [-0.004])
        self.assertEqual(max_object_penetration(backend), 0.0)


if __name__ == "__main__":
    unittest.main()
