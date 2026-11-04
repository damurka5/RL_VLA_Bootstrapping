"""A short lift used to pass while each catalog slipped out within seconds."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

# The catalog switches at import time. Keep this test isolated from CDPR tests.
ROOT = Path(__file__).resolve().parents[1]


class SustainedGripTests(unittest.TestCase):
    def test_all_graspables_survive_twenty_second_physical_hold(self):
        code = r'''
from tools.audit import audit_widowx200_object_scale as a
import numpy as np
config = a.eval_config(a.pipeline_view(a.DEFAULT_CONFIG), "put_into", True)
spec = a.control_spec(config)
cell = a.workspace.sector_lattice()[3]
for name in config["task"]["metadata"]["catchable_object_pool"]:
    model = a.mixed_model([name])
    data, controller = a.prepare(model, [name], [cell], spec)
    body = a.body_id(model, 0)
    start = data.xpos[body].copy()
    a.move(controller, start)
    a.close_gripper(controller, a.catalog.OBJECT_VARIANTS[name].fitted_gripper_opening
                   - a.catalog.CAUGHT_START_GRIP_SQUEEZE)
    a.move(controller, [*cell, 0.25])
    assert data.xpos[body, 2] - start[2] >= 0.04, (name, "no initial lift")
    for _ in range(400):
        controller.apply_normalized_action([0, 0, 0, 0, 0])
        gap = float(np.linalg.norm(data.xpos[body] - controller.get_end_effector_position()))
        assert gap < 0.01, (name, "grip lost", gap)
    assert data.xpos[body, 2] - start[2] >= 0.04, (name, "object dropped")
'''
        env = dict(os.environ, RLVLA_EMBODIMENT="widowx200", PYTHONPATH=str(ROOT))
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                text=True, capture_output=True)
        if "No module named 'mujoco'" in result.stderr or "requires the curated RoboCasa" in result.stderr:
            self.skipTest("MuJoCo and the staged catalog are required")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
