"""Kinematic trace recorder and its summary: frustum test, slip typing, miss offsets."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from tools.audit.summarize_kinematic_traces import main as summarize_main


def _backend(worlds: int):
    eye = torch.eye(3).expand(worlds, 2, 3, 3).clone()
    return SimpleNamespace(
        overview_camera_id=0,
        wrist_camera_id=1,
        host_model=SimpleNamespace(cam_fovy=np.array([45.0, 45.0])),
        config=SimpleNamespace(render_width=320, render_height=240),
        wp=SimpleNamespace(to_torch=lambda value: value),
        # Both cameras at the origin looking along -z.
        data=SimpleNamespace(cam_xpos=torch.zeros(worlds, 2, 3), cam_xmat=eye),
    )


@unittest.skipIf(torch is None, "torch is not installed")
class FrustumTests(unittest.TestCase):
    def test_in_frame(self):
        from tools.audit.episode_kinematic_trace import KinematicTrace

        with tempfile.TemporaryDirectory() as tmp:
            trace = KinematicTrace(backend=_backend(4), output_dir=Path(tmp), torch=torch)
            points = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.5, 0.0, -1.0], [0.0, 0.5, -1.0]])
            # tan(22.5) = 0.414 vertical; x 4/3 = 0.552 horizontal.
            self.assertEqual(trace._in_frame("overview", points).tolist(), [True, False, True, False])


@unittest.skipIf(torch is None, "torch is not installed")
class SummaryTests(unittest.TestCase):
    def test_slip_typing_and_miss_offsets(self):
        from tools.audit.episode_kinematic_trace import KinematicTrace

        worlds, steps = 4, 30
        # world 0: strict.  world 1: slip while commanded open (premature
        # release).  world 2: passive slip high above the band.  world 3:
        # never grasps, closest approach offset +2 cm in y.
        with tempfile.TemporaryDirectory() as tmp:
            trace = KinematicTrace(backend=_backend(worlds), output_dir=Path(tmp) / "trace", torch=torch)
            scenes = [SimpleNamespace(scene_uid=f"s{i}", destination="plate", target_catalog="apple",
                                      destination_success_radius=0.09) for i in range(worlds)]
            trace.start_round(round_index=0, scenes=scenes, target_slots=torch.zeros(worlds, dtype=torch.long),
                              reference_slots=torch.ones(worlds, dtype=torch.long))
            false = torch.zeros(worlds, dtype=torch.bool)
            for t in range(steps):
                grasped = torch.tensor([t >= 5, t >= 5, t >= 5, False])
                lifted = torch.tensor([t >= 8, t >= 8, t >= 8, False])
                slip = torch.tensor([False, t >= 20, t >= 22, False])
                ee = torch.zeros(worlds, 3)
                ee[:, 2] = 0.25
                ee[2, 2] = 0.45 if t >= 10 else 0.25
                objects = torch.zeros(worlds, 2, 3)
                objects[:, 0, :2] = torch.tensor([0.1, 0.10])
                objects[:, 1, :2] = torch.tensor([0.0, -0.05])
                ee[3, :2] = torch.tensor([0.1, 0.10 + 0.02 + 0.01 * abs(t - 15)])
                command = torch.full((worlds, 5), -0.5)
                if t >= 18:
                    command[1, 4] = 0.8
                if t % 4 == 0:
                    # Prior pushes up (+1.0 pre-tanh) in every world; the
                    # residual pushes down by 0.25 pre-tanh.
                    prior = torch.zeros(worlds, 8, 5)
                    prior[..., 2] = 1.0
                    final = torch.tanh(prior[:, :4] - 0.25)
                    trace.record_decision(decision_index=t // 4, prior=prior, chunk=final,
                                          active=torch.ones(worlds, dtype=torch.bool))
                trace.record_step(
                    decision_index=t // 4,
                    action=command,
                    low_dim=SimpleNamespace(ee_position=ee, target_position=ee + 0.01, object_positions=objects,
                                            gripper_opening=torch.full((worlds,), 0.2)),
                    active=torch.ones(worlds, dtype=torch.bool),
                    physical_grasp=grasped & ~slip,
                    bilateral_contact=grasped & ~slip,
                    release_in_progress=false,
                    outcome=SimpleNamespace(grasped=grasped, lifted=lifted, released=false, carry_slip=slip,
                                            wrong_place=false, native=torch.tensor([t >= 25, False, False, False])),
                )
            trace.finish_round(strict=torch.tensor([True, False, False, False]), non_finite=false)
            out = Path(tmp) / "summary"
            summarize_main([str(Path(tmp) / "trace"), "--output", str(out)])
            summary = json.loads((out / "trace_summary.json").read_text())
            slips = summary["slip_types"]["all"]
            self.assertEqual(slips["slips"], 2)
            self.assertAlmostEqual(slips["commanded_open"], 0.5)
            self.assertAlmostEqual(slips["passive"], 0.5)
            miss = summary["grasp_miss"]["all"]
            self.assertEqual(miss["episodes"], 1)
            self.assertAlmostEqual(miss["dy"]["median"], 0.02, places=4)
            self.assertAlmostEqual(miss["dx"]["median"], 0.0, places=4)
            high = summary["post_lift"]["slip|all"]
            self.assertGreater(high["peak_ee_z_after_lift"]["mean"], 0.3)
            carry = summary["z_attribution"]["carry|slip|all"]
            self.assertAlmostEqual(carry["z_prior_only"]["mean"], float(np.tanh(1.0)), places=4)
            self.assertAlmostEqual(carry["z_residual_push"]["mean"], -0.25, places=4)
            self.assertAlmostEqual(carry["z_final"]["mean"], float(np.tanh(0.75)), places=4)
            # The never-grasp world hovers within 6 cm of the object.
            self.assertIn("hover|no_grasp|all", summary["z_attribution"])
            dims = summary["dim_attribution"]["carry|slip|all"]
            self.assertAlmostEqual(dims["z_push"], -0.25, places=4)
            self.assertAlmostEqual(dims["x_prior_only"], 0.0, places=4)
            self.assertIn("preslip|slip|all", summary["dim_attribution"])


class CeilingOverrideTests(unittest.TestCase):
    def test_override_keeps_floor_and_rejects_bad_ceiling(self):
        from tools.audit.xy_approach_probe import override_controller_z_ceiling

        args = SimpleNamespace(controller_workspace_z_bounds=[0.18, 0.60])
        override_controller_z_ceiling(args, 0.40)
        self.assertEqual(args.controller_workspace_z_bounds, [0.18, 0.40])
        with self.assertRaises(SystemExit):
            override_controller_z_ceiling(args, 0.10)


if __name__ == "__main__":
    unittest.main()
