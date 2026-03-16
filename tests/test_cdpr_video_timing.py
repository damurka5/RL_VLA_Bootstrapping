from __future__ import annotations

import types
import unittest


class CDPRVideoTimingTests(unittest.TestCase):
    def test_estimate_video_fps_from_capture_timestamps(self):
        try:
            from robots.cdpr.cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation
        except Exception as exc:  # pragma: no cover - depends on mujoco runtime
            self.skipTest(f"headless_cdpr_egl import unavailable: {exc}")

        sim = HeadlessCDPRSimulation.__new__(HeadlessCDPRSimulation)
        sim.frame_capture_timestamps = [0.1833333333, 0.3666666667, 0.55]
        sim.overview_frames = [object(), object(), object()]
        sim.ee_camera_frames = [object(), object(), object()]
        sim.data = types.SimpleNamespace(time=0.55)
        sim.controller = types.SimpleNamespace(dt=1.0 / 60.0)

        fps = sim._estimate_video_fps()

        self.assertAlmostEqual(fps, 60.0 / 11.0, places=5)


if __name__ == "__main__":
    unittest.main()
