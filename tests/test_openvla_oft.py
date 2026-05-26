from __future__ import annotations

import unittest

from rl_vla_bootstrapping.policy.openvla_oft import _extract_cdpr_env_overrides


class OpenVLAOFTEnvOverrideTests(unittest.TestCase):
    def test_extract_cdpr_env_overrides_maps_record_trajectory_to_env(self):
        injected = {
            "record_trajectory": False,
            "randomize_ee_start": True,
            "randomize_ee_yaw": True,
            "ee_yaw_bounds": [-0.5, 0.5],
        }

        env = _extract_cdpr_env_overrides(injected)

        self.assertEqual(env["RLVLA_CDPR_RECORD_TRAJECTORY"], "0")
        self.assertEqual(env["RLVLA_CDPR_RANDOMIZE_EE_START"], "1")
        self.assertEqual(env["RLVLA_CDPR_RANDOMIZE_EE_YAW"], "1")
        self.assertEqual(env["RLVLA_CDPR_EE_YAW_BOUNDS"], "[-0.5, 0.5]")
        self.assertNotIn("record_trajectory", injected)
        self.assertNotIn("randomize_ee_start", injected)
        self.assertNotIn("randomize_ee_yaw", injected)
        self.assertNotIn("ee_yaw_bounds", injected)


if __name__ == "__main__":
    unittest.main()
