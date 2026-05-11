from __future__ import annotations

import unittest

import numpy as np

from robots.cdpr.cdpr_dataset.cdpr_lchol_spec import CDPRLCHOLSpec


class LCHOLHindsightRelabelingTests(unittest.TestCase):
    def test_relabel_record_stops_at_first_achievement(self):
        spec = CDPRLCHOLSpec()
        trajectory = [
            {
                "instruction_type": "put_into_plate",
                "source_instruction": "put apple into plate",
                "target_object_catalog": "ycb_apple",
                "distance_ee_to_object_xy": 0.20,
                "gripper_closed": 0.0,
                "caught_object_is_target": 0.0,
                "action": np.zeros((4,), dtype=np.float32),
            },
            {
                "instruction_type": "put_into_plate",
                "source_instruction": "put apple into plate",
                "target_object_catalog": "ycb_apple",
                "distance_ee_to_object_xy": 0.03,
                "gripper_closed": 1.0,
                "caught_object_is_target": 1.0,
                "action": np.ones((4,), dtype=np.float32),
            },
            {
                "instruction_type": "put_into_plate",
                "source_instruction": "put apple into plate",
                "target_object_catalog": "ycb_apple",
                "distance_ee_to_object_xy": 0.30,
                "gripper_closed": 1.0,
                "caught_object_is_target": 1.0,
                "action": np.full((4,), 2.0, dtype=np.float32),
            },
        ]

        records = spec.build_hindsight_records(trajectory, prefix_max_steps=16)
        grab_records = [record for record in records if record.option_name == "grab_object"]

        self.assertEqual(len(grab_records), 1)
        record = grab_records[0]
        self.assertEqual(record.first_timestep, 1)
        self.assertEqual(record.instruction, "grab apple")
        self.assertEqual(len(record.prefix_actions), 2)
        np.testing.assert_array_equal(record.action, np.ones((4,), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
