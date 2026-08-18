"""The SFT half: joining frames to rows, and the checkpoint that goes back.

Both are places where a mistake is silent. A bad join trains the vision path on
another episode's pictures; a bad payload hands a resumed GRPO run optimizer
moments from a supervised loss, or drops the curriculum caps the iteration just
earned.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.sil_sft import (  # noqa: E402
    build_resume_payload,
    frames_for_rows,
    load_frame_index,
    resolve_frame_rows,
)


def _frames(stem, *, worlds, decisions, height=2, width=3):
    overview = np.zeros((decisions, len(worlds), height, width, 3), np.uint8)
    wrist = np.zeros_like(overview)
    for d in range(decisions):
        for i, w in enumerate(worlds):
            overview[d, i] = (d * 16 + w) % 256
            wrist[d, i] = (d * 16 + w + 100) % 256
    return {
        stem: {
            "path": f"frames_{stem}.npz",
            "overview": overview,
            "wrist": wrist,
            "world_column": {int(w): i for i, w in enumerate(worlds)},
            "decisions": decisions,
        }
    }


class FrameJoinTests(unittest.TestCase):
    def test_rows_resolve_to_their_own_world_and_decision(self):
        frames = _frames("cap_0.030_record_00", worlds=[2, 5], decisions=4)
        uid = np.array(
            [
                "cap_0.030_record_00/r0w2",
                "cap_0.030_record_00/r0w5",
                "cap_0.030_record_00/r0w2",
            ]
        )
        decision = np.array([0, 1, 3])
        keep, lookups = resolve_frame_rows(uid, decision, frames)
        self.assertTrue(keep.all())
        self.assertEqual(
            lookups,
            [
                ("cap_0.030_record_00", 0, 0),
                ("cap_0.030_record_00", 1, 1),
                ("cap_0.030_record_00", 3, 0),
            ],
        )
        overview, _ = frames_for_rows(lookups, frames, [0, 1, 2])
        self.assertEqual(int(overview[0, 0, 0, 0]), 2)
        self.assertEqual(int(overview[1, 0, 0, 0]), 16 + 5)
        self.assertEqual(int(overview[2, 0, 0, 0]), 3 * 16 + 2)

    def test_a_world_without_frames_is_dropped_not_filled(self):
        """--frame-worlds caps the set; inventing a frame trains on the wrong episode."""

        frames = _frames("s", worlds=[2], decisions=4)
        uid = np.array(["s/r0w2", "s/r0w9"])
        keep, lookups = resolve_frame_rows(uid, np.array([0, 0]), frames)
        self.assertEqual(list(keep), [True, False])
        self.assertEqual(len(lookups), 1)

    def test_a_decision_past_the_recorded_horizon_is_dropped(self):
        frames = _frames("s", worlds=[0], decisions=2)
        keep, _ = resolve_frame_rows(
            np.array(["s/r0w0", "s/r0w0"]), np.array([1, 5]), frames
        )
        self.assertEqual(list(keep), [True, False])

    def test_rows_from_another_harvest_do_not_match(self):
        """The join key is the file stem, and it has to be exact.

        Every rung numbers its rounds from zero and two families share a rung
        label, so anything coarser than the stem collides -- which in phase 3
        merged distinct episodes and split one across train and validation.
        """

        frames = _frames("cap_0.030_record_00", worlds=[0], decisions=2)
        keep, _ = resolve_frame_rows(
            np.array(["cap_0.050_record_00/r0w0"]), np.array([0]), frames
        )
        self.assertFalse(keep.any())

    def test_the_loader_rejects_a_file_that_is_not_a_frames_npz(self):
        with self.assertRaises(SystemExit):
            load_frame_index([Path("replay_cap_0.030_record_00.npz")])


class ResumePayloadTests(unittest.TestCase):
    """Neither existing loader does the right thing; this is the third way."""

    def _source(self):
        return {
            "policy_type": "smolvla_cdpr_grpo",
            "global_step": 6_262_944,
            "policy": {"actor.net.0.weight": np.zeros(2)},
            "vla_lora": {"lora_a": np.zeros(2)},
            "optimizer": {"state": "stale adam moments"},
            "vla_lora_optimizer": {"state": "stale adam moments"},
            "extra_state": {
                "approach_curriculum": {"move_to_object": {"cap": 0.19}}
            },
            "simulator_metadata": {"xml_sha256": "abc"},
            "args": {"vla_lr": 1e-5},
        }

    def test_both_optimizer_states_are_dropped(self):
        """An SFT AdamW carries moments from a different loss at another scale."""

        out = build_resume_payload(
            self._source(),
            policy_state={"actor.net.0.weight": np.ones(2)},
            lora_state=None,
            note={},
        )
        self.assertNotIn("optimizer", out)
        self.assertNotIn("vla_lora_optimizer", out)

    def test_the_curriculum_and_step_survive(self):
        """load_weights_only drops extra_state, which is where the caps live.

        A warm start would put the cap back on the first rung and undo the very
        iteration that earned the promotion.
        """

        out = build_resume_payload(
            self._source(),
            policy_state={"actor.net.0.weight": np.ones(2)},
            lora_state=None,
            note={},
        )
        self.assertEqual(
            out["extra_state"]["approach_curriculum"]["move_to_object"]["cap"],
            0.19,
        )
        self.assertEqual(out["global_step"], 6_262_944)
        self.assertEqual(out["simulator_metadata"]["xml_sha256"], "abc")

    def test_lora_is_replaced_when_given_and_kept_when_not(self):
        """Residual-only SFT must not overwrite the adapter it never trained."""

        source = self._source()
        kept = build_resume_payload(
            source,
            policy_state={"actor.net.0.weight": np.ones(2)},
            lora_state=None,
            note={},
        )
        self.assertIs(kept["vla_lora"], source["vla_lora"])
        replaced = build_resume_payload(
            source,
            policy_state={"actor.net.0.weight": np.ones(2)},
            lora_state={"lora_a": np.ones(2)},
            note={},
        )
        self.assertEqual(list(replaced["vla_lora"]["lora_a"]), [1.0, 1.0])

    def test_the_source_payload_is_not_mutated(self):
        source = self._source()
        build_resume_payload(
            source,
            policy_state={"actor.net.0.weight": np.ones(2)},
            lora_state={"lora_a": np.ones(2)},
            note={"trained": "x"},
        )
        self.assertIn("optimizer", source)
        self.assertEqual(list(source["vla_lora"]["lora_a"]), [0.0, 0.0])


@unittest.skipUnless(
    __import__("importlib.util", fromlist=["util"]).find_spec("torch"),
    "torch is required",
)
class VisionIntegrityTests(unittest.TestCase):
    """M5: the only block of the state that can be compared at all."""

    def test_it_measures_the_vision_block_only(self):
        import torch

        from tools.audit.sil_sft import check_recomputed_vision

        recorded = torch.zeros((3, 10))
        recomputed = recorded.clone()
        # A difference in the PROPRIO block must not be reported: it is copied
        # through, so it is equal by construction and would mask a real change.
        recomputed[:, :4] += 5.0
        out = check_recomputed_vision(
            recomputed, recorded, vision_dim=6, torch=torch
        )
        self.assertEqual(out["vision_max_abs_diff"], 0.0)

    def test_a_vision_mismatch_shows_up(self):
        import torch

        from tools.audit.sil_sft import check_recomputed_vision

        recorded = torch.ones((3, 10))
        recomputed = recorded.clone()
        recomputed[:, -6:] += 0.25
        out = check_recomputed_vision(
            recomputed, recorded, vision_dim=6, torch=torch
        )
        self.assertAlmostEqual(out["vision_max_abs_diff"], 0.25, places=5)
        self.assertAlmostEqual(
            out["vision_relative_mean_abs_diff"], 0.25, places=5
        )

    def test_no_vision_feature_is_not_an_error(self):
        import torch

        from tools.audit.sil_sft import check_recomputed_vision

        out = check_recomputed_vision(
            torch.zeros((2, 6)), torch.zeros((2, 6)), vision_dim=0, torch=torch
        )
        self.assertEqual(out["vision_dim"], 0.0)


if __name__ == "__main__":
    unittest.main()
