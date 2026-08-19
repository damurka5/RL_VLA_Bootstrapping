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


class JoinKeyAgreementTests(unittest.TestCase):
    """The two sides are written by different code paths and must still meet.

    The first version of this join compared the frames stem against the raw
    episode_uid prefix. Both tests fabricated the two sides with the same
    string, so nothing could detect that the real writers disagree -- and they
    do: the replay writes frames_<X>.npz while the dataset keys episodes by
    <parent>/replay_<X>. It matched 0 of 33102 rows on the first real harvest.

    These derive both names the way sil_record actually derives them.
    """

    # sil_record --mode replay: stem = "<rung dir>_<record stem>"
    REPLAY_STEM = "cap_0.030_record_00"

    def test_the_writers_still_use_the_names_this_join_assumes(self):
        import inspect

        from tools.audit import sil_record

        source = inspect.getsource(sil_record.main)
        # replay writes both files from one stem...
        self.assertIn('f"replay_{stem}.npz"', source)
        self.assertIn('f"frames_{stem}.npz"', source)
        # ...and dataset keys episodes by parent AND stem of the replay.
        self.assertIn('f"{path.parent.name}/{path.stem}"', source)

    def test_the_real_pair_of_names_resolves_to_one_key(self):
        from tools.audit.sil_sft import frame_join_key

        frames_file = Path(f"frames_{self.REPLAY_STEM}.npz").stem
        dataset_key = f"replay/replay_{self.REPLAY_STEM}"
        self.assertEqual(
            frame_join_key(frames_file), frame_join_key(dataset_key)
        )
        self.assertEqual(frame_join_key(frames_file), self.REPLAY_STEM)

    def test_the_real_pair_of_names_joins_end_to_end(self):
        from tools.audit.sil_sft import resolve_frame_rows

        frames = _frames(self.REPLAY_STEM, worlds=[5], decisions=3)
        uid = np.array([f"replay/replay_{self.REPLAY_STEM}/r0w5"])
        keep, lookups = resolve_frame_rows(uid, np.array([1]), frames)
        self.assertTrue(keep.all(), "the real naming pair failed to join")
        self.assertEqual(lookups, [(self.REPLAY_STEM, 1, 0)])

    def test_two_rungs_do_not_collide_after_normalising(self):
        """Stripping prefixes must not strip the part that disambiguates.

        Every rung numbers its rounds from zero, so the rung directory is the
        only thing separating cap_0.030_record_00 from cap_0.050_record_00.
        """

        from tools.audit.sil_sft import frame_join_key

        self.assertNotEqual(
            frame_join_key("frames_cap_0.030_record_00"),
            frame_join_key("frames_cap_0.050_record_00"),
        )

    def test_a_bare_name_passes_through(self):
        from tools.audit.sil_sft import frame_join_key

        self.assertEqual(frame_join_key("cap_0.030_record_00"),
                         "cap_0.030_record_00")

    def test_the_gather_survives_the_full_round_trip(self):
        """resolve -> gather, with the real names. The KeyError this caught.

        The lookup tuple is what frames_for_rows indexes the frame index with,
        so a resolve that returns the raw uid prefix resolves cleanly and then
        raises on the gather -- past every check and into the training loop.
        """

        from tools.audit.sil_sft import frames_for_rows, resolve_frame_rows

        frames = _frames(self.REPLAY_STEM, worlds=[5, 9], decisions=3)
        uid = np.array(
            [
                f"replay/replay_{self.REPLAY_STEM}/r0w9",
                f"replay/replay_{self.REPLAY_STEM}/r0w5",
            ]
        )
        keep, lookups = resolve_frame_rows(uid, np.array([2, 0]), frames)
        self.assertTrue(keep.all())
        overview, wrist = frames_for_rows(lookups, frames, [0, 1])
        self.assertEqual(int(overview[0, 0, 0, 0]), (2 * 16 + 9) % 256)
        self.assertEqual(int(overview[1, 0, 0, 0]), 5)
        self.assertEqual(int(wrist[1, 0, 0, 0]), 105)


class PolicyKeySpaceTests(unittest.TestCase):
    """The written policy must occupy the key space of the one it replaces.

    Two different modules reach build_resume_payload: a bare ResidualChunkActor
    with keys "net.net.*", and the trainer's SmolVLAGRPOPolicy with keys
    "log_std" and "actor.net.net.*". The LoRA branch prefixed the second one a
    second time, producing "actor.log_std" and "actor.actor.net.net.*". Nothing
    complained until sil_record tried to load the result two tools later, after
    a full harvest and an SFT had been paid for.
    """

    SOURCE_KEYS = ("log_std", "actor.net.net.0.weight", "actor.net.net.0.bias")

    def _source(self):
        return {
            "policy": {key: 0 for key in self.SOURCE_KEYS},
            "optimizer": {},
            "extra_state": {},
            "args": {},
        }

    def test_the_matching_key_space_is_accepted(self):
        out = build_resume_payload(
            self._source(),
            policy_state={key: 1 for key in self.SOURCE_KEYS},
            lora_state=None,
            note={},
        )
        self.assertEqual(set(out["policy"]), set(self.SOURCE_KEYS))

    def test_the_double_prefix_is_refused_where_it_is_made(self):
        """The exact shape of the bug, named in the error."""

        with self.assertRaises(ValueError) as caught:
            build_resume_payload(
                self._source(),
                policy_state={
                    f"actor.{key}": 1 for key in self.SOURCE_KEYS
                },
                lora_state=None,
                note={},
            )
        message = str(caught.exception)
        self.assertIn("actor.actor.net.net.0.bias", message)
        self.assertIn("log_std", message)

    def test_a_missing_key_is_refused_too(self):
        """The residual stage must not drop log_std by forgetting to copy it."""

        with self.assertRaises(ValueError):
            build_resume_payload(
                self._source(),
                policy_state={
                    key: 1 for key in self.SOURCE_KEYS if key != "log_std"
                },
                lora_state=None,
                note={},
            )

    def test_a_source_without_a_policy_is_not_second_guessed(self):
        out = build_resume_payload(
            {"args": {}}, policy_state={"anything": 1}, lora_state=None, note={}
        )
        self.assertEqual(set(out["policy"]), {"anything"})


class LoraStageWiringTests(unittest.TestCase):
    """The two loads in the LoRA branch, pinned by source.

    strict=False on the load turned a total key mismatch into silence: the
    stage ran from an untrained residual and the loss curve looked normal.
    """

    def _source(self):
        import inspect

        from tools.audit import sil_sft

        return inspect.getsource(sil_sft.main)

    def test_the_residual_is_loaded_strictly(self):
        import re

        source = self._source()
        self.assertIn(
            "base.load_state_dict(best_policy_state, strict=True)", source
        )
        # Matched as a CALL, not as a substring: the comment above that line
        # names strict=False as the bug it fixes, and a bare substring test
        # would fail on the explanation rather than on the code.
        self.assertEqual(
            re.findall(r"load_state_dict\([^)]*strict\s*=\s*False", source), []
        )

    def test_the_policy_is_saved_without_a_second_prefix(self):
        source = self._source()
        self.assertIn("policy_state=base.state_dict()", source)

    def test_the_vision_tower_can_be_turned_on_for_the_sft_stage(self):
        """The RL config leaves it off; the design turns it on here."""

        import inspect

        from tools.audit import sil_sft

        source = inspect.getsource(sil_sft.build_runtime_and_trainer)
        self.assertIn('values["train_vla_vision_lora"] = True', source)
        # The tower's leaf names differ from the expert's, and reusing the
        # expert list matches almost nothing rather than raising.
        self.assertIn("out_proj", source)
        self.assertIn("fc1,fc2", source)


@unittest.skipUnless(
    __import__("importlib.util", fromlist=["util"]).find_spec("torch"),
    "torch is required",
)
class IntegrityControlTests(unittest.TestCase):
    """A difference with no control is the mistake phase 3 paid a week for.

    This recompute differs from the rollout in two expected ways -- the frames
    went through a uint8 round trip, and the batch is a few rows against the
    rollout's hundreds, which selects different bf16 kernels -- and in one fatal
    way, the frames being the wrong pictures. The control shares the first two
    and not the third.
    """

    def test_without_a_control_it_refuses_to_call_it(self):
        import torch

        from tools.audit.sil_sft import check_recomputed_vision

        out = check_recomputed_vision(
            torch.ones((2, 8)), torch.zeros((2, 8)), vision_dim=4, torch=torch
        )
        self.assertIn("uninterpretable", out["verdict"])

    def test_a_difference_near_the_floor_is_the_round_trip(self):
        import torch

        from tools.audit.sil_sft import check_recomputed_vision

        recorded = torch.zeros((1, 8))
        recomputed = recorded.clone()
        recomputed[:, -4:] += 0.10
        control = recomputed.clone()
        control[:, -4:] += 0.05  # floor of the same order as the headline
        out = check_recomputed_vision(
            recomputed, recorded, vision_dim=4, torch=torch,
            control_state=control,
        )
        self.assertIn("consistent with", out["verdict"])
        self.assertAlmostEqual(out["headline_over_control"], 2.0, places=3)

    def test_a_difference_far_above_the_floor_accuses_the_frames(self):
        import torch

        from tools.audit.sil_sft import check_recomputed_vision

        recorded = torch.zeros((1, 8))
        recomputed = recorded.clone()
        recomputed[:, -4:] += 1.00
        control = recomputed.clone()
        control[:, -4:] += 0.001  # a tight floor makes the headline damning
        out = check_recomputed_vision(
            recomputed, recorded, vision_dim=4, torch=torch,
            control_state=control,
        )
        self.assertIn("NOT explained", out["verdict"])
