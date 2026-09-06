"""An instruction scored at a cap it never earned reports the evaluator, not the policy.

`--start-distance-cap` applies to EVERY instruction in the config, but the
approach ladders are per instruction and end at different rungs. A composed
put_into evaluation at 0.20 is correct for the container families and at the
same time puts pick_up -- whose earned cap was 0.080 -- at a start distance it
has never seen.

It scored 0/328 and was read as "pick_up destroyed" twice in a row, once by me
in a recommendation to abandon a checkpoint. Measured at its own 0.06 the same
checkpoint scores 0.1465, and the lineage it was being compared against scores
0.1745 -- statistically tied, where the reading said one was dead.

This is the campaign's §7.1 rule in a new place: a held-out evaluator that
ignores curriculum state evaluates a different reset distribution and can
invert the conclusion. move_to reads 0.630 at its earned cap and 0.172
uncapped, a 3.7x difference on one checkpoint.
"""

from __future__ import annotations

import unittest
import warnings

from tools.audit.sil_record import cap_verdicts, earned_caps


class EarnedCapsFromTheCheckpointTests(unittest.TestCase):
    PAYLOAD = {
        "extra_state": {
            "approach_curriculum": {
                "put_into_plate": {"cap": 0.17, "pass_rate_ema": 0.36},
                "put_into_bowl": {"cap": 0.10, "pass_rate_ema": 0.24},
                "pick_up": {"cap": 0.08, "pass_rate_ema": 0.24},
                "move_to_object": {"cap": 0.14, "pass_rate_ema": 0.47},
            }
        }
    }

    def test_it_reads_the_per_instruction_ladders(self) -> None:
        self.assertEqual(
            earned_caps(self.PAYLOAD),
            {"put_into_plate": 0.17, "put_into_bowl": 0.10,
             "pick_up": 0.08, "move_to_object": 0.14},
        )

    def test_a_checkpoint_without_approach_state_yields_nothing(self) -> None:
        # An SFT seed never ran a ladder. "unknown" is the honest verdict;
        # inventing a comparison would be worse than none.
        self.assertEqual(earned_caps({}), {})
        self.assertEqual(earned_caps(None), {})
        self.assertEqual(earned_caps({"extra_state": {}}), {})

    def test_a_malformed_entry_is_skipped_not_guessed(self) -> None:
        payload = {"extra_state": {"approach_curriculum": {
            "pick_up": {"cap": "not a number"},
            "put_into_plate": {"pass_rate_ema": 0.4},
            "put_into_bowl": {"cap": 0.10},
        }}}
        self.assertEqual(earned_caps(payload), {"put_into_bowl": 0.10})


class TheVerdictTests(unittest.TestCase):
    EARNED = {"put_into_plate": 0.17, "pick_up": 0.08, "move_to_object": 0.14}

    def test_the_case_that_cost_two_wrong_conclusions(self) -> None:
        """A composed eval at 0.20 across all four instructions."""

        v = cap_verdicts(
            ["put_into_plate", "pick_up", "move_to_object"],
            requested_cap=0.20, earned=self.EARNED,
        )
        self.assertEqual(v["put_into_plate"]["verdict"], "above_earned_cap")
        self.assertEqual(v["pick_up"]["verdict"], "above_earned_cap")
        self.assertEqual(v["move_to_object"]["verdict"], "above_earned_cap")
        self.assertEqual(v["pick_up"]["earned_cap"], 0.08)
        self.assertEqual(v["pick_up"]["requested_cap"], 0.2)

    def test_scoring_an_instruction_at_its_own_cap_is_clean(self) -> None:
        v = cap_verdicts(["pick_up"], requested_cap=0.06, earned=self.EARNED)
        # 0.06 is BELOW the earned 0.08 -- easier, which is fine and not the
        # dangerous direction.
        self.assertEqual(v["pick_up"]["verdict"], "below_earned_cap")
        v = cap_verdicts(["pick_up"], requested_cap=0.08, earned=self.EARNED)
        self.assertEqual(v["pick_up"]["verdict"], "at_earned_cap")

    def test_unknown_when_either_side_is_missing(self) -> None:
        self.assertEqual(
            cap_verdicts(["pick_up"], requested_cap=None, earned=self.EARNED)
            ["pick_up"]["verdict"], "unknown")
        self.assertEqual(
            cap_verdicts(["pick_up"], requested_cap=0.2, earned={})
            ["pick_up"]["verdict"], "unknown")

    def test_floating_point_equality_is_not_flagged(self) -> None:
        v = cap_verdicts(["pick_up"], requested_cap=0.08 + 1e-12,
                         earned={"pick_up": 0.08})
        self.assertEqual(v["pick_up"]["verdict"], "at_earned_cap")


class TheWarningTests(unittest.TestCase):
    def test_only_above_earned_cap_warns(self) -> None:
        from tools.audit.sil_record import _report_cap_verdicts

        clean = cap_verdicts(["pick_up"], requested_cap=0.06,
                             earned={"pick_up": 0.08})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _report_cap_verdicts(clean)
        self.assertEqual([w for w in caught if w.category is RuntimeWarning], [])

        risky = cap_verdicts(["pick_up"], requested_cap=0.20,
                             earned={"pick_up": 0.08})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _report_cap_verdicts(risky)
        messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
        self.assertEqual(len(messages), 1)
        self.assertIn("pick_up", messages[0])
        self.assertIn("0.2", messages[0])
        self.assertIn("0.08", messages[0])

    def test_the_message_names_the_consequence_not_just_the_mismatch(self) -> None:
        from tools.audit.sil_record import _report_cap_verdicts

        risky = cap_verdicts(["pick_up"], requested_cap=0.20,
                             earned={"pick_up": 0.08})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _report_cap_verdicts(risky)
        message = str(caught[0].message)
        self.assertIn("reset distribution", message)
        self.assertIn("0.630", message)


class TheCaveatTravelsWithTheNumberTests(unittest.TestCase):
    def test_the_summary_carries_it(self) -> None:
        """A console warning is missed; the json is what gets read later."""

        import inspect

        from tools.audit.sil_record import main

        source = inspect.getsource(main)
        self.assertIn('summary["cap_check"] = _cap_verdicts', source)
        self.assertIn("_report_cap_verdicts(_cap_verdicts)", source)
        # And it must run for the recording modes, where a rate is produced.
        self.assertIn('if args.mode in ("record", "oracle"):', source)

    def test_it_reads_the_payload_already_loaded(self) -> None:
        import inspect

        from tools.audit.sil_record import main

        self.assertIn("earned=earned_caps(world.payload)", inspect.getsource(main))


if __name__ == "__main__":
    unittest.main()
