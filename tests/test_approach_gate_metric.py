"""Which metric promotes the approach curriculum.

A gate wired to a metric the task never emits does not fail -- it freezes. The
pass rate is a clean 0.0, the EMA decays to 0.0, promotion is never due, and
every other curve looks healthy. move_to_object ran 1046 updates and 1.9M steps
that way: instruction_grasps_normal_start/move_to_object was 0 on every single
update (a move_to episode grasps nothing), so the cap never left 0.03 while the
instruction's own success rate climbed 0.115 -> 0.894.
"""

from __future__ import annotations

import unittest
import warnings


class GateMetricSelectionTests(unittest.TestCase):
    def test_only_pick_up_gates_on_the_grasp(self):
        """Success is the default; grasp is the narrow exception.

        Stated as a positive allowlist because the direction of the membership
        test is the whole bug: "placement on success, everything else on grasp"
        and "pick_up on grasp, everything else on success" differ only for the
        instructions nobody was thinking about when it was written.
        """

        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            _GRASP_GATED_INSTRUCTIONS,
        )
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            ACTIVE_INSTRUCTION_TYPES,
        )

        self.assertEqual(set(_GRASP_GATED_INSTRUCTIONS), {"pick_up"})
        for name in ACTIVE_INSTRUCTION_TYPES:
            if name == "pick_up":
                continue
            self.assertNotIn(
                name,
                _GRASP_GATED_INSTRUCTIONS,
                f"{name} would be gated on a grasp it never performs",
            )

    def test_move_to_is_not_routed_to_a_metric_it_never_emits(self):
        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        source = inspect.getsource(mod.main)
        self.assertIn("_GRASP_GATED_INSTRUCTIONS", source)
        # The old wording, which routed move_to to the grasp metric.
        self.assertNotIn('placement_names = {"put_into_plate"', source)


class DeadGateDetectorTests(unittest.TestCase):
    """The detector that would have caught this in minutes instead of hours."""

    @staticmethod
    def _metrics(name, *, worlds, successes):
        return {
            f"instruction_worlds_normal_start/{name}": float(worlds),
            f"instruction_successes_normal_start/{name}": float(successes),
        }

    def test_it_fires_on_the_run_that_actually_happened(self):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            _DEAD_GATE_UPDATES,
            _warn_on_structurally_dead_gate,
        )

        # The real numbers: 916 successes of 1024 worlds, gate reading zero.
        metrics = self._metrics("move_to_object", worlds=1024, successes=916)
        state: dict[str, int] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(_DEAD_GATE_UPDATES):
                _warn_on_structurally_dead_gate(
                    metrics,
                    pass_rates={"move_to_object": 0.0},
                    state=state,
                    promote_threshold=0.30,
                )
        self.assertEqual(len(caught), 1, "expected exactly one warning")
        self.assertIn("move_to_object", str(caught[0].message))

    def test_it_stays_quiet_while_the_instruction_is_genuinely_failing(self):
        """Early training reads zero everywhere and must not be accused."""

        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            _DEAD_GATE_UPDATES,
            _warn_on_structurally_dead_gate,
        )

        metrics = self._metrics("move_to_object", worlds=1024, successes=20)
        state: dict[str, int] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(_DEAD_GATE_UPDATES * 3):
                _warn_on_structurally_dead_gate(
                    metrics,
                    pass_rates={"move_to_object": 0.0},
                    state=state,
                    promote_threshold=0.30,
                )
        self.assertEqual(caught, [])

    def test_a_single_zero_update_resets_the_counter(self):
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            _DEAD_GATE_UPDATES,
            _warn_on_structurally_dead_gate,
        )

        metrics = self._metrics("pick_up", worlds=1024, successes=900)
        state: dict[str, int] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for index in range(_DEAD_GATE_UPDATES * 2):
                rate = 0.4 if index == _DEAD_GATE_UPDATES - 2 else 0.0
                _warn_on_structurally_dead_gate(
                    metrics,
                    pass_rates={"pick_up": rate},
                    state=state,
                    promote_threshold=0.30,
                )
        self.assertEqual(len(caught), 1)

    def test_no_worlds_is_not_evidence(self):
        """Instruction sampling is random per group; an absent slice is silent."""

        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            _DEAD_GATE_UPDATES,
            _warn_on_structurally_dead_gate,
        )

        metrics = self._metrics("move_to_object", worlds=0, successes=0)
        state: dict[str, int] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(_DEAD_GATE_UPDATES * 2):
                _warn_on_structurally_dead_gate(
                    metrics,
                    pass_rates={"move_to_object": 0.0},
                    state=state,
                    promote_threshold=0.30,
                )
        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
