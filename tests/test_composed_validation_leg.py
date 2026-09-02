"""The validation leg that measures the composed task instead of the curriculum's.

`validate_round` already refuses the pre-grasped pick_up stage
(`allow_prelifted=False`) with the right reason in its docstring: a held-out
rate measured on episodes that were handed the grasp moves with the training
knob rather than with the policy. Placement had the identical knob --
`caught_container_fraction` -- and no such refusal.

Phase 6 paid for that. `phase6_compose_iter0` annealed the caught fraction only
1.0 -> 0.9 -> 0.8, so the metric that steered 2.25M steps and fired the stop
rule was 80-90% carry-only episodes. Its 0.6240 peak was a caught-task reading.
The same checkpoint, measured afterwards under an explicit uncaught protocol,
scored composed plate 59/856 = 0.0689 and bowl 10/680 = 0.0147 against the
seed's 0.0935 and 0.0265.

These cover the three ways the new leg can be wired up wrong and still look
like it works: a flag that never reaches the resetter, a gate that turns it on
when the base validation is off, and a metric rename that collides with the
caught leg's keys.
"""

from __future__ import annotations

import inspect
import unittest


class ResetterAcceptsTheFlagTests(unittest.TestCase):
    def test_reset_and_validate_round_take_force_uncaught_container(self) -> None:
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
            RankLocalMJWarpGRPOCollector,
        )

        for owner, name in (
            (BatchedReverseFrontierResetter, "reset"),
            (RankLocalMJWarpGRPOCollector, "validate_round"),
        ):
            signature = inspect.signature(getattr(owner, name))
            self.assertIn(
                "force_uncaught_container",
                signature.parameters,
                f"{owner.__name__}.{name} cannot be asked for the composed task",
            )
            parameter = signature.parameters["force_uncaught_container"]
            self.assertIs(
                parameter.default,
                False,
                "the composed leg must be opt-in; every existing run has to "
                "behave exactly as it did before",
            )

    def test_validate_round_forwards_the_flag_to_the_reset(self) -> None:
        # A flag accepted and dropped is the worst outcome: the leg runs, the
        # metric appears, and it reports the caught task under a composed name.
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            RankLocalMJWarpGRPOCollector,
        )

        source = inspect.getsource(RankLocalMJWarpGRPOCollector.validate_round)
        self.assertIn("force_uncaught_container=bool(force_uncaught_container)", source)

    def test_forcing_does_not_consume_the_generator(self) -> None:
        """The composed scenes must be the caught scenes minus the grasp.

        The resetter draws from a seeded generator, and every branch in it is
        written to draw nothing while its stage is off precisely so the stream
        stays byte-identical. If the forcing branch called `torch.rand` it
        would ALSO reshuffle which scene each world gets, and the composed
        number would then differ from the caught number for two reasons at
        once -- with no way to separate them.
        """

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
        )

        source = inspect.getsource(BatchedReverseFrontierResetter.reset)
        start = source.index("if force_uncaught_container:")
        end = source.index("elif caught_fraction < 1.0:")
        forced_branch = source[start:end]
        self.assertNotIn("torch.rand", forced_branch)
        self.assertIn("is_container.clone()", forced_branch)

    def test_the_uncaught_horizon_floor_still_applies(self) -> None:
        # A composed episode has to make a grasp first, which costs ~30 env
        # steps however close the receptacle is; the floor is keyed off
        # `uncaught_container`, so forcing it on must pick the floor up for
        # free. Measured on the reference harness: 0/6 at a 64-step budget
        # against 4/6 at 128.
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
        )

        source = inspect.getsource(BatchedReverseFrontierResetter.reset)
        floor = source.index("placement_grasp_horizon_min_decisions")
        guard = source.rindex("if bool(uncaught_container.any().item()):", 0, floor)
        self.assertGreater(guard, source.index("uncaught_container = is_container.clone()"))


class ComposedGateTests(unittest.TestCase):
    class _Args:
        def __init__(self, every: int, base: int, composed: int) -> None:
            self.validation_every_steps = every
            self.validation_episodes_per_instruction = base
            self.composed_validation_episodes_per_instruction = composed

    def test_the_gate_requires_both(self) -> None:
        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            _composed_validation_enabled,
        )

        self.assertTrue(_composed_validation_enabled(self._Args(250000, 256, 128)))
        # Off by itself.
        self.assertFalse(_composed_validation_enabled(self._Args(250000, 256, 0)))
        # And never on when the base validation is off -- the composed leg
        # shares its collector, its seed and its trainer-eval bracket.
        self.assertFalse(_composed_validation_enabled(self._Args(0, 256, 128)))
        self.assertFalse(_composed_validation_enabled(self._Args(250000, 0, 128)))

    def test_a_run_without_the_flag_is_unchanged(self) -> None:
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        self.assertEqual(
            parse_args([]).composed_validation_episodes_per_instruction, 0
        )

    def test_the_config_key_matches_the_flag(self) -> None:
        """YAML keys become flags by `_` -> `-`, and a mismatch is silent.

        `composed_validation_episodes_per_instruction` in the config has to
        land on `--composed-validation-episodes-per-instruction`; a key named
        after the METRIC prefix (`validation_composed_...`) parses as valid
        YAML, produces a flag nothing defines, and the leg never runs.
        """

        from pathlib import Path

        from rl_vla_bootstrapping.core.commands import option_name
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import parse_args

        key = "composed_validation_episodes_per_instruction"
        namespace = parse_args([option_name(key), "128"])
        self.assertEqual(getattr(namespace, key), 128)

        config = Path(__file__).resolve().parents[1] / (
            "configs/examples/cdpr_smolvla_phase5_compose_loop.yaml"
        )
        self.assertIn(f"{key}:", config.read_text(encoding="utf-8"))


class MetricPrefixTests(unittest.TestCase):
    def test_every_key_is_renamed_so_nothing_collides(self) -> None:
        """Both legs land in ONE dict via `update()`.

        `_synchronize_validation_rounds` emits bare timing keys beside the
        `validation/`-prefixed ones -- render_time_s, smolvla_time_s,
        policy_time_s, physics_time_s, reward_time_s. Renaming only the
        prefixed half lets the composed leg's timings overwrite the caught
        leg's, silently, with numbers that still look like plausible timings.
        """

        prefix = "validation_composed"
        caught = {
            "validation/episodes": 512.0,
            "validation/success_rate": 0.62,
            "validation/by_instruction/put_into_bowl/success_rate": 0.46,
            "render_time_s": 1.5,
            "policy_time_s": 2.5,
        }
        renamed = {
            (
                f"{prefix}/{key[len('validation/'):]}"
                if key.startswith("validation/")
                else f"{prefix}/{key}"
            ): value
            for key, value in caught.items()
        }
        self.assertEqual(len(renamed), len(caught))
        self.assertEqual(set(renamed) & set(caught), set())
        self.assertIn("validation_composed/success_rate", renamed)
        self.assertIn("validation_composed/render_time_s", renamed)

    def test_the_trainer_uses_that_rename(self) -> None:
        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as module

        source = inspect.getsource(module._run_gpu_validation)
        self.assertIn('else f"{metric_prefix}/{key}"', source)

    def test_the_composed_leg_is_actually_called(self) -> None:
        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as module

        source = inspect.getsource(module.main)
        self.assertIn("_composed_validation_enabled(args)", source)
        self.assertIn("force_uncaught_container=True", source)
        self.assertIn('metric_prefix="validation_composed"', source)


if __name__ == "__main__":
    unittest.main()
