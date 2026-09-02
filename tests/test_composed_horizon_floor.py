"""The composed episode has to fit in the budget it is given.

`placement_grasp_horizon_min_decisions` defaulted to 32 -- 128 env steps -- and
was sized on the uncaught branch's own smoke measurement: 0/6 at a 64-step
budget against 4/6 at 128, taking 77-119 steps, "9 steps of margin at the worst
case". Decomposed over the 24 576-episode oracle harvest, the margin was not
there:

    plate no_release   291 episodes, 291 ran the WHOLE budget  (1.000)
    bowl  no_release  2095 episodes, 1735 ran the whole budget (0.828)
    first grasp        plate p50 90  p90 116  |  bowl p50 81  p90 112
    ended still holding  plate 279 of 291     |  bowl 1549 of 2095

Those episodes never reached the release. Under the oracle `success | settled`
is 1.0000 for both receptacles and the object does not bounce out, so this is
the clock and nothing else.

These pin the value, the arithmetic behind it, and the path from the metadata
key to the number of env steps an episode actually gets -- because a floor that
is read but not applied, or applied and then overwritten, looks exactly like a
floor that works.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSED_CONFIGS = (
    "configs/examples/cdpr_smolvla_phase5_compose_loop.yaml",
    "configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml",
)


def _metadata(name: str) -> dict:
    from rl_vla_bootstrapping.core.config import load_project_config

    return dict(load_project_config(ROOT / name).task.metadata or {})


class TheFloorIsSetWhereCompositionRunsTests(unittest.TestCase):
    def test_every_composed_config_carries_it_explicitly(self) -> None:
        # Explicitly, not by default: the default is the value that was measured
        # to be too short, and a config that inherits it is running the old
        # experiment while claiming to run the new one.
        for name in COMPOSED_CONFIGS:
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("placement_grasp_horizon_min_decisions:", text, name)
            self.assertEqual(
                int(_metadata(name)["placement_grasp_horizon_min_decisions"]),
                40,
                name,
            )

    def test_forty_covers_the_measured_p90(self) -> None:
        # p90 first grasp is 116 env steps (plate) and the carry-and-release
        # after it needs ~40, so 156 steps = 39 decisions is the floor; 40 is
        # that with a little margin.
        decisions = int(_metadata(COMPOSED_CONFIGS[0])["placement_grasp_horizon_min_decisions"])
        self.assertGreaterEqual(decisions * 4, 116 + 40)

    def test_the_caught_placement_config_is_left_alone(self) -> None:
        """Only the COMPOSED task pays for this.

        The rollout loop runs `max_decisions = reset.horizons.max()` for the
        whole batch, so the longer floor costs ~25% wall clock on any update
        that contains an uncaught group. A caught-start placement run performs
        no grasp and must not pay it.
        """

        metadata = _metadata("configs/examples/cdpr_smolvla_phase4_placement_loop.yaml")
        self.assertEqual(
            int(metadata.get("placement_grasp_horizon_min_decisions", 32)), 32
        )


class TheFloorReachesTheEpisodeTests(unittest.TestCase):
    def test_it_is_a_maximum_against_the_curriculum_not_a_replacement(self) -> None:
        # The approach curriculum couples the horizon to the cap. The floor has
        # to raise a short coupled horizon without capping a long one, or a
        # far-rung episode would lose budget it had earned.
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
        )

        source = inspect.getsource(BatchedReverseFrontierResetter.reset)
        start = source.index("if bool(uncaught_container.any().item()):")
        block = source[start : start + 600]
        self.assertIn("torch.maximum(", block)
        self.assertIn("placement_grasp_horizon_min_decisions", block)

    def test_the_floor_is_applied_before_horizons_is_published(self) -> None:
        """Order is the whole correctness here.

        `horizons` is what the rollout loop bounds on
        (`decision < reset.horizons`) and what the recording stores. A floor
        applied after that assignment would raise a value nothing reads.
        """

        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            BatchedReverseFrontierResetter,
        )

        source = inspect.getsource(BatchedReverseFrontierResetter.reset)
        applied = source.index("placement_grasp_horizon_min_decisions")
        published = source.index("horizons=horizon_group.repeat_interleave")
        self.assertLess(applied, published)

    def test_the_predicate_timeout_does_not_shadow_it(self) -> None:
        # evaluate_active_sparse_tasks takes max_steps and times out on it. The
        # collector passes 10_000, so the real bound is reset.horizons; if that
        # were ever lowered to something near 128 the floor would be silently
        # capped by it.
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
            RankLocalMJWarpGRPOCollector,
        )

        source = inspect.getsource(RankLocalMJWarpGRPOCollector.collect_round)
        self.assertIn("max_steps=10_000", source)


class TheReharvestScriptTests(unittest.TestCase):
    SCRIPT = ROOT / "scripts/run_cdpr_phase7_reharvest.sh"

    def test_it_exists_and_is_executable(self) -> None:
        import os

        self.assertTrue(self.SCRIPT.is_file())
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))

    def test_it_refuses_to_reharvest_on_the_old_budget(self) -> None:
        """The precondition the whole script rests on.

        Harvesting again at 32 would spend hours of GPU reproducing the data
        already under o6_*. The guard reads the RESOLVED config value rather
        than trusting that the edit was made.
        """

        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn('if [[ "$HORIZON" -lt 40 ]]; then', text)
        self.assertIn("exit 2", text)

    def test_it_harvests_into_a_new_directory_and_pools_both(self) -> None:
        # A successful 128-step composed episode is not made wrong by a longer
        # budget existing, so the bank grows rather than churns.
        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("o7_demos", text)
        self.assertIn('"$BANK"/o6_demos/replay_*.npz', text)
        self.assertIn('"$BANK"/o7_demos/replay_*.npz', text)

    def test_it_checks_the_harvest_before_paying_for_the_sft(self) -> None:
        # The falsifiable claim is that the horizon was binding. Decomposing
        # both harvests costs seconds on a CPU and comes before the SFT.
        text = self.SCRIPT.read_text(encoding="utf-8")
        decomp = text.index("placement_failure_decomposition.py")
        sft = text.index("sil_sft.py")
        self.assertLess(decomp, sft)

    def test_it_guards_frames_not_replays(self) -> None:
        # sil_record writes the replay npz first and the frames npz last, so a
        # kill during frame compression leaves a complete replay beside a
        # truncated frames file. Phase 6 paid for this once.
        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn('[[ -f "$BANK/o7_demos/frames_o7_${cap}_${stem}.npz" ]] && continue', text)

    def test_it_starts_from_the_seed_not_the_rl_peak(self) -> None:
        # phase6_compose_iter0's peak is the better CAUGHT checkpoint and the
        # worse COMPOSED one (composed plate 0.0689 against the seed's 0.0935).
        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("sft_phase6/sil_sft_adapter.pt", text)
        self.assertNotIn("phase6_compose_iter0", text.split("# ---")[1])


if __name__ == "__main__":
    unittest.main()
