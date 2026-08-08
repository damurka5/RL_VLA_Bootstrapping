"""The XY approach probe has to discriminate before it is allowed to conclude.

The probe cannot be checked on this machine -- it needs MJWarp and a CUDA card --
so every part of it that does not need the GPU is checked here instead, and the
GPU run gets to be about physics rather than about typos.

Three things are worth a test:

* **The import surface.** Every repo symbol the probe reaches for exists, with
  the signature it is called with. This is the whole class of failures that
  otherwise shows up twenty minutes into a run on a booked box.
* **The discriminators actually discriminate.** ``_analyze_policy_trace``
  reports statistics whose only purpose is to separate "the policy servos" from
  "the policy drifts". Synthetic traces of each are fed in and the statistics
  are required to come out different in the stated direction. A statistic that
  reads the same for both would let the probe confirm whatever it was pointed
  at, which is the failure mode this whole exercise exists to avoid.
* **The sampled-cosine artefact is real.** The claim that the campaign's
  0.11-against-0.05 headline is a noise artefact is itself a claim, and it is
  reproduced here on a trace whose mean action is perfectly aimed by
  construction: at sigma 0.333 and the policy's own action magnitude, a cosine
  of 1.0 has to come out near the number the trainer logs.

Plus the plant-gain arithmetic, on a fake plant with a gain the test chose, and
the monkeypatch mechanics, on a fake collector.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tools" / "audit" / "xy_approach_probe.py"

try:
    import torch
except Exception:  # pragma: no cover - torch-less interpreter
    torch = None


def _load_probe():
    spec = importlib.util.spec_from_file_location("xy_approach_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


# --------------------------------------------------------------------------
# Synthetic traces
# --------------------------------------------------------------------------


def _trace(
    *,
    decisions: int,
    worlds: int,
    command: str,
    magnitude: float = 0.5,
    seed: int = 7,
):
    """A trace whose policy behaviour is known by construction.

    ``command`` is "servo" (aimed at the object), "drift" (a fixed world-frame
    direction, object ignored) or "zero".
    """

    rng = np.random.default_rng(seed)
    target = np.zeros((decisions, worlds, 3), dtype=np.float32)
    ee = np.zeros((decisions, worlds, 3), dtype=np.float32)
    # Objects scattered around the workspace; end-effectors start near them, as
    # the 5 cm curriculum cap arranges.
    object_xy = rng.uniform(-0.25, 0.25, size=(worlds, 2))
    start_offset = rng.normal(0.0, 0.035, size=(worlds, 2))
    drift = np.array([0.6, 0.4], dtype=np.float32)
    drift = drift / np.linalg.norm(drift)

    position = object_xy + start_offset
    rows = []
    for step in range(decisions):
        target[step, :, :2] = object_xy
        target[step, :, 2] = 0.20
        ee[step, :, :2] = position
        ee[step, :, 2] = 0.21
        rel = object_xy - position
        unit = rel / np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-9)
        if command == "servo":
            action_xy = unit * magnitude
        elif command == "drift":
            # A fixed world-frame direction plus state-INDEPENDENT jitter. The
            # jitter matters: a command that is constant to the last bit has no
            # variance for R^2 to explain, and a real policy fed a noisy vision
            # feature always has some. Testing against the exactly-constant case
            # only would leave the realistic one unchecked.
            action_xy = np.tile(drift, (worlds, 1)) * magnitude
            action_xy = action_xy + rng.normal(0.0, 0.1, size=(worlds, 2))
        else:
            action_xy = np.zeros((worlds, 2))
        mean = np.zeros((worlds, 5), dtype=np.float32)
        mean[:, :2] = action_xy
        prior = np.zeros((worlds, 5), dtype=np.float32)
        rows.append(
            {
                "ee_xyz": ee[step].copy(),
                "target_xyz": target[step].copy(),
                "prior0": prior,
                "policy_mean0": mean,
                "holding": np.zeros((worlds,), dtype=np.float32),
            }
        )
        # Move by the commanded step so the trajectory is consistent with the
        # command; a servo closes on the object and a drift runs away from it.
        position = position + action_xy * 0.015 * 4.0
    out = probe._Trace()
    out.rows = rows
    return out


class AnalysisDiscriminatesTest(unittest.TestCase):
    def test_servo_trace_reads_as_a_servo(self) -> None:
        metrics = probe._analyze_policy_trace(
            _trace(decisions=20, worlds=256, command="servo"),
            sigma=0.333,
            rng=np.random.default_rng(0),
        )
        self.assertGreater(metrics["mean_cosine_all_decisions"], 0.9)
        self.assertGreater(metrics["state_r2"], 0.9)
        # Objects lie in every direction, so a servo has no preferred world-frame
        # direction and no net mean command.
        self.assertLess(metrics["direction_concentration"], 0.2)
        self.assertLess(metrics["command_mean_norm"], metrics["command_spread"])

    def test_drift_trace_reads_as_a_drift(self) -> None:
        metrics = probe._analyze_policy_trace(
            _trace(decisions=20, worlds=256, command="drift"),
            sigma=0.333,
            rng=np.random.default_rng(0),
        )
        # Decision 0 is the comparable quantity -- it is where the trainer takes
        # its cosine, and where the end-effector has not yet run away.
        self.assertLess(abs(metrics["mean_cosine_decision0"]), 0.20)
        self.assertLess(metrics["state_r2"], 0.2)
        # Pooled over the episode the same drift reads strongly NEGATIVE, not
        # zero: a sustained fixed command ends up pointing away from wherever
        # the object was. Asserted so nobody later reads the pooled figure as if
        # it were the trainer's decision-0 one.
        self.assertLess(metrics["mean_cosine_all_decisions"], -0.5)
        # The signature of a state-independent command: every world commanded the
        # same way, all the length in the mean and none in the spread.
        self.assertGreater(metrics["direction_concentration"], 0.95)
        self.assertGreater(metrics["command_mean_norm"], 2.0 * metrics["command_spread"])
        # And it must travel, coherently, away from where the objects are.
        self.assertGreater(metrics["travel_direction_concentration"], 0.95)
        self.assertGreater(metrics["net_xy_travel_mean_m"], 0.05)

    def test_the_two_are_not_the_same_number(self) -> None:
        """The point of the leg: these traces must not report alike."""

        servo = probe._analyze_policy_trace(
            _trace(decisions=20, worlds=256, command="servo"),
            sigma=0.333,
            rng=np.random.default_rng(0),
        )
        drift = probe._analyze_policy_trace(
            _trace(decisions=20, worlds=256, command="drift"),
            sigma=0.333,
            rng=np.random.default_rng(0),
        )
        for key in (
            "mean_cosine_decision0",
            "state_r2",
            "direction_concentration",
        ):
            self.assertGreater(
                abs(servo[key] - drift[key]),
                0.5,
                msg=f"{key} does not separate a servo from a drift",
            )


class HuggingFaceEnvironmentTest(unittest.TestCase):
    """``RLVLA_HF_OFFLINE=1`` on the command line has to actually do something.

    It is read by ``scripts/huggingface_public_models.sh``, which only the
    launchers source -- so a tool invoked directly inherits neither the offline
    pin nor the credential strip, and the run dies after the weights have loaded
    with a 401 on a public repo. Checked by calling the mirror against a
    scratch environment rather than by trusting that the variable is honoured.
    """

    def setUp(self) -> None:
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "RLVLA_HF_OFFLINE",
                "RLVLA_HF_PUBLIC_MODELS_ONLY",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            )
        }
        for name in self._saved:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_offline_sets_both_switches_the_loader_reads(self) -> None:
        os.environ["RLVLA_HF_OFFLINE"] = "1"
        probe._configure_huggingface()
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

    def test_offline_is_off_by_default(self) -> None:
        probe._configure_huggingface()
        self.assertIsNone(os.environ.get("HF_HUB_OFFLINE"))
        self.assertIsNone(os.environ.get("TRANSFORMERS_OFFLINE"))

    def test_an_inherited_token_is_dropped(self) -> None:
        os.environ["HF_TOKEN"] = "stale"
        os.environ["HUGGING_FACE_HUB_TOKEN"] = "stale"
        probe._configure_huggingface()
        self.assertIsNone(os.environ.get("HF_TOKEN"))
        self.assertIsNone(os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        self.assertEqual(
            os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"), "1"
        )

    def test_a_genuinely_private_checkpoint_can_opt_out(self) -> None:
        os.environ["RLVLA_HF_PUBLIC_MODELS_ONLY"] = "0"
        os.environ["HF_TOKEN"] = "wanted"
        probe._configure_huggingface()
        self.assertEqual(os.environ.get("HF_TOKEN"), "wanted")

    def test_a_typo_in_either_switch_is_refused_not_ignored(self) -> None:
        os.environ["RLVLA_HF_OFFLINE"] = "true"
        with self.assertRaises(SystemExit):
            probe._configure_huggingface()
        os.environ["RLVLA_HF_OFFLINE"] = "0"
        os.environ["RLVLA_HF_PUBLIC_MODELS_ONLY"] = "yes"
        with self.assertRaises(SystemExit):
            probe._configure_huggingface()

    def test_it_matches_the_shell_helper_it_mirrors(self) -> None:
        """The shell script stays the source of truth for the variable names."""

        shell = (ROOT / "scripts" / "huggingface_public_models.sh").read_text()
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        ):
            self.assertIn(name, shell, msg=f"{name} is no longer in the helper")


class StartDistanceCapTest(unittest.TestCase):
    """The cap override has to be able to reproduce held-out validation.

    ``set_random_start_max_goal_distance`` is called on the training resetter
    only (smolvla_grpo_mjwarp_cdpr.py) and never on ``validation_resetter``, so
    validation runs at the resetter's ``inf`` default -- the full-workspace
    start distribution -- while training runs at the earned cap. Reproducing the
    run's own validation therefore means DISABLING the cap, and the override has
    to make that reachable from the command line.
    """

    class _Recorder:
        def __init__(self) -> None:
            self.caps = None

        def set_random_start_max_goal_distance(self, value) -> None:
            self.caps = value

        def set_prelifted_group_fraction(self, value) -> None:
            pass

    def _restore(self, override):
        recorder = self._Recorder()
        probe._restore_approach_curriculum(
            recorder,
            args=SimpleNamespace(instruction_types=("pick_up",)),
            task_metadata={},
            extra_state={},
            cap_override=override,
        )
        return recorder.caps

    def test_an_override_replaces_every_instruction_cap(self) -> None:
        caps = self._restore(0.05)
        self.assertTrue(caps)
        self.assertTrue(all(value == 0.05 for value in caps.values()))

    def test_infinity_is_representable_so_validation_can_be_reproduced(self) -> None:
        caps = self._restore(float("inf"))
        self.assertTrue(all(value == float("inf") for value in caps.values()))

    def test_no_override_leaves_the_checkpoint_caps_alone(self) -> None:
        caps = self._restore(None)
        self.assertIsNotNone(caps)

    def test_the_resetter_default_really_is_uncapped(self) -> None:
        """The premise of the whole override: unset means full workspace."""

        import inspect

        from rl_vla_bootstrapping.policy import mjwarp_rank_local_collector as mod

        source = inspect.getsource(mod.BatchedReverseFrontierResetter)
        self.assertIn('self.random_start_max_goal_distance = float("inf")', source)

    def test_the_trainer_caps_the_validation_resetter_too(self) -> None:
        """The regression guard for the bug this whole probe surfaced.

        For 52M steps the cap reached the training resetter only, so held-out
        validation ran full-workspace starts (and, through the coupling's
        uncapped branch, the full 26-decision budget) while training ran 5 cm
        starts and 17 decisions. Every validation number the campaign steered by
        was measuring a task the policy is never trained on. If the validation
        call site is ever dropped again, that silently comes back.
        """

        import inspect

        from rl_vla_bootstrapping.policy import smolvla_grpo_mjwarp_cdpr as mod

        lines = inspect.getsource(mod).splitlines()
        calls = [
            line.strip()
            for line in lines
            if "set_random_start_max_goal_distance(" in line
            and not line.strip().startswith("#")
        ]
        self.assertEqual(
            len(calls), 2, msg=f"expected two cap call sites, found {calls}"
        )
        self.assertTrue(
            any("validation" in line for line in calls),
            msg=f"no validation-resetter cap call among {calls}",
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class SampledSourceTest(unittest.TestCase):
    """The promote gate reads the sampled rate, so the probe has to produce it.

    Reporting only the deterministic rate invites exactly the comparison that
    was made once already: a deterministic 0.33 held up against a 0.30 gate that
    is fed a sampled number.
    """

    def _world(self):
        return probe._World(
            torch=torch,
            device=torch.device("cpu"),
            args=SimpleNamespace(),
            payload={},
            project=None,
            task_metadata={},
            backend=None,
            layout=SimpleNamespace(worlds_per_rank=4, group_size=2),
            resetter=None,
            action_step_xyz=0.015,
        )

    def test_noise_is_added_at_the_configured_width(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(3)
        source = probe._make_sampled_source(self._world(), sigma=0.333)
        chunk = torch.zeros((20000, 1, 5))
        out = source(runner=SimpleNamespace(_rng=generator), chunk=chunk)
        self.assertAlmostEqual(float(out.std()), 0.333, delta=0.02)
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.02)

    def test_the_action_box_is_respected(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(3)
        source = probe._make_sampled_source(self._world(), sigma=0.333)
        chunk = torch.full((5000, 1, 5), 0.95)
        out = source(runner=SimpleNamespace(_rng=generator), chunk=chunk)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreaterEqual(float(out.min()), -1.0)

    def test_it_is_not_the_deterministic_arm(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(3)
        source = probe._make_sampled_source(self._world(), sigma=0.333)
        chunk = torch.full((100, 1, 5), 0.2)
        out = source(runner=SimpleNamespace(_rng=generator), chunk=chunk)
        self.assertFalse(torch.allclose(out, chunk))


@unittest.skipIf(torch is None, "torch is unavailable")
class VisionAblationTest(unittest.TestCase):
    """Destroy the vision input and nothing else.

    The decisive test of where the policy's aim comes from. Its decision-0
    cosine reads ~0.19-0.40 while the feature probe puts the decodable
    direction at ~0.07; those cannot both describe aiming from vision. A
    success-vs-failure comparison cannot separate them, because selecting on
    success selects on alignment for ANY policy including a blind one.

    The ablation only answers it if it is surgical. Touch a proprioception
    column and the policy loses its own pose; touch a prior column and it loses
    the action it is a residual on; either way a collapse would be read as
    "vision mattered".
    """

    PROPRIO = 6
    VISION = 12
    STATE = PROPRIO + VISION

    def _world(self):
        return probe._World(
            torch=torch,
            device=torch.device("cpu"),
            args=SimpleNamespace(),
            payload={},
            project=None,
            task_metadata={},
            backend=None,
            layout=SimpleNamespace(worlds_per_rank=8, group_size=2),
            resetter=None,
            trainer=SimpleNamespace(state_dim=self.STATE),
            collector=SimpleNamespace(vision_feature_dim=self.VISION),
            action_step_xyz=0.015,
        )

    def _runner(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(5)
        return SimpleNamespace(_rng=generator)

    def _states(self):
        states = torch.arange(8 * self.STATE, dtype=torch.float32)
        return states.reshape(8, self.STATE)

    def test_zero_blanks_only_the_vision_block(self) -> None:
        transform = probe._make_vision_ablation(self._world(), mode="zero")
        states = self._states()
        out = transform(runner=self._runner(), states=states)
        self.assertEqual(float(out[:, self.PROPRIO :].abs().sum()), 0.0)
        self.assertTrue(
            torch.equal(out[:, : self.PROPRIO], states[:, : self.PROPRIO])
        )

    def test_shuffle_preserves_the_feature_distribution_exactly(self) -> None:
        """The point of shuffling: only the correspondence is destroyed.

        Zeroing is off-distribution -- the residual has never seen an all-zero
        vision feature -- so a collapse there could be shock rather than lost
        information. A permutation keeps every value the batch contained.
        """

        transform = probe._make_vision_ablation(self._world(), mode="shuffle")
        states = self._states()
        out = transform(runner=self._runner(), states=states)
        before = torch.sort(states[:, self.PROPRIO :].flatten()).values
        after = torch.sort(out[:, self.PROPRIO :].flatten()).values
        self.assertTrue(torch.equal(before, after))

    def test_shuffle_actually_moves_the_rows(self) -> None:
        transform = probe._make_vision_ablation(self._world(), mode="shuffle")
        states = self._states()
        out = transform(runner=self._runner(), states=states)
        self.assertFalse(
            torch.equal(out[:, self.PROPRIO :], states[:, self.PROPRIO :])
        )

    def test_shuffle_leaves_proprioception_with_its_own_world(self) -> None:
        """Permuting proprioception too would ablate the pose, not vision."""

        transform = probe._make_vision_ablation(self._world(), mode="shuffle")
        states = self._states()
        out = transform(runner=self._runner(), states=states)
        self.assertTrue(
            torch.equal(out[:, : self.PROPRIO], states[:, : self.PROPRIO])
        )

    def test_the_input_is_not_modified_in_place(self) -> None:
        """validate_round reuses the state tensor after the policy call."""

        transform = probe._make_vision_ablation(self._world(), mode="zero")
        states = self._states()
        before = states.clone()
        transform(runner=self._runner(), states=states)
        self.assertTrue(torch.equal(states, before))

    def test_a_run_without_a_vision_feature_is_refused(self) -> None:
        world = self._world()
        world.collector = SimpleNamespace(vision_feature_dim=0)
        with self.assertRaises(ValueError):
            probe._make_vision_ablation(world, mode="shuffle")

    def test_an_unknown_mode_is_refused(self) -> None:
        transform = probe._make_vision_ablation(self._world(), mode="nonsense")
        with self.assertRaises(ValueError):
            transform(runner=self._runner(), states=self._states())

    def test_the_transform_runs_before_the_policy_forward(self) -> None:
        """A post-hoc edit of the ACTION cannot answer what the policy does
        without the feature; the intervention has to be on the input."""

        collector = _FakeCollector(torch, worlds=64, decisions=3)
        world = probe._World(
            torch=torch,
            device=torch.device("cpu"),
            args=SimpleNamespace(),
            payload={},
            project=None,
            task_metadata={},
            backend=collector.backend,
            layout=SimpleNamespace(worlds_per_rank=64, group_size=8),
            resetter=collector.resetter,
            collector=collector,
            action_step_xyz=0.015,
        )
        seen: list = []

        def transform(*, runner, states):
            seen.append(states.clone())
            return states * 0.0

        with probe._ArmRunner(
            world, source=None, state_transform=transform
        ) as runner:
            runner.run(round_index=0)
        self.assertEqual(len(seen), 3)


class CommandCosineTest(unittest.TestCase):
    """The ladder priced localization in metres; the logs report a cosine.

    "5 cm of position error" cannot be read against
    `residual_target_cosine_mean = 0.055`. Measuring the alignment each arm
    actually COMMANDED puts the arms and the policy on one axis, which is what
    turns the ladder into a spec for how well the feature has to localize.
    """

    def _trace(self, command_fn, decisions=6, worlds=64, holding=False):
        rng = np.random.default_rng(4)
        object_xy = rng.uniform(-0.2, 0.2, size=(worlds, 2))
        rows = []
        for _ in range(decisions):
            ee = np.zeros((worlds, 3), dtype=np.float32)
            ee[:, 2] = 0.21
            target = np.zeros((worlds, 3), dtype=np.float32)
            target[:, :2] = object_xy
            target[:, 2] = 0.20
            commanded = np.zeros((worlds, 5), dtype=np.float32)
            commanded[:, :2] = command_fn(object_xy - ee[:, :2])
            rows.append(
                {
                    "ee_xyz": ee,
                    "target_xyz": target,
                    "commanded0": commanded,
                    "holding": np.full(
                        (worlds,), 1.0 if holding else 0.0, dtype=np.float32
                    ),
                }
            )
        trace = probe._Trace()
        trace.rows = rows
        return trace

    def test_a_perfect_servo_reads_one(self) -> None:
        trace = self._trace(lambda rel: rel / np.linalg.norm(rel, axis=-1, keepdims=True))
        self.assertAlmostEqual(
            probe._command_cosine(trace)["command_cosine"], 1.0, places=5
        )

    def test_an_inverted_servo_reads_minus_one(self) -> None:
        trace = self._trace(lambda rel: -rel)
        self.assertAlmostEqual(
            probe._command_cosine(trace)["command_cosine"], -1.0, places=5
        )

    def test_a_constant_command_reads_about_zero(self) -> None:
        """Objects lie in every direction, so a fixed command averages out."""

        trace = self._trace(lambda rel: np.tile(np.array([1.0, 0.0]), (len(rel), 1)))
        self.assertLess(
            abs(probe._command_cosine(trace)["command_cosine"]), 0.2
        )

    def test_a_corrupted_servo_lands_between(self) -> None:
        """The ladder's whole point: more error, lower cosine, monotonically."""

        rng = np.random.default_rng(11)
        previous = 1.1
        for sigma in (0.02, 0.05, 0.10, 0.20):
            trace = self._trace(
                lambda rel, s=sigma: rel + rng.normal(0.0, s, size=rel.shape)
            )
            value = probe._command_cosine(trace)["command_cosine"]
            self.assertLess(value, previous)
            self.assertGreater(value, -0.5)
            previous = value

    def test_holding_steps_are_excluded(self) -> None:
        """A world already holding has no meaningful direction to the object."""

        trace = self._trace(lambda rel: rel, holding=True)
        self.assertTrue(
            np.isnan(probe._command_cosine(trace)["command_cosine"])
        )

    def test_the_ladder_reaches_the_region_the_feature_sits_in(self) -> None:
        """A spec that only covers the comfortable end answers nothing.

        The feature probe puts the decodable direction at ~0.07 cosine. At the
        ~3.8 cm start distance, cos ~ d/sqrt(d^2+sigma^2), so reaching 0.07
        needs sigma around 0.5 m -- the ladder has to go that far or it cannot
        say whether what the encoder offers is survivable.
        """

        self.assertGreaterEqual(max(probe.DEFAULT_LOCALIZATION_ERRORS), 0.5)


class GraspTimingTest(unittest.TestCase):
    """Horizon starvation and bad grasps must not report alike.

    ``oracle_xy`` reaches an ever-grasped rate of 0.85 and converts half of it.
    Whether that is "the grasp latched too late to lift in the remaining budget"
    or "grasps earned during an approach are worse" decides whether the fix is
    the curriculum cap or the grasp itself, so the statistic has to separate
    them on traces where the answer is built in.
    """

    def _trace_and_success(self, *, grasp_decision, converts):
        """``grasp_decision[w]`` is when world w latches (-1 for never)."""

        decisions = 16
        worlds = len(grasp_decision)
        rows = []
        for step in range(decisions):
            holding = np.array(
                [
                    1.0 if 0 <= g <= step else 0.0
                    for g in grasp_decision
                ],
                dtype=np.float32,
            )
            rows.append(
                {
                    "ee_xyz": np.zeros((worlds, 3), dtype=np.float32),
                    "target_xyz": np.zeros((worlds, 3), dtype=np.float32),
                    "prior0": np.zeros((worlds, 5), dtype=np.float32),
                    "policy_mean0": np.zeros((worlds, 5), dtype=np.float32),
                    "holding": holding,
                }
            )
        trace = probe._Trace()
        trace.rows = rows
        return trace, np.array(converts, dtype=bool)

    def test_horizon_starvation_shows_conversion_climbing_with_budget(self) -> None:
        # Early grasps convert, late ones do not -- the starvation signature.
        grasp = [1] * 40 + [14] * 40
        converts = [True] * 40 + [False] * 40
        trace, success = self._trace_and_success(
            grasp_decision=grasp, converts=converts
        )
        out = probe._grasp_timing(trace, success, horizon=16)
        self.assertAlmostEqual(out["ever_grasped_rate"], 1.0, places=6)
        self.assertAlmostEqual(out["conversion_given_grasp"], 0.5, places=6)
        by = {
            item["decisions_remaining"]: item
            for item in out["conversion_by_decisions_remaining"]
        }
        self.assertAlmostEqual(by["0-4"]["conversion"], 0.0, places=6)
        self.assertAlmostEqual(by["12-+"]["conversion"], 1.0, places=6)

    def test_bad_grasps_show_conversion_flat_in_budget(self) -> None:
        # Half convert regardless of when they latched.
        grasp = [1, 1, 14, 14] * 20
        converts = [True, False, True, False] * 20
        trace, success = self._trace_and_success(
            grasp_decision=grasp, converts=converts
        )
        out = probe._grasp_timing(trace, success, horizon=16)
        by = {
            item["decisions_remaining"]: item
            for item in out["conversion_by_decisions_remaining"]
        }
        self.assertAlmostEqual(by["0-4"]["conversion"], 0.5, places=6)
        self.assertAlmostEqual(by["12-+"]["conversion"], 0.5, places=6)

    def test_worlds_that_never_grasp_are_excluded_not_counted_as_decision_zero(
        self,
    ) -> None:
        """``argmax`` on an all-False row returns 0, which is a real decision.

        Without the ``ever`` mask those worlds would land in the highest
        decisions-remaining bucket with success False, manufacturing exactly the
        flat-conversion signature that means "the grasps are bad".
        """

        grasp = [-1] * 60 + [2] * 20
        converts = [False] * 60 + [True] * 20
        trace, success = self._trace_and_success(
            grasp_decision=grasp, converts=converts
        )
        out = probe._grasp_timing(trace, success, horizon=16)
        self.assertAlmostEqual(out["ever_grasped_rate"], 0.25, places=6)
        self.assertAlmostEqual(out["conversion_given_grasp"], 1.0, places=6)
        counted = sum(
            item["worlds"]
            for item in out["conversion_by_decisions_remaining"]
        )
        self.assertEqual(counted, 20)


class LoraDetectionTest(unittest.TestCase):
    """The adapted prior is part of the policy, not an optional extra.

    Skipping the LoRA restore raises no error -- shapes match and every arm just
    silently measures a policy trained against a different prior. So the
    detection is asserted here, including the empty-but-present case that
    ``save`` writes when no adapter was attached.
    """

    def test_weights_present(self) -> None:
        self.assertTrue(
            probe._checkpoint_has_lora({"vla_lora": {"lora_A": [1.0]}})
        )

    def test_key_absent(self) -> None:
        self.assertFalse(probe._checkpoint_has_lora({"policy": {}}))

    def test_key_present_but_empty_is_not_weights(self) -> None:
        self.assertFalse(probe._checkpoint_has_lora({"vla_lora": None}))
        self.assertFalse(probe._checkpoint_has_lora({"vla_lora": {}}))


class DegenerateR2Test(unittest.TestCase):
    """A command with no variance must not score as perfectly explained.

    R^2 divides by the command's own variance. A policy emitting the same XY
    action in every world has none, so the score becomes float noise over float
    noise -- and it came out at 1.0, i.e. "the object direction explains this
    command perfectly", for a trace built to ignore the object entirely. Left
    unguarded that single number would have endorsed whichever conclusion it was
    read in support of.
    """

    def test_a_perfectly_constant_command_reports_nan_not_one(self) -> None:
        worlds, decisions = 128, 10
        rng = np.random.default_rng(5)
        object_xy = rng.uniform(-0.25, 0.25, size=(worlds, 2))
        rows = []
        for _ in range(decisions):
            mean = np.zeros((worlds, 5), dtype=np.float32)
            mean[:, 0] = 0.5  # identical in every world, every decision
            ee = np.zeros((worlds, 3), dtype=np.float32)
            ee[:, 2] = 0.21
            target = np.zeros((worlds, 3), dtype=np.float32)
            target[:, :2] = object_xy
            target[:, 2] = 0.20
            rows.append(
                {
                    "ee_xyz": ee,
                    "target_xyz": target,
                    "prior0": np.zeros((worlds, 5), dtype=np.float32),
                    "policy_mean0": mean,
                    "holding": np.zeros((worlds,), dtype=np.float32),
                }
            )
        trace = probe._Trace()
        trace.rows = rows
        metrics = probe._analyze_policy_trace(
            trace, sigma=0.333, rng=np.random.default_rng(0)
        )
        self.assertTrue(math.isnan(metrics["state_r2"]))
        self.assertLess(metrics["command_variance_per_sample"], 1.0e-8)
        # The statistics that DO stay meaningful must still call it a drift.
        self.assertGreater(metrics["direction_concentration"], 0.99)

    def test_a_command_with_real_variance_still_gets_a_score(self) -> None:
        metrics = probe._analyze_policy_trace(
            _trace(decisions=10, worlds=128, command="servo"),
            sigma=0.333,
            rng=np.random.default_rng(0),
        )
        self.assertFalse(math.isnan(metrics["state_r2"]))
        self.assertGreater(metrics["command_variance_per_sample"], 1.0e-8)


class SampledCosineArtefactTest(unittest.TestCase):
    """A perfectly aimed mean, read the way the trainer reads it.

    ``policy_target_cosine_mean`` is taken on the SAMPLED action, so it is the
    mean's alignment attenuated by sigma 0.333 relative to the mean's own
    magnitude. At the policy's measured XY magnitude a cosine of 1.0 comes out
    near the 0.11 the campaign quotes -- which is why 0.11 cannot be read as
    "the policy does not aim" without the magnitude reported next to it.
    """

    def test_small_but_perfect_mean_reads_as_noise_when_sampled(self) -> None:
        metrics = probe._analyze_policy_trace(
            _trace(decisions=20, worlds=512, command="servo", magnitude=0.05),
            sigma=0.333,
            rng=np.random.default_rng(3),
        )
        self.assertGreater(metrics["mean_cosine_all_decisions"], 0.99)
        self.assertLess(metrics["sampled_cosine_all_decisions"], 0.30)

    def test_a_large_mean_survives_the_noise(self) -> None:
        """The attenuation is not unconditional -- it is a magnitude effect."""

        metrics = probe._analyze_policy_trace(
            _trace(decisions=20, worlds=512, command="servo", magnitude=0.9),
            sigma=0.333,
            rng=np.random.default_rng(3),
        )
        self.assertGreater(metrics["sampled_cosine_all_decisions"], 0.85)


class TanhRecoveryTest(unittest.TestCase):
    def test_pre_tanh_and_residual_are_recovered_from_prior_and_mean(self) -> None:
        worlds = 64
        prior_value = 0.30
        residual_value = 1.00
        rows = []
        for _ in range(4):
            prior = np.full((worlds, 5), prior_value, dtype=np.float32)
            mean = np.full(
                (worlds, 5),
                math.tanh(prior_value + residual_value),
                dtype=np.float32,
            )
            rows.append(
                {
                    "ee_xyz": np.tile(
                        np.array([0.0, 0.0, 0.21], dtype=np.float32), (worlds, 1)
                    ),
                    "target_xyz": np.tile(
                        np.array([0.1, 0.1, 0.20], dtype=np.float32), (worlds, 1)
                    ),
                    "prior0": prior,
                    "policy_mean0": mean,
                    "holding": np.zeros((worlds,), dtype=np.float32),
                }
            )
        trace = probe._Trace()
        trace.rows = rows
        metrics = probe._analyze_policy_trace(
            trace, sigma=0.333, rng=np.random.default_rng(0)
        )
        self.assertAlmostEqual(
            metrics["pre_tanh_abs_xy_mean"], prior_value + residual_value, places=4
        )
        self.assertAlmostEqual(
            metrics["residual_abs_xy_mean"], residual_value, places=4
        )
        expected_slope = 1.0 - math.tanh(prior_value + residual_value) ** 2
        self.assertAlmostEqual(
            metrics["tanh_slope_xy_mean"], expected_slope, places=4
        )


# --------------------------------------------------------------------------
# Plant leg, against a plant whose gain the test picked
# --------------------------------------------------------------------------


class _FakeBackend:
    """A plant with a chosen gain, an optional dead zone and a hard clamp."""

    def __init__(self, torch_module, *, worlds: int, gain: float, dead_zone: float):
        self.torch = torch_module
        self.worlds = worlds
        self.gain = float(gain)
        self.dead_zone = float(dead_zone)
        self.config = SimpleNamespace(
            workspace_x=(-0.28, 0.28), workspace_y=(-0.28, 0.28)
        )
        self._ee = torch_module.zeros((worlds, 3), dtype=torch_module.float32)
        self._ee[:, 2] = 0.21

    def low_dim_observations(self):
        return SimpleNamespace(ee_position=self._ee.clone())

    def step(self, action, active):
        delta = action[:, :3].clone()
        delta[delta.abs() < self.dead_zone] = 0.0
        self._ee = self._ee + delta * 0.015 * self.gain
        self._ee[:, 0].clamp_(-0.28, 0.28)
        self._ee[:, 1].clamp_(-0.28, 0.28)
        return self.low_dim_observations()

    def pop_nonfinite_world_events(self) -> int:
        return 0


def _plant_world(torch_module, *, worlds: int, gain: float, dead_zone: float):
    backend = _FakeBackend(
        torch_module, worlds=worlds, gain=gain, dead_zone=dead_zone
    )
    return probe._World(
        torch=torch_module,
        device=torch_module.device("cpu"),
        args=SimpleNamespace(),
        payload={},
        project=None,
        task_metadata={},
        backend=backend,
        layout=SimpleNamespace(worlds_per_rank=worlds, group_size=8),
        resetter=SimpleNamespace(reset=lambda **_: SimpleNamespace()),
        action_step_xyz=0.015,
    )


@unittest.skipIf(torch is None, "torch is unavailable")
class PlantGainTest(unittest.TestCase):
    def test_a_linear_plant_reports_unit_gain(self) -> None:
        world = _plant_world(torch, worlds=32, gain=1.0, dead_zone=0.0)
        row = probe._run_plant_arm(
            world, axis=0, amplitude=0.20, steps=16, round_index=0,
            allow_prelifted=False,
        )
        self.assertAlmostEqual(row["gain_fraction"], 1.0, places=3)
        self.assertAlmostEqual(
            row["mean_m_per_step"], 0.015 * 0.20, places=6
        )

    def test_a_dead_zone_shows_up_as_zero_gain_below_threshold(self) -> None:
        world = _plant_world(torch, worlds=32, gain=1.0, dead_zone=0.25)
        low = probe._run_plant_arm(
            world, axis=0, amplitude=0.10, steps=16, round_index=0,
            allow_prelifted=False,
        )
        world = _plant_world(torch, worlds=32, gain=1.0, dead_zone=0.25)
        high = probe._run_plant_arm(
            world, axis=0, amplitude=0.30, steps=16, round_index=0,
            allow_prelifted=False,
        )
        self.assertAlmostEqual(low["gain_fraction"], 0.0, places=4)
        self.assertAlmostEqual(high["gain_fraction"], 1.0, places=3)

    def test_a_half_gain_plant_is_not_mistaken_for_a_healthy_one(self) -> None:
        world = _plant_world(torch, worlds=32, gain=0.5, dead_zone=0.0)
        row = probe._run_plant_arm(
            world, axis=0, amplitude=0.30, steps=16, round_index=0,
            allow_prelifted=False,
        )
        self.assertAlmostEqual(row["gain_fraction"], 0.5, places=3)

    def test_clamped_samples_are_excluded_rather_than_read_as_dead(self) -> None:
        """A saturated command runs into the wall; that is not a dead zone.

        Without the exclusion, a large amplitude would report near-zero gain for
        the same reason a dead zone does, and the sweep would manufacture the
        very shape it exists to detect.
        """

        world = _plant_world(torch, worlds=32, gain=1.0, dead_zone=0.0)
        row = probe._run_plant_arm(
            world, axis=0, amplitude=1.0, steps=64, round_index=0,
            allow_prelifted=False,
        )
        self.assertGreater(row["clamped_fraction"], 0.5)
        self.assertAlmostEqual(row["gain_fraction"], 1.0, places=3)

    def test_zero_command_reports_the_uncommanded_drift(self) -> None:
        world = _plant_world(torch, worlds=32, gain=1.0, dead_zone=0.0)
        row = probe._run_plant_arm(
            world, axis=0, amplitude=0.0, steps=16, round_index=0,
            allow_prelifted=False,
        )
        self.assertAlmostEqual(row["mean_m_per_step"], 0.0, places=9)


# --------------------------------------------------------------------------
# Action sources
# --------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch is unavailable")
class OracleSourceTest(unittest.TestCase):
    def _world(self):
        return probe._World(
            torch=torch,
            device=torch.device("cpu"),
            args=SimpleNamespace(),
            payload={},
            project=None,
            task_metadata={},
            backend=None,
            layout=SimpleNamespace(worlds_per_rank=4, group_size=2),
            resetter=None,
            action_step_xyz=0.015,
        )

    def test_servo_saturates_beyond_one_step_and_is_proportional_inside(self) -> None:
        rel = torch.tensor([[0.15, 0.0], [0.0075, 0.0], [-0.15, 0.0]])
        out = probe._servo_xy(rel, 0.015, torch, max_command=1.0)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(out[1, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(out[2, 0]), -1.0, places=6)

    def test_the_command_cap_binds_without_touching_the_linear_region(self) -> None:
        """The cap keeps the arm out of the regime that resets worlds.

        A saturated sustained command drives the cable configuration singular
        and the backend restores that world to its base pose mid-episode, far
        from its object -- so an uncapped arm manufactures failures and reports
        them as the substitution not helping. The cap must bite on the far
        commands and leave the near ones proportional, or it trades one artefact
        for another.
        """

        rel = torch.tensor([[0.20, 0.0], [0.0030, 0.0], [-0.20, 0.0]])
        out = probe._servo_xy(rel, 0.015, torch, max_command=0.35)
        self.assertAlmostEqual(float(out[0, 0]), 0.35, places=6)
        self.assertAlmostEqual(float(out[2, 0]), -0.35, places=6)
        # 0.003 / 0.015 = 0.2, inside the cap and therefore untouched.
        self.assertAlmostEqual(float(out[1, 0]), 0.2, places=6)

    def test_oracle_xy_replaces_only_the_xy_channels(self) -> None:
        world = self._world()
        source = probe._make_oracle_xy_source(world)
        chunk = torch.full((4, 4, 5), 0.7)
        low_dim = SimpleNamespace(
            ee_position=torch.tensor(
                [
                    [0.0, 0.0, 0.21],
                    [0.0, 0.0, 0.21],
                    [0.1, 0.1, 0.21],
                    [0.0, 0.0, 0.21],
                ]
            )
        )
        target = torch.tensor(
            [
                [0.10, 0.00, 0.20],
                [-0.10, 0.00, 0.20],
                [0.10, 0.10, 0.20],
                [0.0075, 0.0, 0.20],
            ]
        )
        runner = SimpleNamespace(_rng=None)
        out = source(
            runner=runner, chunk=chunk, low_dim=low_dim, target=target
        )
        # z, yaw and gripper are the policy's, untouched.
        self.assertTrue(torch.allclose(out[:, :, 2:], chunk[:, :, 2:]))
        # Saturated toward +x, saturated toward -x, already there, half a step.
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(out[1, 0, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(out[2, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[3, 0, 0]), 0.5, places=6)
        # The command is held across the whole open-loop chunk, as replan_every
        # dictates.
        for index in range(4):
            self.assertAlmostEqual(
                float(out[0, index, 0]), float(out[0, 0, 0]), places=6
            )

    def test_localization_error_actually_corrupts_the_handed_over_position(self) -> None:
        world = self._world()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(11)
        source = probe._make_oracle_xy_source(world, position_error_std=0.20)
        chunk = torch.zeros((256, 1, 5))
        low_dim = SimpleNamespace(ee_position=torch.zeros((256, 3)))
        target = torch.zeros((256, 3))
        target[:, 0] = 0.10
        runner = SimpleNamespace(_rng=generator, position_error=None)
        out = source(runner=runner, chunk=chunk, low_dim=low_dim, target=target)
        # With 20 cm of error against a 10 cm offset, a good fraction of worlds
        # must be commanded the wrong way; a source that quietly ignored the
        # error would send every world to +1.
        wrong_way = float((out[:, 0, 0] < 0.0).float().mean())
        self.assertGreater(wrong_way, 0.10)
        self.assertLess(wrong_way, 0.50)

    def test_the_localization_error_is_held_for_the_whole_episode(self) -> None:
        """A per-decision redraw would let the servo average the error away.

        A feature that mislocalizes is wrong in a consistent direction, and over
        ~20 decisions independent redraws would cancel -- pricing a bad feature
        as far more usable than it is. Same shape as the z arc's finding that
        i.i.d. per-step noise explores a sustained bias only at sigma/sqrt(N).
        """

        world = self._world()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(11)
        source = probe._make_oracle_xy_source(world, position_error_std=0.20)
        chunk = torch.zeros((64, 1, 5))
        low_dim = SimpleNamespace(ee_position=torch.zeros((64, 3)))
        target = torch.zeros((64, 3))
        target[:, 0] = 0.10
        runner = SimpleNamespace(_rng=generator, position_error=None)
        first = source(
            runner=runner, chunk=chunk, low_dim=low_dim, target=target
        ).clone()
        held = runner.position_error.clone()
        second = source(
            runner=runner, chunk=chunk, low_dim=low_dim, target=target
        )
        self.assertTrue(torch.equal(runner.position_error, held))
        self.assertTrue(torch.allclose(first, second))
        # And a reset must clear it, or every episode inherits the first one's.
        runner.position_error = None
        third = source(
            runner=runner, chunk=chunk, low_dim=low_dim, target=target
        )
        self.assertFalse(torch.allclose(first, third))


# --------------------------------------------------------------------------
# Runner mechanics
# --------------------------------------------------------------------------


class _FakeTrainer:
    def __init__(self, torch_module, worlds: int):
        self.torch = torch_module
        self.worlds = worlds
        self.calls = 0

    def deterministic_action_chunks_tensor(self, *, states, priors, action_count):
        self.calls += 1
        return self.torch.full(
            (self.worlds, action_count, 5), 0.25, dtype=self.torch.float32
        )


class _FakeCollector:
    """Mimics validate_round's call pattern, not its physics."""

    def __init__(self, torch_module, *, worlds: int, decisions: int):
        self.torch = torch_module
        self.worlds = worlds
        self.decisions = decisions
        self.trainer = _FakeTrainer(torch_module, worlds)
        self.received: list = []
        self.resetter = SimpleNamespace(reset=self._reset)
        self.backend = SimpleNamespace(
            low_dim_observations=self._low_dim,
            pop_nonfinite_world_events=lambda: 0,
        )

    def _low_dim(self):
        objects = self.torch.zeros((self.worlds, 4, 3))
        objects[:, 0, 0] = 0.10
        return SimpleNamespace(
            ee_position=self.torch.zeros((self.worlds, 3)),
            object_positions=objects,
            gripper_opening=self.torch.ones((self.worlds,)),
        )

    def _reset(self, **_):
        return SimpleNamespace(
            task_state=SimpleNamespace(
                target_slots=self.torch.zeros(
                    (self.worlds,), dtype=self.torch.int64
                )
            ),
            physical_grasp=self.torch.zeros(
                (self.worlds,), dtype=self.torch.bool
            ),
            group_target_catalog_ids=self.torch.zeros(
                (self.worlds // 8,), dtype=self.torch.int64
            ),
            horizons=self.torch.full(
                (self.worlds,), self.decisions, dtype=self.torch.int64
            ),
        )

    def validate_round(self, *, round_index: int):
        self.resetter.reset(
            update_index=0, round_index=round_index, allow_prelifted=False
        )
        for _ in range(self.decisions):
            chunk = self.trainer.deterministic_action_chunks_tensor(
                states=self.torch.zeros((self.worlds, 8)),
                priors=self.torch.zeros((self.worlds, 4, 5)),
                action_count=4,
            )
            self.received.append(chunk.clone())
        shape = (self.worlds // 8, 8)
        return SimpleNamespace(
            candidate_success=self.torch.zeros(shape, dtype=self.torch.bool),
            candidate_rewards=self.torch.zeros(shape),
            final_xy_distance=self.torch.full(shape, 0.40),
            final_ee_z=self.torch.full(shape, 0.2006),
            min_ee_z=self.torch.full(shape, 0.174),
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class ArmRunnerTest(unittest.TestCase):
    def _world(self, collector):
        return probe._World(
            torch=torch,
            device=torch.device("cpu"),
            args=SimpleNamespace(),
            payload={},
            project=None,
            task_metadata={},
            backend=collector.backend,
            layout=SimpleNamespace(
                worlds_per_rank=collector.worlds, group_size=8
            ),
            resetter=collector.resetter,
            collector=collector,
            action_step_xyz=0.015,
        )

    def test_the_substituted_command_is_what_the_loop_receives(self) -> None:
        collector = _FakeCollector(torch, worlds=64, decisions=5)
        world = self._world(collector)
        source = probe._make_oracle_xy_source(world)
        with probe._ArmRunner(world, source=source) as runner:
            summary = runner.run(round_index=0)
        # Objects sit at +0.10 in x with the end-effector at the origin, so the
        # servo saturates to +1 and the untouched channels keep the policy's
        # 0.25. If the substitution had silently not taken, x would read 0.25.
        for chunk in collector.received:
            self.assertAlmostEqual(float(chunk[0, 0, 0]), 1.0, places=6)
            self.assertAlmostEqual(float(chunk[0, 0, 2]), 0.25, places=6)
        self.assertEqual(summary["decisions"], 5)
        self.assertEqual(summary["episodes"], 64)
        self.assertAlmostEqual(summary["final_distance_m"], 0.40, places=4)

    def test_the_baseline_arm_passes_the_policy_through_untouched(self) -> None:
        collector = _FakeCollector(torch, worlds=64, decisions=3)
        world = self._world(collector)
        with probe._ArmRunner(world, source=None) as runner:
            runner.run(round_index=0)
        for chunk in collector.received:
            self.assertTrue(torch.allclose(chunk, torch.full_like(chunk, 0.25)))
        self.assertEqual(len(runner.trace.rows), 3)

    def test_the_patch_is_removed_afterwards(self) -> None:
        collector = _FakeCollector(torch, worlds=64, decisions=2)
        world = self._world(collector)
        with probe._ArmRunner(world, source=None) as runner:
            runner.run(round_index=0)
            # The patch lives in the instance __dict__, shadowing the class
            # method. Identity on the bound method cannot be used here: every
            # attribute access on a class method mints a fresh bound object.
            self.assertIn(
                "deterministic_action_chunks_tensor",
                collector.trainer.__dict__,
            )
        self.assertNotIn(
            "deterministic_action_chunks_tensor", collector.trainer.__dict__
        )
        # The fake's resetter is a SimpleNamespace, so its reset IS an instance
        # attribute and must be put back rather than deleted. Compared with ==:
        # bound methods are equal when they wrap the same function and instance,
        # but each access mints a distinct object, so `is` would never hold.
        self.assertEqual(collector.resetter.reset, collector._reset)

    def test_the_trace_records_one_row_per_decision_with_the_right_widths(self) -> None:
        collector = _FakeCollector(torch, worlds=64, decisions=6)
        world = self._world(collector)
        with probe._ArmRunner(world, source=None) as runner:
            runner.run(round_index=0)
        self.assertEqual(runner.trace.stack("ee_xyz").shape, (6, 64, 3))
        self.assertEqual(runner.trace.stack("policy_mean0").shape, (6, 64, 5))
        self.assertEqual(runner.trace.stack("holding").shape, (6, 64))

    def test_the_horizon_override_rewrites_the_budget_and_is_reported(self) -> None:
        """C2 says conversion is budget-bound, so the budget has to be variable.

        ``horizons`` is what validate_round reads for both its decision count
        and its per-step active mask, so rewriting it in place is the whole
        override -- and the reported figure has to follow it, or a longer run
        would still be filed against the coupled budget.
        """

        collector = _FakeCollector(torch, worlds=64, decisions=4)
        world = self._world(collector)
        with probe._ArmRunner(
            world, source=None, horizon_override=26
        ) as runner:
            summary = runner.run(round_index=0)
        self.assertEqual(summary["horizon_decisions"], 26)
        self.assertEqual(runner.horizon_decisions, 26)

    def test_no_override_keeps_the_coupled_budget(self) -> None:
        collector = _FakeCollector(torch, worlds=64, decisions=4)
        world = self._world(collector)
        with probe._ArmRunner(world, source=None) as runner:
            summary = runner.run(round_index=0)
        self.assertEqual(summary["horizon_decisions"], 4)

    def test_analysis_runs_end_to_end_on_a_recorded_trace(self) -> None:
        """The recorded shape and the analysis's expected shape must agree."""

        collector = _FakeCollector(torch, worlds=64, decisions=6)
        world = self._world(collector)
        with probe._ArmRunner(world, source=None) as runner:
            runner.run(round_index=0)
        metrics = probe._analyze_policy_trace(
            runner.trace, sigma=0.333, rng=np.random.default_rng(0)
        )
        for key in (
            "mean_cosine_all_decisions",
            "state_r2",
            "direction_concentration",
            "start_xy_distance_mean_m",
            "tanh_saturated_fraction_xy",
        ):
            self.assertIn(key, metrics)


# --------------------------------------------------------------------------
# Import surface
# --------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch is unavailable")
class ImportSurfaceTest(unittest.TestCase):
    """Everything the probe reaches into the repo for, checked without a GPU."""

    def test_collector_symbols_exist(self) -> None:
        from rl_vla_bootstrapping.policy import mjwarp_rank_local_collector as mod

        for name in (
            "_FITTED_GRIPPER",
            "BatchedReverseFrontierResetter",
            "RankLocalCurriculum",
            "RankLocalMJWarpGRPOCollector",
        ):
            self.assertTrue(hasattr(mod, name), msg=name)
        self.assertTrue(
            hasattr(mod.RankLocalMJWarpGRPOCollector, "validate_round")
        )
        self.assertTrue(
            hasattr(mod.BatchedReverseFrontierResetter, "set_random_start_max_goal_distance")
        )
        self.assertTrue(
            hasattr(mod.BatchedReverseFrontierResetter, "set_prelifted_group_fraction")
        )

    def test_curriculum_symbols_exist_with_the_called_signatures(self) -> None:
        import inspect

        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
            PerInstructionApproachCurriculum,
            PreliftedStageCurriculum,
        )

        signature = inspect.signature(PerInstructionApproachCurriculum.__init__)
        self.assertIn("instruction_types", signature.parameters)
        approach = PerInstructionApproachCurriculum(
            {}, instruction_types=("pick_up",)
        )
        approach.load_state_dict(None)
        self.assertIsInstance(approach.caps_by_instruction_id(), dict)
        prelifted = PreliftedStageCurriculum({})
        prelifted.load_state_dict(None)
        self.assertIsInstance(prelifted.enabled, bool)

    def test_trainer_exposes_the_method_the_probe_substitutes(self) -> None:
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            SmolVLAGRPOTrainer,
        )

        self.assertTrue(
            hasattr(SmolVLAGRPOTrainer, "deterministic_action_chunks_tensor")
        )

    def test_backend_exposes_the_gripper_setpoint_the_oracle_reads(self) -> None:
        """full_oracle drives the gripper toward its fitted opening.

        It needs the COMMANDED opening, not the measured one, because the action
        is a delta on the set-point. That set-point lives on the backend as
        ``_controller_gripper``; if it is ever renamed, this is where it is
        noticed rather than mid-run.
        """

        import inspect

        from rl_vla_bootstrapping.simulation import mjlab_mjwarp_backend as mod

        source = inspect.getsource(mod)
        self.assertIn("_controller_gripper", source)

    def test_every_training_argument_the_probe_reads_exists(self) -> None:
        """The probe rebuilds the stack from ``payload["args"]``.

        Those keys are whatever ``parse_args`` produced when the checkpoint was
        written, so a name the probe reads but the parser never defines is an
        AttributeError that lands only after SmolVLA has finished loading on a
        booked GPU. Checked here against the parser's own dests instead.
        """

        from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import parse_args

        # Parsed on the default backend: argparse defines every dest regardless,
        # and mjlab_mjwarp additionally demands an XML path and a consistent
        # world/group split that have nothing to do with the question here.
        args = parse_args(["--device", "cpu", "--no-distributed"])
        required = (
            "hold_steps",
            "action_step_xyz",
            "action_step_yaw",
            "action_step_gripper",
            "lock_non_commanded_axes",
            "lock_non_commanded_axes_threshold",
            "render_width",
            "render_height",
            "object_slots",
            "mjwarp_nconmax",
            "mjwarp_njmax",
            "mjwarp_nccdmax",
            "reverse_frontier_promotion_success",
            "reverse_frontier_demotion_success",
            "reverse_frontier_validation_episodes",
            "reverse_frontier_min_train_updates",
            "reverse_frontier_saturation_abort_threshold",
            "validation_seed",
            "instruction_types",
            "allowed_objects",
            "base_checkpoint",
            "mixed_precision",
            "image_size",
            "state_dim",
            "image_feature_keys",
            "include_wrist",
            "include_aux_camera",
            "chunk_size",
            "action_dim",
            "smolvla_action_indices",
            "smolvla_action_normalization",
            "smolvla_model_image_size",
            "smolvla_compile_mode",
            "replan_every",
            "smolvla_inference_microbatch_size",
        )
        missing = [name for name in required if not hasattr(args, name)]
        self.assertEqual(missing, [], msg=f"parse_args does not define {missing}")

    def test_the_trainer_exposes_the_lora_restore_path_the_probe_uses(self) -> None:
        from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
            SmolVLAGRPOTrainer,
        )

        self.assertTrue(hasattr(SmolVLAGRPOTrainer, "attach_vla_lora"))
        self.assertTrue(hasattr(SmolVLAGRPOTrainer, "_load_vla_lora_state"))

    def test_reward_metadata_carries_the_pad_offset_the_oracle_aims_at(self) -> None:
        from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
            BatchedCatchReleaseDenseReward,
        )

        reward = BatchedCatchReleaseDenseReward()
        self.assertAlmostEqual(reward.pick_grasp_height_offset, 0.0075, places=6)


if __name__ == "__main__":
    unittest.main()
