"""Measure why a latched pick-up grasp does not become a lift.

Three questions, none of which needs a checkpoint or a training run:

**M1 -- is the loaded z axis dead-zoned?** Drive the scripted oracle to a real
latched grasp, then replace its lift phase with a CONSTANT commanded ``a_z`` and
sweep that constant. If the response is linear from zero, a policy can discover
the lift by perturbing its mean. If there is a dead zone, small sustained
commands do nothing and the lift is only reachable by a large sustained bias --
which per-step i.i.d. Gaussian exploration explores with std ``sigma/sqrt(N)``.

**M2 -- can exploration reach the lifting region?** Same latched grasp, but the
post-grasp command is the trainer's own sampler: ``N(mean_z, sigma)`` resampled
every policy decision, optionally plus a per-episode offset held for the whole
episode (``--episode-offset-std``, the exploration change this probe was written
to justify). Reports the peak-lift distribution and the success rate, so the
change can be priced before it costs GPU-days.

**M3 -- does the 8 mm pose test ever reject a real grasp?** Every arm reports
slip CONDITIONED on the pads being loaded. The trainer's own
``relative_position_slip_mean_m`` is not conditioned on contact, so it is
dominated by free-space approach steps and falls whenever the policy moves less.
This is the conditioned version, and it is the number that decides whether
``_GRASP_MAX_RELATIVE_POSITION_SLIP_M`` is implicated at all.

Reference readings on the CPU engine (``--physics mujoco_cpu``), 6-16 episodes
per arm, 64-step budget:

    sustained a_z    0.05   0.10   0.20   0.30   0.40   0.60   0.80
    median lift mm    1.8    1.8   17.8   46.7   74.1  124.9  186.5
    success/latched   0/6    0/6    0/6   7/14    6/6  14/14    6/6

    Gaussian mean_z 0.00, sigma 0.333: 0/14 success, median peak lift 10.5 mm
    max slip while loaded, over every arm: 0.46-3.52 mm against an 8 mm bound

Run it on MJWarp before trusting any of that: the CPU reference reads the same
MJCF but is a different integrator, and the dead-zone threshold can move.

Usage::

    python tools/audit/lift_barrier_probe.py --physics mjlab_mjwarp \\
        --output runs/lift_barrier_probe
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "render_cdpr_task_reference_episodes.py"
# The phases whose commands the probe replaces. Everything before them stays the
# scripted oracle, so every arm starts from a physically identical latched grasp
# and the arms differ only in what is commanded after the pads close.
DRIVEN_PHASES = frozenset({"lift_object", "hold_lifted"})
ACTIONS_PER_DECISION = 4
# Restricted pool: the full default catalog includes a banana the scripted
# oracle drops on its own, which adds a failure mode this probe is not about.
DEFAULT_CATALOGS = (
    "robocasa_apple",
    "robocasa_tomato",
    "robocasa_orange",
    "robocasa_potato",
)


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cdpr_oracle_reference_harness", HARNESS
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the oracle harness at {HARNESS}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Command:
    """A post-grasp command source with an explicit episode boundary."""

    def start_episode(self, rng: np.random.Generator) -> None:
        """Called once per episode, before its first post-latch decision."""

    def decide(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class _ConstantCommand(_Command):
    """A fixed sustained a_z -- the M1 plant-response probe."""

    def __init__(self, mean_z: float) -> None:
        self._vector = np.array((0.0, 0.0, float(mean_z)))

    def decide(self, _rng: np.random.Generator) -> np.ndarray:
        return self._vector


class _GaussianCommand(_Command):
    """The trainer's sampler: per-decision noise plus a per-episode offset.

    ``episode_offset_std`` 0 is exactly today's behaviour -- i.i.d. per-decision
    noise around a fixed mean. Above 0 an offset is drawn ONCE per episode and
    held, which is what ``--episode-offset-std`` adds to
    ``sample_action_chunks_tensor``: it widens the distribution over SUSTAINED
    bias from ``sigma/sqrt(N)`` to roughly the offset std itself.
    """

    def __init__(
        self, mean_z: float, sigma: float, episode_offset_std: float
    ) -> None:
        self._mean = np.array((0.0, 0.0, float(mean_z)))
        self._sigma = max(float(sigma), 0.0)
        self._offset_std = max(float(episode_offset_std), 0.0)
        self._offset = np.zeros(3)

    def start_episode(self, rng: np.random.Generator) -> None:
        self._offset = (
            rng.normal(loc=0.0, scale=self._offset_std, size=3)
            if self._offset_std > 0.0
            else np.zeros(3)
        )

    def decide(self, rng: np.random.Generator) -> np.ndarray:
        return (
            self._mean
            + self._offset
            + rng.normal(loc=0.0, scale=self._sigma, size=3)
        )


class _PostGraspDriver:
    """Replaces the oracle's lift command with a probe command.

    Holds whatever grip the latch established, exactly as the oracle does, so an
    arm that fails to lift fails on the z command and not on a grip that opened.
    """

    def __init__(
        self,
        harness: Any,
        *,
        command: _Command,
        rng: np.random.Generator,
    ) -> None:
        self._original = harness.oracle_action
        self._command = command
        self._rng = rng
        self._step = 0
        self._sample = np.zeros(3)
        self._in_episode = False

    def __call__(
        self,
        ctx: Any,
        obs: dict[str, Any],
        phase: Any,
        action_step_xyz: float,
        action_step_gripper: float,
    ) -> np.ndarray:
        if phase.name not in DRIVEN_PHASES:
            # Every episode passes through the oracle's approach phases before
            # it latches, so this is the episode boundary: reset the decision
            # clock and arm a fresh per-episode offset for the next latch.
            self._step = 0
            self._in_episode = False
            return self._original(
                ctx, obs, phase, action_step_xyz, action_step_gripper
            )
        if not self._in_episode:
            self._command.start_episode(self._rng)
            self._in_episode = True
        if self._step % ACTIONS_PER_DECISION == 0:
            self._sample = self._command.decide(self._rng)
        self._step += 1
        action = np.zeros(5, dtype=np.float64)
        action[:3] = np.clip(self._sample, -1.0, 1.0)
        action[3] = 0.0
        hold = float(ctx.frozen.get("hold_grip", ctx.close_opening))
        commanded = float(obs["commanded_gripper"])
        action[4] = np.clip(
            (hold - commanded) / float(action_step_gripper), -1.0, 1.0
        )
        return action


def _run_arm(
    harness: Any,
    *,
    label: str,
    command: _Command,
    episodes: int,
    seed: int,
    output: Path,
    physics: str,
    device: str,
    start_distance_cap: float | None,
    catalogs: Sequence[str],
) -> Path:
    driver = _PostGraspDriver(
        harness, command=command, rng=np.random.default_rng(seed)
    )
    original = harness.oracle_action
    harness.oracle_action = driver
    argv = [
        str(HARNESS),
        "--instructions", "pick_up",
        "--episodes-per-instruction", str(int(episodes)),
        "--physics", physics,
        "--device", device,
        "--no-video",
        "--continue-after-terminal",
        "--target-catalogs", *catalogs,
        "--output", str(output),
    ]
    if start_distance_cap is not None:
        argv += ["--start-distance-cap", str(float(start_distance_cap))]
    saved_argv = sys.argv
    sys.argv = argv
    try:
        status = harness.main()
    finally:
        sys.argv = saved_argv
        harness.oracle_action = original
    if int(status or 0) != 0:
        raise RuntimeError(f"Arm {label!r} exited with status {status}.")
    return output / "telemetry.csv"


def _summarize(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise RuntimeError(f"{path} is empty.")
    peaks: list[float] = []
    slips: list[float] = []
    successes = 0
    latched = 0
    budget = 0
    for episode in sorted({row["episode"] for row in rows}, key=int):
        episode_rows = [row for row in rows if row["episode"] == episode]
        budget = max(budget, len(episode_rows))
        held = [
            row
            for row in episode_rows
            if int(float(row["physical_grasp"]))
        ]
        if not held:
            continue
        latched += 1
        peaks.append(max(float(row["target_lift"]) for row in held) * 1000.0)
        slips.extend(
            float(row["relative_position_slip_m"]) * 1000.0 for row in held
        )
        if any(int(float(row["success"])) for row in episode_rows):
            successes += 1
    return {
        "episodes": len({row["episode"] for row in rows}),
        "latched": latched,
        "peaks_mm": peaks,
        "median_peak_mm": statistics.median(peaks) if peaks else 0.0,
        "successes": successes,
        "max_slip_while_held_mm": max(slips) if slips else 0.0,
        "mean_slip_while_held_mm": statistics.mean(slips) if slips else 0.0,
        "env_step_budget": budget,
    }


def _print_table(title: str, rows: Iterable[tuple[str, dict[str, Any]]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'arm':<34} {'latched':>7} {'median peak mm':>15} "
        f"{'success/latched':>16} {'max slip held mm':>17}"
    )
    for label, summary in rows:
        share = f"{summary['successes']}/{summary['latched']}"
        print(
            f"{label:<34} {summary['latched']:>7} "
            f"{summary['median_peak_mm']:>15.1f} {share:>16} "
            f"{summary['max_slip_while_held_mm']:>17.2f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--physics",
        choices=("auto", "mjlab_mjwarp", "mujoco_cpu"),
        default="auto",
        help="Engine. Use mjlab_mjwarp: that is what training runs.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "lift_barrier_probe")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--sweep",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.60],
        help="Sustained a_z values for the M1 plant-response sweep.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.333,
        help=(
            "Per-decision exploration std for M2. The default is exp(-1.10), "
            "the max_log_std ceiling the pick_up config pins log_std to."
        ),
    )
    parser.add_argument(
        "--policy-mean-z",
        type=float,
        default=0.0,
        help="Mean post-grasp a_z for the M2 arms.",
    )
    parser.add_argument(
        "--episode-offset-std",
        type=float,
        nargs="+",
        default=[0.0, 0.25],
        help=(
            "Per-episode offset stds to compare in M2. 0.0 is the current "
            "sampler; the others are what --episode-offset-std would buy."
        ),
    )
    parser.add_argument(
        "--start-distance-cap",
        type=float,
        default=0.20,
        help=(
            "Approach-curriculum cap. The episode horizon is coupled to it, so "
            "the default asks for the long horizon rather than the 3 cm "
            "starting cap's short one. The reported env_step_budget is the "
            "horizon actually used."
        ),
    )
    parser.add_argument(
        "--target-catalogs", nargs="+", default=list(DEFAULT_CATALOGS)
    )
    parser.add_argument(
        "--skip-plant-sweep",
        action="store_true",
        help="Run only the M2 exploration arms.",
    )
    args = parser.parse_args(argv)

    harness = _load_harness()
    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "episodes": int(args.episodes),
        "physics": str(args.physics),
        "device": str(args.device),
        "start_distance_cap": float(args.start_distance_cap),
        "catalogs": list(args.target_catalogs),
    }

    plant_rows: list[tuple[str, dict[str, Any]]] = []
    if not args.skip_plant_sweep:
        for index, mean_z in enumerate(args.sweep):
            label = f"sustained a_z={mean_z:.2f}"
            path = _run_arm(
                harness,
                label=label,
                command=_ConstantCommand(mean_z),
                seed=int(args.seed) + index,
                output=args.output / f"plant_az_{mean_z:.2f}",
                **common,
            )
            plant_rows.append((label, _summarize(path)))

    policy_rows: list[tuple[str, dict[str, Any]]] = []
    for index, offset_std in enumerate(args.episode_offset_std):
        label = (
            f"gauss mean={args.policy_mean_z:.2f} sd={args.sigma:.3f} "
            f"offset={offset_std:.2f}"
        )
        path = _run_arm(
            harness,
            label=label,
            command=_GaussianCommand(
                float(args.policy_mean_z), float(args.sigma), float(offset_std)
            ),
            seed=int(args.seed) + 1000 + index,
            output=args.output / f"policy_offset_{offset_std:.2f}",
            **common,
        )
        policy_rows.append((label, _summarize(path)))

    if plant_rows:
        _print_table(
            "M1 -- loaded plant response to a SUSTAINED z command", plant_rows
        )
        print(
            "\n  A dead zone (near-zero lift below some a_z, then a sharp rise) "
            "means the\n  lift is only reachable by a large sustained bias. A "
            "linear response from\n  zero falsifies the exploration-barrier "
            "reading and points back at the reward."
        )
    _print_table(
        "M2 -- what per-step vs per-episode exploration actually reaches",
        policy_rows,
    )
    print(
        "\n  offset=0.00 is today's sampler. If the offset arms lift materially "
        "further,\n  --episode-offset-std is worth the training run; if they do "
        "not, it is not."
    )

    all_rows = plant_rows + policy_rows
    worst_slip = max(
        (row["max_slip_while_held_mm"] for _, row in all_rows), default=0.0
    )
    budget = max((row["env_step_budget"] for _, row in all_rows), default=0)
    print(
        "\nM3 -- pose test, conditioned on the pads being loaded"
        "\n----------------------------------------------------"
        f"\n  worst slip while physically grasped, over every arm: "
        f"{worst_slip:.2f} mm"
        f"\n  detector bound (_GRASP_MAX_RELATIVE_POSITION_SLIP_M): 8.00 mm"
    )
    if worst_slip >= 8.0:
        print(
            "  -> the 8 mm bound DOES reject loaded grasps on this engine; the "
            "slip test is implicated."
        )
    else:
        print(
            "  -> the bound is never approached, so the grasp predicate is not "
            "what blocks the lift."
        )
    print(f"\nenv_step_budget actually used: {budget}")
    print(f"telemetry: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
