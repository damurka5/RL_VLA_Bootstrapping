#!/usr/bin/env python3
"""Phase-4 loop driver: watch an RL run, fire the harvest when it stalls.

Design in CDPR_PHASE4_LOOP_DESIGN.md. This is the outer driver -- RL runs as
its own two-rank torchrun process and this reads its checkpoints from outside.


Why the trigger is a stall and not a promotion
----------------------------------------------

The brief fired the harvest on "the cap went up". The arithmetic is against it:
a fifteen-update cooldown across nine rungs is a 120-update climb, against a
run of roughly one to four thousand updates, so the loop would fire nine times
in the first tenth of the run at 1.5-2 hours each.

Under the cost there is a better reason. A rising cap means RL is working. SFT
is for where it stops.


Why the checkpoint and not the logs
-----------------------------------

Every checkpoint carries ``extra_state["approach_curriculum"]``, which is
``{cap, pass_rate_ema, cooldown, dwell}`` per instruction -- the exact state the
gate acts on, written by the code that acts on it. Reading TensorBoard event
files instead would mean re-deriving that from a different serialization of a
subset of it, and this campaign has already paid for one metric that was
technically correct and meant something else.

``save_every_steps`` is 200k, which at this horizon is a checkpoint every two to
six updates -- coarse for counting updates, which is why every threshold below
is expressed in STEPS. Steps per update swing fourfold along the ladder (4096 at
horizon 8 against 16384 at 32), so an update-based threshold would mean
different things at the bottom and the top of the same run.


Modes
-----

``watch`` polls a run directory and reports the trigger state. Read-only; safe
to point at a live run, and the way to see how far a run is from firing.

``harvest`` runs the collection half of one iteration: a ladder of record
rounds, a smoothed replay of each with frames, and the pooled dataset.

The SFT half is not here yet -- it needs sil_sft extended to train LoRA from
frames, which is the next piece.

Usage::

    python tools/audit/sil_loop.py --mode watch --run-dir runs/<run>/rl \\
      --instruction move_to_object

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab \\
      python tools/audit/sil_loop.py --mode harvest --run-dir runs/<run>/rl \\
        --instruction move_to_object --config <config> --output runs/phase4/iter_1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# Reading the curriculum out of a checkpoint
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumSample:
    """One checkpoint's view of one instruction's approach curriculum."""

    global_step: int
    cap: float
    pass_rate_ema: float
    cooldown: float = 0.0
    dwell: float = 0.0


def read_curriculum_sample(
    checkpoint: Path, instruction: str
) -> CurriculumSample:
    """Pull ``extra_state["approach_curriculum"][instruction]`` out of a file.

    ``weights_only=False`` because the payload carries an argparse Namespace.
    The tensors are never touched -- only the small state dicts -- but torch
    still materializes them, so this is not free and the caller should not poll
    it faster than checkpoints appear.
    """

    import torch

    try:
        payload = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=False
        )
    except TypeError:  # pragma: no cover - PyTorch < 2.6
        payload = torch.load(Path(checkpoint), map_location="cpu")
    extra = dict(payload.get("extra_state") or {})
    approach = extra.get("approach_curriculum") or {}
    entry = approach.get(instruction)
    if entry is None and "cap" in approach:
        # Legacy single-curriculum checkpoints stored a flat dict, and the
        # trainer replays it into every instruction on load, so reading it that
        # way here matches what the run would actually do with it.
        entry = approach
    if not isinstance(entry, Mapping):
        raise SystemExit(
            f"{checkpoint} carries no approach-curriculum state for "
            f"{instruction!r}. Present: {sorted(approach) or '(nothing)'}. "
            "A checkpoint from a run that trained a different instruction "
            "cannot drive this loop."
        )
    return CurriculumSample(
        global_step=int(payload.get("global_step", 0)),
        cap=float(entry.get("cap", float("nan"))),
        pass_rate_ema=float(entry.get("pass_rate_ema", float("nan"))),
        cooldown=float(entry.get("cooldown", 0.0)),
        dwell=float(entry.get("dwell", 0.0)),
    )


def checkpoint_paths(run_dir: Path) -> list[Path]:
    """Step checkpoints in step order. ``latest.pt`` is deliberately excluded.

    latest.pt is a duplicate of whichever step directory was written last, so
    including it would put the newest sample in the history twice and let a
    single checkpoint satisfy a "held for N steps" test on its own.
    """

    found: list[tuple[int, Path]] = []
    for path in Path(run_dir).glob("step_*/smolvla_grpo_adapter.pt"):
        try:
            step = int(path.parent.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        found.append((step, path))
    return [path for _, path in sorted(found)]


# --------------------------------------------------------------------------
# The trigger
# --------------------------------------------------------------------------


@dataclass
class StallPolicy:
    """Thresholds for §2.4a, all in global steps."""

    # The cap must have been still for this long. Comfortably longer than the
    # 15-update cooldown, which at this horizon is 60k-250k steps, so a rung
    # that is merely waiting out its cooldown does not read as a stall.
    cap_still_steps: int = 600_000
    # And this much must have passed since the previous SFT, so two stalls in
    # quick succession do not both fire.
    min_steps_since_sft: int = 1_000_000
    # The gate the run is failing to cross. Read from the config rather than
    # assumed, because a run whose promote threshold was retuned would
    # otherwise be judged against a bar it was never given.
    promote_pass_rate: float = 0.30


@dataclass
class TriggerDecision:
    fire: bool
    reasons: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    cap: float = float("nan")
    pass_rate_ema: float = float("nan")
    global_step: int = 0
    steps_since_cap_change: int = 0
    at_ladder_top: bool = False


def evaluate_trigger(
    history: Sequence[CurriculumSample],
    *,
    policy: StallPolicy,
    ladder_top: float,
    last_sft_step: int = 0,
    cap_promoted_since_sft: bool = False,
) -> TriggerDecision:
    """Should the harvest fire, given the checkpoint history so far?

    Every condition is reported whether it passed or not. A trigger that only
    says "no" teaches nothing about how far away it is, and the whole point of
    watch mode is to answer that while the run is going.
    """

    if not history:
        return TriggerDecision(
            fire=False, blocked_by=["no checkpoints yet"]
        )
    ordered = sorted(history, key=lambda item: item.global_step)
    latest = ordered[-1]
    at_top = latest.cap >= float(ladder_top) - 1.0e-9

    # How long the cap has held its current value, measured from the earliest
    # checkpoint that already had it. Walking backwards rather than tracking a
    # change event keeps this a pure function of the history, so watch mode and
    # a resumed driver reach the same answer from the same files.
    still_since = latest.global_step
    for sample in reversed(ordered):
        if abs(sample.cap - latest.cap) > 1.0e-9:
            break
        still_since = sample.global_step
    steps_still = int(latest.global_step - still_since)

    window = [
        sample
        for sample in ordered
        if sample.global_step >= still_since
    ]
    ema_below = all(
        sample.pass_rate_ema < float(policy.promote_pass_rate)
        for sample in window
    )
    # Not merely below the bar -- not climbing toward it either. Compared
    # across the window's own endpoints, because an EMA that is still rising
    # will cross on its own and does not need a dataset.
    ema_rising = len(window) >= 2 and (
        window[-1].pass_rate_ema > window[0].pass_rate_ema + 1.0e-4
    )

    decision = TriggerDecision(
        fire=False,
        cap=latest.cap,
        pass_rate_ema=latest.pass_rate_ema,
        global_step=latest.global_step,
        steps_since_cap_change=steps_still,
        at_ladder_top=at_top,
    )

    def check(passed: bool, message: str) -> None:
        (decision.reasons if passed else decision.blocked_by).append(message)

    # A rung nobody has reached yet has no successes to harvest. At the ladder
    # top there is no further promotion to wait for, so the requirement lapses.
    check(
        at_top or cap_promoted_since_sft,
        "cap has promoted since the last SFT"
        if (at_top or cap_promoted_since_sft)
        else "cap has not promoted since the last SFT (nothing new to harvest)",
    )
    check(
        steps_still >= policy.cap_still_steps,
        f"cap still for {steps_still} steps (>= {policy.cap_still_steps})"
        if steps_still >= policy.cap_still_steps
        else f"cap still for only {steps_still} steps "
        f"(needs {policy.cap_still_steps})",
    )
    check(
        ema_below,
        f"pass-rate EMA below {policy.promote_pass_rate} across the window"
        if ema_below
        else f"pass-rate EMA reached {policy.promote_pass_rate} in the window; "
        "RL is about to promote on its own",
    )
    check(
        not ema_rising,
        "pass-rate EMA is flat or falling"
        if not ema_rising
        else f"pass-rate EMA is still rising "
        f"({window[0].pass_rate_ema:.4f} -> {window[-1].pass_rate_ema:.4f})",
    )
    since_sft = int(latest.global_step - int(last_sft_step))
    check(
        since_sft >= policy.min_steps_since_sft,
        f"{since_sft} steps since the last SFT "
        f"(>= {policy.min_steps_since_sft})"
        if since_sft >= policy.min_steps_since_sft
        else f"only {since_sft} steps since the last SFT "
        f"(needs {policy.min_steps_since_sft})",
    )
    # A gate reading a metric the task never emits pins the EMA at exactly zero
    # forever, which satisfies "below the threshold and not rising" perfectly.
    # That is the failure that cost this project ten hours, and a loop driver
    # that harvests on it would turn a frozen run into a frozen loop.
    dead_gate = (
        latest.pass_rate_ema == 0.0
        and steps_still > policy.cap_still_steps
        and all(sample.pass_rate_ema == 0.0 for sample in ordered)
    )
    check(
        not dead_gate,
        "pass-rate EMA has been nonzero at some point"
        if not dead_gate
        else "pass-rate EMA is EXACTLY 0.0 in every checkpoint -- this is a "
        "dead gate, not a stall; fix the gate rather than harvesting",
    )

    decision.fire = not decision.blocked_by
    return decision


# --------------------------------------------------------------------------
# Loop state
# --------------------------------------------------------------------------


@dataclass
class LoopState:
    iteration: int = 0
    last_sft_step: int = 0
    cap_at_last_sft: float = float("nan")
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "LoopState":
        if not Path(path).is_file():
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            iteration=int(payload.get("iteration", 0)),
            last_sft_step=int(payload.get("last_sft_step", 0)),
            cap_at_last_sft=float(
                payload.get("cap_at_last_sft", float("nan"))
            ),
            history=list(payload.get("history") or []),
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------
# Harvest
# --------------------------------------------------------------------------


def harvest_ladder(cap: float, *, rungs: Sequence[float]) -> list[float]:
    """The rungs to collect at, never above the cap actually reached.

    The brief's rule -- from nearly-finished to as far as the policy has got,
    and no farther -- is exactly this clip. Reported as a list so the caller
    can see that a low cap yields a short ladder rather than silently
    collecting one rung and calling it a ladder.
    """

    kept = [float(rung) for rung in rungs if rung <= float(cap) + 1.0e-9]
    return kept or [float(cap)]


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"[loop] $ {printable}", flush=True)
    if dry_run:
        return
    result = subprocess.run(list(command), check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"[loop] command failed with {result.returncode}: {printable}"
        )


def harvest_iteration(
    *,
    checkpoint: Path,
    config: Path,
    output: Path,
    instruction: str,
    rungs: Sequence[float],
    rounds: int,
    smooth_window: int,
    seed_torch: int,
    frame_worlds: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Record a ladder, replay each round smoothed with frames, pool a dataset."""

    output = Path(output)
    harvest_dir = output / "harvest"
    replay_dir = output / "replay"
    dataset_dir = output / "dataset"
    python = sys.executable
    tool = str(ROOT / "tools" / "audit" / "sil_record.py")
    recorded: list[str] = []

    for rung in rungs:
        rung_dir = harvest_dir / f"cap_{rung:.3f}"
        _run(
            [
                python, tool, "--mode", "record",
                "--rounds", str(int(rounds)),
                "--seed-torch", str(int(seed_torch)),
                "--start-distance-cap", str(float(rung)),
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--output", str(rung_dir),
            ],
            dry_run=dry_run,
        )
        for index in range(int(rounds)):
            recorded.append(str(rung_dir / f"record_{index:02d}.npz"))

    for actions in recorded:
        rung = Path(actions).parent.name.split("_", 1)[1]
        _run(
            [
                python, tool, "--mode", "replay",
                # Moving average, which strictly dominated ema and median on
                # both families in phase 3: zero phase lag against a controller
                # that re-anchors on the measured pose every step.
                "--smooth", "moving_average",
                "--smooth-window", str(int(smooth_window)),
                "--actions", actions,
                "--seed-torch", str(int(seed_torch)),
                "--start-distance-cap", rung,
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--record-frames",
                "--frame-worlds", str(int(frame_worlds)),
                "--output", str(replay_dir),
            ],
            dry_run=dry_run,
        )

    # Expanded here rather than passed as a glob: subprocess.run does not go
    # through a shell, so "replay_*.npz" would arrive at sil_record verbatim,
    # be taken as one nonexistent path, and the dataset would be built from
    # nothing -- or, worse, from whichever single file happened to match.
    replays = sorted(replay_dir.glob("replay_*.npz"))
    if not replays and not dry_run:
        raise SystemExit(
            f"No replay_*.npz under {replay_dir}; the smoothing pass produced "
            "nothing to build a dataset from."
        )
    _run(
        [
            python, tool, "--mode", "dataset",
            "--inputs", *[str(path) for path in replays],
            "--output", str(dataset_dir),
        ],
        dry_run=dry_run,
    )
    return {
        "instruction": instruction,
        "rungs": [float(rung) for rung in rungs],
        "rounds_per_rung": int(rounds),
        "recorded": recorded,
        "dataset": str(dataset_dir / "demonstrations.npz"),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _promote_threshold(config: Path | None, default: float) -> float:
    if config is None:
        return float(default)
    from rl_vla_bootstrapping.core.config import load_project_config

    metadata = dict(load_project_config(Path(config)).task.metadata or {})
    return float(
        metadata.get(
            "random_workspace_start_distance_promote_pass_rate", default
        )
    )


def _ladder(config: Path, instruction: str) -> list[float]:
    from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
        PerInstructionApproachCurriculum,
    )
    from rl_vla_bootstrapping.core.config import load_project_config

    metadata = dict(load_project_config(Path(config)).task.metadata or {})
    curriculum = PerInstructionApproachCurriculum(
        metadata, instruction_types=(instruction,)
    )
    item = curriculum._by_name[instruction]
    if item.ladder:
        return [float(rung) for rung in item.ladder]
    rungs: list[float] = []
    value = float(item.initial)
    while value <= float(item.final) + 1.0e-9 and len(rungs) < 64:
        rungs.append(round(value, 6))
        value += float(item.increment)
    return rungs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("watch", "harvest"), default="watch")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--instruction", default="move_to_object")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--cap-still-steps", type=int, default=600_000)
    parser.add_argument("--min-steps-since-sft", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--seed-torch", type=int, default=0)
    parser.add_argument("--frame-worlds", type=int, default=0)
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.0,
        help="watch only: keep polling every N seconds instead of reporting once.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    state_path = (
        args.state.expanduser().resolve()
        if args.state
        else run_dir.parent / "loop_state.json"
    )
    state = LoopState.load(state_path)
    policy = StallPolicy(
        cap_still_steps=int(args.cap_still_steps),
        min_steps_since_sft=int(args.min_steps_since_sft),
        promote_pass_rate=_promote_threshold(args.config, 0.30),
    )
    rungs = _ladder(args.config, args.instruction) if args.config else []
    ladder_top = max(rungs) if rungs else float("inf")

    def sample_history() -> list[CurriculumSample]:
        return [
            read_curriculum_sample(path, args.instruction)
            for path in checkpoint_paths(run_dir)
        ]

    def report() -> TriggerDecision:
        history = sample_history()
        decision = evaluate_trigger(
            history,
            policy=policy,
            ladder_top=ladder_top,
            last_sft_step=state.last_sft_step,
            cap_promoted_since_sft=(
                bool(history)
                and (
                    state.iteration == 0
                    or history[-1].cap > float(state.cap_at_last_sft) + 1.0e-9
                )
            ),
        )
        print(
            f"[loop] step={decision.global_step} cap={decision.cap:.4f} "
            f"ema={decision.pass_rate_ema:.4f} "
            f"still_for={decision.steps_since_cap_change} steps "
            f"checkpoints={len(history)} "
            f"{'FIRE' if decision.fire else 'hold'}",
            flush=True,
        )
        for line in decision.reasons:
            print(f"[loop]   ok      {line}", flush=True)
        for line in decision.blocked_by:
            print(f"[loop]   blocked {line}", flush=True)
        return decision

    if args.mode == "watch":
        while True:
            decision = report()
            if args.poll_seconds <= 0 or decision.fire:
                return 0 if decision.fire else 1
            time.sleep(float(args.poll_seconds))

    if args.config is None or args.output is None:
        raise SystemExit("--mode harvest needs --config and --output.")
    decision = report()
    checkpoints = checkpoint_paths(run_dir)
    if not checkpoints:
        raise SystemExit(f"No step checkpoints under {run_dir}.")
    ladder = harvest_ladder(decision.cap, rungs=rungs)
    print(
        f"[loop] harvesting {len(ladder)} rungs {ladder} at "
        f"{args.rounds} rounds each from {checkpoints[-1]}",
        flush=True,
    )
    report_payload = harvest_iteration(
        checkpoint=checkpoints[-1],
        config=args.config.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        instruction=args.instruction,
        rungs=ladder,
        rounds=int(args.rounds),
        smooth_window=int(args.smooth_window),
        seed_torch=int(args.seed_torch),
        frame_worlds=int(args.frame_worlds),
        dry_run=bool(args.dry_run),
    )
    report_payload["trigger"] = asdict(decision)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "harvest_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[loop] wrote {output / 'harvest_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
