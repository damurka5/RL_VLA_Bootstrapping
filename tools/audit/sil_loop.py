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

``harvest`` runs a whole iteration: the ladder, the smoothed replays with
frames, the pooled dataset, the two SFT stages, the action-drift measurement,
and the verdict.

The verdict is a paired-by-round test against the pre-SFT checkpoint, and the
top rung's own harvest is reused as its baseline -- same checkpoint, same cap,
same seed. A single 512-world round cannot separate less than about five
points, so an unresolved result is reported as "no evidence" and the loop keeps
the checkpoint it already had. ``resume_checkpoint`` in the report is what RL
should resume from either way.

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
    # How much the pass-rate EMA may improve across the window and still count
    # as flat, comparing the two halves' means rather than the endpoints -- a
    # single noisy checkpoint at either end should not decide an iteration.
    #
    # 0.02 against the +0.09-per-100-updates a genuinely climbing top rung was
    # measured at, so a run in that state is nowhere near being called flat.
    plateau_delta: float = 0.02
    # And the plateau is judged over a RECENT span, not over everything since
    # the cap last moved. Those are different questions, and using one window
    # for both is backwards: the still-window only grows, so the climb that
    # follows a promotion stays in it forever and a LONGER stall becomes harder
    # to detect. Measured on iteration 0 at cap 0.19 -- across the whole
    # 8.2M-step still-window the EMA halves differ by +0.078 and the trigger
    # blocks, while the last ten checkpoints differ by -0.003 and it is plainly
    # flat.
    plateau_window_steps: int = 2_000_000


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
    # A micron of tolerance, not a float epsilon. The cap is built by repeated
    # `+= increment` from the initial rung, so the top arrives as
    # 0.18999999999999997 rather than 0.19, and a config that spells its ladder
    # out in YAML can differ in the last bits again. At 1e-9 the top rung is
    # simply never recognised, which silently disables the one condition that
    # lets the loop fire when there is no further promotion to wait for. 1e-6 m
    # is meaningless as a distance and safe as a tolerance -- the rungs are
    # 0.02 m apart.
    at_top = latest.cap >= float(ladder_top) - 1.0e-6

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
    # The plateau half-mean test reads only the recent tail of that window; the
    # "cap still for N steps" test above keeps the whole of it.
    recent_from = max(
        still_since, latest.global_step - int(policy.plateau_window_steps)
    )
    recent = [
        sample for sample in window if sample.global_step >= recent_from
    ] or window[-2:]
    ema_below = all(
        sample.pass_rate_ema < float(policy.promote_pass_rate)
        for sample in window
    )
    # Not merely below the bar -- not climbing toward it either. Halves rather
    # than endpoints, so one noisy checkpoint cannot decide an iteration.
    if len(recent) >= 4:
        half = len(recent) // 2
        first = sum(s.pass_rate_ema for s in recent[:half]) / half
        second = sum(s.pass_rate_ema for s in recent[half:]) / (
            len(recent) - half
        )
        ema_rising = (second - first) > float(policy.plateau_delta)
    else:
        ema_rising = len(recent) >= 2 and (
            recent[-1].pass_rate_ema
            > recent[0].pass_rate_ema + float(policy.plateau_delta)
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
    # Only below the top rung. There, "the EMA cannot reach the promote gate"
    # is what stalled means -- the cap is stuck because the policy cannot clear
    # the rung. AT the top there is no promotion to gate, and requiring the EMA
    # to sit under 0.30 would demand the policy get WORSE before the loop would
    # help it. Measured on iteration 0: at cap 0.19 the EMA reached 0.587 while
    # success was still climbing at +0.097 per 100 updates, and the trigger as
    # first specified would have refused for the rest of the run. At the top,
    # the plateau test below is the whole condition.
    if not at_top:
        check(
            ema_below,
            f"pass-rate EMA below {policy.promote_pass_rate} across the window"
            if ema_below
            else f"pass-rate EMA reached {policy.promote_pass_rate} in the "
            "window; RL is about to promote on its own",
        )
    check(
        not ema_rising,
        "pass-rate EMA is flat or falling across the window"
        if not ema_rising
        else f"pass-rate EMA is still rising "
        f"({recent[0].pass_rate_ema:.4f} -> {recent[-1].pass_rate_ema:.4f} "
        f"over the last {policy.plateau_window_steps} steps, more than "
        f"{policy.plateau_delta} across the halves) -- RL is still getting "
        "better on its own",
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
    lora_epochs: int,
    lora_row_fraction: float,
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
    # The SFT half. Frames are passed explicitly for the same reason the
    # dataset inputs are: no shell, so a glob would arrive verbatim.
    frames = sorted(replay_dir.glob("frames_*.npz"))
    if not frames and not dry_run:
        raise SystemExit(
            f"No frames_*.npz under {replay_dir}. The replay pass ran without "
            "--record-frames, so there are no pictures to train LoRA from."
        )
    sft_dir = output / "sft"
    _run(
        [
            python, str(ROOT / "tools" / "audit" / "sil_sft.py"),
            "--dataset", str(dataset_dir / "demonstrations.npz"),
            "--checkpoint", str(checkpoint),
            "--frames", *[str(path) for path in frames],
            "--train-vision-lora",
            "--lora-epochs", str(int(lora_epochs)),
            "--lora-row-fraction", str(float(lora_row_fraction)),
            "--output", str(sft_dir),
        ],
        dry_run=dry_run,
    )
    # M4, on the demonstrations this iteration is about to train on. Run on the
    # SMOOTHED replays and successes only, because that is what the dataset is
    # built from -- measuring the harvest instead would score trajectories that
    # never became demonstrations.
    stats_dir = output / "stats"
    _run(
        [
            python, str(ROOT / "tools" / "audit" / "sil_action_stats.py"),
            "--recordings", *[str(path) for path in replays],
            "--successes-only", "--no-plots",
            "--output", str(stats_dir),
        ],
        dry_run=dry_run,
    )

    # The verdict. The top rung's harvest IS a record run of the pre-SFT
    # checkpoint at that cap, so it is reused as the baseline rather than paid
    # for twice -- same checkpoint, same cap, same seed, same rounds.
    top_rung = max(rungs)
    baseline_dir = harvest_dir / f"cap_{top_rung:.3f}"
    candidate_dir = output / "eval" / "candidate"
    sft_checkpoint = sft_dir / "sil_sft_adapter.pt"
    _run(
        [
            python, tool, "--mode", "record",
            "--rounds", str(int(rounds)),
            "--seed-torch", str(int(seed_torch)),
            "--start-distance-cap", str(float(top_rung)),
            "--checkpoint", str(sft_checkpoint),
            "--config", str(config),
            "--output", str(candidate_dir),
        ],
        dry_run=dry_run,
    )
    verdict: dict[str, Any] = {"accepted": False, "reason": "dry run"}
    drift: dict[str, Any] = {"halt": False, "reason": "dry run"}
    if not dry_run:
        verdict = accept_or_reject(
            baseline_dir, candidate_dir, instruction=instruction
        )
        print(
            f"[loop] verdict: {'ACCEPT' if verdict['accepted'] else 'REJECT'} "
            f"-- {verdict.get('reason')}",
            flush=True,
        )
    return {
        "instruction": instruction,
        "rungs": [float(rung) for rung in rungs],
        "rounds_per_rung": int(rounds),
        "recorded": recorded,
        "dataset": str(dataset_dir / "demonstrations.npz"),
        "frames": [str(path) for path in frames],
        "sft_checkpoint": str(sft_checkpoint),
        "action_stats": str(stats_dir / "action_stats.json"),
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "verdict": verdict,
        "drift": drift,
        # The checkpoint RL should resume from. On a reject that is the one it
        # already had: an unresolved difference is "no evidence", not "no
        # effect", and accepting on it would let the loop drift on noise.
        "resume_checkpoint": (
            str(sft_checkpoint) if verdict.get("accepted") else str(checkpoint)
        ),
    }


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def select_candidate(
    candidates: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Which SFT epoch gets tested in simulation. Chosen BEFORE the rollout.

    Not the best of three simulated points. Picking the maximum of three noisy
    numbers and then testing that maximum inflates the result by exactly the
    amount the selection gained -- with a per-round spread that swings the plate
    rate 0.466 to 0.690 at one cap, that is not a small correction.

    So the candidate is pre-registered by validation MSE, which costs nothing
    and is computed without touching the simulator. The other epochs are still
    rolled out and reported, but as diagnostics: they do not choose.
    """

    scored = [
        item
        for item in candidates
        if item.get("val_mse") is not None
        and not (isinstance(item["val_mse"], float) and item["val_mse"] != item["val_mse"])
    ]
    if not scored:
        return None
    return min(scored, key=lambda item: float(item["val_mse"]))


def accept_or_reject(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    instruction: str,
) -> dict[str, Any]:
    """Paired-by-round test of one candidate against the pre-SFT checkpoint.

    Both sides are read through sil_eval_table's own collector and paired test
    rather than a copy of them, because the pairing is the whole point: round i
    seeds the same resets on both sides, and an unpaired comparison measures
    the between-round spread instead of the policy difference.
    """

    from tools.audit.sil_eval_table import _collect, _paired, _rate

    base = _collect(str(baseline_dir))
    cand = _collect(str(candidate_dir))
    b = base["by_instruction"].get(instruction)
    c = cand["by_instruction"].get(instruction)
    if b is None or c is None:
        return {
            "accepted": False,
            "reason": f"{instruction} is absent from one side of the comparison",
        }
    if b["per_round_episodes"] != c["per_round_episodes"]:
        # Different denominators mean different resets, and a delta across them
        # measures the reset distribution rather than the policy.
        return {
            "accepted": False,
            "reason": "per-round episode counts differ; not comparable",
            "baseline_episodes": b["per_round_episodes"],
            "candidate_episodes": c["per_round_episodes"],
        }
    paired = _paired(b["rates"], c["rates"])
    if int(paired["n"]) < 2:
        # _paired cannot compute a spread from one pair, so it returns
        # resolved=False whatever the numbers are. Reported as the structural
        # fact it is: a one-round comparison is not a weak measurement, it is
        # no measurement, and "REJECT" beside a visible delta reads as one.
        return {
            "accepted": False,
            "instruction": instruction,
            "baseline_rate": round(_rate(b), 5),
            "candidate_rate": round(_rate(c), 5),
            "delta": round(float(paired["mean"]), 5),
            "rounds": int(paired["n"]),
            "reason": (
                f"only {int(paired['n'])} round on each side -- the paired "
                "test needs at least 2 to have a spread at all, so this can "
                "never accept regardless of the delta. Re-run the evaluation "
                "with --rounds 4."
            ),
        }
    accepted = bool(
        paired["resolved"]
        and paired["all_same_sign"]
        and float(paired["mean"]) > 0.0
    )
    return {
        "accepted": accepted,
        "instruction": instruction,
        "baseline_rate": round(_rate(b), 5),
        "candidate_rate": round(_rate(c), 5),
        "delta": round(float(paired["mean"]), 5),
        "t": paired["t"],
        "resolved": paired["resolved"],
        "all_same_sign": paired["all_same_sign"],
        "rounds": paired["n"],
        "deltas": [round(float(d), 5) for d in paired["deltas"]],
        "reason": (
            "candidate beats the pre-SFT checkpoint on a paired test"
            if accepted
            else "the difference does not resolve; keeping the pre-SFT "
            "checkpoint. A single 512-world round cannot separate less than "
            "about five points, so an unresolved result is 'no evidence', not "
            "'no effect'."
        ),
    }


# --------------------------------------------------------------------------
# M4: is the loop eating itself?
# --------------------------------------------------------------------------


@dataclass
class DriftPolicy:
    """When to stop the loop because it is training on its own collapse."""

    # aim must fall this much, twice running, before it counts as falling.
    # The calibration is on synthetic policies: 0.000 knows nothing, 0.163 is
    # half the rows aiming, 0.327 is a clean servo. 0.01 is well inside that
    # scale and well outside the +-0.002 null spread.
    aim_drop: float = 0.01
    concentration_rise: float = 0.02


def check_action_drift(
    history: Sequence[Mapping[str, Any]], *, policy: DriftPolicy
) -> dict[str, Any]:
    """Halt when aim falls two iterations running while concentration rises.

    Each iteration trains on its own successes, so a policy that is collapsing
    toward a constant will keep producing demonstrations that agree with it.
    aim is the measurement that sees this: cosine to the goal minus the same
    cosine with the commands permuted across rows, which is the only null that
    preserves both marginals and breaks only the pairing.

    Concentration alone accuses nothing -- a genuine servo scores 0.82 when the
    arm sits systematically to one side of its goals -- so it is read only
    together with aim, and only in the rising direction.
    """

    rows = [
        item
        for item in history
        if item.get("aim") is not None
        and item.get("direction_concentration") is not None
    ]
    if len(rows) < 3:
        return {
            "halt": False,
            "reason": f"only {len(rows)} iterations with action stats; "
            "two consecutive falls need three points",
        }
    a, b, c = rows[-3], rows[-2], rows[-1]
    falls = (
        float(b["aim"]) < float(a["aim"]) - policy.aim_drop
        and float(c["aim"]) < float(b["aim"]) - policy.aim_drop
    )
    rises = (
        float(c["direction_concentration"])
        > float(a["direction_concentration"]) + policy.concentration_rise
    )
    halt = bool(falls and rises)
    return {
        "halt": halt,
        "aim": [round(float(x["aim"]), 5) for x in (a, b, c)],
        "direction_concentration": [
            round(float(x["direction_concentration"]), 5) for x in (a, b, c)
        ],
        "reason": (
            "aim has fallen for two consecutive iterations while direction "
            "concentration rose -- the loop is collapsing toward a constant "
            "and the next dataset would teach it that constant"
            if halt
            else "aim is not in a sustained fall, or concentration is not "
            "rising with it"
        ),
    }


def read_action_stats(path: Path, instruction: str) -> dict[str, Any]:
    """Pull aim and concentration for one instruction out of action_stats.json."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = (payload.get("by_instruction") or {}).get(instruction)
    if entry is None:
        entry = payload.get("overall") or {}
    return {
        "aim": entry.get("aim"),
        "direction_concentration": entry.get("direction_concentration"),
        "rows": entry.get("rows"),
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
    parser.add_argument("--lora-epochs", type=int, default=8)
    parser.add_argument("--lora-row-fraction", type=float, default=0.3)
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
        lora_epochs=int(args.lora_epochs),
        lora_row_fraction=float(args.lora_row_fraction),
        dry_run=bool(args.dry_run),
    )
    report_payload["trigger"] = asdict(decision)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # State carried to the next iteration. Written whether the verdict accepted
    # or not: a rejected iteration still consumed steps, and last_sft_step is
    # what stops the trigger firing again immediately on the same stall.
    stats_path = Path(report_payload["action_stats"])
    entry: dict[str, Any] = {
        "iteration": state.iteration + 1,
        "global_step": decision.global_step,
        "cap": decision.cap,
        "accepted": bool(report_payload["verdict"].get("accepted")),
    }
    if stats_path.is_file():
        entry.update(read_action_stats(stats_path, args.instruction))
    state.history.append(entry)
    state.iteration += 1
    state.last_sft_step = int(decision.global_step)
    state.cap_at_last_sft = float(decision.cap)
    drift = check_action_drift(state.history, policy=DriftPolicy())
    report_payload["drift"] = drift
    if drift["halt"]:
        print(
            f"[loop] HALT: {drift['reason']}\n"
            f"[loop]   aim {drift['aim']} "
            f"concentration {drift['direction_concentration']}",
            flush=True,
        )
    else:
        print(f"[loop] drift check: {drift['reason']}", flush=True)
    state.save(state_path)
    print(f"[loop] state -> {state_path}", flush=True)
    (output / "harvest_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[loop] wrote {output / 'harvest_report.json'}", flush=True)
    print(
        f"[loop] resume RL from {report_payload['resume_checkpoint']}",
        flush=True,
    )
    # Non-zero on a halt so a shell loop driving this stops rather than
    # cheerfully starting the iteration that would train on the collapse.
    return 2 if drift["halt"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
