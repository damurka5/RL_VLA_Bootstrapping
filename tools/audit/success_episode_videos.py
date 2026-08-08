"""Watch the pick_up successes, and record whether they are aim or luck.

This is the seed check for the self-imitation loop. The plan is to keep the
policy's own successful episodes and train the vision tower on them densely,
which only works if those episodes contain the behaviour we want to distil. They
might not: at the 0.13 m cap the rollout budget allows ~0.78 m of travel, so a
success from a far start can be a random walk that stumbled onto the object
rather than an approach that aimed at it. Distilling luck teaches luck.

So every successful episode gets two things.

**A video**, because the fastest way to tell a servo from a stumble is to watch
one. Frames are teed off the exact tensors the policy was given -- the backend's
own ``render_policy_cameras`` is wrapped, not called a second time -- so the
video is what the policy saw, not a re-render from a different angle.

**Its decision-0 aiming cosine**, the same quantity the trainer logs, reported
separately for successes and failures and split by start distance. That is the
number the loop stands on: if successful far-start episodes aim no better than
the run's ~0.055 average, they are not carrying a signal worth imitating and the
pipeline has no seed.

Frames are kept only for a tracked SUBSET of worlds (``--track-worlds``), on the
host, because 512 worlds x ~104 steps of RGB does not fit anywhere. Rounds are
cheap to add, so collect successes by running more of them rather than by
tracking more worlds.

Usage::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab \\
      python tools/audit/success_episode_videos.py \\
        --checkpoint runs/<run>/smolvla_grpo_adapter.pt \\
        --rounds 8 --output runs/pick_up_successes
"""

from __future__ import annotations

import os
import sys


def _configure_huggingface() -> None:
    """Mirror scripts/huggingface_public_models.sh; see xy_approach_probe."""

    public_only = os.environ.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1")
    if public_only not in {"0", "1"}:
        raise SystemExit("RLVLA_HF_PUBLIC_MODELS_ONLY must be 0 or 1.")
    if public_only == "1":
        removed = [
            name
            for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
            if os.environ.pop(name, None)
        ]
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        if removed:
            print(f"[huggingface] ignoring inherited {', '.join(removed)}")
    offline = os.environ.get("RLVLA_HF_OFFLINE", "0")
    if offline not in {"0", "1"}:
        raise SystemExit("RLVLA_HF_OFFLINE must be 0 or 1.")
    if offline == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[huggingface] offline: using the local cache only")


_configure_huggingface()

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_xy_probe() -> Any:
    """Reuse the probe's stack builder rather than rebuilding it here.

    _build_world already restores the approach curriculum, attaches and loads
    the action-expert LoRA, and refuses a checkpoint whose vision width does not
    match. Duplicating it is how the two drift apart.
    """

    spec = importlib.util.spec_from_file_location(
        "xy_approach_probe", Path(__file__).with_name("xy_approach_probe.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import xy_approach_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Mp4:
    """Minimal ffmpeg pipe. Same encoder settings as the video evaluator."""

    def __init__(self, path: Path, *, fps: float, height: int, width: int) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to encode episode MP4s.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{int(width)}x{int(height)}",
                "-r", f"{float(fps):.6f}", "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(
            np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        )

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        details = (
            self.process.stderr.read().decode("utf-8", "replace")
            if self.process.stderr is not None
            else ""
        )
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited {code} for {self.path}: {details}")


def _to_rgb(camera: Any, world: int) -> np.ndarray:
    """[B, C, H, W] float in [0,1] -> one HxWx3 uint8 frame."""

    array = camera[int(world)].permute(1, 2, 0).detach().float().cpu().numpy()
    return np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)


class _FrameTap:
    """Tee the frames the policy is given, for a subset of worlds.

    Wraps the backend's own render call rather than rendering again: a second
    render would cost as much as the rollout and, more importantly, would not be
    guaranteed to show the same thing the policy acted on.
    """

    def __init__(self, backend: Any, worlds: Sequence[int], both_cameras: bool):
        self.backend = backend
        self.worlds = list(worlds)
        self.both = bool(both_cameras)
        self.frames: dict[int, list[np.ndarray]] = {w: [] for w in self.worlds}
        self._original = backend.render_policy_cameras

    def __enter__(self) -> "_FrameTap":
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            cameras = self._original(*args, **kwargs)
            for world in self.worlds:
                overview = _to_rgb(cameras.overview, world)
                if self.both:
                    wrist = _to_rgb(cameras.wrist, world)
                    overview = np.concatenate([overview, wrist], axis=1)
                self.frames[world].append(overview)
            return cameras

        self.backend.render_policy_cameras = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        if "render_policy_cameras" in vars(self.backend):
            del self.backend.render_policy_cameras
        else:  # pragma: no cover - bound-method backends
            self.backend.render_policy_cameras = self._original

    def reset(self) -> None:
        for world in self.worlds:
            self.frames[world].clear()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "examples"
        / "cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "pick_up_successes"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--smolvla-microbatch", type=int, default=256)
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Validation rounds to run. More rounds is the cheap way to more "
        "successes; more tracked worlds is the expensive way.",
    )
    parser.add_argument(
        "--track-worlds",
        type=int,
        default=16,
        help="How many worlds keep frames. 512 worlds of RGB fits nowhere.",
    )
    parser.add_argument("--max-videos", type=int, default=24)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--wrist",
        action="store_true",
        help="Append the wrist view beside the overview in each frame.",
    )
    parser.add_argument(
        "--far-start-m",
        type=float,
        default=0.06,
        help=(
            "Episodes starting at least this far from the object count as FAR. "
            "The seed check is whether far-start SUCCESSES aim better than the "
            "run's average; close-start successes are explained by proximity."
        ),
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Roll out the sampled policy instead of the deterministic mean -- "
        "the distribution self-imitation would actually collect from.",
    )
    parser.add_argument(
        "--start-distance-cap", type=float, default=None,
        help="Override the checkpoint's approach cap (m); inf disables it.",
    )
    parser.add_argument("--sigma", type=float, default=0.333)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    probe = _load_xy_probe()
    cap = args.start_distance_cap
    if cap is not None and (cap <= 0.0 or cap == float("inf")):
        cap = float("inf")
    world = probe._build_world(
        checkpoint=checkpoint,
        config_path=config_path,
        device_str=str(args.device),
        worlds=int(args.worlds),
        group_size=int(args.group_size),
        microbatch=int(args.smolvla_microbatch),
        load_policy=True,
        run_dir=output,
        start_distance_cap=cap,
    )

    tracked = list(range(min(int(args.track_worlds), int(args.worlds))))
    source = (
        probe._make_sampled_source(world, sigma=float(args.sigma))
        if args.sampled
        else None
    )
    rows: list[dict[str, Any]] = []
    videos = 0
    videos_dir = output / "videos"

    for round_index in range(int(args.rounds)):
        with _FrameTap(world.backend, tracked, bool(args.wrist)) as tap:
            with probe._ArmRunner(
                world, source=source, seed_offset=round_index
            ) as runner:
                summary = runner.run(round_index=round_index)
            frames = {w: list(v) for w, v in tap.frames.items()}

        trace = runner.trace
        ee = trace.stack("ee_xyz")
        target = trace.stack("target_xyz")
        commanded = trace.stack("commanded0")
        rel0 = (target[0] - ee[0])[:, :2]
        start = np.linalg.norm(rel0, axis=-1)
        cos0 = probe._cosine(commanded[0][:, :2], rel0)
        success = runner.world_success > 0.5
        grasped = (trace.stack("holding") > 0.5).any(axis=0)

        for world_index in range(len(success)):
            rows.append(
                {
                    "round": round_index,
                    "world": world_index,
                    "success": bool(success[world_index]),
                    "ever_grasped": bool(grasped[world_index]),
                    "start_distance_m": float(start[world_index]),
                    "far_start": bool(start[world_index] >= args.far_start_m),
                    "cosine_decision0": float(cos0[world_index]),
                    "tracked": world_index in frames,
                }
            )

        print(
            f"[successes] round {round_index}: success "
            f"{summary['success_rate']:.3f}  grasped "
            f"{summary['ever_grasped_rate']:.3f}  horizon "
            f"{summary['horizon_decisions']}  diverged "
            f"{summary['diverged_worlds']}",
            flush=True,
        )

        for world_index in tracked:
            if videos >= int(args.max_videos):
                break
            if not success[world_index]:
                continue
            stack = frames.get(world_index) or []
            if not stack:
                continue
            far = "far" if start[world_index] >= args.far_start_m else "near"
            name = (
                f"r{round_index:02d}_w{world_index:03d}_{far}"
                f"_d{start[world_index] * 1000:.0f}mm"
                f"_cos{cos0[world_index]:+.2f}.mp4"
            )
            writer = _Mp4(
                videos_dir / name,
                fps=float(args.fps),
                height=stack[0].shape[0],
                width=stack[0].shape[1],
            )
            for frame in stack:
                writer.write(frame)
            writer.close()
            videos += 1
            print(f"[successes]   wrote {name}", flush=True)

        if videos >= int(args.max_videos):
            print("[successes] video budget reached", flush=True)
            break

    keys = list(rows[0]) if rows else []
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    _report(rows, far_start_m=float(args.far_start_m))
    (output / "summary.json").write_text(
        json.dumps({"episodes": len(rows), "videos": videos}, indent=2)
    )
    print(f"\nwrote {output}  ({videos} videos)", flush=True)
    return 0


def _report(rows: Sequence[dict[str, Any]], *, far_start_m: float) -> None:
    """Is a successful far-start episode aiming, or lucky?"""

    if not rows:
        print("no episodes")
        return
    cos = np.array([row["cosine_decision0"] for row in rows])
    success = np.array([row["success"] for row in rows], dtype=bool)
    far = np.array([row["far_start"] for row in rows], dtype=bool)

    print("\nSEED CHECK -- decision-0 aiming cosine by outcome and start")
    print("-----------------------------------------------------------")
    print(f"{'group':<28} {'episodes':>9} {'cosine':>9} {'+-':>7}")
    for label, mask in (
        ("all", np.ones_like(success)),
        ("success", success),
        ("failure", ~success),
        (f"success, far (>={far_start_m:.2f} m)", success & far),
        (f"success, near (<{far_start_m:.2f} m)", success & ~far),
        (f"failure, far (>={far_start_m:.2f} m)", (~success) & far),
    ):
        count = int(mask.sum())
        if not count:
            print(f"{label:<28} {count:>9}       n/a")
            continue
        print(
            f"{label:<28} {count:>9} {cos[mask].mean():>+9.3f} "
            f"{cos[mask].std():>7.3f}"
        )
    print(
        "\n  The loop needs `success, far` to aim materially better than "
        "`failure, far`.\n  If the two match, those successes are a random "
        "walk that happened to end on the\n  object -- at the 0.13 m cap the "
        "budget allows ~0.78 m of travel, which is ample\n  for that -- and "
        "imitating them teaches luck rather than approach.\n"
        "\n  Compare the absolute level against the run's own "
        "residual_target_cosine_mean\n  (~0.055 across every run so far). "
        "Successes at that level are not a seed."
    )


if __name__ == "__main__":
    raise SystemExit(main())
