#!/usr/bin/env python3
"""move_to validation on the TRAINING distribution, and the videos that show it.

Two evaluators already record move_to videos and neither of them measures
validation.

``rl_vla_bootstrapping/cli/evaluate_cdpr_smolvla_mjwarp_videos.py`` builds its
resetter with a task_metadata of exactly two keys -- the scene-object bounds --
so ``random_workspace_gripper_start`` is absent, the whole approach-curriculum
block (``mjwarp_rank_local_collector.py:1579``) never executes, and the start
comes from either a Reverse Frontier shell (a schedule phase 4 does not run) or
a uniform draw over the full workspace at Z in [0.32, 0.52]. Its own report says
so: *"intentionally harder than the checkpoint's validation reset and must not
be reported as the original validation metric"*. ``tools/audit/success_episode_
videos.py`` does restore the cap -- it builds through ``xy_approach_probe.
_build_world`` -- but it cannot vary the scene, and it reports the pick_up seed
check rather than a validation table.

What the training configuration actually is, for the record, because every
number below is a consequence of it:

* start distance <= the cap the checkpoint EARNED (``extra_state.approach_
  curriculum``), 3-D, because ``curriculum_cap_includes_z`` is true;
* ``random_workspace_start_distance_final: 0.19`` is the ladder top, and it was
  set by the wrist camera, not by reach: 0.19 is the last rung where the target
  is never CERTAINLY out of the wrist frame;
* ``ee_workspace_z_bounds: [0.27, 0.40]``, ceiling set by the overview camera --
  above 0.441 m the gripper leaves the overview image entirely;
* objects on a 3x3 grid of +-0.18 m plus <=0.025 m of shift and jitter, so every
  object is inside the overview frame and inside the +-0.24 m EE workspace.

So "the object is in at least one camera and the robot can reach it" is a
property of the training reset, not a filter applied on top of it. This tool
therefore MEASURES it per episode rather than assuming it: ``target_in_overview``
comes from the MJCF's own overview camera, and ``wrist_angle_deg`` from the
orientation-free nadir bound both taken from ``tools/audit/start_distance_
probe.py`` so there is one definition and not two. If the headline rate and the
rate over the visible-and-reachable subset ever differ, the run says so instead
of quietly averaging the two.

The harder legs are the same harness with two knobs moved, which is the point:
``--metadata-override min_scene_objects=2 --metadata-override
max_scene_objects=3`` puts two or three objects on the desk, and a
``--start-distance-cap`` above the earned one starts the gripper far enough away
that the wrist camera no longer contains the object the instruction names. That
is the only camera an object CAN be hidden from here -- the object grid is fixed
and lies inside the overview frustum at every cell -- so "not visible in one of
the cameras" means the wrist, and ``--video-filter target_out_of_wrist`` selects
exactly those episodes. Their CSV is written to its own directory with
``counts_toward_validation_metric: false`` in the manifest.

Both legs run ``collector.validate_round`` -- the trainer's own validation, with
the deterministic residual mean -- with the action source untouched. Nothing
here reimplements the rollout, the reward, the horizon or the success predicate.

Note on repeatability: the frozen SmolVLA prior samples fresh flow-matching
noise on every forward, so ~6.6% of per-episode verdicts move between two runs
of the same seed. Read the rate at the sample size, not at the third decimal.

Usage::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab \\
      python tools/audit/move_to_validation_videos.py \\
        --checkpoint runs/<run>/rl/<step>/smolvla_grpo_adapter.pt \\
        --config configs/examples/cdpr_smolvla_phase4_move_to_loop.yaml \\
        --output runs/move_to_validation/train_config \\
        --rounds 2 --max-videos 5
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
from pathlib import Path  # noqa: E402
from typing import Any, Iterable, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = (
    ROOT / "configs" / "examples" / "cdpr_smolvla_phase4_move_to_loop.yaml"
)
# Parked slots sit at |4.0| m by construction (mjwarp_rank_local_collector.py:
# "Inactive slots return to the XML park poses far outside the desk"), so a
# metre is a generous line between "on the desk" and "parked".
_ACTIVE_SLOT_RADIUS_M = 1.0


def _sibling(name: str) -> Any:
    """Load a tool module from this directory by path.

    The same reuse rule the other audit tools follow: the stack builder, the
    arm runner and the camera geometry each have one definition, and it is the
    one the trainer or the preflight already uses. Duplicating any of them is
    how two tools start measuring two different things under one name.
    """

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ResetTap:
    """Snapshot the scene the reset produced, before any action is taken.

    Wraps whatever ``resetter.reset`` currently is, so it composes on top of the
    arm runner's own patch rather than replacing it -- enter it INSIDE the
    runner's context. The snapshot is read back through
    ``low_dim_observations``, i.e. the poses the simulator holds, not the poses
    the resetter commanded; the two have disagreed before.
    """

    def __init__(
        self,
        resetter: Any,
        backend: Any,
        *,
        on_reset: Any = None,
    ) -> None:
        self.resetter = resetter
        self.backend = backend
        # Called with this tap once the snapshot is taken and BEFORE the first
        # render, which is the only window in which the frame tap can still be
        # pointed at the worlds this reset happens to have made interesting.
        self.on_reset = on_reset
        self.ee: np.ndarray | None = None
        self.objects: np.ndarray | None = None
        self.target_slots: np.ndarray | None = None
        self.instructions: tuple[str, ...] = ()
        self.horizons: np.ndarray | None = None
        self.target_catalog_ids: np.ndarray | None = None
        self._original: Any = None
        self._owned = False

    def __enter__(self) -> "_ResetTap":
        self._original = self.resetter.reset
        self._owned = "reset" in vars(self.resetter)

        def wrapped(**kwargs: Any) -> Any:
            reset = self._original(**kwargs)
            low_dim = self.backend.low_dim_observations()

            def host(value: Any) -> np.ndarray:
                return value.detach().float().cpu().numpy().copy()

            self.ee = host(low_dim.ee_position)
            self.objects = host(low_dim.object_positions)
            self.target_slots = (
                reset.task_state.target_slots.detach().cpu().numpy().copy()
            )
            self.instructions = tuple(reset.instructions)
            self.horizons = reset.horizons.detach().cpu().numpy().copy()
            catalogs = reset.group_target_catalog_ids
            self.target_catalog_ids = (
                None if catalogs is None else catalogs.detach().cpu().numpy().copy()
            )
            if self.on_reset is not None:
                self.on_reset(self)
            return reset

        self.resetter.reset = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._owned:
            self.resetter.reset = self._original
        else:
            vars(self.resetter).pop("reset", None)


def _per_world(values: np.ndarray, worlds: int) -> np.ndarray:
    """Expand a per-group column to per-world, or pass a per-world one through."""

    flat = np.asarray(values).reshape(-1)
    if flat.shape[0] == worlds:
        return flat
    if worlds % flat.shape[0]:
        raise ValueError(
            f"Cannot expand {flat.shape[0]} groups to {worlds} worlds."
        )
    return np.repeat(flat, worlds // flat.shape[0])


def _visibility(
    torch: Any,
    geometry: Any,
    *,
    ee: np.ndarray,
    objects: np.ndarray,
    target_slots: np.ndarray,
    camera: dict[str, Any] | None,
    aspect: float,
) -> dict[str, np.ndarray]:
    """Per-episode camera geometry for the target and for the whole scene.

    ``certainly_in`` / ``certainly_out`` are BOUNDS, not a projection: the wrist
    camera hangs off a ball joint, so its realized orientation is not a function
    of the commanded pose and an exact projection would be exact about the wrong
    thing. Between the two bounds the answer is "depends on how the wrist is
    hanging", and the columns say so rather than guessing.
    """

    worlds, slots = objects.shape[0], objects.shape[1]
    rows = np.arange(worlds)
    target = objects[rows, target_slots]
    active = (
        np.linalg.norm(objects[:, :, :2], axis=-1) < _ACTIVE_SLOT_RADIUS_M
    )

    ee_t = torch.tensor(ee, dtype=torch.float32)
    target_t = torch.tensor(target, dtype=torch.float32)
    objects_t = torch.tensor(objects.reshape(-1, 3), dtype=torch.float32)
    ee_repeat = ee_t.repeat_interleave(slots, dim=0)

    certainly_in, certainly_out = geometry.wrist_bounds_deg(float(aspect))
    target_angle = (
        geometry.wrist_angle_from_nadir(torch, ee_t, target_t).numpy()
    )
    scene_angle = (
        geometry.wrist_angle_from_nadir(torch, ee_repeat, objects_t)
        .numpy()
        .reshape(worlds, slots)
    )
    if camera is None:
        target_in_overview = np.ones(worlds, dtype=bool)
        scene_in_overview = np.ones((worlds, slots), dtype=bool)
    else:
        target_in_overview = (
            geometry.overview_in_frame(
                torch, target_t, camera, aspect=float(aspect)
            )
            .numpy()
            .astype(bool)
        )
        scene_in_overview = (
            geometry.overview_in_frame(
                torch, objects_t, camera, aspect=float(aspect)
            )
            .numpy()
            .astype(bool)
            .reshape(worlds, slots)
        )
    target_out_of_wrist = target_angle > certainly_out
    target_in_wrist = target_angle <= certainly_in
    return {
        "scene_object_count": active.sum(axis=1).astype(np.int64),
        "target_x": target[:, 0],
        "target_y": target[:, 1],
        "target_z": target[:, 2],
        "target_in_overview": target_in_overview,
        "target_wrist_angle_deg": target_angle,
        "target_certainly_in_wrist": target_in_wrist,
        "target_certainly_out_of_wrist": target_out_of_wrist,
        "target_in_any_camera": target_in_overview | target_in_wrist,
        "scene_objects_in_overview": (scene_in_overview & active).sum(axis=1),
        "scene_objects_certainly_out_of_wrist": (
            (scene_angle > certainly_out) & active
        ).sum(axis=1),
        "all_scene_objects_out_of_wrist": (
            ((scene_angle > certainly_out) | ~active).all(axis=1)
        ),
        "wrist_certainly_in_deg": np.full(worlds, certainly_in),
        "wrist_certainly_out_deg": np.full(worlds, certainly_out),
    }


def _named_vs_nearest(
    ee: np.ndarray, objects: np.ndarray, target_slots: np.ndarray
) -> dict[str, np.ndarray]:
    """Distance to the object the instruction NAMES, and to the nearest other.

    The grounding question this run exists to answer is not "did it move" but
    "did it move to the named one". With one object on the desk the instruction
    is decorative and the two distances are the same number; with two or three
    they separate, and the episodes where the named object was NOT the nearest
    at reset are the only ones that test language at all.
    """

    worlds, slots = objects.shape[0], objects.shape[1]
    rows = np.arange(worlds)
    active = (
        np.linalg.norm(objects[:, :, :2], axis=-1) < _ACTIVE_SLOT_RADIUS_M
    )
    planar = np.linalg.norm(
        objects[:, :, :2] - ee[:, None, :2], axis=-1
    )
    named = planar[rows, target_slots]
    others = planar.copy()
    others[rows, target_slots] = np.inf
    others[~active] = np.inf
    nearest_other = others.min(axis=1)
    return {
        "named": named,
        "nearest_other": nearest_other,
        "named_is_nearest": named <= nearest_other,
    }


def _reachable(
    target: np.ndarray, x_bounds: Sequence[float], y_bounds: Sequence[float]
) -> np.ndarray:
    """Can the gripper hover over this object without leaving its workspace?"""

    return (
        (target[:, 0] >= float(x_bounds[0]))
        & (target[:, 0] <= float(x_bounds[1]))
        & (target[:, 1] >= float(y_bounds[0]))
        & (target[:, 1] <= float(y_bounds[1]))
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(
    rows: Sequence[dict[str, Any]], *, scope: str, value: Any
) -> dict[str, Any]:
    def column(key: str) -> np.ndarray:
        return np.array([float(row[key]) for row in rows], dtype=np.float64)

    successes = int(sum(bool(row["success"]) for row in rows))
    episodes = len(rows)
    return {
        "scope": scope,
        "value": value,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else float("nan"),
        "mean_start_xy_distance_m": (
            float(column("start_xy_distance_m").mean()) if rows else float("nan")
        ),
        "median_start_xy_distance_m": (
            float(np.median(column("start_xy_distance_m")))
            if rows
            else float("nan")
        ),
        "mean_best_xy_distance_m": (
            float(column("best_xy_distance_m").mean()) if rows else float("nan")
        ),
        "mean_final_dense_distance_m": (
            float(np.nanmean(column("final_dense_distance_m")))
            if rows
            else float("nan")
        ),
        "mean_cosine_decision0": (
            float(column("cosine_decision0").mean()) if rows else float("nan")
        ),
        "target_in_any_camera_rate": (
            float(np.mean([bool(row["target_in_any_camera"]) for row in rows]))
            if rows
            else float("nan")
        ),
        "target_out_of_wrist_rate": (
            float(
                np.mean(
                    [bool(row["target_certainly_out_of_wrist"]) for row in rows]
                )
            )
            if rows
            else float("nan")
        ),
        "target_reachable_rate": (
            float(np.mean([bool(row["target_reachable"]) for row in rows]))
            if rows
            else float("nan")
        ),
        "ended_closer_to_named_rate": (
            float(
                np.mean([bool(row["ended_closer_to_named"]) for row in rows])
            )
            if rows
            else float("nan")
        ),
        "within_curriculum_cap_rate": (
            float(
                np.mean(
                    [bool(row["start_within_curriculum_cap"]) for row in rows]
                )
            )
            if rows
            else float("nan")
        ),
    }


def _summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The validation table: overall, then every split that can hide a failure."""

    if not rows:
        return []
    summary = [_group_rows(rows, scope="all", value="")]
    qualified = [
        row
        for row in rows
        if bool(row["target_in_any_camera"]) and bool(row["target_reachable"])
    ]
    summary.append(
        _group_rows(
            qualified, scope="training_configuration", value="visible_reachable"
        )
    )
    # The only episodes that test language: more than one object on the desk
    # AND the named one was not already the closest thing to the gripper. A
    # policy that ignores the instruction and servos to whatever is nearest
    # scores the same as a grounded one everywhere else.
    grounding = [
        row
        for row in rows
        if int(row["scene_object_count"]) > 1
        and not bool(row["named_is_nearest_at_start"])
    ]
    if grounding:
        summary.append(
            _group_rows(grounding, scope="grounding_test", value="named_not_nearest")
        )
    for catalog in sorted({str(row["target_catalog"]) for row in rows}):
        summary.append(
            _group_rows(
                [row for row in rows if str(row["target_catalog"]) == catalog],
                scope="target_object",
                value=catalog,
            )
        )
    for count in sorted({int(row["scene_object_count"]) for row in rows}):
        summary.append(
            _group_rows(
                [row for row in rows if int(row["scene_object_count"]) == count],
                scope="scene_objects",
                value=count,
            )
        )
    for flag in (False, True):
        selected = [
            row
            for row in rows
            if bool(row["target_certainly_out_of_wrist"]) == flag
        ]
        if selected:
            summary.append(
                _group_rows(
                    selected,
                    scope="target_certainly_out_of_wrist",
                    value=str(flag).lower(),
                )
            )
    return summary


_FILTER_KEYS = {
    "target_out_of_wrist": "target_certainly_out_of_wrist",
    "all_objects_out_of_wrist": "all_scene_objects_out_of_wrist",
}


def _filter_indices(
    visibility: dict[str, np.ndarray], video_filter: str, worlds: int
) -> list[int]:
    """Which worlds of this reset are worth spending frame memory on."""

    if video_filter == "any":
        return list(range(worlds))
    column = visibility[_FILTER_KEYS[video_filter]]
    return [index for index in range(worlds) if bool(column[index])]


def _passes_filter(row: dict[str, Any], video_filter: str) -> bool:
    if video_filter == "any":
        return True
    if video_filter == "target_out_of_wrist":
        return bool(row["target_certainly_out_of_wrist"])
    if video_filter == "all_objects_out_of_wrist":
        return bool(row["all_scene_objects_out_of_wrist"])
    raise ValueError(f"Unknown video filter {video_filter!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label",
        default="train_config",
        help="Stamped into the manifest and every video filename.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--smolvla-microbatch", type=int, default=256)
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help=(
            "Validation rounds. worlds x rounds is the episode count; training "
            "validation_episodes_per_instruction is 1024, i.e. 512 x 2."
        ),
    )
    parser.add_argument(
        "--track-worlds",
        type=int,
        default=32,
        help=(
            "How many worlds keep frames. 512 worlds of RGB fits nowhere. The "
            "tracked set is chosen per reset from the episodes --video-filter "
            "asks for, so the budget is not spent on episodes this leg is not "
            "about."
        ),
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=5,
        help="Budget for SUCCESSFUL episode videos, across all rounds.",
    )
    parser.add_argument(
        "--failure-videos",
        type=int,
        default=0,
        help=(
            "Separate budget for the closest-approach FAILURES, written to "
            "videos/near_miss. A harder leg that scores nothing still has to "
            "show what it did."
        ),
    )
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--no-wrist",
        action="store_true",
        help="Overview only. By default each frame is overview|wrist, which is "
        "what makes a hidden-from-the-wrist episode visible as one.",
    )
    parser.add_argument(
        "--video-filter",
        choices=("any", "target_out_of_wrist", "all_objects_out_of_wrist"),
        default="any",
        help=(
            "Which episodes may be recorded. The CSV always keeps every "
            "episode; this only selects what is worth watching."
        ),
    )
    parser.add_argument(
        "--start-distance-cap",
        type=float,
        default=None,
        help=(
            "Override the approach-curriculum cap (m). Omit to use the cap the "
            "checkpoint EARNED, which is the training configuration. inf or a "
            "non-positive value disables the cap entirely."
        ),
    )
    parser.add_argument(
        "--metadata-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override task.metadata before the resetter and reward are built, "
            "e.g. min_scene_objects=2. Same contract as the xy_approach_probe "
            "flag of the same name."
        ),
    )
    parser.add_argument(
        "--counts-toward-metric",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Recorded in the manifest. Defaults to true only for a run with no "
            "cap override and no metadata override -- i.e. the training "
            "configuration and nothing else."
        ),
    )
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    if args.worlds % args.group_size:
        parser.error("--worlds must be a multiple of --group-size.")
    if args.rounds < 1:
        parser.error("--rounds must be positive.")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    probe = _sibling("xy_approach_probe")
    geometry = _sibling("start_distance_probe")
    recorder = _sibling("success_episode_videos")

    import torch

    cap = args.start_distance_cap
    if cap is not None and (cap <= 0.0 or cap == float("inf")):
        cap = float("inf")
    counts_toward_metric = (
        bool(args.counts_toward_metric)
        if args.counts_toward_metric is not None
        else (cap is None and not args.metadata_override)
    )

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
        metadata_overrides=list(args.metadata_override or []),
    )
    instructions = tuple(world.args.instruction_types or ())
    if instructions != ("move_to_object",):
        raise SystemExit(
            "This tool measures move_to_object validation; the config runs "
            f"{list(instructions)}."
        )

    if not bool(getattr(world.resetter, "random_workspace_gripper_start", False)):
        raise SystemExit(
            "random_workspace_gripper_start is false for this config, so the "
            "approach-curriculum cap never reaches the simulator "
            "(mjwarp_rank_local_collector.py:1579) and the start distribution "
            "is not the one the checkpoint trained on. Refusing to report a "
            "validation number measured on a different task."
        )

    from rl_vla_bootstrapping.simulation.cdpr_object_catalog import (
        ACTIVE_CDPR_CATALOGS,
    )

    aspect = float(world.args.render_width) / float(world.args.render_height)
    try:
        camera = geometry.load_overview_camera(Path(world.args.mjwarp_xml_path))
    except SystemExit as error:  # no overview camera in this scene
        print(f"[move-to-validation] {error}; overview framing not measured")
        camera = None
    scene_min, scene_max = world.resetter.scene_object_range
    realized_caps = {
        "move_to_object": round(
            float(
                world.resetter._start_cap_table[
                    int(world.resetter.instruction_ids[0].item())
                ].item()
            ),
            4,
        )
    }
    print(
        f"[move-to-validation] label={args.label} cap={realized_caps} "
        f"scene_objects={scene_min}-{scene_max} worlds={args.worlds} "
        f"rounds={args.rounds} counts_toward_metric={counts_toward_metric}",
        flush=True,
    )

    # The same seeding the trainer's validation does
    # (smolvla_grpo_mjwarp_cdpr.py:_run_gpu_validation): the reset sampler has
    # its own generator, while the frozen SmolVLA flow sampler consumes this
    # CUDA stream. It does not make the verdicts repeatable -- the prior draws
    # fresh noise per forward -- it makes them comparable.
    validation_seed = int(world.args.validation_seed)
    torch.manual_seed(validation_seed)
    torch.cuda.manual_seed(validation_seed)
    # The point the cap is a radius around: target XY at the hover height, in
    # 3-D because curriculum_cap_includes_z is on. Rebuilding it is how a cap
    # audit goes wrong, so it comes from the same metadata the reward reads.
    hover_z = float(
        world.task_metadata.get("move_to_object_approach_z", 0.27)
    )
    cap_value = float(realized_caps["move_to_object"])

    # One tracked world per GRPO GROUP, not the first N worlds.
    #
    # The eight candidates of a group share their scene AND their start pose --
    # the reset samples both per group -- so tracking worlds 0..31 keeps eight
    # copies each of four resets, and the video budget then fills with clips
    # that are, correctly, indistinguishable from one another. Striding by the
    # group size buys 32 different scenes for the same frame memory.
    group_stride = max(1, int(args.group_size))
    tracked = list(range(0, int(args.worlds), group_stride))[
        : max(1, int(args.track_worlds))
    ]
    videos_dir = output / "videos"
    near_miss_dir = videos_dir / "near_miss"
    rows: list[dict[str, Any]] = []
    success_videos = 0
    failure_videos = 0
    # Objects already on film. A budget of five spent on five apples shows one
    # scene five times; this spreads it over the catalog before it doubles up.
    filmed_catalogs: set[str] = set()
    diverged_total = 0

    for round_index in range(int(args.rounds)):
        round_state: dict[str, Any] = {}
        with recorder._FrameTap(
            world.backend, tracked, not bool(args.no_wrist)
        ) as tap:

            def _on_reset(snapshot: _ResetTap, _state=round_state) -> None:
                """Spend the frame budget on the episodes this leg is about.

                The reset has happened and nothing has been rendered yet, so the
                tracked set can still be chosen from the scene rather than fixed
                at worlds 0..N. It matters for the wrist-blind leg: only ~14% of
                episodes hide the target from the wrist at cap 0.33, so a fixed
                prefix of worlds would spend six frame budgets in seven on
                episodes the leg is not asking about. Falls back to the prefix
                when a reset produces no qualifying world at all.
                """

                visibility = _visibility(
                    torch,
                    geometry,
                    ee=snapshot.ee,
                    objects=snapshot.objects,
                    target_slots=snapshot.target_slots,
                    camera=camera,
                    aspect=aspect,
                )
                _state["visibility"] = visibility
                count = int(snapshot.ee.shape[0])
                qualifying = _filter_indices(
                    visibility, str(args.video_filter), count
                )
                # First qualifying world of each group, for the same reason the
                # default tracked set strides: the other seven are the same
                # reset.
                unique: list[int] = []
                claimed: set[int] = set()
                for index in qualifying:
                    group = index // group_stride
                    if group in claimed:
                        continue
                    claimed.add(group)
                    unique.append(index)
                chosen = (unique or list(tracked))[: len(tracked)]
                tap.worlds = list(chosen)
                tap.frames = {index: [] for index in chosen}

            with probe._ArmRunner(
                world, source=None, seed_offset=round_index
            ) as runner:
                with _ResetTap(
                    world.collector.resetter,
                    world.backend,
                    on_reset=_on_reset,
                ) as reset_tap:
                    result = world.collector.validate_round(
                        round_index=round_index
                    )
            frames = {index: list(value) for index, value in tap.frames.items()}

        # The state the episode ended in. Every group here runs the same
        # instruction at the same cap, so the horizons are equal and this is the
        # terminal step for every world rather than a mid-episode snapshot of
        # the ones that ran longer.
        terminal = world.backend.low_dim_observations()
        terminal_ee = terminal.ee_position.detach().float().cpu().numpy().copy()
        terminal_objects = (
            terminal.object_positions.detach().float().cpu().numpy().copy()
        )

        diverged = int(world.backend.pop_nonfinite_world_events())
        diverged_total += diverged
        worlds = int(world.layout.worlds_per_rank)
        def host(value: Any) -> np.ndarray:
            return value.reshape(-1).detach().float().cpu().numpy().copy()

        success = host(result.candidate_success) > 0.5
        final_dense = host(result.final_xy_distance)
        final_ee_z = host(result.final_ee_z)
        rewards = host(result.candidate_rewards)

        trace = runner.trace
        ee_track = trace.stack("ee_xyz")  # [decisions, worlds, 3]
        target_track = trace.stack("target_xyz")
        commanded = trace.stack("commanded0")
        relative0 = (target_track[0] - ee_track[0])[:, :2]
        start_distance = np.linalg.norm(relative0, axis=-1)
        cosine0 = probe._cosine(commanded[0][:, :2], relative0)
        xy_track = np.linalg.norm(
            (target_track - ee_track)[:, :, :2], axis=-1
        )
        best_distance = xy_track.min(axis=0)

        if reset_tap.objects is None or reset_tap.ee is None:
            raise RuntimeError("The reset tap recorded no scene.")
        visibility = round_state["visibility"]
        start_3d = np.linalg.norm(
            np.stack(
                (
                    reset_tap.ee[:, 0] - visibility["target_x"],
                    reset_tap.ee[:, 1] - visibility["target_y"],
                    reset_tap.ee[:, 2] - hover_z,
                ),
                axis=-1,
            ),
            axis=-1,
        )
        within_cap = start_3d <= cap_value + 1.0e-4
        at_start = _named_vs_nearest(
            reset_tap.ee, reset_tap.objects, reset_tap.target_slots
        )
        at_end = _named_vs_nearest(
            terminal_ee, terminal_objects, reset_tap.target_slots
        )
        reachable = _reachable(
            np.stack(
                (
                    visibility["target_x"],
                    visibility["target_y"],
                    visibility["target_z"],
                ),
                axis=-1,
            ),
            world.resetter.workspace_x_bounds,
            world.resetter.workspace_y_bounds,
        )
        catalog_ids = _per_world(
            reset_tap.target_catalog_ids
            if reset_tap.target_catalog_ids is not None
            else result.group_target_catalog_ids.detach().cpu().numpy(),
            worlds,
        )

        round_rows: list[dict[str, Any]] = []
        for index in range(worlds):
            round_rows.append(
                {
                    "leg": str(args.label),
                    "round": round_index,
                    "world": index,
                    "instruction": (
                        reset_tap.instructions[index]
                        if index < len(reset_tap.instructions)
                        else ""
                    ),
                    "target_catalog": ACTIVE_CDPR_CATALOGS[
                        int(catalog_ids[index])
                    ],
                    "success": bool(success[index]),
                    "scene_object_count": int(
                        visibility["scene_object_count"][index]
                    ),
                    "start_xy_distance_m": float(start_distance[index]),
                    "start_3d_distance_to_hover_m": float(start_3d[index]),
                    "start_xy_distance_to_nearest_other_m": float(
                        at_start["nearest_other"][index]
                    ),
                    "named_is_nearest_at_start": bool(
                        at_start["named_is_nearest"][index]
                    ),
                    "final_xy_distance_to_named_m": float(
                        at_end["named"][index]
                    ),
                    "final_xy_distance_to_nearest_other_m": float(
                        at_end["nearest_other"][index]
                    ),
                    "ended_closer_to_named": bool(
                        at_end["named_is_nearest"][index]
                    ),
                    "start_within_curriculum_cap": bool(within_cap[index]),
                    "best_xy_distance_m": float(best_distance[index]),
                    "final_dense_distance_m": float(final_dense[index]),
                    "final_ee_z_m": float(final_ee_z[index]),
                    "dense_reward": float(rewards[index]),
                    "cosine_decision0": float(cosine0[index]),
                    "horizon_decisions": int(runner.horizon_decisions),
                    "start_ee_x": float(reset_tap.ee[index, 0]),
                    "start_ee_y": float(reset_tap.ee[index, 1]),
                    "start_ee_z": float(reset_tap.ee[index, 2]),
                    "target_x": float(visibility["target_x"][index]),
                    "target_y": float(visibility["target_y"][index]),
                    "target_z": float(visibility["target_z"][index]),
                    "target_reachable": bool(reachable[index]),
                    "target_in_overview": bool(
                        visibility["target_in_overview"][index]
                    ),
                    "target_wrist_angle_deg": float(
                        visibility["target_wrist_angle_deg"][index]
                    ),
                    "target_certainly_in_wrist": bool(
                        visibility["target_certainly_in_wrist"][index]
                    ),
                    "target_certainly_out_of_wrist": bool(
                        visibility["target_certainly_out_of_wrist"][index]
                    ),
                    "target_in_any_camera": bool(
                        visibility["target_in_any_camera"][index]
                    ),
                    "scene_objects_in_overview": int(
                        visibility["scene_objects_in_overview"][index]
                    ),
                    "scene_objects_certainly_out_of_wrist": int(
                        visibility["scene_objects_certainly_out_of_wrist"][index]
                    ),
                    "all_scene_objects_out_of_wrist": bool(
                        visibility["all_scene_objects_out_of_wrist"][index]
                    ),
                    "tracked": index in frames,
                    "video": "",
                }
            )

        print(
            f"[move-to-validation] round {round_index}: success "
            f"{success.mean():.3f} ({int(success.sum())}/{worlds})  "
            f"start median {np.median(start_distance) * 1000:.0f} mm  "
            f"best median {np.median(best_distance) * 1000:.0f} mm  "
            f"horizon {runner.horizon_decisions}  "
            f"out-of-wrist {visibility['target_certainly_out_of_wrist'].mean():.3f}"
            f"  over-cap {1.0 - within_cap.mean():.4f}"
            f"  diverged {diverged}",
            flush=True,
        )

        def _write_video(index: int, *, directory: Path, kind: str) -> str:
            stack = frames.get(index) or []
            if not stack:
                return ""
            row = round_rows[index]
            wrist = (
                "wristblind"
                if row["target_certainly_out_of_wrist"]
                else f"wrist{row['target_wrist_angle_deg']:.0f}deg"
            )
            name = (
                f"{args.label}_r{round_index:02d}_w{index:03d}_{kind}"
                f"_{row['target_catalog']}"
                f"_obj{row['scene_object_count']}"
                f"_d{row['start_xy_distance_m'] * 1000:.0f}mm_{wrist}.mp4"
            )
            writer = recorder._Mp4(
                directory / name,
                fps=float(args.fps),
                height=stack[0].shape[0],
                width=stack[0].shape[1],
            )
            for frame in stack:
                writer.write(frame)
            writer.close()
            print(f"[move-to-validation]   wrote {kind}/{name}", flush=True)
            return str((directory / name).relative_to(output))

        eligible = [
            index
            for index in sorted(frames)
            if _passes_filter(round_rows[index], str(args.video_filter))
        ]

        def _spread(indices: Iterable[int]) -> list[int]:
            """Unfilmed objects first, then world order.

            NOT ranked by how well the episode went: these clips are the
            evidence for the rate in the CSV, and a best-first order would make
            them a highlight reel of the same run.
            """

            return sorted(
                indices,
                key=lambda i: (
                    str(round_rows[i]["target_catalog"]) in filmed_catalogs,
                    i,
                ),
            )

        for index in _spread(i for i in eligible if success[i]):
            if success_videos >= int(args.max_videos):
                break
            path = _write_video(index, directory=videos_dir, kind="success")
            if path:
                round_rows[index]["video"] = path
                filmed_catalogs.add(str(round_rows[index]["target_catalog"]))
                success_videos += 1
        # Failures ARE ranked, by closest approach: the question a near-miss
        # answers is how close the policy got, so the closest ones are the
        # informative ones.
        for index in sorted(
            (i for i in eligible if not success[i]),
            key=lambda i: float(best_distance[i]),
        ):
            if failure_videos >= int(args.failure_videos):
                break
            path = _write_video(index, directory=near_miss_dir, kind="nearmiss")
            if path:
                round_rows[index]["video"] = path
                failure_videos += 1

        rows.extend(round_rows)

    summary = _summary_rows(rows)
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "validation_summary.csv", summary)

    episodes = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    violations = [
        row
        for row in rows
        if not (row["target_in_any_camera"] and row["target_reachable"])
    ]
    manifest = {
        "label": str(args.label),
        "checkpoint": str(checkpoint),
        "checkpoint_global_step": int(world.payload.get("global_step", 0)),
        "config": str(config_path),
        "counts_toward_validation_metric": counts_toward_metric,
        "start_distance_cap_m": realized_caps,
        "start_distance_cap_override": (
            None if args.start_distance_cap is None else float(cap)
        ),
        "metadata_overrides": list(args.metadata_override or []),
        "scene_object_range": [int(scene_min), int(scene_max)],
        "worlds": int(args.worlds),
        "rounds": int(args.rounds),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else float("nan"),
        "validation_seed": int(world.args.validation_seed),
        "policy_mode": "deterministic_residual_mean_over_sampled_smolvla_prior",
        "success_predicate": "trainer validate_round, move_to xy tolerance",
        "success_distance_m": float(
            world.task_metadata.get("move_to_object_xy_tolerance", 0.02)
        ),
        "overview_camera": camera,
        "wrist_bounds_deg": list(geometry.wrist_bounds_deg(aspect)),
        "render_aspect": aspect,
        "episodes_outside_training_preconditions": len(violations),
        "episodes_over_curriculum_cap": sum(
            1 for row in rows if not row["start_within_curriculum_cap"]
        ),
        "curriculum_hover_z_m": hover_z,
        "diverged_worlds": diverged_total,
        "success_videos": success_videos,
        "near_miss_videos": failure_videos,
        "video_filter": str(args.video_filter),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print("\nVALIDATION -- move_to_object")
    print("---------------------------")
    print(f"  label                 {args.label}")
    print(f"  counts toward metric  {counts_toward_metric}")
    print(f"  episodes              {episodes}")
    print(
        f"  success rate          {successes / episodes:.4f} "
        f"({successes}/{episodes})"
        if episodes
        else "  success rate          n/a"
    )
    if violations:
        print(
            f"  PRECONDITION         {len(violations)} episode(s) had the "
            "target outside every camera or outside the reachable workspace; "
            "see the training_configuration row of validation_summary.csv for "
            "the rate over the qualifying subset only."
        )
    else:
        print(
            "  precondition          every episode had the target inside the "
            "overview or wrist frame AND inside the reachable workspace"
        )
    over_cap = sum(1 for row in rows if not row["start_within_curriculum_cap"])
    print(
        f"  start cap             {cap_value:.4f} m (3-D to the hover point); "
        f"{over_cap} episode(s) started outside it"
    )
    for row in summary:
        if row["scope"] in {"all", "training_configuration"}:
            continue
        print(
            f"    {row['scope']:<32} {str(row['value']):<20} "
            f"{row['episodes']:>6} eps  {row['success_rate']:.4f}  "
            f"best {row['mean_best_xy_distance_m']:.4f} m"
        )
    print(
        f"\nwrote {output}  ({success_videos} success video(s), "
        f"{failure_videos} near-miss video(s))"
    )
    if not success_videos:
        print(
            "  no successful tracked episode: raise --rounds or "
            "--track-worlds, or pass --failure-videos to see what it did "
            "instead."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
