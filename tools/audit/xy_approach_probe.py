"""Why the pick_up policy never servos horizontally: plant, command, or knowledge?

Written to test the campaign's §4.10/§4.11 chain -- *the projection is blind, so
the policy cannot servo, so it misses by 0.40 m, so the cap never promotes* -- by
breaking it in three places. FIRST RUN, 512 worlds, step 7505256. What it found
is recorded here, because two of the three legs came back against the premise
they were written to test.

**The headline: held-out validation measures a different task.** The approach
curriculum's cap reaches the resetter through
``set_random_start_max_goal_distance``, and the trainer calls that on the
TRAINING resetter only -- never on ``validation_resetter``, whose cap therefore
stays at the ``inf`` default that "restores the full-workspace start
distribution". With ``random_workspace_min_goal_xy_distance: 0.10`` also in
force, validation starts are >= 0.10 m from the goal while training starts are
<= 0.05 m. The two distributions do not overlap at any point. Run through the
trainer's own ``validate_round`` with the earned cap restored, the same
deterministic checkpoint scores **success 0.266, final distance 0.190 m** --
against the 0.001 and 0.39-0.42 m the run has logged for 52M steps. Reproduce
either side with ``--start-distance-cap``.

**A -- the XY plant (`--legs plant`).** Sustained ``a_x``/``a_y``, signed,
through zero, free and loaded, open-loop from real resets.

    MEASURED: gain 0.44-0.54 flat across every amplitude from 0.05 to 0.60, both
    signs, both axes, free and loaded alike; drift at ``a = 0`` is 0.012 mm/step,
    0.8 mm over an episode against the 400 mm under investigation. No dead zone,
    no rectification, no uncommanded bias. The XY-plant hypothesis is dead and
    the z arc was not aimed at the wrong axis.

    The one substantive reading: the realized gain is ~0.50, not 1.0. The
    effective step is ~0.0075 m, half the nominal ``action_step_xyz``. That
    halves the reach of any horizon budget and is why the oracle arms need
    ``--servo-max-command``.

**B -- what the checkpoint commands (`--legs policy`).** The deterministic mean
action, the prior, the true direction and the end-effector track, every decision.

    MEASURED: not a blind drift. ``direction_concentration`` 0.34 (a fixed drift
    reads ~1.0), ``command_spread`` 0.70 against ``command_mean_norm`` 0.30 (a
    fixed drift is the reverse), ``travel_cosine_to_object`` +0.47 -- the net
    displacement over an episode points TOWARD the object -- and
    ``mean_cosine_decision0`` +0.23 against the prior's +0.13. It is a weak,
    noisy, partially aimed servo, not a policy that ignores the object.
    ``tanh_saturated_fraction_xy`` is 0.000, so the composed action is nowhere
    near its bounds and the gradient is not being squashed.

    ``state_r2`` 0.009 is the honest limit: a linear read of the true direction
    explains almost none of the command. The aiming is real but small next to a
    0.76 command magnitude.

    Note on the campaign's headline, which stands. ``policy_target_cosine_mean``
    is taken on ``actions`` (mjwarp_rank_local_collector.py:2592) -- POST
    exploration noise at sigma 0.333 -- while ``prior_target_cosine`` is taken on
    the clean prior, and the block comment above them (line 2455) says the
    opposite. 0.11-against-0.05 compares a noise-attenuated quantity with a clean
    one. This leg reports both forms.

**C -- hand it the answer (`--legs oracle`).** Same resets, reward, grasp
detector and horizon; only the two XY channels are replaced by a servo on the
true object position.

    FIRST RUN WAS INSTRUMENT-BOUND, and is the reason ``--servo-max-command``
    exists. ``full_oracle`` -- the designated ceiling -- scored LOWEST of every
    arm (0.012, ever-grasped 0.010), the oracle arms ended farther out and higher
    than the untouched policy, and they diverged 35-54 worlds of 512 against the
    policy arm's 8. A ceiling arm below its own baseline is a broken instrument,
    not a finding. The mechanism: the parameter-free servo saturates at +-1 for
    anything past 1.5 cm, the plant realizes only half of it (leg A), so the
    saturated command is held long enough to drive the cable configuration
    singular -- and ``_contain_nonfinite_worlds`` then restores that world to its
    calibrated base pose, mid-episode, far from its object.

    So the pre-registered falsifier -- ``oracle_xy`` not beating ``policy`` means
    localization is not the constraint -- DID NOT FIRE, because its precondition
    failed. Re-run with the command cap before reading it.

    What survived: the ``oracle_xy_err_*`` ladder degrades monotonically in both
    success (0.188, 0.125, 0.074, 0.070, 0.047) and ever-grasped (0.340, 0.238,
    0.121, 0.084, 0.051) as the handed-over position is corrupted by 0 to 0.20 m.
    The substitution does something real, and localization accuracy does matter,
    inside an arm that is otherwise confounded.

Guards, both of which earned their place on the first run:

* Realized start distance comes from the reset itself, never from the logged cap
  -- which is how the validation-cap finding surfaced at all.
* Diverged-world counts are reported per arm, which is what exposed leg C as
  instrument-bound rather than informative.

Leg A needs no SmolVLA forward and no policy, so it skips loading the runtime.
Legs B and C share one build.

Usage::

    RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab \\
      python tools/audit/xy_approach_probe.py \\
        --checkpoint runs/<run>/smolvla_grpo_adapter.pt \\
        --output runs/xy_approach_probe

Reproduce the run's own held-out validation (uncapped starts)::

    ... --legs policy --start-distance-cap inf
"""

from __future__ import annotations

import os
import sys


def _configure_huggingface() -> None:
    """Mirror both halves of ``scripts/huggingface_public_models.sh``.

    Only the launchers source that script, so a tool run directly gets neither
    half and ``RLVLA_HF_OFFLINE=1`` on the command line does nothing at all --
    it is read by the shell helper, not by huggingface_hub. The failure lands
    after the weights have finished loading, inside the processor fetch, and
    reports as a 401/RepositoryNotFound on a public repo, which reads as a
    missing model and is not one.

    ``RLVLA_HF_PUBLIC_MODELS_ONLY`` (default 1) drops an inherited credential so
    the anonymous fetch stays anonymous. ``RLVLA_HF_OFFLINE`` (default 0) pins
    huggingface_hub and transformers to the local cache, for a box that holds
    the model files but has lost its route to huggingface.co.

    Called at import, before anything else here imports huggingface_hub: both
    switches are read into module constants on first import, so setting them
    later is silently too late.
    """

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
import json  # noqa: E402
from argparse import Namespace  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Amplitudes for the leg-A sweep. Signed and straddling zero: the zero arm is the
# uncommanded-drift control, and the negative arms are what would expose a
# rectifying plant (one direction moves, the other does not), which is the
# failure mode that would produce a monotone sideways run all by itself.
DEFAULT_PLANT_SWEEP = (
    -0.60, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.60,
)
# Object-position error stds for the oracle_xy pricing arms, in metres. 0.02 is
# roughly the grasp tolerance; 0.20 is most of the workspace.
DEFAULT_LOCALIZATION_ERRORS = (0.02, 0.05, 0.10, 0.20)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


@dataclass
class _World:
    """Everything a leg needs, built once and shared."""

    torch: Any
    device: Any
    args: Namespace
    payload: Mapping[str, Any]
    project: Any
    task_metadata: dict[str, Any]
    backend: Any
    layout: Any
    resetter: Any
    runtime: Any = None
    trainer: Any = None
    collector: Any = None
    grasp_offset: float = 0.0075
    action_step_xyz: float = 0.015
    action_step_gripper: float = 0.05
    fitted_gripper: Any = None


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch before weights_only.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "policy" not in payload:
        raise ValueError(f"{path} is not a GRPO policy checkpoint.")
    if not isinstance(payload.get("args"), Mapping):
        raise ValueError(
            f"{path} has no saved training arguments, so its MJWarp runtime "
            "cannot be reproduced."
        )
    return payload


def _checkpoint_has_lora(payload: Mapping[str, Any]) -> bool:
    """Whether the checkpoint stores action-expert LoRA weights to restore.

    ``save`` writes the key unconditionally and leaves it None/empty when no
    adapter was attached, so presence of the key is not the question -- presence
    of weights under it is.
    """

    state = payload.get("vla_lora")
    return bool(state)


def _probe_args(
    payload: Mapping[str, Any],
    *,
    config_path: Path,
    xml_path: Path,
    device: str,
    worlds: int,
    group_size: int,
    microbatch: int,
) -> Namespace:
    """The checkpoint's own arguments, with only the batch shape narrowed."""

    values = dict(payload["args"])
    values.update(
        {
            "config": str(config_path),
            "device": str(device),
            "distributed": False,
            "simulator_backend": "mjlab_mjwarp",
            "worlds_per_rank": int(worlds),
            "groups_per_rank": int(worlds) // int(group_size),
            "grpo_group_size": int(group_size),
            "mjwarp_xml_path": str(xml_path),
            "smolvla_inference_microbatch_size": int(microbatch),
            "smolvla_compile_model": False,
            "resume_checkpoint": None,
        }
    )
    return Namespace(**values)


def _build_world(
    *,
    checkpoint: Path,
    config_path: Path,
    device_str: str,
    worlds: int,
    group_size: int,
    microbatch: int,
    load_policy: bool,
    run_dir: Path,
    start_distance_cap: float | None = None,
) -> _World:
    """Reproduce the training stack. ``load_policy`` False skips SmolVLA."""

    import torch

    from rl_vla_bootstrapping.core.config import load_project_config
    from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
        _FITTED_GRIPPER,
        BatchedReverseFrontierResetter,
        RankLocalCurriculum,
        RankLocalMJWarpGRPOCollector,
    )
    from rl_vla_bootstrapping.policy.rank_local_grpo import RankLocalGroupLayout
    from rl_vla_bootstrapping.simulation.cdpr_backend import (
        CDPRBackendConfig,
        create_cdpr_backend,
    )
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        BatchedCatchReleaseDenseReward,
        BatchedMoveToDistanceReward,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This probe measures the MJWarp plant training runs on; it needs a "
            "CUDA GPU in the cdpr-mjlab environment."
        )
    device = torch.device(device_str)
    project = load_project_config(config_path)
    if str(project.simulator.backend) != "mjlab_mjwarp":
        raise ValueError(
            f"Expected an mjlab_mjwarp config, got {project.simulator.backend!r}."
        )
    xml_path = project.resolve_path(project.simulator.fixed_scene_xml)
    if xml_path is None:
        raise ValueError("The config does not define simulator.fixed_scene_xml.")

    payload = _load_checkpoint(checkpoint)
    args = _probe_args(
        payload,
        config_path=config_path,
        xml_path=xml_path,
        device=str(device),
        worlds=worlds,
        group_size=group_size,
        microbatch=microbatch,
    )
    task_metadata = dict(project.task.metadata or {})

    layout = RankLocalGroupLayout(
        worlds_per_rank=int(args.worlds_per_rank),
        groups_per_rank=int(args.groups_per_rank),
        group_size=int(args.grpo_group_size),
    )
    layout.validate()

    backend_config = CDPRBackendConfig(
        backend="mjlab_mjwarp",
        worlds_per_rank=int(args.worlds_per_rank),
        groups_per_rank=int(args.groups_per_rank),
        grpo_group_size=int(args.grpo_group_size),
        hold_steps=int(args.hold_steps),
        action_step_xyz=float(args.action_step_xyz),
        action_step_yaw=float(args.action_step_yaw),
        action_step_gripper=float(args.action_step_gripper),
        lock_non_commanded_axes=bool(args.lock_non_commanded_axes),
        lock_non_commanded_axes_threshold=float(
            args.lock_non_commanded_axes_threshold
        ),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        object_slots=int(args.object_slots),
        nconmax=int(args.mjwarp_nconmax),
        njmax=int(args.mjwarp_njmax),
        nccdmax=args.mjwarp_nccdmax,
        device=str(device),
        xml_path=Path(args.mjwarp_xml_path),
        **(
            {
                "workspace_z": (
                    float(args.controller_workspace_z_bounds[0]),
                    float(args.controller_workspace_z_bounds[1]),
                )
            }
            if getattr(args, "controller_workspace_z_bounds", None)
            else {}
        ),
    )
    print(
        f"[xy-probe] allocating {layout.worlds_per_rank} worlds "
        f"({layout.groups_per_rank} groups of {layout.group_size}) on {device}",
        flush=True,
    )
    backend = create_cdpr_backend(backend_config)

    curriculum = RankLocalCurriculum(
        device=device,
        promotion_success=float(args.reverse_frontier_promotion_success),
        demotion_success=float(args.reverse_frontier_demotion_success),
        validation_rollouts_per_shell=int(
            args.reverse_frontier_validation_episodes
        ),
        min_updates=int(args.reverse_frontier_min_train_updates),
        saturation_abort_threshold=float(
            args.reverse_frontier_saturation_abort_threshold
        ),
    )
    extra_state = dict(payload.get("extra_state") or {})
    curriculum_state = extra_state.get("curriculum")
    if not isinstance(curriculum_state, Mapping):
        curriculum_state = extra_state.get("complex_runtime")
    if isinstance(curriculum_state, Mapping):
        curriculum.restore(curriculum_state)

    # The validation resetter, exactly as training builds it: frontier only, no
    # rehearsal, balanced catalogs, validation seed. Held-out validation is the
    # regime whose 0.001 and 0.40 m this probe is explaining, so the arms have to
    # share its reset distribution or they are answering a different question.
    resetter = BatchedReverseFrontierResetter(
        backend=backend,
        layout=layout,
        curriculum=curriculum,
        rank=0,
        base_seed=int(args.validation_seed),
        instruction_types=args.instruction_types,
        allowed_objects=args.allowed_objects,
        frontier_probability=1.0,
        rehearsal_probability=0.0,
        balanced_target_catalogs=True,
        task_metadata=task_metadata,
    )
    _restore_approach_curriculum(
        resetter,
        args=args,
        task_metadata=task_metadata,
        extra_state=extra_state,
        cap_override=start_distance_cap,
    )

    world = _World(
        torch=torch,
        device=device,
        args=args,
        payload=payload,
        project=project,
        task_metadata=task_metadata,
        backend=backend,
        layout=layout,
        resetter=resetter,
        grasp_offset=float(
            task_metadata.get("pick_grasp_height_offset", 0.0075)
        ),
        action_step_xyz=float(args.action_step_xyz),
        action_step_gripper=float(args.action_step_gripper),
        fitted_gripper=torch.tensor(
            _FITTED_GRIPPER, dtype=torch.float32, device=device
        ),
    )
    if not load_policy:
        return world

    from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime
    from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
        SmolVLAGRPOTrainer,
    )

    print(
        f"[xy-probe] loading frozen SmolVLA {args.base_checkpoint}", flush=True
    )
    runtime = load_smolvla_runtime(
        checkpoint=str(args.base_checkpoint),
        device=str(device),
        mixed_precision=str(args.mixed_precision),
        image_size=int(args.image_size),
        state_dim=int(args.state_dim),
        image_feature_keys=(
            None
            if args.image_feature_keys is None
            else tuple(args.image_feature_keys)
        ),
        include_wrist=bool(args.include_wrist),
        include_aux_camera=bool(args.include_aux_camera),
        mask_empty_aux_camera=bool(
            getattr(args, "mask_empty_aux_camera", False)
        ),
        chunk_size=int(args.chunk_size),
        action_dim=int(args.action_dim),
        action_indices=(
            None
            if args.smolvla_action_indices is None
            else tuple(int(value) for value in args.smolvla_action_indices)
        ),
        action_normalization=str(args.smolvla_action_normalization),
        model_image_size=(
            None
            if int(args.smolvla_model_image_size) <= 0
            else int(args.smolvla_model_image_size)
        ),
        compile_model=False,
        compile_mode=str(args.smolvla_compile_mode),
        vision_pooling=str(
            getattr(args, "residual_vision_pooling", "flat_random")
        ),
    )
    trainer = SmolVLAGRPOTrainer(
        args=args,
        state_dim=int(payload["state_dim"]),
        action_dim=int(payload["action_dim"]),
        chunk_size=int(payload["chunk_size"]),
        run_dir=run_dir,
        device=device,
    )
    # The action-expert LoRA, before the residual weights.
    #
    # The residual was trained on top of the ADAPTED prior, so running it
    # against the stock SmolVLA is running it against an input it never saw.
    # There is no error if this is skipped -- the shapes all match and every arm
    # quietly measures a different policy -- which is why it is asserted rather
    # than attempted. The recorded drift is small (`vla_lora/kl` ~0.0001 in
    # every run), but "small enough to ignore" is a claim about the answer, and
    # this probe exists because that class of claim has been wrong here before.
    if _checkpoint_has_lora(payload):
        if not bool(getattr(args, "train_vla_lora", False)):
            raise RuntimeError(
                "The checkpoint carries action-expert LoRA weights but its own "
                "saved args have train_vla_lora False. The prior cannot be "
                "reproduced; refusing to measure a different policy."
            )
        info = trainer.attach_vla_lora(runtime)
        trainer._load_vla_lora_state(payload)
        runtime.policy.eval()
        print(
            "[xy-probe] restored action-expert LoRA: "
            f"{info['vla_lora/modules']:.0f} modules, "
            f"{info['vla_lora/trainable_params']:.0f} params",
            flush=True,
        )
    else:
        print(
            "[xy-probe] checkpoint carries no LoRA; prior is the stock frozen "
            "SmolVLA",
            flush=True,
        )

    trainer._unwrap(trainer.actor).load_state_dict(payload["policy"])
    trainer.actor.eval()

    include_relative_target = bool(
        getattr(args, "residual_relative_target", False)
    )
    vision_feature_dim = (
        int(getattr(args, "residual_vision_dim", 0))
        if bool(getattr(args, "residual_vision_features", False))
        else 0
    )
    move_to_reward = None
    catch_release_reward = None
    reward_mode = str(
        task_metadata.get("reward_mode", "sparse_binary")
    ).strip().lower()
    instructions = tuple(args.instruction_types or ())
    if reward_mode == "dense":
        if "move_to_object" in instructions:
            move_to_reward = BatchedMoveToDistanceReward.from_metadata(
                task_metadata
            )
        if {"put_into_plate", "put_into_bowl", "pick_up"}.intersection(
            instructions
        ):
            catch_release_reward = (
                BatchedCatchReleaseDenseReward.from_metadata(task_metadata)
            )
    collector = RankLocalMJWarpGRPOCollector(
        backend=backend,
        smolvla_runtime=runtime,
        trainer=trainer,
        resetter=resetter,
        layout=layout,
        actions_per_policy_decision=int(args.replan_every),
        smolvla_microbatch_size=int(args.smolvla_inference_microbatch_size),
        move_to_distance_reward=move_to_reward,
        catch_release_dense_reward=catch_release_reward,
        include_relative_target=include_relative_target,
        vision_feature_dim=vision_feature_dim,
        dynamic_sampling=False,
        group_selection="uniform",
    )
    world.runtime = runtime
    world.trainer = trainer
    world.collector = collector
    return world


def _restore_approach_curriculum(
    resetter: Any,
    *,
    args: Namespace,
    task_metadata: Mapping[str, Any],
    extra_state: Mapping[str, Any],
    cap_override: float | None = None,
) -> None:
    """Put the earned start-distance cap back on the resetter.

    Reported separately from the realized start distance on purpose. The cap is
    what the trainer logs; the realized distance is what the resetter produces,
    and the two have disagreed before because the curriculum lives on a base
    class a subclass can override.

    ``cap_override`` forces a cap instead of restoring the checkpoint's. A
    non-finite value disables the cap entirely, which is what the TRAINING
    validator does by omission: ``set_random_start_max_goal_distance`` is called
    on the training resetter only (smolvla_grpo_mjwarp_cdpr.py:1600) and never
    on ``validation_resetter``, whose cap therefore stays at its ``inf``
    default. Passing ``inf`` here reproduces held-out validation exactly.
    """

    from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import (
        PerInstructionApproachCurriculum,
        PreliftedStageCurriculum,
    )

    names = tuple(args.instruction_types or ("pick_up",))
    approach = PerInstructionApproachCurriculum(
        task_metadata, instruction_types=names
    )
    approach.load_state_dict(extra_state.get("approach_curriculum"))
    caps = approach.caps_by_instruction_id()
    if cap_override is not None:
        caps = {key: float(cap_override) for key in caps}
    resetter.set_random_start_max_goal_distance(caps)
    prelifted = PreliftedStageCurriculum(task_metadata)
    prelifted.load_state_dict(extra_state.get("prelifted_curriculum"))
    if prelifted.enabled:
        resetter.set_prelifted_group_fraction(prelifted.current_fraction())
    print(
        f"[xy-probe] restored approach caps (m): "
        f"{ {k: round(float(v), 4) for k, v in caps.items()} }",
        flush=True,
    )


# --------------------------------------------------------------------------
# Leg A -- the XY plant
# --------------------------------------------------------------------------


def _run_plant_arm(
    world: _World,
    *,
    axis: int,
    amplitude: float,
    steps: int,
    round_index: int,
    allow_prelifted: bool,
) -> dict[str, Any]:
    """Hold one constant XY command open-loop and measure realized motion.

    No SmolVLA, no policy, no reward -- this is a kinematic question about the
    plant and nothing else. The gripper channel commands "hold whatever you have"
    so a loaded arm does not drop its object and turn into the free arm.
    """

    torch = world.torch
    backend = world.backend
    reset = world.resetter.reset(
        update_index=0, round_index=round_index, allow_prelifted=allow_prelifted
    )
    worlds = int(world.layout.worlds_per_rank)
    active = torch.ones((worlds,), dtype=torch.bool, device=world.device)

    backend.pop_nonfinite_world_events()
    action = torch.zeros((worlds, 5), dtype=torch.float32, device=world.device)
    action[:, axis] = float(amplitude)
    track: list[Any] = []
    low_dim = backend.low_dim_observations()
    track.append(low_dim.ee_position.detach().float().clone())
    for _ in range(int(steps)):
        # Hold the current commanded opening: delta 0 leaves _controller_gripper
        # untouched, which is exactly "keep holding".
        action[:, 4] = 0.0
        low_dim = backend.step(action, active)
        track.append(low_dim.ee_position.detach().float().clone())
    diverged = int(backend.pop_nonfinite_world_events())

    positions = torch.stack(track, dim=0)  # [steps+1, worlds, 3]
    deltas = (positions[1:] - positions[:-1])[..., axis]  # [steps, worlds]

    # Exclude samples pinned against the workspace clamp: there the realized
    # motion is zero for a reason that has nothing to do with the gain, and
    # including them would manufacture a dead zone at large amplitudes.
    bound = float(
        max(
            backend.config.workspace_x
            if axis == 0
            else backend.config.workspace_y
        )
    )
    margin = 0.02
    near_clamp = positions[:-1, :, axis].abs() > (bound - margin)
    # Drop the first decision's worth of steps as controller transient.
    warmup = min(4, int(steps) - 1)
    usable = torch.zeros_like(near_clamp, dtype=torch.bool)
    usable[warmup:] = True
    usable &= ~near_clamp

    selected = deltas[usable]
    count = int(selected.numel())
    if count == 0:
        return {
            "axis": "xy"[axis],
            "amplitude": float(amplitude),
            "loaded": bool(allow_prelifted),
            "samples": 0,
            "mean_m_per_step": float("nan"),
            "median_m_per_step": float("nan"),
            "std_m_per_step": float("nan"),
            "gain_fraction": float("nan"),
            "clamped_fraction": 1.0,
            "diverged_worlds": diverged,
        }
    ideal = float(world.action_step_xyz) * float(amplitude)
    mean = float(selected.mean().item())
    return {
        "axis": "xy"[axis],
        "amplitude": float(amplitude),
        "loaded": bool(allow_prelifted),
        "samples": count,
        "mean_m_per_step": mean,
        "median_m_per_step": float(selected.median().item()),
        "std_m_per_step": float(selected.std().item()) if count > 1 else 0.0,
        # Realized over commanded. 1.0 is a perfect plant; 0.0 is a dead zone.
        "gain_fraction": (mean / ideal) if ideal != 0.0 else float("nan"),
        "clamped_fraction": float(near_clamp[warmup:].float().mean().item()),
        "diverged_worlds": diverged,
    }


def _run_plant_leg(
    world: _World,
    *,
    sweep: Sequence[float],
    steps: int,
    loaded: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = [False, True] if loaded else [False]
    round_index = 0
    for allow_prelifted in conditions:
        for axis in (0, 1):
            for amplitude in sweep:
                row = _run_plant_arm(
                    world,
                    axis=axis,
                    amplitude=float(amplitude),
                    steps=int(steps),
                    round_index=round_index,
                    allow_prelifted=allow_prelifted,
                )
                round_index += 1
                rows.append(row)
                print(
                    f"[xy-probe][plant] {'loaded' if allow_prelifted else 'free '} "
                    f"a_{row['axis']}={amplitude:+.2f}  "
                    f"realized={row['mean_m_per_step'] * 1000.0:+8.3f} mm/step  "
                    f"gain={row['gain_fraction']:+.3f}  "
                    f"clamped={row['clamped_fraction']:.2f}  "
                    f"n={row['samples']}",
                    flush=True,
                )
    return rows


# --------------------------------------------------------------------------
# Legs B and C -- action sources over the real validation loop
# --------------------------------------------------------------------------


@dataclass
class _Trace:
    """Per-decision record, batched over worlds and kept on the host."""

    rows: list[dict[str, np.ndarray]] = field(default_factory=list)

    def stack(self, key: str) -> np.ndarray:
        return np.stack([row[key] for row in self.rows], axis=0)


class _ArmRunner:
    """Runs ``collector.validate_round`` with the action source substituted.

    Substituting rather than reimplementing is the point. The reset, the reward,
    the grasp detector, the horizon, the termination and the reported metrics are
    the trainer's own code, so an arm differs from the deterministic policy in
    exactly one respect -- the five numbers it commands -- and nothing else can
    drift between this probe and training.
    """

    def __init__(
        self,
        world: _World,
        *,
        source: Callable[..., Any] | None,
        seed_offset: int = 0,
        horizon_override: int = 0,
    ) -> None:
        self.world = world
        self.source = source
        self.seed_offset = int(seed_offset)
        # Rewrite the rollout budget in place. `horizons` is the tensor
        # validate_round reads for both its decision count and its per-step
        # active mask, and nothing else consumes it -- the sparse-task evaluator
        # is called with max_steps 10_000, so episode length is governed here
        # and only here. Lets the horizon be varied without a training run.
        self.horizon_override = max(0, int(horizon_override))
        self.trace = _Trace()
        self.reset: Any = None
        self.decision = 0
        self.ever_grasped: Any = None
        # Per-episode localization error, drawn on first use and cleared at each
        # reset. See _make_oracle_xy_source for why it is not per-decision.
        self.position_error: Any = None
        self.horizon_decisions = 0
        self.world_success: Any = None
        self._original_action = None
        self._original_reset = None
        # Whether the name was already an INSTANCE attribute before the patch.
        # Both substituted names are ordinary class methods, so restoring by
        # assignment would leave a bound method shadowing the class for the rest
        # of the process -- harmless in one arm, but the probe runs several arms
        # back to back over one live trainer.
        self._owned_action = False
        self._owned_reset = False
        self._rng: Any = None

    # -- installation ---------------------------------------------------

    def __enter__(self) -> "_ArmRunner":
        collector = self.world.collector
        trainer = collector.trainer
        resetter = collector.resetter
        torch = self.world.torch

        self._original_action = trainer.deterministic_action_chunks_tensor
        self._original_reset = resetter.reset
        self._owned_action = (
            "deterministic_action_chunks_tensor" in vars(trainer)
        )
        self._owned_reset = "reset" in vars(resetter)
        self._rng = torch.Generator(device=self.world.device)
        self._rng.manual_seed(20260806 + self.seed_offset)

        def patched_reset(**kwargs: Any) -> Any:
            reset = self._original_reset(**kwargs)
            if self.horizon_override:
                reset.horizons.fill_(self.horizon_override)
            self.reset = reset
            self.decision = 0
            self.position_error = None
            self.ever_grasped = torch.zeros(
                (int(self.world.layout.worlds_per_rank),),
                dtype=torch.bool,
                device=self.world.device,
            )
            # The rollout budget is coupled to the approach-curriculum cap, so a
            # scripted arm that runs out of steps and one that cannot do the
            # task look identical in the success column. Recorded so they can be
            # told apart.
            self.horizon_decisions = int(reset.horizons.max().item())
            self.world.backend.pop_nonfinite_world_events()
            return reset

        def patched_action(*, states: Any, priors: Any, action_count: int) -> Any:
            chunk = self._original_action(
                states=states, priors=priors, action_count=action_count
            )
            low_dim = self.world.backend.low_dim_observations()
            rows = torch.arange(
                low_dim.object_positions.shape[0], device=self.world.device
            )
            target = low_dim.object_positions[
                rows, self.reset.task_state.target_slots
            ]
            holding = self.reset.physical_grasp.to(dtype=torch.bool)
            self.ever_grasped |= holding
            commanded = (
                chunk
                if self.source is None
                else self.source(
                    runner=self,
                    chunk=chunk,
                    priors=priors,
                    low_dim=low_dim,
                    target=target,
                    holding=holding,
                )
            )
            self._record(
                low_dim=low_dim,
                target=target,
                priors=priors,
                policy_chunk=chunk,
                commanded=commanded,
                holding=holding,
            )
            self.decision += 1
            return commanded

        trainer.deterministic_action_chunks_tensor = patched_action
        resetter.reset = patched_reset
        return self

    def __exit__(self, *exc: Any) -> None:
        trainer = self.world.collector.trainer
        resetter = self.world.collector.resetter
        if self._owned_action:
            trainer.deterministic_action_chunks_tensor = self._original_action
        else:
            vars(trainer).pop("deterministic_action_chunks_tensor", None)
        if self._owned_reset:
            resetter.reset = self._original_reset
        else:
            vars(resetter).pop("reset", None)

    # -- recording ------------------------------------------------------

    def _record(
        self,
        *,
        low_dim: Any,
        target: Any,
        priors: Any,
        policy_chunk: Any,
        commanded: Any,
        holding: Any,
    ) -> None:
        def host(value: Any) -> np.ndarray:
            return value.detach().float().cpu().numpy().copy()

        self.trace.rows.append(
            {
                "decision": np.full(
                    (int(low_dim.ee_position.shape[0]),),
                    float(self.decision),
                    dtype=np.float32,
                ),
                "ee_xyz": host(low_dim.ee_position),
                "target_xyz": host(target),
                "gripper_opening": host(low_dim.gripper_opening),
                # Action index 0 of each chunk only: that is the decision the
                # campaign's cosine metrics are taken at, and keeping the whole
                # chunk would quadruple the trace for no extra question.
                "prior0": host(priors[:, 0]),
                "policy_mean0": host(policy_chunk[:, 0]),
                "commanded0": host(commanded[:, 0]),
                "holding": host(holding.to(dtype=self.world.torch.float32)),
            }
        )

    def run(self, *, round_index: int) -> dict[str, Any]:
        collector = self.world.collector
        result = collector.validate_round(round_index=round_index)
        diverged = int(self.world.backend.pop_nonfinite_world_events())
        success = result.candidate_success.reshape(-1).float()
        self.world_success = success.detach().cpu().numpy().copy()
        return {
            "success_rate": float(success.mean().item()),
            "ever_grasped_rate": float(
                self.ever_grasped.float().mean().item()
            ),
            "final_distance_m": float(
                result.final_xy_distance.reshape(-1).mean().item()
            ),
            "final_ee_z_m": float(result.final_ee_z.reshape(-1).mean().item()),
            "min_ee_z_m": float(result.min_ee_z.reshape(-1).mean().item()),
            "reward_mean": float(
                result.candidate_rewards.reshape(-1).mean().item()
            ),
            "episodes": int(success.numel()),
            "decisions": int(len(self.trace.rows)),
            "horizon_decisions": int(self.horizon_decisions),
            "diverged_worlds": diverged,
        }


# -- the action sources -------------------------------------------------


def _servo_xy(
    rel_xy: Any, step: float, torch: Any, *, max_command: float = 1.0
) -> Any:
    """Proportional XY servo at the controller's own natural gain.

    ``rel / step`` commands exactly the displacement needed and saturates when
    further away than one step, so the proportional part is parameter-free.

    ``max_command`` caps that saturation, and the default is NOT 1.0 for a
    measured reason. Leg A puts the realized XY gain at ~0.50, so a saturated
    +-1 command is held for twice as many steps as the geometry suggests, and
    the backend restores any world whose cable configuration goes singular
    "under a large action" (mjlab_mjwarp_backend.py:1176) to its calibrated base
    pose -- mid-episode, far from its object. An arm that saturates therefore
    manufactures its own failures and reports them as the substitution not
    helping. Keep this inside the range leg A measured as linear.
    """

    limit = abs(float(max_command))
    return torch.clamp(rel_xy / float(step), -limit, limit)


def _make_sampled_source(world: _World, *, sigma: float) -> Callable[..., Any]:
    """The deterministic mean plus the trainer's exploration noise.

    The approach curriculum's promote gate does not read the deterministic rate
    this probe otherwise reports. It reads the SAMPLED, normal-start pass rate
    (smolvla_grpo_mjwarp_cdpr.py, instruction_successes_normal_start), so the
    number that decides whether the cap ever moves is this one -- and comparing
    a deterministic 0.33 against a 0.30 gate is comparing two different
    quantities.

    Reproduces sample_action_chunks_tensor: mean + N(0, sigma) per dimension,
    clamped to the action box. The per-episode offset is deliberately omitted;
    it is gated on already holding the object and configured only for z, so it
    does not touch the approach rate this gate is about.
    """

    torch = world.torch

    def source(*, runner: "_ArmRunner", chunk: Any, **_: Any) -> Any:
        noise = torch.randn(
            chunk.shape,
            dtype=chunk.dtype,
            device=chunk.device,
            generator=runner._rng,
        )
        return torch.clamp(chunk + noise * float(sigma), -1.0, 1.0)

    return source


def _make_oracle_xy_source(
    world: _World,
    *,
    position_error_std: float = 0.0,
    max_command: float = 1.0,
) -> Callable[..., Any]:
    """Keep the policy's z/yaw/gripper; replace only the XY channels.

    ``position_error_std`` corrupts the handed-over object position with an
    error drawn ONCE PER EPISODE and held, not resampled each decision. A
    feature that localizes badly is wrong in a consistent direction for as long
    as the scene does not change; per-decision resampling would instead let the
    servo average the error away over ~20 decisions and would price a bad
    feature as far more usable than it is. This is the same distinction the z
    arc turned on -- per-step i.i.d. noise explores a SUSTAINED bias only with
    sigma/sqrt(N) -- applied to the input side.
    """

    torch = world.torch

    def source(
        *, runner: "_ArmRunner", chunk: Any, low_dim: Any, target: Any, **_: Any
    ) -> Any:
        believed = target
        if position_error_std > 0.0:
            if runner.position_error is None:
                runner.position_error = torch.randn(
                    target.shape,
                    dtype=target.dtype,
                    device=target.device,
                    generator=runner._rng,
                ) * float(position_error_std)
            believed = target + runner.position_error
        rel_xy = (believed - low_dim.ee_position)[:, :2]
        command = _servo_xy(
            rel_xy, world.action_step_xyz, torch, max_command=max_command
        )
        out = chunk.clone()
        out[:, :, 0] = command[:, None, 0]
        out[:, :, 1] = command[:, None, 1]
        return out

    return source


def _make_full_oracle_source(
    world: _World,
    *,
    align_tolerance: float = 0.010,
    lift_command: float = 0.60,
    max_command: float = 1.0,
) -> Callable[..., Any]:
    """Scripted servo -> descend -> close -> lift, through the training env.

    The ceiling arm. ``lift_command`` is 0.60 because the measured loaded plant
    needs a sustained a_z of ~0.30 before it moves at all (§4.9) -- a scripted
    lift at 0.10 would fail for the reason the campaign already understands and
    would say nothing about the approach.
    """

    torch = world.torch

    def source(
        *,
        runner: "_ArmRunner",
        chunk: Any,
        low_dim: Any,
        target: Any,
        holding: Any,
        **_: Any,
    ) -> Any:
        step = float(world.action_step_xyz)
        grasp_point = target.clone()
        grasp_point[:, 2] = grasp_point[:, 2] + float(world.grasp_offset)
        xy_err = (grasp_point - low_dim.ee_position)[:, :2]
        z_err = grasp_point[:, 2] - low_dim.ee_position[:, 2]
        aligned = torch.linalg.vector_norm(xy_err, dim=-1) < float(
            align_tolerance
        )
        # Descended to or below the grasp height -- a HALF-SPACE, not a band.
        # The realized step is ~0.0075 m (leg A puts the gain at 0.50), so a
        # +-5 mm seating band is narrower than one step and the descent can pass
        # straight through it without ever arming the close. The first run of
        # this arm reached 0.022 m of the grasp point -- the closest of any arm
        # -- and grasped 4% of the time, which is that bug and not the task.
        seated = aligned & (z_err >= -0.008)
        engaged = holding | runner.ever_grasped

        a_xy = _servo_xy(xy_err, step, torch, max_command=max_command)
        # Stay on a hover plane until aligned, so the gripper does not descend
        # into the desk beside the object and push it away.
        hover_err = (grasp_point[:, 2] + 0.05) - low_dim.ee_position[:, 2]
        a_z = torch.where(
            engaged,
            torch.full_like(z_err, float(lift_command)),
            torch.where(
                aligned,
                # z is deliberately NOT capped by max_command. That cap exists
                # to keep sustained LATERAL commands out of the cable-singularity
                # regime; the z axis is the one the campaign has already
                # characterized, and throttling the descent to 0.35 costs ~19 env
                # steps of a 68-step budget for no safety it buys.
                torch.clamp(z_err / step, -1.0, 1.0),
                torch.clamp(hover_err / step, -1.0, 1.0),
            ),
        )
        # group_target_catalog_ids is per GROUP, not per world -- the resetter
        # draws one target catalog for each group of eight and broadcasts it.
        catalog = runner.reset.group_target_catalog_ids.reshape(-1).long()
        catalog = catalog.repeat_interleave(int(world.layout.group_size))
        fitted = world.fitted_gripper.index_select(0, catalog)
        commanded_open = world.backend._controller_gripper
        goal = torch.where(
            engaged | seated,
            (fitted - (0.001 / 0.03)).clamp(0.0, 1.0),
            torch.ones_like(fitted),
        )
        a_grip = torch.clamp(
            (goal - commanded_open) / float(world.action_step_gripper),
            -1.0,
            1.0,
        )
        out = torch.zeros_like(chunk)
        out[:, :, 0] = a_xy[:, None, 0]
        out[:, :, 1] = a_xy[:, None, 1]
        out[:, :, 2] = a_z[:, None]
        out[:, :, 3] = 0.0
        out[:, :, 4] = a_grip[:, None]
        return out

    return source


# --------------------------------------------------------------------------
# Leg B analysis
# --------------------------------------------------------------------------


def _grasp_timing(
    trace: _Trace, success: np.ndarray, *, horizon: int
) -> dict[str, Any]:
    """Does a grasp become a success, and does WHEN it happens decide that?

    ``oracle_xy`` reaches an ever-grasped rate of 0.85 -- essentially the
    ``success | pre-grasped`` 0.83 the campaign measured -- while converting only
    half of those grasps into successes. Two very different things produce that,
    and they need opposite fixes:

    * **Horizon starvation.** A world that latches at decision 12 of 17 has 20
      env steps left, and at the measured ~1.3 mm/step loaded lift rate a 50 mm
      lift needs about 38. The grasp is fine; the budget is not. The signature is
      a conversion rate that climbs with decisions remaining.
    * **Grasp quality.** Grasps earned during an approach are worse than the
      seated ones a pre-grasped reset manufactures. The signature is a
      conversion rate flat in decisions remaining.

    The horizon here is 17 because the approach cap is stuck at 0.05 m and
    ``curriculum_horizon_coupling_enabled`` interpolates the budget from the cap
    -- so if it is starvation, the cap and the horizon are one bottleneck, not
    two.
    """

    holding = trace.stack("holding") > 0.5  # [D, W]
    decisions, worlds = holding.shape
    ever = holding.any(axis=0)
    # argmax on a boolean gives the first True, and 0 where there is none; the
    # `ever` mask is what keeps those apart.
    first = np.argmax(holding, axis=0)
    remaining = int(horizon) - first

    out: dict[str, Any] = {
        "ever_grasped_rate": float(np.mean(ever)),
        "success_rate": float(np.mean(success)),
        "conversion_given_grasp": (
            float(np.mean(success[ever])) if ever.any() else float("nan")
        ),
        "first_grasp_decision_mean": (
            float(np.mean(first[ever])) if ever.any() else float("nan")
        ),
        "horizon_decisions": int(horizon),
    }
    buckets: list[dict[str, Any]] = []
    edges = [(0, 4), (4, 8), (8, 12), (12, 100)]
    for low, high in edges:
        mask = ever & (remaining >= low) & (remaining < high)
        buckets.append(
            {
                "decisions_remaining": f"{low}-{high if high < 100 else '+'}",
                "worlds": int(mask.sum()),
                "conversion": (
                    float(np.mean(success[mask])) if mask.any() else float("nan")
                ),
            }
        )
    out["conversion_by_decisions_remaining"] = buckets
    return out


def _unit(vectors: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(_unit(a) * _unit(b), axis=-1)


def _analyze_policy_trace(
    trace: _Trace, *, sigma: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Everything leg B asks, from one recorded run of the real policy."""

    ee = trace.stack("ee_xyz")  # [D, W, 3]
    target = trace.stack("target_xyz")
    prior = trace.stack("prior0")  # [D, W, 5]
    mean = trace.stack("policy_mean0")
    holding = trace.stack("holding") > 0.5

    rel_xy = (target - ee)[..., :2]
    mean_xy = mean[..., :2]
    prior_xy = prior[..., :2]

    # Approach steps only. A world already holding its object has no meaningful
    # "direction to the object", and half the TRAINING groups start pre-grasped
    # -- which is exactly why the campaign's absolute cosines move with the
    # prelifted fraction. Validation runs allow_prelifted=False, so this filter
    # should barely bite; it is here so the number cannot silently pick up the
    # same contamination if the arm is ever re-run on training resets.
    approach = ~holding
    far = np.linalg.norm(rel_xy, axis=-1) > 0.005
    usable = approach & far

    sampled_xy = mean_xy + rng.normal(0.0, sigma, size=mean_xy.shape)

    def pooled(values: np.ndarray) -> float:
        selected = values[usable]
        return float(np.mean(selected)) if selected.size else float("nan")

    first = usable[0]
    metrics: dict[str, Any] = {
        # The mean action's alignment -- the number the campaign meant to quote.
        "mean_cosine_all_decisions": pooled(_cosine(mean_xy, rel_xy)),
        "mean_cosine_decision0": float(
            np.mean(_cosine(mean_xy[0], rel_xy[0])[first])
        ),
        "prior_cosine_all_decisions": pooled(_cosine(prior_xy, rel_xy)),
        # The alignment the trainer actually logs: same mean, plus exploration
        # noise. If this reads ~0.11 while the row above reads much higher, the
        # campaign's headline comparison was an artefact of the noise.
        "sampled_cosine_all_decisions": pooled(_cosine(sampled_xy, rel_xy)),
        "mean_xy_magnitude": pooled(np.linalg.norm(mean_xy, axis=-1)),
        "prior_xy_magnitude": pooled(np.linalg.norm(prior_xy, axis=-1)),
    }

    # Is the command state-dependent at all? Compare the length of the AVERAGE
    # command against the average deviation from it. A constant drift has all its
    # length in the mean; a servo has it in the spread.
    flat = mean_xy[usable]
    if flat.size:
        grand = flat.mean(axis=0)
        metrics["command_mean_vector"] = [float(v) for v in grand]
        metrics["command_mean_norm"] = float(np.linalg.norm(grand))
        metrics["command_spread"] = float(
            np.mean(np.linalg.norm(flat - grand, axis=-1))
        )
        # 1.0 = every world commanded the same way in world frame regardless of
        # its object, i.e. a fixed drift. 0.0 = no preferred direction.
        metrics["direction_concentration"] = float(
            np.linalg.norm(_unit(flat).mean(axis=0))
        )
        # Least squares a_xy ~ A @ unit(rel_xy) + b, pooled. R^2 is how much of
        # the command the true direction explains: a policy that servos scores
        # high even at a small gain.
        #
        # The guard is load-bearing, not defensive. R^2 divides by the command's
        # own variance, and a state-INDEPENDENT command has none -- so the ratio
        # becomes float-noise over float-noise and comes out at 1.0, reporting
        # "the object direction explains this perfectly" for a policy that
        # ignores the object completely. That is the exact shape of a
        # measurement that confirms whatever it is pointed at, so a command with
        # no variance to explain reports NaN and the variance is published next
        # to the score rather than left implicit.
        design = np.concatenate(
            [_unit(rel_xy[usable]), np.ones((flat.shape[0], 1))], axis=1
        )
        solution, *_ = np.linalg.lstsq(design, flat, rcond=None)
        residual = flat - design @ solution
        total = flat - flat.mean(axis=0)
        denom = float(np.sum(total**2))
        variance = denom / float(flat.shape[0])
        metrics["command_variance_per_sample"] = variance
        metrics["state_r2"] = (
            float(1.0 - np.sum(residual**2) / denom)
            if variance > 1.0e-8
            else float("nan")
        )
    else:
        metrics["command_mean_vector"] = []

    # tanh headroom. The actor computes tanh(prior + scale * tanh(net)), so the
    # pre-tanh sum is recoverable from the prior and the emitted mean, and the
    # residual's OWN saturation with it: |pre_tanh - prior| = scale means the
    # residual's inner tanh is pinned and its gradient is gone.
    clipped = np.clip(mean, -1.0 + 1e-6, 1.0 - 1e-6)
    pre_tanh = np.arctanh(clipped)
    effective_residual = pre_tanh - prior
    metrics["pre_tanh_abs_xy_mean"] = float(
        np.mean(np.abs(pre_tanh[..., :2])[usable])
    )
    metrics["tanh_slope_xy_mean"] = float(
        np.mean((1.0 - clipped[..., :2] ** 2)[usable])
    )
    metrics["tanh_saturated_fraction_xy"] = float(
        np.mean(((1.0 - clipped[..., :2] ** 2) < 0.1)[usable])
    )
    metrics["residual_abs_xy_mean"] = float(
        np.mean(np.abs(effective_residual[..., :2])[usable])
    )

    # Where it starts, where it ends, and whether it ran into the wall. The start
    # distance is read off the reset, not off the logged curriculum cap.
    start = np.linalg.norm(rel_xy[0], axis=-1)
    last = ee[-1]
    metrics["start_xy_distance_mean_m"] = float(np.mean(start))
    metrics["start_xy_distance_p95_m"] = float(np.percentile(start, 95))
    metrics["start_xy_distance_max_m"] = float(np.max(start))
    metrics["final_xy_distance_mean_m"] = float(
        np.mean(np.linalg.norm((target - ee)[-1][..., :2], axis=-1))
    )
    metrics["net_xy_travel_mean_m"] = float(
        np.mean(np.linalg.norm((last - ee[0])[..., :2], axis=-1))
    )
    metrics["path_xy_length_mean_m"] = float(
        np.mean(np.sum(np.linalg.norm(np.diff(ee[..., :2], axis=0), axis=-1), axis=0))
    )
    travel = (last - ee[0])[..., :2]
    metrics["travel_direction_concentration"] = float(
        np.linalg.norm(_unit(travel).mean(axis=0))
    )
    metrics["travel_cosine_to_object"] = float(
        np.mean(_cosine(travel, rel_xy[0]))
    )
    metrics["final_at_xy_clamp_fraction"] = float(
        np.mean(np.max(np.abs(last[..., :2]), axis=-1) > 0.26)
    )
    return metrics


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _report_plant(rows: Sequence[Mapping[str, Any]], *, steps: int) -> None:
    print("\nA -- XY plant response to a SUSTAINED command")
    print("--------------------------------------------")
    print(
        f"{'cond':<7} {'axis':<5} {'a':>6} {'realized mm/step':>17} "
        f"{'gain':>7} {'clamped':>8} {'diverged':>9}"
    )
    for row in rows:
        print(
            f"{'loaded' if row['loaded'] else 'free':<7} "
            f"{row['axis']:<5} {row['amplitude']:>+6.2f} "
            f"{row['mean_m_per_step'] * 1000.0:>17.3f} "
            f"{row['gain_fraction']:>+7.3f} {row['clamped_fraction']:>8.2f} "
            f"{row['diverged_worlds']:>9d}"
        )
    zeros = [r for r in rows if r["amplitude"] == 0.0]
    drift = max((abs(r["mean_m_per_step"]) for r in zeros), default=0.0)
    print(
        f"\n  Uncommanded drift at a=0: {drift * 1000.0:.4f} mm/step "
        f"({drift * 1000.0 * int(steps):.1f} mm over this arm's {int(steps)} "
        "steps). The horizontal miss under\n  investigation is 400 mm."
    )
    print(
        "  A linear, sign-symmetric curve through zero means the XY plant is "
        "healthy and\n  the horizontal miss is not the plant's fault. A dead "
        "zone, an asymmetry between\n  signs, or a nonzero drift at a=0 each "
        "point somewhere very different."
    )


def _report_arms(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\nC -- success when the object position is handed over")
    print("----------------------------------------------------")
    print(
        f"{'arm':<26} {'success':>8} {'grasped':>8} {'final d (m)':>12} "
        f"{'final z':>8} {'min z':>7} {'reward':>8} {'diverged':>9}"
    )
    for row in rows:
        print(
            f"{row['arm']:<26} {row['success_rate']:>8.3f} "
            f"{row['ever_grasped_rate']:>8.3f} {row['final_distance_m']:>12.3f} "
            f"{row['final_ee_z_m']:>8.3f} {row['min_ee_z_m']:>7.3f} "
            f"{row['reward_mean']:>8.3f} {row['diverged_worlds']:>9d}"
        )
    horizon = max((int(row.get("horizon_decisions", 0)) for row in rows), default=0)
    print(
        f"\n  rollout budget: {horizon} decisions "
        f"(~{horizon * 4} env steps), set by the approach-curriculum cap. A "
        "scripted arm\n  that ran out of budget and one that cannot do the "
        "task score the same here; the\n  ever-grasped column separates them."
    )
    baseline = next((r for r in rows if r["arm"] == "policy"), None)
    oracle = next((r for r in rows if r["arm"] == "oracle_xy"), None)
    if baseline is not None and oracle is not None:
        print(
            f"\n  policy {baseline['success_rate']:.3f} -> oracle_xy "
            f"{oracle['success_rate']:.3f}. The campaign's own "
            "`success | pre-grasped` is 0.83,\n  which is what oracle_xy should "
            "reach if horizontal localization is the whole gap.\n  If it does "
            "not, localization is not the binding constraint and the projection "
            "is\n  not the thing to fix."
        )


def _report_timing(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    print("\nC2 -- does a grasp become a success, and does timing decide it?")
    print("---------------------------------------------------------------")
    print(
        f"{'arm':<26} {'grasped':>8} {'success':>8} {'convert':>8} "
        f"{'1st grasp':>10}   conversion by decisions remaining"
    )
    for row in rows:
        buckets = "  ".join(
            f"{item['decisions_remaining']}:"
            + (
                "  n/a"
                if item["worlds"] == 0
                else f"{item['conversion']:.2f}({item['worlds']})"
            )
            for item in row["conversion_by_decisions_remaining"]
        )
        print(
            f"{row['arm']:<26} {row['ever_grasped_rate']:>8.3f} "
            f"{row['success_rate']:>8.3f} {row['conversion_given_grasp']:>8.3f} "
            f"{row['first_grasp_decision_mean']:>10.1f}   {buckets}"
        )
    print(
        "\n  A conversion rate that CLIMBS with decisions remaining is horizon "
        "starvation:\n  the grasp is fine and the budget is not, and since the "
        "budget is interpolated\n  from the approach cap, the stuck cap and the "
        "short horizon are one bottleneck.\n  A conversion rate FLAT in "
        "decisions remaining means grasps earned during an\n  approach are "
        "simply worse than the seated ones a pre-grasped reset makes, and\n  no "
        "amount of extra budget fixes it."
    )


def _report_policy(metrics: Mapping[str, Any]) -> None:
    print("\nB -- what the deterministic policy commands")
    print("------------------------------------------")
    for key in (
        "start_xy_distance_mean_m",
        "start_xy_distance_p95_m",
        "start_xy_distance_max_m",
        "final_xy_distance_mean_m",
        "net_xy_travel_mean_m",
        "path_xy_length_mean_m",
        "final_at_xy_clamp_fraction",
        "travel_direction_concentration",
        "travel_cosine_to_object",
        "mean_cosine_decision0",
        "mean_cosine_all_decisions",
        "prior_cosine_all_decisions",
        "sampled_cosine_all_decisions",
        "mean_xy_magnitude",
        "prior_xy_magnitude",
        "command_mean_norm",
        "command_spread",
        "direction_concentration",
        "command_variance_per_sample",
        "state_r2",
        "pre_tanh_abs_xy_mean",
        "tanh_slope_xy_mean",
        "tanh_saturated_fraction_xy",
        "residual_abs_xy_mean",
    ):
        value = metrics.get(key)
        if value is None:
            continue
        print(f"  {key:<34} {float(value):+.4f}")
    print(f"  {'command_mean_vector':<34} {metrics.get('command_mean_vector')}")
    print(
        "\n  Compare against the campaign only at DECISION 0. The trainer's "
        "cosine is a\n  decision-0 probe, and the all-decisions figure is not "
        "the same quantity: any\n  sustained drift ends up running away from "
        "wherever the object was, so it goes\n  NEGATIVE over an episode while "
        "reading ~0 at the start. A drift and a servo are\n  separated by "
        "direction_concentration and state_r2, not by the pooled cosine."
        "\n\n  mean_cosine vs sampled_cosine is the size of the noise artefact "
        "in the\n  0.11-against-0.05 headline. command_mean_norm >> "
        "command_spread with\n  direction_concentration near 1 is a fixed "
        "drift; the reverse, with a high\n  state_r2, is a policy that servos. "
        "state_r2 is NaN when the command has no\n  variance to explain -- read "
        "command_variance_per_sample before trusting it."
    )


# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "examples"
        / "cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "xy_approach_probe"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--smolvla-microbatch", type=int, default=256)
    parser.add_argument(
        "--legs",
        default="plant,policy,oracle",
        help="Comma-separated subset of plant,policy,oracle.",
    )
    parser.add_argument(
        "--plant-sweep",
        type=float,
        nargs="+",
        default=list(DEFAULT_PLANT_SWEEP),
        help="Sustained XY amplitudes for leg A. Keep 0.0 in it.",
    )
    parser.add_argument(
        "--plant-steps",
        type=int,
        default=64,
        help="Env steps per leg-A arm.",
    )
    parser.add_argument(
        "--plant-skip-loaded",
        action="store_true",
        help="Free-flying arms only; skip the pre-grasped (loaded) condition.",
    )
    parser.add_argument(
        "--localization-errors",
        type=float,
        nargs="*",
        default=list(DEFAULT_LOCALIZATION_ERRORS),
        help="Object-position error stds (m) for the oracle_xy pricing arms.",
    )
    parser.add_argument(
        "--skip-full-oracle",
        action="store_true",
        help="Skip the scripted ceiling arm.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.333,
        help=(
            "Exploration std used to reconstruct the trainer's SAMPLED cosine "
            "from the mean. exp(-1.10), the pick_up max_log_std ceiling."
        ),
    )
    parser.add_argument(
        "--start-distance-cap",
        type=float,
        default=None,
        help=(
            "Override the checkpoint's approach-curriculum cap (m). Omit to "
            "restore what the checkpoint earned. Pass a non-positive value or "
            "inf to DISABLE the cap, which is what held-out validation does by "
            "omission -- set_random_start_max_goal_distance is called on the "
            "training resetter only, never on validation_resetter -- so `inf` "
            "reproduces the run's own validation start distribution."
        ),
    )
    parser.add_argument(
        "--servo-max-command",
        type=float,
        default=0.35,
        help=(
            "Cap on the oracle arms' commanded magnitude. Leg A measures the "
            "realized XY gain at ~0.50, so a saturated +-1 command is held "
            "twice as long as the geometry implies and drives worlds into the "
            "cable-singularity reset. 1.0 restores the saturating servo."
        ),
    )
    parser.add_argument(
        "--horizon-decisions",
        type=int,
        default=0,
        help=(
            "Override the rollout budget in decisions (0 keeps the coupled "
            "one). The budget is interpolated from the approach cap, so at the "
            "stuck 0.05 m cap it is 17 decisions -- and C2 shows conversion "
            "climbing with decisions remaining in every arm. This tests whether "
            "a longer budget alone lifts success, without a training run. "
            "curriculum_horizon_max is 26."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args(argv)

    legs = {name.strip() for name in str(args.legs).split(",") if name.strip()}
    unknown = legs.difference({"plant", "policy", "oracle"})
    if unknown:
        parser.error(f"Unknown legs: {sorted(unknown)}")
    if args.worlds % args.group_size:
        parser.error("--worlds must be a multiple of --group-size.")

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    start_cap = args.start_distance_cap
    if start_cap is not None and (start_cap <= 0.0 or start_cap == float("inf")):
        start_cap = float("inf")
    needs_policy = bool(legs & {"policy", "oracle"})
    world = _build_world(
        checkpoint=checkpoint,
        config_path=config_path,
        device_str=str(args.device),
        worlds=int(args.worlds),
        group_size=int(args.group_size),
        microbatch=int(args.smolvla_microbatch),
        load_policy=needs_policy,
        run_dir=output,
        start_distance_cap=start_cap,
    )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "worlds": int(args.worlds),
    }

    plant_rows: list[dict[str, Any]] = []
    if "plant" in legs:
        plant_rows = _run_plant_leg(
            world,
            sweep=tuple(args.plant_sweep),
            steps=int(args.plant_steps),
            loaded=not bool(args.plant_skip_loaded),
        )
        _write_csv(output / "plant_xy.csv", plant_rows)
        summary["plant"] = plant_rows

    arm_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    policy_metrics: dict[str, Any] | None = None
    if needs_policy:
        rng = np.random.default_rng(int(args.seed))
        arms: list[tuple[str, Callable[..., Any] | None]] = [
            ("policy", None),
            # What the promote gate actually reads.
            (
                "policy_sampled",
                _make_sampled_source(world, sigma=float(args.sigma)),
            ),
        ]
        if "oracle" in legs:
            servo_cap = float(args.servo_max_command)
            arms.append(
                (
                    "oracle_xy",
                    _make_oracle_xy_source(world, max_command=servo_cap),
                )
            )
            for error in args.localization_errors:
                arms.append(
                    (
                        f"oracle_xy_err_{float(error):.02f}m",
                        _make_oracle_xy_source(
                            world,
                            position_error_std=float(error),
                            max_command=servo_cap,
                        ),
                    )
                )
            if not args.skip_full_oracle:
                arms.append(
                    (
                        "full_oracle",
                        _make_full_oracle_source(world, max_command=servo_cap),
                    )
                )

        for index, (name, source) in enumerate(arms):
            print(f"[xy-probe] arm {name}", flush=True)
            with _ArmRunner(
                world,
                source=source,
                seed_offset=index,
                horizon_override=int(args.horizon_decisions),
            ) as runner:
                row = runner.run(round_index=0)
            row["arm"] = name
            arm_rows.append(row)
            print(
                f"[xy-probe][{name}] success={row['success_rate']:.3f} "
                f"grasped={row['ever_grasped_rate']:.3f} "
                f"final_d={row['final_distance_m']:.3f} m "
                f"diverged={row['diverged_worlds']}",
                flush=True,
            )
            if runner.trace.rows and runner.world_success is not None:
                timing = _grasp_timing(
                    runner.trace,
                    runner.world_success > 0.5,
                    horizon=int(runner.horizon_decisions),
                )
                timing["arm"] = name
                timing_rows.append(timing)
            if name == "policy" and runner.trace.rows:
                policy_metrics = _analyze_policy_trace(
                    runner.trace, sigma=float(args.sigma), rng=rng
                )
                np.savez_compressed(
                    output / "policy_trace.npz",
                    **{
                        key: runner.trace.stack(key)
                        for key in runner.trace.rows[0]
                    },
                )
        _write_csv(output / "arms.csv", arm_rows)
        summary["arms"] = arm_rows
        summary["grasp_timing"] = timing_rows
        if policy_metrics is not None:
            summary["policy"] = policy_metrics

    if plant_rows:
        _report_plant(plant_rows, steps=int(args.plant_steps))
    if policy_metrics is not None and "policy" in legs:
        _report_policy(policy_metrics)
    if arm_rows and "oracle" in legs:
        _report_arms(arm_rows)
        _report_timing(timing_rows)

    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
