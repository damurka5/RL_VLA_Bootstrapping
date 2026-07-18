#!/usr/bin/env python3
"""Rank-local, GPU-resident CDPR GRPO training on MJLab/MuJoCo Warp.

This is deliberately a separate entrypoint from the established CPU trainer.
Each torchrun rank owns a complete simulator batch and complete GRPO groups;
the only distributed communication in the rollout/update loop is update-level
schedule, curriculum, metric, and DDP gradient synchronization.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import (
    BatchedReverseFrontierResetter,
    RankLocalCurriculum,
    RankLocalMJWarpGRPOCollector,
    concatenate_collector_rounds,
)
from rl_vla_bootstrapping.policy.rank_local_grpo import (
    RankLocalGroupLayout,
    synchronize_equal_ddp_schedule,
)
from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import (
    _configure_distributed,
    _destroy_distributed,
    _log,
    _make_run_dir,
    _require_torch,
    _set_quiet_env,
    _set_seed,
    _silence_output,
    _write_json,
)
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import (
    SmolVLAGRPOTrainer,
    _resolve_checkpoint,
    parse_args,
)
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
    BatchedMoveToDistanceReward,
)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, default=str))
        stream.write("\n")


def _task_metadata(args: Any) -> dict[str, Any]:
    raw = os.environ.get("RLVLA_TASK_METADATA_JSON", "").strip()
    if raw:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("RLVLA_TASK_METADATA_JSON must encode an object.")
        return dict(payload)
    if args.config:
        config = load_project_config(Path(args.config).expanduser().resolve())
        return dict(config.task.metadata or {})
    return {}


def _synchronize_update_metrics_once(
    metrics: Mapping[str, float],
    *,
    device: Any,
) -> dict[str, float]:
    """Synchronize all metrics at the update boundary, never per group."""

    import torch
    import torch.distributed as dist

    keys = tuple(sorted(metrics))
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    world_size = 1
    wall_keys = ("rollout_time_s", "update_time_s")
    wall_values = torch.tensor(
        [float(metrics.get(key, 0.0)) for key in wall_keys],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(wall_values, op=dist.ReduceOp.MAX)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    summed = {
        key: float(value)
        for key, value in zip(keys, values.detach().cpu().tolist())
    }
    summed["distributed_world_size"] = float(world_size)
    # Counts and work are global sums. Component times and scalar optimizer
    # diagnostics are reported as rank means from the same collective.
    for key in keys:
        if (
            key.endswith("_time_s")
            or key.endswith("_mean")
            or key.endswith("_max")
            or key.endswith("_std")
            or key.endswith("_rate")
            or "_mean_" in key
            or key.startswith("loss_")
            or key
            in {
                "entropy_mean",
                "approx_kl_mean",
                "clip_fraction_mean",
                "smolvla_batch_size",
                "smolvla_inference_microbatch_size",
                "complete_groups_per_rank",
                "group_pass_rate_mean",
                "padded_records",
                "backward_collectives",
                "optimizer_steps",
                "timers_cuda_synchronized",
                "profiled_update",
                "dense_move_to_distance_reward",
            }
        ):
            summed[key] /= float(world_size)
    for key, value in zip(
        wall_keys, wall_values.detach().cpu().tolist()
    ):
        summed[key] = float(value)
    rollout_wall = max(summed.get("rollout_time_s", 0.0), 1.0e-9)
    summed["sampled_actions_per_second_global"] = (
        summed.get("sampled_environment_actions", 0.0) / rollout_wall
    )
    summed["selected_actions_per_second_global"] = (
        summed.get("selected_environment_actions", 0.0) / rollout_wall
    )
    summed["trajectory_work_amplification"] = (
        summed.get("sampled_environment_actions", 0.0)
        / max(1.0, summed.get("selected_environment_actions", 0.0))
    )
    return summed


def _runtime_metadata(args: Any, backend: Any) -> dict[str, Any]:
    metadata = backend.metadata()
    metadata.update(
        {
            "entrypoint": "smolvla_grpo_mjwarp_cdpr",
            "global_step_definition": (
                "cumulative selected environment actions summed across ranks"
            ),
            "smolvla_precision": str(args.mixed_precision),
            "smolvla_inference_microbatch_size": int(
                args.smolvla_inference_microbatch_size
            ),
        }
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if str(args.simulator_backend) != "mjlab_mjwarp":
        raise SystemExit(
            "This entrypoint requires --simulator-backend mjlab_mjwarp. "
            "Use smolvla_grpo_finetune_cdpr.py for the CPU backend."
        )
    if not bool(args.grpo_trajectory_groups):
        raise SystemExit(
            "MJWarp training requires --grpo-trajectory-groups true because "
            "worlds are terminal continuation candidates."
        )

    _require_torch()
    import torch

    dist_ctx = _configure_distributed(args)
    _set_quiet_env(args, dist_ctx)
    _set_seed(int(args.seed) + int(dist_ctx.rank) * 1_000_003)
    device = torch.device(str(dist_ctx.device))
    if device.type != "cuda":
        _destroy_distributed(dist_ctx)
        raise RuntimeError("MJWarp production training requires one CUDA GPU per rank.")

    layout = RankLocalGroupLayout(
        worlds_per_rank=int(args.worlds_per_rank),
        groups_per_rank=int(args.groups_per_rank),
        group_size=int(args.grpo_group_size),
    )
    layout.validate()
    layout.assert_no_cross_rank_group(dist_ctx.rank, dist_ctx.world_size)
    run_dir = _make_run_dir(args.run_root_dir, args.run_id)
    metrics_path = run_dir / "metrics.jsonl"
    if dist_ctx.is_main:
        _write_json(run_dir / "config.json", vars(args))

    backend = None
    runtime = None
    try:
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
        )
        _log(
            dist_ctx,
            "[smolvla-mjwarp] allocating "
            f"{layout.worlds_per_rank} worlds / {layout.groups_per_rank} "
            f"complete groups on rank {dist_ctx.rank} ({device})",
        )
        backend = create_cdpr_backend(backend_config)

        _log(
            dist_ctx,
            f"[smolvla-mjwarp] loading frozen SmolVLA replica on {device}: "
            f"{args.base_checkpoint}",
        )
        with _silence_output(not dist_ctx.is_main):
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
                compile_model=bool(args.smolvla_compile_model),
                compile_mode=str(args.smolvla_compile_mode),
            )

        trainer = SmolVLAGRPOTrainer(
            args=args,
            state_dim=int(args.state_dim),
            action_dim=int(args.action_dim),
            chunk_size=int(args.chunk_size),
            run_dir=run_dir,
            device=device,
            distributed=dist_ctx,
        )
        simulator_metadata = _runtime_metadata(args, backend)
        global_step = 0
        if args.resume_checkpoint:
            checkpoint = _resolve_checkpoint(args.resume_checkpoint)
            global_step = trainer.load(
                checkpoint,
                expected_simulator_metadata=simulator_metadata,
                allow_legacy_simulator_metadata=bool(
                    args.allow_legacy_simulator_checkpoint
                ),
            )
            _log(
                dist_ctx,
                f"[smolvla-mjwarp] resumed {checkpoint} at global step "
                f"{global_step}",
            )

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
        curriculum_state = trainer.loaded_extra_state.get("curriculum")
        if not isinstance(curriculum_state, Mapping):
            legacy_complex = trainer.loaded_extra_state.get("complex_runtime")
            if isinstance(legacy_complex, Mapping):
                curriculum_state = legacy_complex
        if isinstance(curriculum_state, Mapping):
            curriculum.restore(curriculum_state)
        task_metadata = _task_metadata(args)
        reward_mode = str(
            task_metadata.get("reward_mode", "sparse_binary")
        ).strip().lower()
        move_to_distance_reward = None
        if reward_mode == "dense":
            configured_instructions = tuple(args.instruction_types or ())
            if configured_instructions != ("move_to_object",):
                raise RuntimeError(
                    "The MJWarp dense reward path currently supports exactly "
                    "--instruction-types move_to_object; received "
                    f"{configured_instructions!r}."
                )
            move_to_distance_reward = (
                BatchedMoveToDistanceReward.from_metadata(task_metadata)
            )
        resetter = BatchedReverseFrontierResetter(
            backend=backend,
            layout=layout,
            curriculum=curriculum,
            rank=dist_ctx.rank,
            base_seed=int(args.seed),
            instruction_types=args.instruction_types,
            allowed_objects=args.allowed_objects,
            frontier_probability=float(
                args.reverse_frontier_sample_probability
            ),
            rehearsal_probability=float(
                args.reverse_frontier_rehearsal_probability
            ),
        )
        collector = RankLocalMJWarpGRPOCollector(
            backend=backend,
            smolvla_runtime=runtime,
            trainer=trainer,
            resetter=resetter,
            layout=layout,
            actions_per_policy_decision=int(args.replan_every),
            smolvla_microbatch_size=int(args.smolvla_inference_microbatch_size),
            normalize_advantage=bool(args.grpo_normalize_group_advantage),
            advantage_clip_abs=float(args.grpo_clip_advantage_abs),
            dynamic_min_pass_rate=float(args.grpo_dynamic_min_pass_rate),
            dynamic_max_pass_rate=float(args.grpo_dynamic_max_pass_rate),
            dynamic_sampling=bool(args.grpo_dynamic_sampling),
            group_selection=str(args.grpo_group_selection),
            move_to_distance_reward=move_to_distance_reward,
            profile=bool(args.mjwarp_profile_timers),
        )

        update_index = int(curriculum.updates)
        start_update_index = int(update_index)
        last_saved_step = int(global_step)
        while (
            global_step < int(args.max_train_steps)
            and (
                int(args.mjwarp_max_updates) <= 0
                or update_index - start_update_index
                < int(args.mjwarp_max_updates)
            )
        ):
            profile_limit = int(args.mjwarp_profile_updates)
            profile_this_update = bool(args.mjwarp_profile_timers) and (
                profile_limit <= 0
                or update_index - start_update_index < profile_limit
            )
            collector.profile = profile_this_update
            trainer.profile_update = profile_this_update
            update_started = time.perf_counter()
            rounds = []
            local_informative = 0
            max_rounds = max(
                1,
                (
                    int(args.grpo_max_groups_per_update)
                    + int(layout.groups_per_rank)
                    - 1
                )
                // int(layout.groups_per_rank),
            )
            for round_index in range(max_rounds):
                item = collector.collect_round(
                    update_index=update_index,
                    round_index=round_index,
                )
                rounds.append(item)
                local_informative += int(item.loss_mask.sum().item())
                if (
                    int(args.grpo_target_records_per_update) <= 0
                    or local_informative
                    >= int(args.grpo_target_records_per_update)
                ):
                    break
                if (
                    float(args.grpo_max_collection_seconds_per_update) > 0.0
                    and time.perf_counter() - update_started
                    >= float(args.grpo_max_collection_seconds_per_update)
                ):
                    break

            (
                records,
                loss_mask,
                candidate_rewards,
                successes,
                task_ids,
                shell_ids,
                rollout_metrics,
            ) = (
                concatenate_collector_rounds(rounds)
            )
            synchronization_time = 0.0
            synchronization_started = time.perf_counter()
            schedule = synchronize_equal_ddp_schedule(
                local_informative_records=int(loss_mask.sum().item()),
                records_per_minibatch=int(args.minibatch_size),
                ppo_epochs=int(args.ppo_epochs),
                device=device,
            )
            synchronization_time += (
                time.perf_counter() - synchronization_started
            )
            update_timer = time.perf_counter()
            update_metrics = trainer.update_tensor_records(
                records,
                loss_mask=loss_mask,
                schedule=schedule,
            )
            torch.cuda.synchronize(device)
            update_metrics["update_time_s"] = time.perf_counter() - update_timer

            synchronization_started = time.perf_counter()
            curriculum_metrics = curriculum.update_once_per_optimizer_update(
                group_instruction_ids=task_ids,
                group_shell_ids=shell_ids,
                candidate_success=successes,
            )
            synchronization_time += (
                time.perf_counter() - synchronization_started
            )
            capacity = backend.capacity_status()
            local_metrics = {
                **rollout_metrics,
                **update_metrics,
                "candidate_successes": float(successes.sum().item()),
                "candidate_worlds": float(successes.numel()),
                "candidate_reward_sum": float(candidate_rewards.sum().item()),
                "candidate_reward_count": float(candidate_rewards.numel()),
                "groups_collected": float(successes.shape[0]),
                "contacts_rank_sum": float(capacity["contacts"]),
                "max_constraints_per_world_rank_sum": float(
                    capacity["max_constraints_per_world"]
                ),
                "contact_capacity_overflow_ranks": float(
                    capacity["contact_overflow"]
                ),
                "constraint_capacity_overflow_ranks": float(
                    capacity["constraint_overflow"]
                ),
                "updates": 1.0,
            }
            synchronization_started = time.perf_counter()
            synchronized_metrics = _synchronize_update_metrics_once(
                local_metrics, device=device
            )
            synchronization_time += (
                time.perf_counter() - synchronization_started
            )
            global_selected = int(
                synchronized_metrics.get("selected_environment_actions", 0.0)
            )
            global_step += global_selected
            update_index += 1
            if (
                synchronized_metrics["contact_capacity_overflow_ranks"] > 0
                or synchronized_metrics["constraint_capacity_overflow_ranks"] > 0
            ):
                raise RuntimeError(
                    "MJWarp contact/constraint capacity overflow: "
                    f"local={capacity}, synchronized={synchronized_metrics}. "
                    "Increase simulator.nconmax/mjwarp_nconmax or "
                    "simulator.njmax/mjwarp_njmax; the backend will not continue "
                    "with truncated physics."
                )
            synchronized_metrics.update(curriculum_metrics)
            synchronized_metrics.update(
                {
                    "global_step": float(global_step),
                    "update_index": float(update_index),
                    "global_step_increment": float(global_selected),
                    "worlds_per_rank": float(layout.worlds_per_rank),
                    "groups_per_rank": float(layout.groups_per_rank),
                    "grpo_group_size": float(layout.group_size),
                    "synchronization_time_s_rank0": float(
                        synchronization_time
                    ),
                    "contact_capacity": float(capacity["contact_capacity"]),
                    "constraint_capacity_per_world": float(
                        capacity["constraint_capacity_per_world"]
                    ),
                    "profiled_update": float(profile_this_update),
                }
            )
            if profile_this_update:
                profiled_components = {
                    "smolvla_inference": float(
                        synchronized_metrics.get("smolvla_time_s", 0.0)
                    ),
                    "environment_step": float(
                        synchronized_metrics.get("physics_time_s", 0.0)
                    ),
                    "scene_reset": float(
                        synchronized_metrics.get("reset_time_s", 0.0)
                    ),
                    "backpropagation": float(
                        synchronized_metrics.get(
                            "backpropagation_time_s", 0.0
                        )
                    ),
                }
                selected_denominator = max(1.0, float(global_selected))
                for name, seconds in profiled_components.items():
                    synchronized_metrics[
                        f"profile/{name}_time_s"
                    ] = seconds
                    synchronized_metrics[
                        f"profile/{name}_ms_per_selected_action"
                    ] = 1000.0 * seconds / selected_denominator
                dominant_name, dominant_seconds = max(
                    profiled_components.items(), key=lambda item: item[1]
                )
                synchronized_metrics[
                    "profile/dominant_stage"
                ] = dominant_name
                synchronized_metrics[
                    "profile/dominant_stage_time_s"
                ] = dominant_seconds

            if dist_ctx.is_main:
                _append_jsonl(metrics_path, synchronized_metrics)
                if profile_this_update:
                    _write_json(
                        run_dir / "latest_profile.json",
                        {
                            key: value
                            for key, value in synchronized_metrics.items()
                            if key.startswith("profile/")
                            or key
                            in {
                                "global_step",
                                "update_index",
                                "sampled_environment_actions",
                                "selected_environment_actions",
                            }
                        },
                    )
                _log(
                    dist_ctx,
                    "[smolvla-mjwarp] "
                    f"step={global_step} update={update_index} "
                    f"sampled={synchronized_metrics['sampled_actions_per_second_global']:.1f} "
                    f"selected={synchronized_metrics['selected_actions_per_second_global']:.1f} "
                    f"success={synchronized_metrics['candidate_successes']:.0f}/"
                    f"{synchronized_metrics['candidate_worlds']:.0f} "
                    f"records={synchronized_metrics['informative_records']:.0f}",
                )

            save_due = (
                int(args.save_every_steps) > 0
                and global_step - last_saved_step >= int(args.save_every_steps)
            )
            final_update = global_step >= int(args.max_train_steps)
            final_update = final_update or (
                int(args.mjwarp_max_updates) > 0
                and update_index - start_update_index
                >= int(args.mjwarp_max_updates)
            )
            if dist_ctx.is_main and (save_due or final_update):
                trainer.save(
                    global_step=global_step,
                    args=args,
                    latest=False,
                    extra_state={"curriculum": curriculum.snapshot()},
                    simulator_metadata=simulator_metadata,
                )
                last_saved_step = int(global_step)
    finally:
        if backend is not None:
            backend.close()
        runtime = None
        _destroy_distributed(dist_ctx)


if __name__ == "__main__":
    main()
