from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from rl_vla_bootstrapping.core.commands import ensure_directory
from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.embodiments.mujoco import MujocoEmbodiment
from rl_vla_bootstrapping.simulation.scene_builder import build_scene_xml, preview_selection
from robots.cdpr.cdpr_dataset.rl_instruction_tasks import canonical_object_name
from robots.cdpr.cdpr_mujoco.policy_control import (
    CDPRPolicyControlSpec,
    apply_normalized_cdpr_action,
    policy_action_frequency_hz,
    policy_action_period_s,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OpenVLA/OFT policy on the CDPR embodiment.")
    parser.add_argument("--config", required=True, help="Path to bootstrap YAML/JSON/TOML config.")
    parser.add_argument("--adapter-path", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--action-head-path", required=True, help="Path to the RL action-head checkpoint (.pt).")
    parser.add_argument("--base-ckpt", default=None, help="Optional override for the base VLA checkpoint.")
    parser.add_argument("--scene", default=None, help="Scene name override. Defaults to config preview scene.")
    parser.add_argument("--target-object", default=None, help="Primary target object.")
    parser.add_argument(
        "--distractor",
        action="append",
        default=[],
        help="Additional distractor object. Repeat the flag to add more than one.",
    )
    parser.add_argument("--instruction", default=None, help="Instruction text. Defaults to 'pick up <target>'.")
    parser.add_argument("--steps", type=int, default=150, help="Maximum number of policy actions to execute.")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=None,
        help="Open-loop chunk length. Defaults to config action codec chunk size.",
    )
    parser.add_argument("--hold-steps", type=int, default=None, help="Extra simulator substeps per policy action.")
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through OpenVLA center-crop behavior.",
    )
    parser.add_argument("--run-dir", default=None, help="Optional output directory.")
    parser.add_argument("--run-name", default="policy_rollout", help="Artifact name prefix for saved results.")
    parser.add_argument(
        "--action-guard",
        type=float,
        default=1.25,
        help="Warn and clip if the policy outputs absolute values larger than this.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Logging cadence in policy steps.")
    parser.add_argument("--no-save", action="store_true", help="Skip simulator-side artifact export.")
    return parser


def _load_openvla_modules(policy_repo: Path):
    if str(policy_repo) not in sys.path:
        sys.path.insert(0, str(policy_repo))
    try:
        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
        )
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Could not import OpenVLA/OFT runtime dependencies. "
            f"Ensure the policy repo exists and the active environment has its requirements installed: {exc}"
        ) from exc
    return GenerateConfig, get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action, PeftModel


def _default_objects(config, target_object: str | None, distractors: list[str]) -> list[str]:
    if target_object:
        names = [target_object]
        names.extend(str(item) for item in distractors if item)
        return names
    _, preview_objects = preview_selection(config)
    names = list(preview_objects)
    if distractors:
        names.extend(str(item) for item in distractors if item)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _default_instruction(target_object: str | None, object_names: list[str]) -> str:
    target = target_object or (object_names[0] if object_names else "object")
    return f"pick up {canonical_object_name(target)}"


def _control_spec_from_config(config, hold_steps: int | None) -> CDPRPolicyControlSpec:
    limits = config.embodiment.action_adapter.controller_limits
    return CDPRPolicyControlSpec(
        xyz_limits=(
            limits["x"],
            limits["y"],
            limits["z"],
        ),
        action_step_xyz=float(config.training.rl.args.get("action_step_xyz", config.embodiment.action_adapter.controller_scales.get("x", 0.005))),
        action_step_yaw=float(config.training.rl.args.get("action_step_yaw", config.embodiment.action_adapter.controller_scales.get("yaw", 0.25))),
        open_gripper_threshold=float(config.embodiment.action_adapter.open_gripper_threshold),
        close_gripper_threshold=float(config.embodiment.action_adapter.close_gripper_threshold),
        hold_steps=int(config.training.rl.args.get("hold_steps", 0) if hold_steps is None else hold_steps),
    )


def _make_observation(sim, instruction: str, gripper_range: tuple[float, float]) -> tuple[dict[str, np.ndarray], int]:
    full_rgb = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")

    ee = np.asarray(sim.get_end_effector_position(), dtype=np.float32).reshape(-1)[:3]
    yaw = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip_phys = float(sim.get_gripper_opening()) if hasattr(sim, "get_gripper_opening") else float(gripper_range[1])
    g_lo, g_hi = gripper_range
    grip_norm = float(np.clip((grip_phys - g_lo) / max(g_hi - g_lo, 1e-6), 0.0, 1.0))

    state = np.array([ee[0], ee[1], ee[2], yaw, grip_norm], dtype=np.float32)
    obs = {
        "full_image": np.ascontiguousarray(full_rgb),
        "wrist_image": np.ascontiguousarray(wrist_rgb),
        "state": state,
        "task_description": instruction,
    }
    return obs, state.size


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_project_config(args.config)
    policy_repo = config.resolve_path(config.policy.repo_path)
    if policy_repo is None:
        raise RuntimeError("Config is missing `policy.repo_path`.")

    run_dir = (
        ensure_directory(Path(args.run_dir).expanduser().resolve())
        if args.run_dir
        else ensure_directory(
            (config.resolve_path(config.project.output_root) or Path("runs"))
            / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )

    object_names = _default_objects(config, args.target_object, args.distractor)
    if not object_names:
        raise RuntimeError("No objects selected for the rollout scene.")
    scene_name = args.scene or config.simulation.preview_scene or "desk"
    instruction = args.instruction or _default_instruction(args.target_object, object_names)

    xml_path = build_scene_xml(
        config,
        output_dir=run_dir / "scene",
        scene_name=scene_name,
        object_names=object_names,
    )
    embodiment = MujocoEmbodiment(config=config, spec=config.embodiment)
    sim = embodiment.instantiate_controller(xml_path=xml_path, run_dir=run_dir)
    sim.initialize()
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=10)
    setattr(sim, "language_instruction", instruction)

    control_spec = _control_spec_from_config(config, args.hold_steps)
    sim_dt = float(getattr(getattr(sim, "controller", None), "dt", 1.0 / 60.0))
    action_period = policy_action_period_s(sim_dt, control_spec.hold_steps)
    action_hz = policy_action_frequency_hz(sim_dt, control_spec.hold_steps)
    gripper_range = (
        float(getattr(sim, "gripper_min", config.embodiment.action_adapter.controller_limits["gripper"][0])),
        float(getattr(sim, "gripper_max", config.embodiment.action_adapter.controller_limits["gripper"][1])),
    )

    print(f"Run directory: {run_dir}")
    print(f"Scene: {scene_name}")
    print(f"Objects: {object_names}")
    print(f"Instruction: {instruction}")
    print(
        "Control contract: "
        f"dt={sim_dt:.6f}s, hold_steps={control_spec.hold_steps}, "
        f"sim_steps_per_action={control_spec.sim_steps_per_policy_action}, "
        f"policy_period={action_period:.6f}s (~{action_hz:.2f} Hz), "
        f"action_step_xyz={control_spec.action_step_xyz}, action_step_yaw={control_spec.action_step_yaw}"
    )

    (
        GenerateConfig,
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        get_vla_action,
        PeftModel,
    ) = _load_openvla_modules(policy_repo)

    cfg = GenerateConfig(
        pretrained_checkpoint=args.base_ckpt or config.policy.base_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=int(config.policy.num_images_in_input),
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=bool(args.center_crop),
        num_open_loop_steps=int(args.chunk_length or config.policy.action_codec.chunk_size),
        unnorm_key=None,
    )
    cfg.cdpr_action_head_path = str(Path(args.action_head_path).expanduser().resolve())

    vla_base = get_vla(cfg)
    vla_base.eval()
    if Path(args.adapter_path).expanduser().is_dir():
        vla = PeftModel.from_pretrained(vla_base, str(Path(args.adapter_path).expanduser().resolve()))
        vla.eval()
    else:
        raise RuntimeError(f"Adapter path is not a directory: {args.adapter_path}")

    processor = get_processor(cfg)
    param = next(vla.parameters())
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim).to(device=param.device, dtype=param.dtype)
    action_head.eval()

    initial_obs, proprio_dim = _make_observation(sim, instruction, gripper_range)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=proprio_dim)
    proprio_projector = proprio_projector.to(device=param.device, dtype=param.dtype)
    proprio_projector.eval()

    current_chunk = np.asarray(
        get_vla_action(cfg, vla, processor, initial_obs, instruction, action_head, proprio_projector),
        dtype=np.float32,
    )
    if current_chunk.ndim == 1:
        current_chunk = current_chunk.reshape(1, -1)
    if current_chunk.shape[1] < 5:
        raise RuntimeError(f"Expected at least 5 action dimensions, got chunk shape {current_chunk.shape}")
    current_chunk = current_chunk[:, :5]
    chunk_idx = 0

    for step in range(int(args.steps)):
        if chunk_idx >= len(current_chunk):
            obs, _ = _make_observation(sim, instruction, gripper_range)
            current_chunk = np.asarray(
                get_vla_action(cfg, vla, processor, obs, instruction, action_head, proprio_projector),
                dtype=np.float32,
            )
            if current_chunk.ndim == 1:
                current_chunk = current_chunk.reshape(1, -1)
            current_chunk = current_chunk[:, :5]
            chunk_idx = 0

        action = np.asarray(current_chunk[chunk_idx], dtype=np.float32).reshape(5)
        chunk_idx += 1
        max_abs = float(np.max(np.abs(action)))
        if max_abs > float(args.action_guard):
            print(f"[warn] Step {step}: policy action max abs {max_abs:.4f} > {args.action_guard}; clipping to [-1, 1].")

        result = apply_normalized_cdpr_action(
            sim,
            action,
            control_spec,
            capture_last_frame=True,
        )

        if step < 5 or step % max(1, int(args.log_every)) == 0:
            ee = np.asarray(result.get("ee_position", sim.get_end_effector_position()), dtype=np.float32).reshape(-1)[:3]
            yaw = float(result.get("ee_yaw", sim.get_yaw() if hasattr(sim, "get_yaw") else 0.0))
            grip = float(result.get("gripper_opening", sim.get_gripper_opening() if hasattr(sim, "get_gripper_opening") else 0.0))
            print(
                f"step={step:03d} action={np.clip(action, -1.0, 1.0).round(4).tolist()} "
                f"ee=({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}) yaw={yaw:.4f} grip={grip:.4f}"
            )

    if not args.no_save and hasattr(sim, "save_trajectory_results"):
        sim.save_trajectory_results(str(run_dir), args.run_name)
    if hasattr(sim, "cleanup"):
        sim.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
