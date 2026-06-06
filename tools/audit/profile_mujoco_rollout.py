#!/usr/bin/env python3
"""Profile CDPR MuJoCo reset/step/render paths without OpenVLA."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tools" / "audit" / "out"
DEFAULT_METADATA = {
    "instruction_sampling": "uniform_cycle",
    "target_object_pool": ["ycb_apple"],
    "distractor_object_pool": ["plate"],
    "required_scene_object_pool": ["plate"],
    "container_object_pool": ["plate"],
    "min_scene_objects": 2,
    "max_scene_objects": 2,
    "scene_variant_count": 1,
    "goal_center_xy": [0.0, 0.0],
    "goal_height_above_table": 0.10,
    "move_to_object_validation_distance_threshold": 0.025,
    "push_success_displacement": 0.08,
    "put_plate_xy_tolerance": 0.08,
    "put_plate_z_tolerance": 0.10,
    "reward_mode": "sparse_binary",
    "sparse_success_reward": 1.0,
    "sparse_failure_reward": 0.0,
}


def _add_paths() -> None:
    for path in (REPO_ROOT, REPO_ROOT / "robots" / "cdpr"):
        raw = path.as_posix()
        if raw not in sys.path:
            sys.path.insert(0, raw)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 1024 * 1024 * 8:
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def _gpu_vram_mb() -> float | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    values = []
    for line in proc.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            pass
    return max(values) if values else None


def _timer() -> float:
    return time.perf_counter()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _preprocess_images(frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    processed = {}
    for name, frame in frames.items():
        arr = np.asarray(frame, dtype=np.uint8)
        # Cheap OpenVLA-adjacent preprocessing substitute: resize by striding if needed,
        # normalize to float, and put channels first.
        if arr.shape[0] > 224:
            step = max(1, arr.shape[0] // 224)
            arr = arr[::step, ::step]
        arr = arr[:224, :224]
        processed[name] = np.transpose(arr.astype(np.float32) / 255.0, (2, 0, 1))
    return processed


def _write_csv(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "mujoco_rollout_profile.csv"
    fields = [
        "phase",
        "iteration",
        "duration_s",
        "fps",
        "cache_hit",
        "cache_miss",
        "cache_size",
        "rss_mb",
        "gpu_vram_mb",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_summary(summary: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mujoco_rollout_profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )


def _default_cached_wrapper() -> Path:
    candidates = sorted(
        (REPO_ROOT / "robots" / "cdpr" / "cdpr_dataset" / "wrappers").glob(
            "desk__plate-ycb_apple_wrapper/desk__plate-ycb_apple_wrapper.xml"
        )
    )
    if candidates:
        return candidates[0].resolve()
    wrappers = sorted((REPO_ROOT / "robots" / "cdpr" / "cdpr_dataset" / "wrappers").rglob("*_wrapper.xml"))
    if not wrappers:
        raise FileNotFoundError("No cached CDPR wrapper XMLs found under robots/cdpr/cdpr_dataset/wrappers.")
    return wrappers[0].resolve()


def _detect_profile_object_body(model: Any) -> int:
    import mujoco as mj

    robot_prefixes = ("rotor_", "slider_", "ee_", "camera_", "yaw_frame", "finger_")
    for body_id in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if not name or name == "world" or name.startswith(robot_prefixes):
            continue
        joint_count = int(model.body_jntnum[body_id])
        joint_adr = int(model.body_jntadr[body_id])
        if any(int(model.jnt_type[joint_adr + offset]) == int(mj.mjtJoint.mjJNT_FREE) for offset in range(joint_count)):
            return int(body_id)
    return 0


def _run_render_worker(wrapper_xml: Path, render_steps: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / "mujoco_render_worker_summary.json"
    if result_path.exists():
        result_path.unlink()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render-worker",
        "--wrapper-xml",
        str(wrapper_xml),
        "--render-steps",
        str(max(1, int(render_steps))),
        "--render-output",
        str(result_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=max(30, int(render_steps) * 2))
    except Exception as exc:
        return {"error": f"render worker launch failed: {type(exc).__name__}: {exc}"}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except Exception as exc:
            result = {"error": f"render worker output parse failed: {exc}"}
    else:
        result = {"error": "render worker produced no summary"}
    if proc.returncode != 0:
        result["error"] = (
            f"render worker exited with {proc.returncode}; stdout={proc.stdout[-400:]!r}; "
            f"stderr={proc.stderr[-400:]!r}; {result.get('error', '')}"
        )
    return result


def _render_worker_main(args: argparse.Namespace) -> int:
    import mujoco as mj

    wrapper_xml = Path(args.wrapper_xml).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    step_times: list[float] = []
    render_times: list[float] = []
    preprocess_times: list[float] = []
    error = ""
    try:
        model = mj.MjModel.from_xml_path(str(wrapper_xml))
        data = mj.MjData(model)
        renderer = mj.Renderer(model, height=480, width=640)
        overview_cam = mj.MjvCamera()
        overview_cam.type = mj.mjtCamera.mjCAMERA_FREE
        overview_cam.lookat[:] = np.array([0.0, 0.0, 0.10])
        overview_cam.distance = 1.5
        overview_cam.azimuth = 90
        overview_cam.elevation = -30
        ee_cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
        rng = np.random.default_rng(123)
        for idx in range(max(1, int(args.render_steps))):
            if model.nu:
                data.ctrl[:] = 0.0
                slider_noise = rng.uniform(-0.01, 0.01, size=min(4, model.nu))
                data.ctrl[: slider_noise.size] = slider_noise
            start = _timer()
            mj.mj_step(model, data)
            step_duration = _timer() - start
            step_times.append(step_duration)
            render_start = _timer()
            renderer.update_scene(data, camera=overview_cam)
            overview = renderer.render()
            if ee_cam_id != -1:
                renderer.update_scene(data, camera="ee_camera")
            else:
                renderer.update_scene(data)
            wrist = renderer.render()
            render_duration = _timer() - render_start
            render_times.append(render_duration)
            preprocess_start = _timer()
            _preprocess_images({"overview": overview, "ee_camera": wrist})
            preprocess_times.append(_timer() - preprocess_start)
            rows.append(
                {
                    "phase": "step_with_two_camera_render_worker",
                    "iteration": idx,
                    "duration_s": step_duration,
                    "fps": 1.0 / max(step_duration, 1e-9),
                    "rss_mb": _rss_mb(),
                    "gpu_vram_mb": _gpu_vram_mb(),
                }
            )
            rows.append(
                {
                    "phase": "two_camera_render_readback_worker",
                    "iteration": idx,
                    "duration_s": render_duration,
                    "fps": 1.0 / max(render_duration, 1e-9),
                    "rss_mb": _rss_mb(),
                    "gpu_vram_mb": _gpu_vram_mb(),
                }
            )
        renderer.close()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "error": error,
        "rows": rows,
        "step_times": step_times,
        "render_times": render_times,
        "preprocess_times": preprocess_times,
    }
    Path(args.render_output).write_text(json.dumps(result, indent=2, sort_keys=True))
    return 1 if error else 0


def _direct_sim_profile(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco as mj

    from cdpr_mujoco.model_cache import cache_stats, get_compiled_model
    from robots.cdpr.cdpr_dataset.rl_instruction_tasks import (
        InstructionSpec,
        compute_instruction_validation_success,
        init_reward_state,
    )

    wrapper_xml = _default_cached_wrapper()
    rows: list[dict[str, Any]] = []
    reset_times: list[float] = []
    compile_times: list[float] = []
    reward_times: list[float] = []
    step_times: list[float] = []
    render_step_times: list[float] = []
    render_times: list[float] = []
    preprocess_times: list[float] = []
    cache_hits = 0
    cache_misses = 0
    semantic_key = {
        "robot_xml_version": "direct_profile",
        "scene_name": "cached_wrapper",
        "sorted_object_set": tuple(sorted(args.allowed_objects)),
        "texture_variant": "wrapper",
        "object_topology_version": wrapper_xml.name,
    }

    for idx in range(max(1, args.resets)):
        start = _timer()
        model, event_obj = get_compiled_model(
            wrapper_xml,
            enabled=not args.disable_cache,
            timestep=1.0 / 60.0,
            offscreen_width=640,
            offscreen_height=480,
            offscreen_samples=os.environ.get("RLVLA_CDPR_OFFSCREEN_SAMPLES", "1"),
            semantic_key=semantic_key,
        )
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        duration = _timer() - start
        event = event_obj.as_dict()
        compile_time = float(event.get("compile_time_s", 0.0))
        compile_times.append(compile_time)
        reset_times.append(duration)
        hit = bool(event.get("hit", False))
        miss = bool(event.get("miss", False))
        cache_hits += int(hit)
        cache_misses += int(miss)
        rows.append(
            {
                "phase": "reset_direct_sim",
                "iteration": idx,
                "duration_s": duration,
                "cache_hit": int(hit),
                "cache_miss": int(miss),
                "cache_size": event.get("cache_size", ""),
                "rss_mb": _rss_mb(),
                "gpu_vram_mb": _gpu_vram_mb(),
                "note": str(wrapper_xml),
            }
        )

    model, _event_obj = get_compiled_model(
        wrapper_xml,
        enabled=not args.disable_cache,
        timestep=1.0 / 60.0,
        offscreen_width=640,
        offscreen_height=480,
        offscreen_samples=os.environ.get("RLVLA_CDPR_OFFSCREEN_SAMPLES", "1"),
        semantic_key=semantic_key,
    )
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    ee_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ee_base")
    object_body = _detect_profile_object_body(model)
    spec = InstructionSpec(
        instruction_type="move_to_object",
        text="move to object",
        target_object="ycb_apple",
        direction=np.zeros(3, dtype=np.float32),
        target_displacement=0.0,
        lift_target=0.0,
    )
    ee0 = np.asarray(data.xpos[ee_body], dtype=np.float32)
    obj0 = np.asarray(data.xpos[object_body], dtype=np.float32)
    reward_state = init_reward_state(ee0, obj0)
    rng = np.random.default_rng(args.seed)

    for idx in range(max(1, args.steps)):
        if model.nu:
            data.ctrl[:] = 0.0
            slider_noise = rng.uniform(-0.01, 0.01, size=min(4, model.nu))
            data.ctrl[: slider_noise.size] = slider_noise
        start = _timer()
        mj.mj_step(model, data)
        duration = _timer() - start
        step_times.append(duration)
        reward_start = _timer()
        compute_instruction_validation_success(
            spec=spec,
            ee_pos=np.asarray(data.xpos[ee_body], dtype=np.float32),
            reward_state=reward_state,
            task_metadata=DEFAULT_METADATA,
            obj_pos=np.asarray(data.xpos[object_body], dtype=np.float32),
            goal_pos=np.asarray(data.xpos[object_body], dtype=np.float32),
        )
        reward_times.append(_timer() - reward_start)
        rows.append(
            {
                "phase": "step_no_render_direct_sim",
                "iteration": idx,
                "duration_s": duration,
                "fps": 1.0 / max(duration, 1e-9),
                "rss_mb": _rss_mb(),
                "gpu_vram_mb": _gpu_vram_mb(),
            }
        )

    render_result = _run_render_worker(wrapper_xml, args.render_steps)
    for row in render_result.get("rows", []):
        rows.append(row)
    render_step_times.extend(float(x) for x in render_result.get("step_times", []))
    render_times.extend(float(x) for x in render_result.get("render_times", []))
    preprocess_times.extend(float(x) for x in render_result.get("preprocess_times", []))
    if render_result.get("error"):
        rows.append(
            {
                "phase": "two_camera_render_readback_direct_sim",
                "iteration": "",
                "duration_s": "",
                "fps": "",
                "rss_mb": _rss_mb(),
                "gpu_vram_mb": _gpu_vram_mb(),
                "note": str(render_result.get("error")),
            }
        )

    return {
        "mode": "direct_mujoco_fallback",
        "rows": rows,
        "reset_times": reset_times,
        "compile_times": compile_times,
        "reward_times": reward_times,
        "step_times": step_times,
        "render_step_times": render_step_times,
        "render_times": render_times,
        "preprocess_times": preprocess_times,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_stats": cache_stats(),
        "wrapper_xml": str(wrapper_xml),
        "render_error": render_result.get("error", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--stable-objects", action="store_true", default=True)
    parser.add_argument("--allowed-objects", nargs="*", default=["ycb_apple", "plate"])
    parser.add_argument("--render-worker", action="store_true")
    parser.add_argument("--wrapper-xml", default="")
    parser.add_argument("--render-output", default=str(OUT_DIR / "mujoco_render_worker_summary.json"))
    args = parser.parse_args()

    if args.render_worker:
        if not args.wrapper_xml:
            parser.error("--render-worker requires --wrapper-xml")
        return _render_worker_main(args)

    _add_paths()
    os.environ.setdefault("RLVLA_TASK_METADATA_JSON", json.dumps(DEFAULT_METADATA))
    os.environ.setdefault("RLVLA_CDPR_OFFSCREEN_SAMPLES", "1")
    if args.stable_objects:
        os.environ.setdefault("RLVLA_CDPR_USE_STABLE_OBJECTS", "1")
    os.environ["RLVLA_CDPR_COMPILED_MODEL_CACHE"] = "0" if args.disable_cache else "1"

    rows: list[dict[str, Any]] = []
    reset_times: list[float] = []
    compile_times: list[float] = []
    reward_times: list[float] = []
    step_times: list[float] = []
    render_step_times: list[float] = []
    render_times: list[float] = []
    preprocess_times: list[float] = []
    cache_hits = 0
    cache_misses = 0
    error = ""
    profile_mode = "cdpr_language_rl_env"
    fallback_note = ""
    render_error = ""

    try:
        from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation
        from robots.cdpr.cdpr_dataset.rl_cdpr_env import CDPRLanguageRLEnv

        env = CDPRLanguageRLEnv(
            max_steps=max(args.steps, args.render_steps) + 10,
            capture_frames=False,
            record_trajectory=False,
            allowed_objects=args.allowed_objects,
            wrapper_cleanup=False,
            use_wrapper_cache=True,
            reuse_existing_wrapper_variants=True,
            use_compiled_model_cache=not args.disable_cache,
            seed=args.seed,
        )
        rng = np.random.default_rng(args.seed)

        obs = None
        info: dict[str, Any] = {}
        for idx in range(max(1, args.resets)):
            start = _timer()
            obs, info = env.reset(
                seed=args.seed + idx,
                options={
                    "instruction_type": "move_to_object",
                    "target_object": "ycb_apple",
                    "reference_object": "plate",
                },
            )
            duration = _timer() - start
            reset_times.append(duration)
            compile_time = float(info.get("mujoco_model_compile_time_s", 0.0))
            compile_times.append(compile_time)
            hit = bool(info.get("mujoco_model_cache_hit", False))
            miss = bool(info.get("mujoco_model_cache_miss", False))
            cache_hits += int(hit)
            cache_misses += int(miss)
            rows.append(
                {
                    "phase": "reset",
                    "iteration": idx,
                    "duration_s": duration,
                    "cache_hit": int(hit),
                    "cache_miss": int(miss),
                    "cache_size": info.get("mujoco_model_cache_size", ""),
                    "rss_mb": _rss_mb(),
                    "gpu_vram_mb": _gpu_vram_mb(),
                    "note": info.get("wrapper_xml", ""),
                }
            )

        if obs is None:
            raise RuntimeError("No reset completed.")

        for idx in range(max(1, args.steps)):
            action = rng.uniform(-0.4, 0.4, size=(5,)).astype(np.float32)
            start = _timer()
            _, _, terminated, truncated, step_info = env.step(action)
            duration = _timer() - start
            step_times.append(duration)
            reward_times.append(float(step_info.get("reward_compute_time_s", 0.0)))
            rows.append(
                {
                    "phase": "step_no_render",
                    "iteration": idx,
                    "duration_s": duration,
                    "fps": 1.0 / max(duration, 1e-9),
                    "rss_mb": _rss_mb(),
                    "gpu_vram_mb": _gpu_vram_mb(),
                }
            )
            if terminated or truncated:
                env.reset(
                    seed=args.seed + 10_000 + idx,
                    options={"instruction_type": "move_to_object", "target_object": "ycb_apple"},
                )

        env.capture_frames = True
        for idx in range(max(1, args.render_steps)):
            action = rng.uniform(-0.4, 0.4, size=(5,)).astype(np.float32)
            start = _timer()
            _, _, terminated, truncated, _step_info = env.step(action)
            duration = _timer() - start
            render_step_times.append(duration)
            render_start = _timer()
            frames = env.render(["overview", "ee_camera"])
            render_duration = _timer() - render_start
            render_times.append(render_duration)
            preprocess_start = _timer()
            _preprocess_images(frames)
            preprocess_times.append(_timer() - preprocess_start)
            rows.append(
                {
                    "phase": "step_with_two_camera_render",
                    "iteration": idx,
                    "duration_s": duration,
                    "fps": 1.0 / max(duration, 1e-9),
                    "rss_mb": _rss_mb(),
                    "gpu_vram_mb": _gpu_vram_mb(),
                }
            )
            rows.append(
                {
                    "phase": "two_camera_render_readback",
                    "iteration": idx,
                    "duration_s": render_duration,
                    "fps": 1.0 / max(render_duration, 1e-9),
                    "rss_mb": _rss_mb(),
                    "gpu_vram_mb": _gpu_vram_mb(),
                }
            )
            if terminated or truncated:
                env.reset(
                    seed=args.seed + 20_000 + idx,
                    options={"instruction_type": "move_to_object", "target_object": "ycb_apple"},
                )

        cache_stats = HeadlessCDPRSimulation.compiled_model_cache_stats()
        env.close()
    except Exception as exc:
        if isinstance(exc, ImportError) and "gym" in str(exc).lower():
            fallback_note = f"CDPRLanguageRLEnv unavailable ({exc}); profiled raw MuJoCo model/data path instead."
            direct = _direct_sim_profile(args)
            profile_mode = str(direct["mode"])
            rows = list(direct["rows"])
            reset_times = list(direct["reset_times"])
            compile_times = list(direct["compile_times"])
            reward_times = list(direct["reward_times"])
            step_times = list(direct["step_times"])
            render_step_times = list(direct["render_step_times"])
            render_times = list(direct["render_times"])
            preprocess_times = list(direct["preprocess_times"])
            cache_hits = int(direct["cache_hits"])
            cache_misses = int(direct["cache_misses"])
            cache_stats = dict(direct["cache_stats"])
            render_error = str(direct.get("render_error", ""))
        else:
            error = f"{type(exc).__name__}: {exc}"
            cache_stats = {}

    summary = {
        "repo_root": str(REPO_ROOT),
        "profile_mode": profile_mode,
        "fallback_note": fallback_note,
        "error": error,
        "render_error": render_error,
        "resets": len(reset_times),
        "steps_without_render": len(step_times),
        "steps_with_render": len(render_step_times),
        "compile_time_s_mean": mean(compile_times) if compile_times else 0.0,
        "compile_time_s_p95": _percentile(compile_times, 95),
        "reset_time_s_mean": mean(reset_times) if reset_times else 0.0,
        "reset_time_s_p95": _percentile(reset_times, 95),
        "physics_step_time_s_mean": mean(step_times) if step_times else 0.0,
        "physics_step_time_s_p95": _percentile(step_times, 95),
        "reward_success_predicate_time_s_mean": mean(reward_times) if reward_times else 0.0,
        "two_camera_render_readback_time_s_mean": mean(render_times) if render_times else 0.0,
        "two_camera_render_readback_time_s_p95": _percentile(render_times, 95),
        "image_preprocessing_time_s_mean": mean(preprocess_times) if preprocess_times else 0.0,
        "fps_without_render": 1.0 / max(mean(step_times), 1e-9) if step_times else 0.0,
        "fps_with_render": 1.0 / max(mean(render_step_times), 1e-9) if render_step_times else 0.0,
        "cache_hits_observed": int(cache_hits),
        "cache_misses_observed": int(cache_misses),
        "cache_stats": cache_stats,
        "cpu_ram_mb": _rss_mb(),
        "gpu_vram_mb": _gpu_vram_mb(),
        "outputs": {
            "csv": str(OUT_DIR / "mujoco_rollout_profile.csv"),
            "summary_json": str(OUT_DIR / "mujoco_rollout_profile_summary.json"),
        },
    }
    _write_csv(rows)
    _write_summary(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
