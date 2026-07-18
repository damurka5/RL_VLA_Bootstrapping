#!/usr/bin/env python3
"""Two-rank end-to-end CDPR MJWarp sweep with host/GPU telemetry."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASELINE = {
    "backend": "mujoco_cpu_serial_egl",
    "selected_actions_per_second_range": [29.0, 33.0],
    "sampled_actions_per_second_range": [225.0, 240.0],
    "work_amplification_range": [7.2, 7.8],
    "smolvla_inference_fraction_range": [0.54, 0.62],
    "full_env_step_fraction_range": [0.30, 0.37],
    "render_fraction_range": [0.04, 0.05],
    "distributed_sync_fraction": 0.01,
    "note": (
        "Measured historical end-to-end rollout baseline supplied with the "
        "migration request; not a synthetic physics-only measurement."
    ),
}
MOVE_TO_OBJECTS = (
    "robocasa_apple",
    "robocasa_banana",
    "robocasa_tomato",
    "robocasa_orange",
    "robocasa_potato",
    "robocasa_mug",
    "robocasa_plate",
    "robocasa_bowl",
)


def _configure_public_model_environment(
    environment: dict[str, str],
) -> None:
    """Use anonymous access for the public SmolVLA checkpoints by default."""

    public_only = environment.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1").strip()
    if public_only == "1":
        environment.pop("HF_TOKEN", None)
        environment.pop("HUGGING_FACE_HUB_TOKEN", None)
        environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        return
    if public_only != "0":
        raise ValueError("RLVLA_HF_PUBLIC_MODELS_ONLY must be 0 or 1.")


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _maximum(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _gpu_sample() -> list[dict[str, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=5)
    except Exception:
        return []
    rows: list[dict[str, float]] = []
    for line in output.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            continue
        rows.append(
            {
                "index": float(values[0]),
                "gpu_utilization_percent": float(values[1]),
                "power_w": float(values[2]),
                "vram_used_mib": float(values[3]),
                "vram_total_mib": float(values[4]),
            }
        )
    return rows


def _read_last_metrics(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return [], {}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows, (rows[-1] if rows else {})


def _command(
    *,
    args: argparse.Namespace,
    worlds: int,
    microbatch: int,
    run_root: Path,
    run_id: str,
) -> list[str]:
    result = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "-m",
        "rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr",
        "--config",
        str(args.config),
        "--simulator-backend",
        "mjlab_mjwarp",
        "--mjwarp-xml-path",
        str(args.xml.resolve()),
        "--worlds-per-rank",
        str(worlds),
        "--groups-per-rank",
        str(worlds // 8),
        "--grpo-group-size",
        "8",
        "--grpo-trajectory-groups",
        "--grpo-dynamic-sampling",
        "--instruction-types",
        "move_to_object",
        "--allowed-objects",
        *MOVE_TO_OBJECTS,
        "--grpo-target-records-per-update",
        "0",
        "--grpo-max-groups-per-update",
        str(worlds // 8),
        "--base-checkpoint",
        args.base_checkpoint,
        "--run-root-dir",
        str(run_root),
        "--run-id",
        run_id,
        "--device",
        "cuda",
        "--distributed",
        "--mixed-precision",
        "bf16",
        "--chunk-size",
        "8",
        "--replan-every",
        "4",
        "--action-step-xyz",
        "0.015",
        "--action-step-yaw",
        "0.08",
        "--action-step-gripper",
        "0.05",
        "--hold-steps",
        "6",
        "--no-lock-non-commanded-axes",
        "--render-width",
        "320",
        "--render-height",
        "240",
        "--object-slots",
        "4",
        "--smolvla-model-image-size",
        "256",
        "--smolvla-inference-microbatch-size",
        str(min(worlds, max(1, int(microbatch)))),
        "--hidden-dim",
        "1024",
        "--ppo-epochs",
        "1",
        "--minibatch-size",
        "256",
        "--microbatch-size",
        "128",
        "--save-every-steps",
        "0",
        "--max-train-steps",
        "2000000000",
        "--mjwarp-max-updates",
        str(max(2, int(args.updates))),
        "--mjwarp-profile-timers",
        "--no-progress",
        "--progress-only",
    ]
    result.append(
        "--smolvla-compile-model"
        if args.compile_model
        else "--no-smolvla-compile-model"
    )
    result.extend(["--smolvla-compile-mode", "max-autotune"])
    return result


def _run_one(
    args: argparse.Namespace,
    worlds: int,
    microbatch: int,
    output: Path,
) -> dict[str, Any]:
    run_root = output / "runs"
    effective_microbatch = min(worlds, max(1, int(microbatch)))
    run_id = f"worlds_{worlds}_microbatch_{effective_microbatch}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    command = _command(
        args=args,
        worlds=worlds,
        microbatch=effective_microbatch,
        run_root=run_root,
        run_id=run_id,
    )
    log_path = run_dir / "benchmark.log"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    _configure_public_model_environment(environment)
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    telemetry: list[dict[str, Any]] = []
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            psutil = None
        while process.poll() is None:
            sample: dict[str, Any] = {
                "elapsed_s": time.perf_counter() - started,
                "gpus": _gpu_sample(),
            }
            if psutil is not None:
                memory = psutil.virtual_memory()
                sample["cpu_utilization_percent"] = psutil.cpu_percent(interval=None)
                sample["ram_used_gib"] = float(
                    (memory.total - memory.available) / 1024**3
                )
            telemetry.append(sample)
            time.sleep(max(0.2, float(args.sample_interval)))
        return_code = int(process.wait())
    elapsed = time.perf_counter() - started
    rows, measured = _read_last_metrics(run_dir / "metrics.jsonl")
    # The first update is compile/cache warmup; the final row is the reported
    # steady-state sample unless the run failed before producing it.
    gpu_rows = [
        gpu for sample in telemetry for gpu in sample.get("gpus", [])
    ]
    result: dict[str, Any] = {
        "worlds_per_rank": worlds,
        "configured_smolvla_microbatch_size": effective_microbatch,
        "groups_per_rank": worlds // 8,
        "server_candidate_worlds": worlds * 2,
        "return_code": return_code,
        "ok": return_code == 0 and bool(measured),
        "command": command,
        "log": str(log_path),
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "updates_recorded": len(rows),
        "end_to_end_process_time_s": elapsed,
        "measured_update": measured,
        "telemetry": {
            "samples": len(telemetry),
            "gpu_utilization_percent_mean": _mean(
                [row["gpu_utilization_percent"] for row in gpu_rows]
            ),
            "gpu_utilization_percent_max": _maximum(
                [row["gpu_utilization_percent"] for row in gpu_rows]
            ),
            "power_w_mean_per_gpu_sample": _mean(
                [row["power_w"] for row in gpu_rows]
            ),
            "power_w_max": _maximum([row["power_w"] for row in gpu_rows]),
            "vram_used_mib_max": _maximum(
                [row["vram_used_mib"] for row in gpu_rows]
            ),
            "cpu_utilization_percent_mean": _mean(
                [
                    float(sample["cpu_utilization_percent"])
                    for sample in telemetry
                    if "cpu_utilization_percent" in sample
                ]
            ),
            "ram_used_gib_max": _maximum(
                [
                    float(sample["ram_used_gib"])
                    for sample in telemetry
                    if "ram_used_gib" in sample
                ]
            ),
        },
    }
    if measured:
        selected = float(measured.get("selected_actions_per_second_global", 0.0))
        sampled = float(measured.get("sampled_actions_per_second_global", 0.0))
        result["comparison_to_cpu_baseline"] = {
            "selected_vs_baseline_midpoint": selected / 31.0 if selected else 0.0,
            "sampled_vs_baseline_midpoint": sampled / 232.5 if sampled else 0.0,
        }
    return result


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CDPR MJLab/MuJoCo Warp end-to-end benchmark",
        "",
        "The CPU baseline is the measured rollout profile supplied for this "
        "migration. Values below are complete rollout/update measurements; no "
        "physics-only FPS value is labeled as a training speedup.",
        "",
        "| worlds/rank | groups/rank | status | sampled actions/s | selected actions/s | amplification | active SmolVLA batch | inference microbatch | reset s | physics s | render s | SmolVLA s | backprop s | update s | dominant stage | GPU util mean | power mean W | VRAM max MiB | CPU util mean | RAM max GiB |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        metric = row.get("measured_update") or {}
        telemetry = row.get("telemetry") or {}
        lines.append(
            "| {worlds_per_rank} | {groups_per_rank} | {status} | "
            "{sampled:.2f} | {selected:.2f} | {amplification:.2f}× | "
            "{batch:.0f} | {microbatch:.0f} | {reset:.3f} | {physics:.3f} | "
            "{render:.3f} | {smol:.3f} | {backprop:.3f} | {update:.3f} | "
            "{dominant} | {util:.1f}% | {power:.1f} | {vram:.0f} | "
            "{cpu:.1f}% | {ram:.1f} |".format(
                worlds_per_rank=row["worlds_per_rank"],
                groups_per_rank=row["groups_per_rank"],
                status="pass" if row["ok"] else f"fail({row['return_code']})",
                sampled=float(metric.get("sampled_actions_per_second_global", 0.0)),
                selected=float(metric.get("selected_actions_per_second_global", 0.0)),
                amplification=float(
                    metric.get("trajectory_work_amplification", 0.0)
                ),
                batch=float(metric.get("smolvla_batch_size", 0.0)),
                microbatch=float(
                    metric.get("smolvla_inference_microbatch_size", 0.0)
                ),
                reset=float(metric.get("reset_time_s", 0.0)),
                physics=float(metric.get("physics_time_s", 0.0)),
                render=float(metric.get("render_time_s", 0.0)),
                smol=float(metric.get("smolvla_time_s", 0.0)),
                backprop=float(
                    metric.get("backpropagation_time_s", 0.0)
                ),
                update=float(metric.get("update_time_s", 0.0)),
                dominant=str(metric.get("profile/dominant_stage", "n/a")),
                util=float(telemetry.get("gpu_utilization_percent_mean", 0.0)),
                power=float(
                    telemetry.get("power_w_mean_per_gpu_sample", 0.0)
                ),
                vram=float(telemetry.get("vram_used_mib_max", 0.0)),
                cpu=float(
                    telemetry.get("cpu_utilization_percent_mean", 0.0)
                ),
                ram=float(telemetry.get("ram_used_gib_max", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "CPU baseline: selected 29–33 actions/s; sampled 225–240 actions/s; "
            "work amplification 7.2–7.8×.",
            "",
            "Recommended production setting: "
            f"**{payload.get('recommended_worlds_per_rank', 'n/a')} "
            "worlds/rank with SmolVLA microbatch "
            f"{payload.get('recommended_smolvla_microbatch_size', 'n/a')}**. "
            "This is the fastest successful measured end-to-end setting, not "
            "the setting that merely allocates the most VRAM.",
            "",
            f"Machine-readable artifact: `{payload['artifact']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/examples/"
            "cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml"
        ),
    )
    parser.add_argument("--worlds", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--microbatch", nargs="+", type=int, default=[16])
    parser.add_argument("--base-checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/cdpr_mjlab_benchmark")
    )
    args = parser.parse_args()
    if os.environ.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1").strip() not in {
        "0",
        "1",
    }:
        parser.error("RLVLA_HF_PUBLIC_MODELS_ONLY must be 0 or 1")
    args.repo_root = args.repo_root.expanduser().resolve()
    if not args.xml.is_absolute():
        args.xml = args.repo_root / args.xml
    if not args.config.is_absolute():
        args.config = (args.repo_root / args.config).resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = args.repo_root / output
    output.mkdir(parents=True, exist_ok=True)
    for worlds in args.worlds:
        if worlds < 8 or worlds % 8:
            parser.error(f"world count {worlds} is not a positive multiple of eight")

    runs = []
    seen_settings: set[tuple[int, int]] = set()
    for worlds in args.worlds:
        for requested_microbatch in args.microbatch:
            microbatch = min(worlds, max(1, int(requested_microbatch)))
            setting = (worlds, microbatch)
            if setting in seen_settings:
                continue
            seen_settings.add(setting)
            print(
                f"[benchmark] worlds_per_rank={worlds} "
                f"smolvla_microbatch={microbatch}",
                flush=True,
            )
            runs.append(_run_one(args, worlds, microbatch, output))
    successful = [row for row in runs if row["ok"]]
    recommended_run = (
        max(
            successful,
            key=lambda row: float(
                (row.get("measured_update") or {}).get(
                    "selected_actions_per_second_global", 0.0
                )
            ),
        )
        if successful
        else None
    )
    artifact = output / "benchmark.json"
    payload = {
        "schema_version": 1,
        "artifact": str(artifact),
        "baseline": BASELINE,
        "world_size": 2,
        "group_size": 8,
        "camera_resolution": [320, 240],
        "model_input_resolution": [256, 256],
        "task_config": str(args.config),
        "recommended_worlds_per_rank": (
            recommended_run["worlds_per_rank"] if recommended_run else None
        ),
        "recommended_smolvla_microbatch_size": (
            recommended_run["configured_smolvla_microbatch_size"]
            if recommended_run
            else None
        ),
        "largest_successful_worlds_per_rank": (
            max(row["worlds_per_rank"] for row in successful)
            if successful
            else None
        ),
        "runs": runs,
    }
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report = output / "benchmark.md"
    report.write_text(_markdown(payload), encoding="utf-8")
    print(f"benchmark_artifact={artifact}")
    print(f"benchmark_report={report}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
