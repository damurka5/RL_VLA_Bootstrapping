#!/usr/bin/env python3
"""Strict A40/MJLab/MuJoCo-Warp preflight with a machine-readable report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_vla_bootstrapping.core.config import load_project_config
from rl_vla_bootstrapping.simulation.cdpr_backend import (
    CDPRBackendConfig,
    create_cdpr_backend,
)
from rl_vla_bootstrapping.simulation.mjwarp_compat import (
    PINNED_CUDA_RUNTIME,
    execute_mjwarp_compatibility_spike,
    inspect_cdpr_mjcf,
    package_versions,
    pinned_version_mismatches,
)


def _nvidia_smi() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, timeout=15)
    devices: list[dict[str, Any]] = []
    for row in output.strip().splitlines():
        values = [value.strip() for value in row.split(",")]
        if len(values) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi row: {row!r}")
        devices.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "driver_version": values[2],
                "memory_total_mib": float(values[3]),
                "compute_capability": values[4],
            }
        )
    return devices


def _driver_major(version: str) -> int:
    try:
        return int(str(version).split(".", 1)[0])
    except ValueError:
        return 0


def _resolve_xml(config_path: Path) -> tuple[Any, Path]:
    config = load_project_config(config_path)
    xml = config.resolve_path(config.simulator.fixed_scene_xml)
    if xml is None:
        raise ValueError("Config simulator.fixed_scene_xml is required.")
    return config, xml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/examples/"
            "cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml"
        ),
    )
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--require-gpus", type=int, default=2)
    parser.add_argument("--nconmax", type=int, default=256)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("runs/mjlab_preflight.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {
        "ok": False,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "required_cuda_runtime": PINNED_CUDA_RUNTIME,
        "checks": {
            "exact_package_pins": False,
            "static_required_mjcf_features": False,
            "gpu_count": False,
            "two_a40s": False,
            "driver_570_or_newer": False,
            "pytorch_cuda": False,
            "pytorch_gpu_count": False,
            "torch_cuda_12_8": False,
            "warp_gpu_count": False,
            "mjwarp_import": False,
            "mjlab_import": False,
            "put_model": False,
            "world_allocation": False,
            "batched_camera_creation": False,
            "required_mjcf_features": False,
            "camera_tensor_contract": False,
        },
        "errors": [],
    }
    backend = None
    try:
        config, xml_path = _resolve_xml(args.config.resolve())
        report["config"] = str(args.config.resolve())
        report["xml_path"] = str(xml_path)
        static_compatibility = inspect_cdpr_mjcf(xml_path)
        report["mjcf_static"] = static_compatibility.as_dict()
        report["checks"]["static_required_mjcf_features"] = bool(
            static_compatibility.parse_ok
            and not static_compatibility.unsupported
            and all(static_compatibility.required_features.values())
        )
        report["versions"] = package_versions()
        mismatches = pinned_version_mismatches(report["versions"])
        report["checks"]["exact_package_pins"] = not mismatches
        if mismatches:
            report["errors"].append(
                "Version lock mismatch: "
                + ", ".join(
                    f"{name} expected {wanted}, found {actual}"
                    for name, (wanted, actual) in mismatches.items()
                )
            )

        try:
            gpus = _nvidia_smi()
        except Exception as exc:
            gpus = []
            report["errors"].append(
                f"NVIDIA inventory failed: {type(exc).__name__}: {exc}"
            )
        report["gpus"] = gpus
        report["checks"]["gpu_count"] = len(gpus) >= int(args.require_gpus)
        report["checks"]["two_a40s"] = (
            len(gpus) >= int(args.require_gpus)
            and all("A40" in gpu["name"] for gpu in gpus[: int(args.require_gpus)])
        )
        report["checks"]["driver_570_or_newer"] = bool(gpus) and all(
            _driver_major(gpu["driver_version"]) >= 570 for gpu in gpus
        )

        import torch
        import warp as wp
        import mujoco_warp  # noqa: F401
        import mjlab  # noqa: F401

        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        report["checks"]["pytorch_cuda"] = bool(torch.cuda.is_available())
        report["checks"]["pytorch_gpu_count"] = (
            torch.cuda.device_count() >= int(args.require_gpus)
        )
        report["checks"]["torch_cuda_12_8"] = str(torch.version.cuda) == "12.8"
        wp.init()
        warp_devices = [str(device) for device in wp.get_cuda_devices()]
        report["warp_devices"] = warp_devices
        report["checks"]["warp_gpu_count"] = (
            len(warp_devices) >= int(args.require_gpus)
        )
        report["checks"]["mjwarp_import"] = True
        report["checks"]["mjlab_import"] = True

        compatibility = execute_mjwarp_compatibility_spike(
            xml_path,
            nworld=int(args.worlds),
            nconmax=int(args.nconmax),
            njmax=int(args.njmax),
            device="cuda:0",
            create_renderer=True,
            render_width=int(config.simulator.render_width),
            render_height=int(config.simulator.render_height),
        )
        report["mjcf_compatibility"] = compatibility.as_dict()
        report["checks"]["put_model"] = compatibility.put_model_ok
        report["checks"]["world_allocation"] = compatibility.world_allocation_ok
        report["checks"]["batched_camera_creation"] = compatibility.render_context_ok
        report["checks"]["required_mjcf_features"] = (
            all(compatibility.required_features.values())
            and not compatibility.unsupported
        )
        if not compatibility.compatible:
            report["errors"].append(
                "MJCF compatibility spike did not pass; inspect "
                "mjcf_compatibility.error/unsupported/required_features."
            )

        groups = int(args.worlds) // 8
        if groups < 1 or groups * 8 != int(args.worlds):
            raise ValueError("--worlds must be a positive multiple of eight.")
        backend = create_cdpr_backend(
            CDPRBackendConfig(
                backend="mjlab_mjwarp",
                worlds_per_rank=int(args.worlds),
                groups_per_rank=groups,
                grpo_group_size=8,
                hold_steps=6,
                render_width=int(config.simulator.render_width),
                render_height=int(config.simulator.render_height),
                object_slots=4,
                nconmax=int(args.nconmax),
                njmax=int(args.njmax),
                device="cuda:0",
                xml_path=xml_path,
            )
        )
        all_worlds = torch.arange(args.worlds, device="cuda:0")
        backend.reset_worlds(all_worlds)
        cameras = backend.render_policy_cameras()
        expected = (
            int(args.worlds),
            3,
            int(config.simulator.render_height),
            int(config.simulator.render_width),
        )
        camera_contract = (
            tuple(cameras.overview.shape) == expected
            and tuple(cameras.wrist.shape) == expected
            and cameras.overview.device.type == "cuda"
            and cameras.wrist.device.type == "cuda"
            and cameras.overview.dtype == torch.float32
            and cameras.wrist.dtype == torch.float32
            and cameras.aux.data_ptr() == cameras.wrist.data_ptr()
            and bool(torch.isfinite(cameras.overview).all())
            and bool(torch.isfinite(cameras.wrist).all())
        )
        report["camera_contract"] = {
            "overview_shape": list(cameras.overview.shape),
            "wrist_shape": list(cameras.wrist.shape),
            "dtype": str(cameras.overview.dtype),
            "device": str(cameras.overview.device),
            "range": [
                float(
                    torch.minimum(cameras.overview.min(), cameras.wrist.min()).item()
                ),
                float(
                    torch.maximum(cameras.overview.max(), cameras.wrist.max()).item()
                ),
            ],
            "aux_duplicates_wrist_storage": (
                cameras.aux.data_ptr() == cameras.wrist.data_ptr()
            ),
        }
        report["checks"]["camera_tensor_contract"] = bool(camera_contract)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        if backend is not None:
            backend.close()

    failed = [name for name, passed in report["checks"].items() if not bool(passed)]
    report["failed_checks"] = failed
    report["ok"] = not failed and not report["errors"]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"preflight_report={output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
