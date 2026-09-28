"""Shared utilities for the lightweight simulator comparator.

The comparator deliberately stays independent of OpenVLA and policy training.
Backends expose scripted/waypoint task rollouts plus simple geometric predicates.
"""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tools" / "sim_compare" / "out"
TMP_DIR = OUT_DIR / "tmp"

DEFAULT_SEED = 20260606
DEFAULT_RESETS = 20
DEFAULT_STEPS = 300
DEFAULT_RENDER_STEPS = 60
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_CAMERA_COUNT = 2
DEFAULT_RENDER_BACKEND = "auto"

TABLE_TOP_Z = 0.015
TABLE_PENETRATION_TOLERANCE_M = 0.004
MOVE_XY_THRESHOLD_M = 0.04
MOVE_XYZ_THRESHOLD_M = 0.06
PUSH_DISPLACEMENT_THRESHOLD_M = 0.05
RELATION_OFFSET_M = 0.08
RELATION_TOLERANCE_M = 0.035

MAX_CONTACT_FORCE_N = 120.0
MAX_TRANSIENT_LINEAR_VELOCITY_MPS = 5.0
MAX_TRANSIENT_ANGULAR_VELOCITY_RADPS = 50.0
MAX_SETTLED_LINEAR_VELOCITY_MPS = 0.12
MAX_SETTLED_ANGULAR_VELOCITY_RADPS = 2.5

SUPPORTED_RENDER_BACKENDS = ("auto", "egl", "osmesa", "glfw")


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    category: str
    geom_type: str
    size: Tuple[float, ...]
    half_height: float
    mass: float
    rgba: Tuple[float, float, float, float]
    friction: Tuple[float, float, float] = (0.9, 0.01, 0.001)
    solref: Tuple[float, float] = (0.006, 1.0)
    solimp: Tuple[float, float, float] = (0.92, 0.98, 0.001)
    condim: int = 6
    margin: float = 0.0005
    gap: float = 0.0
    bottom_offset: Optional[float] = None

    @property
    def center_z(self) -> float:
        return TABLE_TOP_Z - self.bottom_offset_m + 0.002

    @property
    def bottom_offset_m(self) -> float:
        return -float(self.half_height) if self.bottom_offset is None else float(self.bottom_offset)

    def bottom_z(self, body_z: float) -> float:
        return float(body_z) + self.bottom_offset_m

    @property
    def grasp_width(self) -> float:
        if self.geom_type == "box":
            return 2.0 * float(self.size[0])
        if self.geom_type in {"cylinder", "sphere", "compound_bowl"}:
            return 2.0 * float(self.size[0])
        return 0.0


OBJECT_SPECS: Dict[str, ObjectSpec] = {
    "block": ObjectSpec(
        key="block",
        category="cube/block",
        geom_type="box",
        size=(0.035, 0.035, 0.035),
        half_height=0.035,
        mass=0.12,
        rgba=(0.85, 0.18, 0.12, 1.0),
        friction=(1.0, 0.01, 0.001),
    ),
    "can": ObjectSpec(
        key="can",
        category="cylinder/can",
        geom_type="cylinder",
        size=(0.032, 0.045),
        half_height=0.045,
        mass=0.10,
        rgba=(0.10, 0.38, 0.80, 1.0),
        friction=(1.0, 0.03, 0.006),
    ),
    "sphere": ObjectSpec(
        key="sphere",
        category="sphere/ball",
        geom_type="sphere",
        size=(0.035,),
        half_height=0.035,
        mass=0.09,
        rgba=(0.95, 0.83, 0.22, 1.0),
        friction=(0.85, 0.06, 0.035),
    ),
    "plate": ObjectSpec(
        key="plate",
        category="plate/receptacle",
        geom_type="cylinder",
        size=(0.085, 0.012),
        half_height=0.012,
        mass=0.16,
        rgba=(0.12, 0.35, 0.85, 0.92),
        friction=(1.2, 0.05, 0.012),
    ),
    "bowl": ObjectSpec(
        key="bowl",
        category="bowl/cup proxy",
        geom_type="compound_bowl",
        size=(0.070, 0.014),
        half_height=0.052,
        mass=0.18,
        rgba=(0.88, 0.88, 0.82, 0.88),
        friction=(1.0, 0.04, 0.010),
        bottom_offset=-0.014,
    ),
}

DEFAULT_TASK_OBJECTS = ("block", "can", "sphere")
DEFAULT_CONTACT_OBJECTS = ("block", "can", "sphere", "plate", "bowl")


BACKEND_SUMMARY_FIELDS = [
    "backend_name",
    "status",
    "simulator_version",
    "robot_embodiment",
    "num_environments",
    "reset_time_mean_s",
    "step_fps_no_render",
    "step_fps_with_rgb",
    "render_resolution",
    "render_backend",
    "platform",
    "cpu_ram_mb",
    "gpu_vram_mb",
    "gpu_utilization_percent",
    "success_predicate_correctness",
    "object_stability_pass_rate",
    "contact_anomalies",
    "engineering_notes",
    "missing_features",
    "migration_difficulty",
    "skipped_reason",
]

TASK_RESULT_FIELDS = [
    "backend_name",
    "simulator_version",
    "task_name",
    "object_category",
    "object_name",
    "robot_embodiment",
    "num_environments",
    "validation_scope",
    "episode",
    "reset_time_s",
    "steps",
    "step_time_s",
    "step_fps_no_render",
    "success",
    "success_predicate_correctness",
    "initial_ee_xyz",
    "final_ee_xyz",
    "initial_object_xyz",
    "final_object_xyz",
    "reference_object_xyz",
    "threshold_m",
    "engineering_notes",
]

CONTACT_RESULT_FIELDS = [
    "backend_name",
    "simulator_version",
    "object_category",
    "object_name",
    "test_name",
    "pass_fail",
    "failure_reason",
    "steps",
    "duration_s",
    "max_linear_velocity",
    "max_angular_velocity",
    "settled_linear_velocity",
    "settled_angular_velocity",
    "max_normal_contact_force",
    "min_body_z",
    "min_bottom_z",
    "finger_contact_count",
    "contact_anomalies",
    "engineering_notes",
]

RENDER_PROFILE_FIELDS = [
    "backend_name",
    "simulator_version",
    "robot_embodiment",
    "num_environments",
    "camera_count",
    "render_resolution",
    "render_backend",
    "platform",
    "rendered_rgb_frames",
    "sim_steps",
    "step_time_s",
    "render_time_s",
    "total_time_s",
    "step_fps_during_rgb",
    "rgb_frame_fps",
    "cpu_ram_mb",
    "gpu_vram_mb",
    "gpu_utilization_percent",
    "failure_reason",
    "engineering_notes",
]


def timer() -> float:
    return time.perf_counter()


def format_vec(value: Sequence[float]) -> str:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(x):.4f}" for x in arr[:3]) + "]"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 1024 * 1024 * 8:
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def gpu_vram_mb() -> Optional[float]:
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
    values: List[float] = []
    for line in proc.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def gpu_utilization_percent() -> Optional[float]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
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
    values: List[float] = []
    for line in proc.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def platform_label() -> str:
    return f"{platform.system()} {platform.machine()} {platform.release()}".strip()


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def module_version(module_name: str, dist_name: Optional[str] = None) -> str:
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, "__version__", "")
        if value:
            return str(value)
    except Exception:
        pass
    try:
        return importlib.metadata.version(dist_name or module_name)
    except Exception:
        return ""


def xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa[:2] - bb[:2]))


def xyz_distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa[:3] - bb[:3]))


def move_to_object_success(
    ee_xyz: Sequence[float],
    target_xyz: Sequence[float],
    threshold_m: float = MOVE_XY_THRESHOLD_M,
    metric: str = "xy",
) -> bool:
    if metric == "xyz":
        return xyz_distance(ee_xyz, target_xyz) <= float(threshold_m)
    return xy_distance(ee_xyz, target_xyz) <= float(threshold_m)


def push_success(
    initial_xyz: Sequence[float],
    final_xyz: Sequence[float],
    direction_xyz: Sequence[float],
    threshold_m: float = PUSH_DISPLACEMENT_THRESHOLD_M,
) -> bool:
    direction = np.asarray(direction_xyz, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return False
    unit = direction / norm
    displacement = np.asarray(final_xyz, dtype=np.float64)[:3] - np.asarray(initial_xyz, dtype=np.float64)[:3]
    return float(np.dot(displacement, unit)) >= float(threshold_m)


def relation_success(
    target_xyz: Sequence[float],
    reference_xyz: Sequence[float],
    relation: str,
    offset_m: float = RELATION_OFFSET_M,
    tolerance_m: float = RELATION_TOLERANCE_M,
) -> bool:
    target = np.asarray(target_xyz, dtype=np.float64)
    ref = np.asarray(reference_xyz, dtype=np.float64)
    rel = relation.lower()
    if rel == "left":
        return bool(target[0] <= ref[0] - offset_m and abs(target[1] - ref[1]) <= tolerance_m)
    if rel == "right":
        return bool(target[0] >= ref[0] + offset_m and abs(target[1] - ref[1]) <= tolerance_m)
    if rel == "front":
        return bool(target[1] >= ref[1] + offset_m and abs(target[0] - ref[0]) <= tolerance_m)
    if rel == "behind":
        return bool(target[1] <= ref[1] - offset_m and abs(target[0] - ref[0]) <= tolerance_m)
    raise ValueError(f"Unknown relation: {relation}")


def predicate_self_check() -> Tuple[str, float]:
    cases = [
        move_to_object_success([0.01, 0.01, 0.3], [0.02, 0.02, 0.04]),
        not move_to_object_success([0.20, 0.20, 0.3], [0.02, 0.02, 0.04]),
        push_success([0, 0, 0.05], [0.07, 0.0, 0.05], [1, 0, 0]),
        not push_success([0, 0, 0.05], [0.03, 0.0, 0.05], [1, 0, 0]),
        relation_success([-0.10, 0.0, 0.05], [0.0, 0.0, 0.02], "left"),
        not relation_success([-0.02, 0.0, 0.05], [0.0, 0.0, 0.02], "left"),
        relation_success([0.0, 0.10, 0.05], [0.0, 0.0, 0.02], "front"),
        not relation_success([0.0, -0.02, 0.05], [0.0, 0.0, 0.02], "front"),
    ]
    correct = sum(1 for value in cases if value)
    total = len(cases)
    return f"{correct}/{total}", float(correct / total)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def skip_backend_summary(
    backend_name: str,
    robot_embodiment: str,
    missing_dependency: str,
    engineering_notes: str,
    missing_features: str,
    migration_difficulty: str,
    version: str = "",
) -> Dict[str, Any]:
    return {
        "backend_name": backend_name,
        "status": "skipped",
        "simulator_version": version,
        "robot_embodiment": robot_embodiment,
        "num_environments": "",
        "reset_time_mean_s": "",
        "step_fps_no_render": "",
        "step_fps_with_rgb": "",
        "render_resolution": "",
        "render_backend": "",
        "platform": platform_label(),
        "cpu_ram_mb": f"{rss_mb():.2f}",
        "gpu_vram_mb": "" if gpu_vram_mb() is None else f"{gpu_vram_mb():.2f}",
        "gpu_utilization_percent": "" if gpu_utilization_percent() is None else f"{gpu_utilization_percent():.1f}",
        "success_predicate_correctness": "",
        "object_stability_pass_rate": "",
        "contact_anomalies": "",
        "engineering_notes": engineering_notes,
        "missing_features": missing_features,
        "migration_difficulty": migration_difficulty,
        "skipped_reason": f"missing dependency: {missing_dependency}",
    }


def discover_optional_backend_summaries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not module_available("mani_skill"):
        rows.append(
            skip_backend_summary(
                "maniskill_sapien",
                "Panda/Franka candidate",
                "mani_skill and sapien",
                "Candidate is appropriate for scripted Panda manipulation and vectorized scenes, but is not installed in this Python environment.",
                "No adapter run; CDPR embodiment not ported; task geometry would use Panda end-effector sites.",
                "medium for Panda scaffold, high for CDPR port",
            )
        )
    else:
        rows.append(
            skip_backend_summary(
                "maniskill_sapien",
                "Panda/Franka candidate",
                "adapter_not_enabled",
                "mani_skill is installed, but this lightweight scaffold currently records it as a candidate until a concrete task adapter is selected.",
                "Backend-specific scene/task adapter not implemented in this scaffold.",
                "medium for Panda scaffold, high for CDPR port",
                module_version("mani_skill"),
            )
        )

    isaac_available = module_available("isaaclab") or module_available("omni.isaac.lab") or module_available("omni.isaac.core")
    if not isaac_available:
        rows.append(
            skip_backend_summary(
                "isaac_lab_or_sim",
                "Franka/Panda candidate",
                "Isaac Lab / Isaac Sim Python modules",
                "Skipped by design unless Isaac Lab/Isaac Sim is already installed and importable.",
                "No local Isaac app/runtime detected; CDPR, assets, and rendering setup would require a dedicated port.",
                "high",
            )
        )
    else:
        rows.append(
            skip_backend_summary(
                "isaac_lab_or_sim",
                "Franka/Panda candidate",
                "adapter_not_enabled",
                "Isaac modules are importable, but this scaffold does not launch the heavy Isaac application runtime automatically.",
                "Backend-specific scene/task adapter not implemented in this lightweight run.",
                "high",
                module_version("isaaclab") or module_version("omni.isaac.core"),
            )
        )

    if not module_available("robosuite"):
        rows.append(
            skip_backend_summary(
                "robosuite",
                "Panda/Franka candidate",
                "robosuite",
                "Candidate is useful for scripted Panda tasks, but it is not installed in this Python environment.",
                "No adapter run; CDPR embodiment not ported; object set would be robosuite XML proxies.",
                "medium",
            )
        )
    else:
        rows.append(
            skip_backend_summary(
                "robosuite",
                "Panda/Franka candidate",
                "adapter_not_enabled",
                "robosuite is installed, but this scaffold records it as a candidate until a concrete same-objects scene adapter is selected.",
                "Backend-specific same-object scene adapter not implemented in this scaffold.",
                "medium",
                module_version("robosuite"),
            )
        )

    if not module_available("pybullet"):
        rows.append(
            skip_backend_summary(
                "pybullet_optional",
                "Kinematic point/end-effector candidate",
                "pybullet",
                "Optional low-effort backend not installed. It is kept as a possible quick geometry/contact sanity baseline.",
                "No adapter run; not a primary migration candidate for the current OpenVLA-OFT pipeline.",
                "medium-high",
            )
        )
    else:
        rows.append(
            skip_backend_summary(
                "pybullet_optional",
                "Kinematic point/end-effector candidate",
                "adapter_not_enabled",
                "pybullet is importable, but this scaffold focuses on requested primary candidates and does not run a Bullet adapter yet.",
                "Backend-specific object/renderer adapter not implemented.",
                "medium-high",
                module_version("pybullet"),
            )
        )

    return rows


def _best_backend(
    rows: Sequence[Dict[str, Any]],
    field: str,
    only_status: str = "ran",
) -> Tuple[str, float]:
    best_name = ""
    best_value = 0.0
    for row in rows:
        if row.get("status") != only_status:
            continue
        value = safe_float(row.get(field, ""))
        if value > best_value:
            best_value = value
            best_name = str(row.get("backend_name", ""))
    return best_name, best_value


def _contact_pass_rate(contact_rows: Sequence[Dict[str, Any]], backend_name: str, stability_only: bool = True) -> float:
    tests = {"drop", "rest_on_table", "push"} if stability_only else None
    selected = [
        row
        for row in contact_rows
        if row.get("backend_name") == backend_name and (tests is None or row.get("test_name") in tests)
    ]
    if not selected:
        return 0.0
    good = sum(1 for row in selected if row.get("pass_fail") == "pass")
    return float(good / len(selected))


def _final_recommendation(
    backend_rows: Sequence[Dict[str, Any]],
    contact_rows: Sequence[Dict[str, Any]],
    render_rows: Sequence[Dict[str, Any]],
    settings: Dict[str, Any],
) -> str:
    measured = [row for row in backend_rows if row.get("status") == "ran"]
    if not measured:
        return "INSUFFICIENT_EVIDENCE"
    render_requested = bool(settings.get("render", True))
    if render_requested and not any(safe_float(row.get("rendered_rgb_frames")) > 0 for row in render_rows):
        return "FIX_HEADLESS_RENDERING_FIRST"
    stability_rows = [row for row in contact_rows if row.get("test_name") in {"drop", "rest_on_table", "push"}]
    if stability_rows and any(row.get("pass_fail") != "pass" for row in stability_rows):
        return "FIX_OBJECT_STABILITY_FIRST"
    graspable_gripper_rows = [
        row
        for row in contact_rows
        if row.get("test_name") in {"gripper_squeeze", "lift"}
        and "object wider than gripper" not in str(row.get("failure_reason", ""))
    ]
    if graspable_gripper_rows and any(row.get("pass_fail") != "pass" for row in graspable_gripper_rows):
        return "FIX_GRIPPER_CONTACT_TEST_FIRST"
    if not render_rows and not contact_rows:
        return "INSUFFICIENT_EVIDENCE"
    return "MUJOCO_BASELINE_READY_FOR_EXTERNAL_COMPARISON"


def _brief_text(value: Any, limit: int = 700) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if " Traceback " in text:
        text = text.split(" Traceback ", 1)[0].strip()
    if len(text) > int(limit):
        return text[: int(limit) - 3].rstrip() + "..."
    return text


def _failure_summary(rows: Sequence[Dict[str, Any]], limit: int = 8) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("failure_reason") or row.get("contact_anomalies") or "").strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "- No failures recorded."
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: int(limit)]
    return "\n".join(f"- `{count}` row(s): {reason}" for reason, count in items)


def write_report(
    path: Path,
    backend_rows: Sequence[Dict[str, Any]],
    task_rows: Sequence[Dict[str, Any]],
    contact_rows: Sequence[Dict[str, Any]],
    render_rows: Sequence[Dict[str, Any]],
    settings: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    physics_backend, physics_fps = _best_backend(backend_rows, "step_fps_no_render")
    rgb_backend, rgb_fps = _best_backend(backend_rows, "step_fps_with_rgb")

    measured = [row for row in backend_rows if row.get("status") == "ran"]
    skipped = [row for row in backend_rows if row.get("status") != "ran"]
    stability_scores = [
        (row.get("backend_name", ""), _contact_pass_rate(contact_rows, str(row.get("backend_name", ""))))
        for row in measured
    ]
    stability_scores.sort(key=lambda item: item[1], reverse=True)
    stable_backend = stability_scores[0][0] if stability_scores else ""
    stable_score = stability_scores[0][1] if stability_scores else 0.0

    predicate_backend = ""
    predicate_score = 0.0
    for row in measured:
        correctness = str(row.get("success_predicate_correctness", "0/0"))
        try:
            numerator, denominator = correctness.split("/", 1)
            score = float(numerator) / max(float(denominator), 1.0)
        except Exception:
            score = 0.0
        if score > predicate_score:
            predicate_score = score
            predicate_backend = str(row.get("backend_name", ""))

    successful_tasks = sum(1 for row in task_rows if str(row.get("success")) == "1")
    task_count = len(task_rows)
    contact_passes = sum(1 for row in contact_rows if row.get("pass_fail") == "pass")
    stability_rows = [row for row in contact_rows if row.get("test_name") in {"drop", "rest_on_table", "push"}]
    gripper_rows = [row for row in contact_rows if row.get("test_name") in {"gripper_squeeze", "lift"}]
    gripper_passes = sum(1 for row in gripper_rows if row.get("pass_fail") == "pass")
    relation_rows = [row for row in task_rows if str(row.get("task_name", "")).startswith("place_relation")]
    predicate_only_relations = sum(1 for row in relation_rows if row.get("validation_scope") == "predicate_validation_only")
    recommendation = _final_recommendation(backend_rows, contact_rows, render_rows, settings)
    fair_baseline_text = (
        "yes, for external simulator comparison"
        if recommendation == "MUJOCO_BASELINE_READY_FOR_EXTERNAL_COMPARISON"
        else f"not yet: {recommendation}"
    )

    skip_lines = []
    for row in skipped:
        skip_lines.append(
            f"- {row.get('backend_name')}: {row.get('skipped_reason')} "
            f"({row.get('migration_difficulty')} migration difficulty)."
        )
    skip_text = "\n".join(skip_lines) if skip_lines else "- No optional backends were skipped."

    measured_text = "\n".join(
        [
            (
                f"- {row.get('backend_name')}: physics {safe_float(row.get('step_fps_no_render')):.1f} steps/s, "
                f"RGB {safe_float(row.get('step_fps_with_rgb')):.1f} steps/s, "
                f"reset {safe_float(row.get('reset_time_mean_s')):.4f}s, "
                f"stability {row.get('object_stability_pass_rate')}, "
                f"render backend `{row.get('render_backend') or 'not attempted'}`."
            )
            for row in measured
        ]
    )
    if not measured_text:
        measured_text = "- No backend produced executable measurements."

    render_text = "\n".join(
        [
            (
                f"- {row.get('backend_name')}: {row.get('rendered_rgb_frames')} RGB frames at "
                f"{row.get('render_resolution')}, backend `{row.get('render_backend')}`, "
                f"{safe_float(row.get('rgb_frame_fps')):.1f} frame/s. "
                f"{_brief_text(row.get('failure_reason') or row.get('engineering_notes'))}"
            )
            for row in render_rows
        ]
    )
    if not render_text:
        render_text = "- No RGB render profile completed."

    render_diagnosis_text = "\n".join(
        [
            (
                f"- `{row.get('render_backend') or 'unknown'}` on `{row.get('platform') or 'unknown platform'}`: "
                f"{_brief_text(row.get('engineering_notes') or row.get('failure_reason'), 900)}"
            )
            for row in render_rows
        ]
    )
    if not render_diagnosis_text:
        render_diagnosis_text = "- No render diagnosis was produced."

    gripper_contact_rows = [row for row in gripper_rows if safe_float(row.get("finger_contact_count")) > 0]
    gripper_geometry_rows = [
        row for row in gripper_rows if "object wider than gripper" in str(row.get("failure_reason", ""))
    ]
    gripper_eject_rows = [
        row for row in gripper_rows if "squeeze contact ejects object" in str(row.get("failure_reason", ""))
    ]

    report = f"""# Simulator Comparator Report

Generated by `tools/sim_compare/run_benchmark.py` with deterministic seed `{settings.get('seed')}`.

## Inputs

- Episodes/resets per manipulation task/object: `{settings.get('resets')}`
- Steps per task episode: `{settings.get('steps')}`
- Render steps: `{settings.get('render_steps')}`
- Render resolution: `{settings.get('width')}x{settings.get('height')}`
- Render requested: `{settings.get('render')}`
- Render backend setting: `{settings.get('render_backend')}`
- Camera count: `{settings.get('camera_count')}`
- Manipulation objects: `{', '.join(settings.get('task_objects', []))}`
- Contact objects: `{', '.join(settings.get('contact_objects', []))}`
- OpenVLA checkpoints/training: not used

## Measured Backends

{measured_text}

## Skipped Backends

{skip_text}

## Task Results

- Scripted manipulation rows: `{task_count}`
- Successful scripted task rows: `{successful_tasks}/{task_count}`
- Relation task rows using direct object placement: `{predicate_only_relations}/{len(relation_rows)}` marked as `predicate_validation_only`
- Contact rows: `{len(contact_rows)}`
- Contact pass rows under strengthened criteria: `{contact_passes}/{len(contact_rows)}`
- Drop/rest/push object stability rows: `{sum(1 for row in stability_rows if row.get('pass_fail') == 'pass')}/{len(stability_rows)}`
- Gripper squeeze/lift rows: `{gripper_passes}/{len(gripper_rows)}`

## Render Profile

{render_text}

## Render Diagnosis

{render_diagnosis_text}

## Contact Failure Summary

{_failure_summary([row for row in contact_rows if row.get('pass_fail') != 'pass'])}

## Gripper Test Validity

- Graspable gripper rows with detected finger/object contact: `{len(gripper_contact_rows)}/{len([row for row in gripper_rows if 'object wider than gripper' not in str(row.get('failure_reason', ''))])}`
- Rows reporting gripper width limits: `{len(gripper_geometry_rows)}`
- Rows where squeeze contact ejects the object before a stable lift can be attempted: `{len(gripper_eject_rows)}`
- Interpretation: contact detection is now valid, but the current CDPR gripper contact geometry/actuation is not yet a reliable grasp baseline.

## Answers

1. **Fastest physics-only stepping:** `{physics_backend or 'not measured'}` at approximately `{physics_fps:.1f}` steps/s. With the current local dependencies, only MuJoCo produced executable physics measurements.
2. **Fastest with RGB rendering:** `{rgb_backend or 'not measured'}` at approximately `{rgb_fps:.1f}` simulation steps/s with RGB capture. This is measured separately from physics-only stepping.
3. **Most stable simple objects:** `{stable_backend or 'not measured'}` with a drop/rest/push stability pass rate of `{stable_score:.2%}` among measured backends.
4. **Easiest binary geometric predicates:** `{predicate_backend or 'not measured'}`. The predicates are direct body-position checks in the current MuJoCo scaffold, with self-check score `{predicate_score:.2%}`.
5. **Easiest later OpenVLA-OFT connection:** `mujoco_raw_cdpr`, because it uses the current CDPR MJCF, object scale, camera naming, and geometric state access already adjacent to the existing pipeline.
6. **Migration evidence:** This run does not provide positive evidence for migration because the optional candidate simulators were not installed or not adapter-enabled. Continue improving the MuJoCo setup unless a follow-up run installs ManiSkill/robosuite or uses an already working Isaac Lab runtime and shows a clear render/contact/control advantage.
7. **What scripted task success proves:** scripted success validates deterministic stepping plus geometric predicates. It does not validate OpenVLA learning, action decoding, visual servoing, or robust grasping.
8. **MuJoCo baseline status:** `{recommendation}`.
9. **Fair baseline for external simulator comparison:** `{fair_baseline_text}`.

## Fairness Caveats

- Robot embodiment differs across candidates: this run measures the current CDPR MuJoCo MJCF; skipped candidates would likely start with Panda/Franka scenes.
- Controllers are scripted waypoint or direct-state controllers, not learned policies and not OpenVLA action decoding.
- `place_relation` uses direct object placement in the MuJoCo scaffold and is marked `predicate_validation_only` because robust scripted grasping is outside this benchmark scaffold.
- Object geometry is matched by category and approximate scale/mass/friction, but backend-native collision and solver models are not identical.
- Rendering FPS depends on local OpenGL/EGL/OSMesa availability and camera count; physics-only FPS excludes RGB readback.
- Contact pass/fail thresholds now check table penetration, settled linear/angular velocity, transient explosion/spin, excessive contact force, and missing expected gripper contact.

## Final Recommendation

`{recommendation}`

## Output Files

- `tools/sim_compare/out/backend_summary.csv`
- `tools/sim_compare/out/task_results.csv`
- `tools/sim_compare/out/contact_results.csv`
- `tools/sim_compare/out/render_profile.csv`
"""
    path.write_text(report)


def write_all_outputs(
    backend_rows: Sequence[Dict[str, Any]],
    task_rows: Sequence[Dict[str, Any]],
    contact_rows: Sequence[Dict[str, Any]],
    render_rows: Sequence[Dict[str, Any]],
    settings: Dict[str, Any],
    out_dir: Optional[Path] = None,
) -> None:
    target_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    write_csv(target_dir / "backend_summary.csv", BACKEND_SUMMARY_FIELDS, backend_rows)
    write_csv(target_dir / "task_results.csv", TASK_RESULT_FIELDS, task_rows)
    write_csv(target_dir / "contact_results.csv", CONTACT_RESULT_FIELDS, contact_rows)
    write_csv(target_dir / "render_profile.csv", RENDER_PROFILE_FIELDS, render_rows)
    write_json(
        target_dir / "settings.json",
        {
            "settings": settings,
            "backend_rows": list(backend_rows),
            "task_row_count": len(task_rows),
            "contact_row_count": len(contact_rows),
            "render_row_count": len(render_rows),
        },
    )
    write_report(
        target_dir / "SIMULATOR_COMPARATOR_REPORT.md",
        backend_rows,
        task_rows,
        contact_rows,
        render_rows,
        settings,
    )
