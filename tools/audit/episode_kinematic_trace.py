"""Per-env-step kinematic traces of full-task evaluation episodes.

Videos showed far-side carries where the object stays between the fingers
while the hand climbs out of the overview frame, and is then lost beside the
receptacle. The outcome flags cannot tell those stories apart: ``carry_slip``
is contact lost WITHOUT an exempted release, and the exemption only covers
opening over the goal, so a hand that opens mid-carry and a grip that fails
passively are the same flag.

This records, for every env step of every episode:

* the commanded controller target and the measured EE position (tracking),
* the gripper command, its measured opening, pad contact (physical grasp and
  bilateral contact), and whether a release over the goal was in progress,
* the target object's and the receptacle's positions,
* whether the EE, the object and the receptacle fall inside the overview and
  the wrist camera frustums, computed from each world's live camera pose and
  the model's fovy. This is frustum membership, not occlusion,
* the outcome observer's latched flags, so events can be timed.

One ``trace_round<k>.npz`` per round, arrays shaped (steps, worlds, ...),
float32 or bool. ``summarize_kinematic_traces.py`` turns them into tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class KinematicTrace:
    def __init__(self, *, backend: Any, output_dir: Path, torch: Any) -> None:
        self.backend = backend
        self.torch = torch
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_ids = {
            "overview": int(backend.overview_camera_id),
            "wrist": int(backend.wrist_camera_id),
        }
        host = backend.host_model
        aspect = float(backend.config.render_width) / float(backend.config.render_height)
        self.tan_half = {}
        for name, cam in self.camera_ids.items():
            tan_v = float(np.tan(np.deg2rad(float(host.cam_fovy[cam]) / 2.0)))
            self.tan_half[name] = (tan_v * aspect, tan_v)
        self.frames_available = True
        try:
            self._cam_xpos = backend.wp.to_torch(backend.data.cam_xpos)
            self._cam_xmat = backend.wp.to_torch(backend.data.cam_xmat)
        except Exception as error:  # pragma: no cover - depends on the backend build
            self.frames_available = False
            print(f"[trace] camera poses unavailable ({error}); frame flags will be False", flush=True)
        self._rows: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers

    def _in_frame(self, camera: str, points: Any) -> Any:
        """(worlds,) bool: does each world's point project inside the frustum."""

        torch = self.torch
        if not self.frames_available:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        cam = self.camera_ids[camera]
        position = self._cam_xpos[:, cam].reshape(-1, 3).to(points.dtype)
        rotation = self._cam_xmat[:, cam].reshape(-1, 3, 3).to(points.dtype)
        # MuJoCo: world = xmat @ local; the camera looks along local -z,
        # local +x right, +y up.
        local = torch.einsum("wji,wj->wi", rotation, points - position)
        depth = -local[:, 2]
        tan_h, tan_v = self.tan_half[camera]
        safe = depth.clamp_min(1e-6)
        return (depth > 1e-6) & ((local[:, 0] / safe).abs() <= tan_h) & ((local[:, 1] / safe).abs() <= tan_v)

    # ------------------------------------------------------------------ recording

    def start_round(self, *, round_index: int, scenes: Sequence[Any], target_slots: Any, reference_slots: Any) -> None:
        self._rows = []
        self._meta = {
            "round_index": int(round_index),
            "scene_uid": np.asarray([str(s.scene_uid) for s in scenes]),
            "destination": np.asarray([str(s.destination) for s in scenes]),
            "target_catalog": np.asarray([str(s.target_catalog) for s in scenes]),
            "success_radius": np.asarray([float(s.destination_success_radius) for s in scenes], dtype=np.float32),
        }
        self._target_slots = target_slots
        self._reference_slots = reference_slots

    def record_step(
        self,
        *,
        decision_index: int,
        action: Any,
        low_dim: Any,
        active: Any,
        physical_grasp: Any,
        bilateral_contact: Any,
        release_in_progress: Any,
        outcome: Any,
    ) -> None:
        torch = self.torch
        rows = torch.arange(low_dim.ee_position.shape[0], device=low_dim.ee_position.device)
        target = low_dim.object_positions[rows, self._target_slots]
        receptacle = low_dim.object_positions[rows, self._reference_slots]
        ee = low_dim.ee_position
        record = {
            "decision": torch.full_like(active, int(decision_index), dtype=torch.int32),
            "active": active,
            "ee_xyz": ee,
            "command_xyz": low_dim.target_position,
            "action": action,
            "gripper_command": action[:, 4],
            "opening": low_dim.gripper_opening,
            "physical_grasp": physical_grasp,
            "bilateral_contact": bilateral_contact,
            "release_in_progress": release_in_progress,
            "object_xyz": target,
            "receptacle_xyz": receptacle,
            "grasped": outcome.grasped,
            "lifted": outcome.lifted,
            "released": outcome.released,
            "carry_slip": outcome.carry_slip,
            "wrong_place": outcome.wrong_place,
            "native": outcome.native,
        }
        for camera in ("overview", "wrist"):
            for name, point in (("ee", ee), ("object", target), ("receptacle", receptacle)):
                record[f"{name}_in_{camera}"] = self._in_frame(camera, point)
        self._rows.append({key: value.detach().clone() for key, value in record.items()})

    def finish_round(self, *, strict: Any, non_finite: Any) -> Path:
        out: dict[str, np.ndarray] = dict(self._meta)
        for key in self._rows[0]:
            stacked = self.torch.stack([row[key] for row in self._rows], dim=0)
            array = stacked.cpu().numpy()
            if array.dtype == np.float64:
                array = array.astype(np.float32)
            out[key] = array
        out["strict_final"] = strict.detach().cpu().numpy().astype(bool)
        out["non_finite"] = non_finite.detach().cpu().numpy().astype(bool)
        path = self.output_dir / f"trace_round{int(self._meta['round_index']):03d}.npz"
        np.savez_compressed(path, **out)
        self._rows = []
        return path


def load_traces(paths: Sequence[Path]) -> list[Mapping[str, np.ndarray]]:
    return [dict(np.load(path, allow_pickle=False)) for path in paths]
