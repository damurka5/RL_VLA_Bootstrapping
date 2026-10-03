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
* the outcome observer's latched flags, so events can be timed,
* at every policy decision, the SmolVLA prior and the final action chunk. The
  legacy residual policy acts as ``tanh(prior + residual_scale * residual)``,
  so ``tanh(prior)`` is what the frozen prior alone would command, and
  ``atanh(final) - prior`` is the residual's push. That attributes the
  far-side upward Z to one or the other.
* when the caller passes the actor's direct components, the reference logit,
  the correction logit, the combined logit and the mean action as separate
  arrays (``decision_reference_logit`` etc.), with ``policy_architecture`` in
  the metadata. For ``frozen_reference_logit_correction_v1`` traces
  ``atanh(final) - prior`` is NOT the bounded residual: it mixes the frozen
  reference's push with the learned correction (and, on stochastic arms, the
  exploration noise), so the summarizer uses the direct arrays instead.

One ``trace_round<k>.npz`` per round, arrays shaped (steps, worlds, ...),
float32 or bool. ``summarize_kinematic_traces.py`` turns them into tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class KinematicTrace:
    def __init__(
        self,
        *,
        backend: Any,
        output_dir: Path,
        torch: Any,
        residual_scale: float = float("nan"),
        action_step_xyz: float = float("nan"),
        policy_architecture: str = "bounded_residual_v0",
    ) -> None:
        self.backend = backend
        self.policy_architecture = str(policy_architecture)
        self.residual_scale = float(residual_scale)
        self.action_step_xyz = float(action_step_xyz)
        self._decisions: list[dict[str, Any]] = []
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
        self._decisions = []
        self._meta = {
            "residual_scale": np.float32(self.residual_scale),
            "action_step_xyz": np.float32(self.action_step_xyz),
            "policy_architecture": np.asarray(self.policy_architecture),
            "round_index": int(round_index),
            "scene_uid": np.asarray([str(s.scene_uid) for s in scenes]),
            "destination": np.asarray([str(s.destination) for s in scenes]),
            "target_catalog": np.asarray([str(s.target_catalog) for s in scenes]),
            "success_radius": np.asarray([float(s.destination_success_radius) for s in scenes], dtype=np.float32),
        }
        self._target_slots = target_slots
        self._reference_slots = reference_slots

    def record_decision(
        self, *, decision_index: int, prior: Any, chunk: Any, active: Any,
        components: Mapping[str, Any] | None = None,
    ) -> None:
        """The prior and the executed chunk of one decision, (worlds, per, 5).

        ``components`` (from ``trainer.action_components_tensor``) adds the
        directly computed reference logit, correction logit, combined logit
        and deterministic mean action.
        """

        per = int(chunk.shape[1])
        worlds, dims = int(chunk.shape[0]), int(chunk.shape[-1])
        prior = prior.reshape(worlds, -1, dims)[:, :per]
        record = {
            "decision_index": int(decision_index),
            "prior": prior.detach().to(self.torch.float32).clone(),
            "final": chunk.detach().to(self.torch.float32).clone(),
            "active": active.detach().clone(),
        }
        if components is not None:
            for key, name in (
                ("reference_logit", "reference_logit"),
                ("correction", "correction_logit"),
                ("logit", "combined_logit"),
                ("mean", "mean"),
            ):
                record[name] = components[key][:, :per].detach().to(self.torch.float32).clone()
        self._decisions.append(record)

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
        if self._decisions:
            out["decision_index"] = np.asarray([d["decision_index"] for d in self._decisions], dtype=np.int32)
            keys = ["prior", "final", "active"] + [
                key
                for key in ("reference_logit", "correction_logit", "combined_logit", "mean")
                if all(key in d for d in self._decisions)
            ]
            for key in keys:
                out[f"decision_{key}"] = self.torch.stack(
                    [d[key] for d in self._decisions], dim=0
                ).cpu().numpy()
        out["strict_final"] = strict.detach().cpu().numpy().astype(bool)
        out["non_finite"] = non_finite.detach().cpu().numpy().astype(bool)
        path = self.output_dir / f"trace_round{int(self._meta['round_index']):03d}.npz"
        np.savez_compressed(path, **out)
        self._rows = []
        self._decisions = []
        return path


def load_traces(paths: Sequence[Path]) -> list[Mapping[str, np.ndarray]]:
    return [dict(np.load(path, allow_pickle=False)) for path in paths]
