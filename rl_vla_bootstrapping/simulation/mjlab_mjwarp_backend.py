from __future__ import annotations

import hashlib
import importlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cdpr_backend import (
    CDPRBackendConfig,
    CDPRLowDimBatch,
    CDPRRenderBatch,
    CDPRSimulatorBackend,
    SimulatorDependencyError,
)
from .cdpr_object_catalog import (
    ACTIVE_CDPR_CATALOGS,
    INACTIVE_CATALOG_ID,
    PRIMITIVE_NAMES,
    build_variant_arrays,
)
from .mjwarp_compat import package_versions, require_pinned_versions


_FRAME_ANCHORS = (
    (-0.535, -0.755, 1.309),
    (0.755, -0.525, 1.309),
    (0.535, 0.755, 1.309),
    (-0.755, 0.525, 1.309),
)


def _mjcf_tree_sha256(path: Path) -> str:
    """Content fingerprint for the root MJCF and all recursive includes."""
    hasher = hashlib.sha256()
    visited: set[Path] = set()

    def add(current: Path) -> None:
        resolved = current.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        content = resolved.read_bytes()
        hasher.update(resolved.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(content)
        root = ET.fromstring(content)
        for include in root.findall(".//include"):
            filename = str(include.get("file") or "").strip()
            if filename:
                add(resolved.parent / filename)

    add(path)
    return hasher.hexdigest()


def _required_import(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exercised on remote CUDA host
        raise SimulatorDependencyError(
            f"The explicitly selected MJLab/MuJoCo Warp backend cannot import "
            f"{name!r}: {exc}. {install_hint}"
        ) from exc


def _host_name_id(mujoco: Any, model: Any, objtype: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, objtype, name))
    if value < 0:
        raise RuntimeError(f"Required MJCF element {name!r} is missing.")
    return value


def _calibrate_host_cdpr(mujoco: Any, model: Any) -> dict[str, Any]:
    """Reproduce the CPU controller's tendon preload and finite-difference model."""

    import numpy as np

    data = mujoco.MjData(model)
    slider_joint_ids = [
        _host_name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"slider_{index}")
        for index in range(1, 5)
    ]
    slider_qadr = np.asarray(
        [int(model.jnt_qposadr[index]) for index in slider_joint_ids],
        dtype=np.int64,
    )
    slider_dofadr = np.asarray(
        [int(model.jnt_dofadr[index]) for index in slider_joint_ids],
        dtype=np.int64,
    )
    tendon_ids = np.asarray(
        [
            _host_name_id(mujoco, model, mujoco.mjtObj.mjOBJ_TENDON, f"rope_{index}")
            for index in range(1, 5)
        ],
        dtype=np.int64,
    )
    slider_actuator_ids = np.asarray(
        [
            _host_name_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"slider_{index}_pos",
            )
            for index in range(1, 5)
        ],
        dtype=np.int64,
    )
    ee_body_id = _host_name_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ee_base"
    )
    topcenter_site_id = _host_name_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "topcenter"
    )
    mujoco.mj_forward(model, data)

    def estimate_length_jacobian(dq: float = 1.0e-4) -> Any:
        jacobian = np.zeros((4,), dtype=np.float64)
        for index, (qadr, tendon_id) in enumerate(zip(slider_qadr, tendon_ids)):
            q0 = float(data.qpos[qadr])
            length0 = float(data.ten_length[tendon_id])
            data.qpos[qadr] = q0 + dq
            mujoco.mj_forward(model, data)
            jacobian[index] = (float(data.ten_length[tendon_id]) - length0) / dq
            data.qpos[qadr] = q0
            mujoco.mj_forward(model, data)
        if bool(np.any(np.abs(jacobian) < 1.0e-8)):
            raise RuntimeError(
                f"CDPR tendon calibration is singular: dlength_dq={jacobian.tolist()}."
            )
        return jacobian

    for _ in range(8):
        mujoco.mj_forward(model, data)
        current_q = data.qpos[slider_qadr].copy()
        current_lengths = data.ten_length[tendon_ids].copy()
        jacobian = estimate_length_jacobian()
        upper = model.tendon_range[tendon_ids, 1]
        targets = current_q + (upper - current_lengths) / jacobian
        limits = model.actuator_ctrlrange[slider_actuator_ids]
        targets = np.clip(targets, limits[:, 0], limits[:, 1])
        data.qpos[slider_qadr] = targets
        data.qvel[slider_dofadr] = 0.0
        data.ctrl[slider_actuator_ids] = targets
        if float(np.max(np.abs(targets - current_q))) <= 1.0e-6:
            break

    mujoco.mj_forward(model, data)
    jacobian = estimate_length_jacobian()
    attach_offset = (
        data.site_xpos[topcenter_site_id] - data.xpos[ee_body_id]
    ).astype(np.float32)
    model.qpos0[:] = data.qpos
    if hasattr(mujoco, "mj_setConst"):
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)
    return {
        "slider_joint_ids": slider_joint_ids,
        "slider_qadr": slider_qadr,
        "slider_dofadr": slider_dofadr,
        "tendon_ids": tendon_ids,
        "slider_actuator_ids": slider_actuator_ids,
        "dlength_dq": jacobian.astype(np.float32),
        "slider_q_per_length": (1.0 / jacobian).astype(np.float32),
        "attach_offset": attach_offset,
        "base_qpos": data.qpos.astype(np.float32).copy(),
        "base_qvel": data.qvel.astype(np.float32).copy(),
        "base_ctrl": data.ctrl.astype(np.float32).copy(),
    }


class MJLabMJWarpCDPRBackend(CDPRSimulatorBackend):
    """GPU-resident, fixed-topology CDPR worlds backed by native MJWarp.

    MJLab supplies the pinned PyTorch/MJWarp runtime.  This backend uses the
    native MJWarp model/data objects directly because the existing GRPO trainer
    has a custom lifecycle rather than an Isaac-Lab-style manager lifecycle.
    No Python environment objects are allocated per world.
    """

    def __init__(
        self,
        *,
        config: CDPRBackendConfig,
        create_renderer: bool = True,
        require_mjlab: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self.torch = _required_import(
            "torch", "Run scripts/setup_cdpr_mjlab_remote.sh first."
        )
        self.wp = _required_import(
            "warp", "Install the pinned warp-lang package."
        )
        self.mujoco = _required_import(
            "mujoco", "Install the pinned MuJoCo package."
        )
        self.mjw = _required_import(
            "mujoco_warp", "Install the pinned mujoco-warp package."
        )
        self.mjlab = (
            _required_import("mjlab", "Install the pinned MJLab package.")
            if require_mjlab
            else None
        )
        require_pinned_versions()
        if not bool(self.torch.cuda.is_available()):
            raise SimulatorDependencyError(
                "MJLab/MuJoCo Warp backend requires a CUDA-enabled PyTorch runtime."
            )

        self._device = self.torch.device(str(config.device))
        if self._device.type != "cuda":
            raise ValueError(
                f"Production MJWarp rollout requires a CUDA device, got {self._device}."
            )
        self.torch.cuda.set_device(self._device)
        self.wp.init()
        self.wp.set_device(str(self._device))

        xml_path = Path(config.xml_path or "").expanduser().resolve()
        self.host_model = self.mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.host_model.opt.timestep = 1.0 / 60.0
        self._calibration = _calibrate_host_cdpr(self.mujoco, self.host_model)
        self._resolve_host_ids()

        with self.wp.ScopedDevice(str(self._device)):
            self.model = self.mjw.put_model(self.host_model)
            make_kwargs: dict[str, Any] = {
                "nworld": int(config.worlds_per_rank),
                "nconmax": int(config.nconmax),
                "njmax": int(config.njmax),
            }
            if config.nccdmax is not None:
                make_kwargs["nccdmax"] = int(config.nccdmax)
            self.data = self.mjw.make_data(self.host_model, **make_kwargs)
            self._make_model_fields_batched()
            self._bind_torch_views()
            self.mjw.reset_data(self.model, self.data)
            self.mjw.set_const(self.model, self.data)
            self.mjw.forward(self.model, self.data)

        self._initialize_controller_state()
        self._initialize_catalog_tables()
        inactive = self.torch.full(
            (self.worlds_per_rank, 4),
            INACTIVE_CATALOG_ID,
            dtype=self.torch.int64,
            device=self._device,
        )
        self.set_object_catalogs(inactive)

        self.render_context = None
        self._overview_rgb_wp = None
        self._wrist_rgb_wp = None
        self._overview_rgb = None
        self._wrist_rgb = None
        if create_renderer:
            self._initialize_renderer()

    @property
    def device(self) -> Any:
        return self._device

    def _resolve_host_ids(self) -> None:
        mj = self.mujoco
        model = self.host_model
        self.ee_body_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_BODY, "ee_base"
        )
        self.ee_free_joint_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_JOINT, "ee_free"
        )
        self.ee_free_qadr = int(model.jnt_qposadr[self.ee_free_joint_id])
        self.ee_free_dofadr = int(model.jnt_dofadr[self.ee_free_joint_id])
        self.topcenter_site_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_SITE, "topcenter"
        )
        self.yaw_joint_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_JOINT, "ee_yaw"
        )
        self.yaw_qadr = int(model.jnt_qposadr[self.yaw_joint_id])
        self.finger_joint_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_JOINT, "finger_l"
        )
        self.finger_qadr = int(model.jnt_qposadr[self.finger_joint_id])
        self.finger_dofadr = int(model.jnt_dofadr[self.finger_joint_id])
        self.yaw_actuator_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_ACTUATOR, "act_ee_yaw"
        )
        self.gripper_actuator_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper"
        )
        self.overview_camera_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_CAMERA, "overview"
        )
        self.wrist_camera_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera"
        )
        self.desk_geom_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_GEOM, "mjwarp_desk_surface"
        )
        self.background_geom_id = _host_name_id(
            mj, model, mj.mjtObj.mjOBJ_GEOM, "mjwarp_background"
        )
        self.desk_material_ids = tuple(
            _host_name_id(
                mj,
                model,
                mj.mjtObj.mjOBJ_MATERIAL,
                f"mjwarp_desk_mat_{index}",
            )
            for index in range(7)
        )
        self.gripper_surface_geom_ids = tuple(
            _host_name_id(mj, model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "palm",
                "finger_l_shoulder",
                "finger_l_link",
                "finger_l_tip",
                "left_finger_pad",
                "finger_r_shoulder",
                "finger_r_link",
                "finger_r_tip",
                "right_finger_pad",
            )
        )
        self.finger_pad_geom_ids = tuple(
            _host_name_id(mj, model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_finger_pad", "right_finger_pad")
        )
        self.slider_qadr = tuple(
            int(value) for value in self._calibration["slider_qadr"]
        )
        self.slider_dofadr = tuple(
            int(value) for value in self._calibration["slider_dofadr"]
        )
        self.tendon_ids = tuple(
            int(value) for value in self._calibration["tendon_ids"]
        )
        self.slider_actuator_ids = tuple(
            int(value) for value in self._calibration["slider_actuator_ids"]
        )
        self.object_body_ids = tuple(
            _host_name_id(
                mj,
                model,
                mj.mjtObj.mjOBJ_BODY,
                f"mjwarp_object_slot_{slot}",
            )
            for slot in range(4)
        )
        self.object_joint_ids = tuple(
            _host_name_id(
                mj,
                model,
                mj.mjtObj.mjOBJ_JOINT,
                f"mjwarp_object_slot_{slot}_free",
            )
            for slot in range(4)
        )
        self.object_qadr = tuple(
            int(model.jnt_qposadr[joint_id]) for joint_id in self.object_joint_ids
        )
        self.object_dofadr = tuple(
            int(model.jnt_dofadr[joint_id]) for joint_id in self.object_joint_ids
        )
        slot_geom_ids: list[list[int]] = []
        for slot in range(4):
            slot_geom_ids.append(
                [
                    _host_name_id(
                        mj,
                        model,
                        mj.mjtObj.mjOBJ_GEOM,
                        f"mjwarp_slot_{slot}_{primitive}",
                    )
                    for primitive in PRIMITIVE_NAMES
                ]
            )
        self.slot_geom_ids_host = tuple(tuple(row) for row in slot_geom_ids)

    def _wp_array(self, value: Any, dtype: Any) -> Any:
        return self.wp.array(value, dtype=dtype, device=str(self._device))

    def _make_model_fields_batched(self) -> None:
        import numpy as np

        nworld = self.worlds_per_rank
        model = self.host_model

        def repeated(value: Any, *, reshape: tuple[int, ...] | None = None) -> Any:
            array = np.asarray(value)
            if reshape is not None:
                array = array.reshape(reshape)
            return np.repeat(array[None, ...], nworld, axis=0)

        # Allocate every dependent field listed by the official per-world mesh
        # workflow before any optional CUDA graph capture.
        self.model.geom_size = self._wp_array(
            repeated(model.geom_size), self.wp.vec3
        )
        self.model.geom_pos = self._wp_array(
            repeated(model.geom_pos), self.wp.vec3
        )
        self.model.geom_quat = self._wp_array(
            repeated(model.geom_quat), self.wp.quat
        )
        self.model.geom_rgba = self._wp_array(
            repeated(model.geom_rgba), self.wp.vec4
        )
        self.model.geom_matid = self._wp_array(
            repeated(model.geom_matid), self.wp.int32
        )
        self.model.geom_aabb = self._wp_array(
            repeated(model.geom_aabb, reshape=(model.ngeom, 2, 3)),
            self.wp.vec3,
        )
        self.model.geom_rbound = self._wp_array(
            repeated(model.geom_rbound), self.wp.float32
        )
        self.model.body_mass = self._wp_array(
            repeated(model.body_mass), self.wp.float32
        )
        self.model.body_subtreemass = self._wp_array(
            repeated(model.body_subtreemass), self.wp.float32
        )
        self.model.body_inertia = self._wp_array(
            repeated(model.body_inertia), self.wp.vec3
        )
        self.model.body_invweight0 = self._wp_array(
            repeated(model.body_invweight0), self.wp.vec2
        )
        self.model.body_ipos = self._wp_array(
            repeated(model.body_ipos), self.wp.vec3
        )
        self.model.body_iquat = self._wp_array(
            repeated(model.body_iquat), self.wp.quat
        )

    def _bind_torch_views(self) -> None:
        to_torch = self.wp.to_torch
        for name in (
            "qpos",
            "qvel",
            "ctrl",
            "act",
            "history",
            "qacc_warmstart",
            "time",
            "eq_active",
            "xpos",
            "xquat",
            "site_xpos",
            "ten_length",
            "cvel",
            "sensordata",
        ):
            value = getattr(self.data, name, None)
            if value is not None:
                setattr(self, f"_{name}", to_torch(value))
        self._contact_geom = to_torch(self.data.contact.geom)
        self._contact_worldid = to_torch(self.data.contact.worldid)
        self._contact_dist = to_torch(self.data.contact.dist)
        self._nacon = to_torch(self.data.nacon)
        self._nefc = to_torch(self.data.nefc)

        for name in (
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_rgba",
            "geom_matid",
            "geom_aabb",
            "geom_rbound",
            "body_mass",
            "body_subtreemass",
            "body_inertia",
            "body_invweight0",
            "body_ipos",
            "body_iquat",
        ):
            setattr(self, f"_model_{name}", to_torch(getattr(self.model, name)))

        self._slot_geom_ids = self.torch.tensor(
            self.slot_geom_ids_host, dtype=self.torch.int64, device=self._device
        )
        self._object_body_ids_tensor = self.torch.tensor(
            self.object_body_ids, dtype=self.torch.int64, device=self._device
        )
        self._object_qadr_tensor = self.torch.tensor(
            self.object_qadr, dtype=self.torch.int64, device=self._device
        )
        self._object_dofadr_tensor = self.torch.tensor(
            self.object_dofadr, dtype=self.torch.int64, device=self._device
        )
        self._slider_qadr_tensor = self.torch.tensor(
            self.slider_qadr, dtype=self.torch.int64, device=self._device
        )
        self._slider_actuator_ids_tensor = self.torch.tensor(
            self.slider_actuator_ids,
            dtype=self.torch.int64,
            device=self._device,
        )
        self._tendon_ids_tensor = self.torch.tensor(
            self.tendon_ids, dtype=self.torch.int64, device=self._device
        )
        self._desk_material_ids_tensor = self.torch.tensor(
            self.desk_material_ids, dtype=self.torch.int32, device=self._device
        )
        self._gripper_surface_geom_ids_tensor = self.torch.tensor(
            self.gripper_surface_geom_ids,
            dtype=self.torch.int64,
            device=self._device,
        )
        self._finger_pad_geom_ids_tensor = self.torch.tensor(
            self.finger_pad_geom_ids,
            dtype=self.torch.int64,
            device=self._device,
        )

    def _initialize_controller_state(self) -> None:
        torch = self.torch
        nworld = self.worlds_per_rank
        self._frame_anchors = torch.tensor(
            _FRAME_ANCHORS, dtype=torch.float32, device=self._device
        )
        attach_offset = torch.tensor(
            self._calibration["attach_offset"],
            dtype=torch.float32,
            device=self._device,
        )
        slider_q_per_length = torch.tensor(
            self._calibration["slider_q_per_length"],
            dtype=torch.float32,
            device=self._device,
        )
        self._controller_attach_offset = attach_offset.expand(nworld, -1).clone()
        self._slider_q_per_length = slider_q_per_length.expand(nworld, -1).clone()
        self._dlength_dq = (
            torch.tensor(
                self._calibration["dlength_dq"],
                dtype=torch.float32,
                device=self._device,
            )
            .expand(nworld, -1)
            .clone()
        )
        self._tendon_upper = torch.tensor(
            self.host_model.tendon_range[list(self.tendon_ids), 1],
            dtype=torch.float32,
            device=self._device,
        )
        self._controller_target = self._xpos[:, self.ee_body_id].clone()
        self._controller_prev_lengths = self._ten_length.index_select(
            1, self._tendon_ids_tensor
        ).clone()
        self._controller_yaw = self._qpos[:, self.yaw_qadr].clone()
        self._controller_gripper = self._normalized_gripper_opening().clone()
        self._last_actions = torch.zeros(
            (nworld, 5), dtype=torch.float32, device=self._device
        )
        self._catalog_ids = torch.full(
            (nworld, 4),
            INACTIVE_CATALOG_ID,
            dtype=torch.int64,
            device=self._device,
        )
        self._object_rest_height = torch.zeros(
            (nworld, 4), dtype=torch.float32, device=self._device
        )
        self._pinned_target_slots = torch.zeros(
            (nworld,), dtype=torch.int64, device=self._device
        )
        self._pinned_ee_offsets = torch.zeros(
            (nworld, 3), dtype=torch.float32, device=self._device
        )
        self._pinned_mask = torch.zeros(
            (nworld,), dtype=torch.bool, device=self._device
        )
        self._pinned_release_threshold = torch.ones(
            (nworld,), dtype=torch.float32, device=self._device
        )
        slider_limits = self.host_model.actuator_ctrlrange[
            list(self.slider_actuator_ids)
        ]
        self._slider_ctrl_limits = torch.tensor(
            slider_limits, dtype=torch.float32, device=self._device
        )
        yaw_range = self.host_model.jnt_range[self.yaw_joint_id]
        self._yaw_limits = torch.tensor(
            (min(yaw_range), max(yaw_range)),
            dtype=torch.float32,
            device=self._device,
        )
        finger_range = self.host_model.jnt_range[self.finger_joint_id]
        self._finger_limits = torch.tensor(
            (min(finger_range), max(finger_range)),
            dtype=torch.float32,
            device=self._device,
        )
        gripper_ctrl_range = self.host_model.actuator_ctrlrange[
            self.gripper_actuator_id
        ]
        self._gripper_ctrl_limits = torch.tensor(
            (min(gripper_ctrl_range), max(gripper_ctrl_range)),
            dtype=torch.float32,
            device=self._device,
        )
        self._workspace_min = torch.tensor(
            (
                min(self.config.workspace_x),
                min(self.config.workspace_y),
                min(self.config.workspace_z),
            ),
            dtype=torch.float32,
            device=self._device,
        )
        self._workspace_max = torch.tensor(
            (
                max(self.config.workspace_x),
                max(self.config.workspace_y),
                max(self.config.workspace_z),
            ),
            dtype=torch.float32,
            device=self._device,
        )

    def _initialize_catalog_tables(self) -> None:
        import numpy as np

        rows: list[dict[str, Any]] = []
        for object_id in (INACTIVE_CATALOG_ID, *range(len(ACTIVE_CDPR_CATALOGS))):
            ids = np.full((1, 4), INACTIVE_CATALOG_ID, dtype=np.int32)
            ids[0, 0] = object_id
            rows.append(build_variant_arrays(ids))

        def table(name: str) -> Any:
            value = np.stack([row[name][0, 0] for row in rows], axis=0)
            return self.torch.as_tensor(
                value, dtype=self.torch.float32, device=self._device
            )

        self._catalog_geom_size = table("geom_size")
        self._catalog_geom_pos = table("geom_pos")
        self._catalog_geom_quat = table("geom_quat")
        self._catalog_geom_rgba = table("geom_rgba")
        self._catalog_body_mass = table("body_mass")
        self._catalog_body_inertia = table("body_inertia")
        self._catalog_rest_height = table("rest_height")

    def _initialize_renderer(self) -> None:
        width = int(self.config.render_width)
        height = int(self.config.render_height)
        with self.wp.ScopedDevice(str(self._device)):
            self.render_context = self.mjw.create_render_context(
                self.host_model,
                nworld=self.worlds_per_rank,
                cam_res=(width, height),
                render_rgb=True,
                render_depth=False,
                use_textures=True,
                use_shadows=False,
                enabled_geom_groups=[0, 1, 2, 3],
            )
            self._overview_rgb_wp = self.wp.zeros(
                (self.worlds_per_rank, height, width),
                dtype=self.wp.vec3,
                device=str(self._device),
            )
            self._wrist_rgb_wp = self.wp.zeros(
                (self.worlds_per_rank, height, width),
                dtype=self.wp.vec3,
                device=str(self._device),
            )
            self._overview_rgb = self.wp.to_torch(self._overview_rgb_wp)
            self._wrist_rgb = self.wp.to_torch(self._wrist_rgb_wp)

    def _normalize_world_indices(self, world_indices: Any) -> Any:
        indices = self.torch.as_tensor(
            world_indices, dtype=self.torch.int64, device=self._device
        ).reshape(-1)
        if indices.numel() == 0:
            return indices
        if bool((indices < 0).any()) or bool((indices >= self.worlds_per_rank).any()):
            raise IndexError(
                f"World indices must be in [0, {self.worlds_per_rank})."
            )
        return indices

    def _copy_subset_value(self, target: Any, indices: Any, value: Any) -> None:
        source = self.torch.as_tensor(
            value, dtype=target.dtype, device=self._device
        )
        if source.shape == target.shape:
            source = source.index_select(0, indices)
        expected = (int(indices.numel()), *target.shape[1:])
        if tuple(source.shape) != expected:
            raise ValueError(
                f"Subset state has shape {tuple(source.shape)}, expected {expected}."
            )
        target.index_copy_(0, indices, source)

    def reset_worlds(
        self,
        world_indices: Any,
        *,
        qpos: Any | None = None,
        qvel: Any | None = None,
        controller_state: Mapping[str, Any] | None = None,
    ) -> None:
        indices = self._normalize_world_indices(world_indices)
        if indices.numel() == 0:
            return
        reset_mask = self.torch.zeros(
            (self.worlds_per_rank,), dtype=self.torch.bool, device=self._device
        )
        reset_mask[indices] = True
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.reset_data(
                self.model, self.data, self.wp.from_torch(reset_mask)
            )
        if qpos is not None:
            self._copy_subset_value(self._qpos, indices, qpos)
        if qvel is not None:
            self._copy_subset_value(self._qvel, indices, qvel)
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.forward(self.model, self.data)

        state = dict(controller_state or {})
        defaults = {
            "target_position": self._xpos[:, self.ee_body_id],
            "target_yaw": self._qpos[:, self.yaw_qadr],
            "target_gripper": self._normalized_gripper_opening(),
            "previous_tendon_lengths": self._ten_length.index_select(
                1, self._tendon_ids_tensor
            ),
            "attach_offset": self.torch.tensor(
                self._calibration["attach_offset"],
                dtype=self.torch.float32,
                device=self._device,
            ).expand(self.worlds_per_rank, -1),
            "slider_q_per_length": self.torch.tensor(
                self._calibration["slider_q_per_length"],
                dtype=self.torch.float32,
                device=self._device,
            ).expand(self.worlds_per_rank, -1),
        }
        targets = (
            ("target_position", self._controller_target),
            ("target_yaw", self._controller_yaw),
            ("target_gripper", self._controller_gripper),
            ("previous_tendon_lengths", self._controller_prev_lengths),
            ("attach_offset", self._controller_attach_offset),
            ("slider_q_per_length", self._slider_q_per_length),
        )
        for key, target in targets:
            self._copy_subset_value(target, indices, state.get(key, defaults[key]))
        self._dlength_dq.index_copy_(
            0,
            indices,
            self._slider_q_per_length.index_select(0, indices).reciprocal(),
        )
        self._last_actions.index_fill_(0, indices, 0.0)
        self._pinned_mask.index_fill_(0, indices, False)

    def broadcast_group_state(self, base_world_indices: Any) -> None:
        base = self._normalize_world_indices(base_world_indices)
        if int(base.numel()) < 1:
            return
        group_size = int(self.config.grpo_group_size)
        offsets = self.torch.arange(
            group_size, dtype=self.torch.int64, device=self._device
        )
        starts = (base // group_size) * group_size
        destinations = (starts[:, None] + offsets[None, :]).reshape(-1)
        if int(self.torch.unique(starts).numel()) != int(starts.numel()):
            raise ValueError("Each local group may be broadcast only once.")
        sources = base[:, None].expand(-1, group_size).reshape(-1)

        state_names = (
            "_qpos",
            "_qvel",
            "_ctrl",
            "_act",
            "_history",
            "_qacc_warmstart",
            "_time",
            "_eq_active",
        )
        for name in state_names:
            target = getattr(self, name, None)
            if target is not None and target.ndim > 0:
                target.index_copy_(0, destinations, target.index_select(0, sources))
        for target in (
            self._controller_target,
            self._controller_yaw,
            self._controller_gripper,
            self._controller_prev_lengths,
            self._controller_attach_offset,
            self._slider_q_per_length,
            self._dlength_dq,
            self._last_actions,
            self._catalog_ids,
            self._object_rest_height,
            self._pinned_target_slots,
            self._pinned_ee_offsets,
            self._pinned_mask,
            self._pinned_release_threshold,
        ):
            target.index_copy_(0, destinations, target.index_select(0, sources))

        for name in (
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_rgba",
            "geom_matid",
            "geom_aabb",
            "geom_rbound",
            "body_mass",
            "body_subtreemass",
            "body_inertia",
            "body_invweight0",
            "body_ipos",
            "body_iquat",
        ):
            target = getattr(self, f"_model_{name}")
            target.index_copy_(0, destinations, target.index_select(0, sources))
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.set_const(self.model, self.data)
            self.mjw.forward(self.model, self.data)

    def _normalized_gripper_opening(self) -> Any:
        minimum, maximum = self._finger_limits if hasattr(self, "_finger_limits") else (
            float(min(self.host_model.jnt_range[self.finger_joint_id])),
            float(max(self.host_model.jnt_range[self.finger_joint_id])),
        )
        span = maximum - minimum
        return ((self._qpos[:, self.finger_qadr] - minimum) / span).clamp(0.0, 1.0)

    def _write_controller_controls(self) -> None:
        ee_position = self._xpos[:, self.ee_body_id]
        current_slider_q = self._qpos.index_select(1, self._slider_qadr_tensor)
        current_attach = ee_position + self._controller_attach_offset
        target_attach = self._controller_target + self._controller_attach_offset
        current_lengths = self.torch.linalg.vector_norm(
            current_attach[:, None, :] - self._frame_anchors[None, :, :],
            dim=-1,
        )
        target_lengths = self.torch.linalg.vector_norm(
            target_attach[:, None, :] - self._frame_anchors[None, :, :],
            dim=-1,
        )
        slider_target = current_slider_q + self._slider_q_per_length * (
            current_lengths - target_lengths
        )
        is_hold = (
            self.torch.linalg.vector_norm(
                self._controller_target - ee_position, dim=-1
            )
            < 1.0e-6
        )
        slider_target = self.torch.where(
            is_hold[:, None], current_slider_q, slider_target
        )
        slider_target = self.torch.maximum(
            self.torch.minimum(slider_target, self._slider_ctrl_limits[:, 1]),
            self._slider_ctrl_limits[:, 0],
        )
        self._ctrl[:, self._slider_actuator_ids_tensor] = slider_target
        self._ctrl[:, self.yaw_actuator_id] = self._controller_yaw

        ctrl_min, ctrl_max = self._gripper_ctrl_limits
        self._ctrl[:, self.gripper_actuator_id] = (
            ctrl_min + self._controller_gripper * (ctrl_max - ctrl_min)
        )

    def _write_configured_pinned_poses(self) -> None:
        """Write pin constraints after a physics substep without a host branch."""
        torch = self.torch
        rows = torch.arange(
            self.worlds_per_rank, dtype=torch.int64, device=self._device
        )
        qadr = self._object_qadr_tensor.index_select(
            0, self._pinned_target_slots
        )
        dofadr = self._object_dofadr_tensor.index_select(
            0, self._pinned_target_slots
        )
        desired = (
            self._xpos[:, self.ee_body_id] + self._pinned_ee_offsets
        )
        for axis in range(3):
            current = self._qpos[rows, qadr + axis]
            self._qpos[rows, qadr + axis] = torch.where(
                self._pinned_mask, desired[:, axis], current
            )
        for axis in range(6):
            current = self._qvel[rows, dofadr + axis]
            self._qvel[rows, dofadr + axis] = torch.where(
                self._pinned_mask, torch.zeros_like(current), current
            )

    def step(self, actions: Any, active_mask: Any) -> CDPRLowDimBatch:
        torch = self.torch
        action_tensor = torch.as_tensor(
            actions, dtype=torch.float32, device=self._device
        )
        active = torch.as_tensor(
            active_mask, dtype=torch.bool, device=self._device
        ).reshape(-1)
        if tuple(action_tensor.shape) != (self.worlds_per_rank, 5):
            raise ValueError(
                f"MJWarp actions must have shape ({self.worlds_per_rank}, 5), "
                f"got {tuple(action_tensor.shape)}."
            )
        if tuple(active.shape) != (self.worlds_per_rank,):
            raise ValueError(
                f"active_mask must have shape ({self.worlds_per_rank},), "
                f"got {tuple(active.shape)}."
            )
        action_tensor = action_tensor.clamp(-1.0, 1.0)
        masked_action = torch.where(
            active[:, None], action_tensor, torch.zeros_like(action_tensor)
        )
        ee_position = self._xpos[:, self.ee_body_id]
        target_delta = masked_action[:, :3] * float(
            self.config.action_step_xyz
        )
        if self.config.lock_non_commanded_axes:
            active_axes = (
                masked_action[:, :3].abs()
                > float(self.config.lock_non_commanded_axes_threshold)
            )
            proposed_target = self.torch.where(
                active_axes,
                self._controller_target + target_delta,
                self._controller_target,
            )
        else:
            # This is the active production CPU setting: every command is
            # relative to the measured end-effector pose.
            proposed_target = ee_position + target_delta
        proposed_target = torch.maximum(
            torch.minimum(proposed_target, self._workspace_max),
            self._workspace_min,
        )
        self._controller_target = torch.where(
            active[:, None], proposed_target, self._controller_target
        )
        proposed_yaw = (
            self._controller_yaw
            + masked_action[:, 3] * float(self.config.action_step_yaw)
        ).clamp(self._yaw_limits[0], self._yaw_limits[1])
        self._controller_yaw = torch.where(
            active, proposed_yaw, self._controller_yaw
        )
        proposed_gripper = (
            self._controller_gripper
            + masked_action[:, 4] * float(self.config.action_step_gripper)
        ).clamp(0.0, 1.0)
        self._controller_gripper = torch.where(
            active, proposed_gripper, self._controller_gripper
        )
        self._pinned_mask.logical_and_(
            self._controller_gripper <= self._pinned_release_threshold
        )
        self._last_actions.copy_(masked_action)

        with self.wp.ScopedDevice(str(self._device)):
            for _ in range(self.config.physics_substeps):
                self._write_controller_controls()
                self.mjw.step(self.model, self.data)
                self._write_configured_pinned_poses()
            # The pin write happens after each substep. Refresh derived body,
            # contact, tendon, and camera state after the final write.
            self.mjw.forward(self.model, self.data)
        self._controller_prev_lengths.copy_(
            self._ten_length.index_select(1, self._tendon_ids_tensor)
        )
        return self.low_dim_observations()

    def low_dim_observations(self) -> CDPRLowDimBatch:
        object_positions = self._xpos.index_select(
            1, self._object_body_ids_tensor
        )
        object_quaternions = self._xquat.index_select(
            1, self._object_body_ids_tensor
        )
        return CDPRLowDimBatch(
            ee_position=self._xpos[:, self.ee_body_id],
            ee_quaternion=self._xquat[:, self.ee_body_id],
            ee_yaw=self._qpos[:, self.yaw_qadr],
            gripper_opening=self._normalized_gripper_opening(),
            target_position=self._controller_target,
            tendon_lengths=self._ten_length.index_select(
                1, self._tendon_ids_tensor
            ),
            object_positions=object_positions,
            object_quaternions=object_quaternions,
        )

    def _get_rgb(self, camera_id: int, output: Any) -> None:
        try:
            self.mjw.get_rgb(self.render_context, camera_id, output)
        except TypeError:  # 3.9 renderer keyword compatibility
            self.mjw.get_rgb(
                self.render_context, rgb_data=output, cam_id=camera_id
            )

    def render_policy_cameras(self) -> CDPRRenderBatch:
        if self.render_context is None:
            raise RuntimeError("The MJWarp renderer was disabled at backend creation.")
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.refit_bvh(self.model, self.data, self.render_context)
            self.mjw.render(self.model, self.data, self.render_context)
            self._get_rgb(self.overview_camera_id, self._overview_rgb_wp)
            self._get_rgb(self.wrist_camera_id, self._wrist_rgb_wp)
        overview = self._overview_rgb.permute(0, 3, 1, 2)
        wrist = self._wrist_rgb.permute(0, 3, 1, 2)
        expected = (
            self.worlds_per_rank,
            3,
            int(self.config.render_height),
            int(self.config.render_width),
        )
        if tuple(overview.shape) != expected or tuple(wrist.shape) != expected:
            raise RuntimeError(
                f"MJWarp camera contract mismatch: overview={tuple(overview.shape)}, "
                f"wrist={tuple(wrist.shape)}, expected={expected}."
            )
        if overview.device != self._device or wrist.device != self._device:
            raise RuntimeError("MJWarp camera tensors left the rank-local GPU.")
        if overview.dtype != self.torch.float32 or wrist.dtype != self.torch.float32:
            raise RuntimeError("MJWarp RGB output must be normalized float32.")
        return CDPRRenderBatch(overview=overview, wrist=wrist)

    def body_pose(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        ids = self.torch.tensor(
            [
                _host_name_id(
                    self.mujoco,
                    self.host_model,
                    self.mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
                for name in body_names
            ],
            dtype=self.torch.int64,
            device=self._device,
        )
        return self._xpos.index_select(1, ids), self._xquat.index_select(1, ids)

    def body_velocity(self, body_names: Sequence[str]) -> tuple[Any, Any]:
        ids = self.torch.tensor(
            [
                _host_name_id(
                    self.mujoco,
                    self.host_model,
                    self.mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
                for name in body_names
            ],
            dtype=self.torch.int64,
            device=self._device,
        )
        cvel = self._cvel.index_select(1, ids)
        # MuJoCo spatial velocity ordering is angular then linear.
        return cvel[..., 3:6], cvel[..., 0:3]

    def contact_mask(self, geom_a_ids: Any, geom_b_ids: Any) -> Any:
        torch = self.torch
        geom_a = torch.as_tensor(
            geom_a_ids, dtype=torch.int64, device=self._device
        ).reshape(-1)
        geom_b = torch.as_tensor(
            geom_b_ids, dtype=torch.int64, device=self._device
        ).reshape(-1)
        if geom_a.numel() == 1:
            geom_a = geom_a.expand(self.worlds_per_rank)
        if geom_b.numel() == 1:
            geom_b = geom_b.expand(self.worlds_per_rank)
        if geom_a.numel() != self.worlds_per_rank or geom_b.numel() != self.worlds_per_rank:
            raise ValueError("Contact geom ids must be scalar or one pair per world.")

        world = self._contact_worldid.to(dtype=torch.int64)
        contact_index = torch.arange(
            world.numel(), dtype=torch.int64, device=self._device
        )
        valid = (
            (contact_index < self._nacon.reshape(-1)[0].to(dtype=torch.int64))
            & (world >= 0)
            & (world < self.worlds_per_rank)
            & (self._contact_dist <= 0.002)
        )
        safe_world = world.clamp(0, self.worlds_per_rank - 1)
        first = self._contact_geom[:, 0].to(dtype=torch.int64)
        second = self._contact_geom[:, 1].to(dtype=torch.int64)
        wanted_a = geom_a.index_select(0, safe_world)
        wanted_b = geom_b.index_select(0, safe_world)
        matched = valid & (
            ((first == wanted_a) & (second == wanted_b))
            | ((first == wanted_b) & (second == wanted_a))
        )
        counts = torch.zeros(
            (self.worlds_per_rank,), dtype=torch.int32, device=self._device
        )
        counts.scatter_add_(0, safe_world, matched.to(dtype=torch.int32))
        return counts > 0

    def finger_object_contact_mask(self, target_slots: Any) -> Any:
        """Tensorized equivalent of the CPU reverse-frontier pad-contact gate."""
        torch = self.torch
        slots = torch.as_tensor(
            target_slots, dtype=torch.int64, device=self._device
        ).reshape(-1)
        if tuple(slots.shape) != (self.worlds_per_rank,):
            raise ValueError("One target object slot is required per world.")

        world = self._contact_worldid.to(dtype=torch.int64)
        contact_index = torch.arange(
            world.numel(), dtype=torch.int64, device=self._device
        )
        valid = (
            (contact_index < self._nacon.reshape(-1)[0].to(dtype=torch.int64))
            & (world >= 0)
            & (world < self.worlds_per_rank)
            & (self._contact_dist <= 0.002)
        )
        safe_world = world.clamp(0, self.worlds_per_rank - 1)
        first = self._contact_geom[:, 0].to(dtype=torch.int64)
        second = self._contact_geom[:, 1].to(dtype=torch.int64)
        target_geoms = self._slot_geom_ids.index_select(
            0, slots.index_select(0, safe_world)
        )
        first_is_target = (first[:, None] == target_geoms).any(dim=1)
        second_is_target = (second[:, None] == target_geoms).any(dim=1)
        first_is_pad = (
            first[:, None] == self._finger_pad_geom_ids_tensor[None, :]
        ).any(dim=1)
        second_is_pad = (
            second[:, None] == self._finger_pad_geom_ids_tensor[None, :]
        ).any(dim=1)
        matched = valid & (
            (first_is_target & second_is_pad)
            | (second_is_target & first_is_pad)
        )
        counts = torch.zeros(
            (self.worlds_per_rank,), dtype=torch.int32, device=self._device
        )
        counts.scatter_add_(0, safe_world, matched.to(dtype=torch.int32))
        return counts >= 1

    def configure_pinned_objects(
        self,
        target_slots: Any,
        ee_offsets: Any,
        pinned_mask: Any,
        release_threshold: Any,
    ) -> None:
        torch = self.torch
        slots = torch.as_tensor(
            target_slots, dtype=torch.int64, device=self._device
        ).reshape(-1)
        offsets = torch.as_tensor(
            ee_offsets, dtype=torch.float32, device=self._device
        )
        mask = torch.as_tensor(
            pinned_mask, dtype=torch.bool, device=self._device
        ).reshape(-1)
        threshold = torch.as_tensor(
            release_threshold, dtype=torch.float32, device=self._device
        ).reshape(-1)
        expected = self.worlds_per_rank
        if tuple(slots.shape) != (expected,):
            raise ValueError("One pinned target slot is required per world.")
        if tuple(offsets.shape) != (expected, 3):
            raise ValueError(f"ee_offsets must have shape ({expected}, 3).")
        if tuple(mask.shape) != (expected,):
            raise ValueError("pinned_mask must contain one boolean per world.")
        if tuple(threshold.shape) != (expected,):
            raise ValueError("release_threshold must contain one value per world.")
        self._pinned_target_slots.copy_(slots)
        self._pinned_ee_offsets.copy_(offsets)
        self._pinned_mask.copy_(mask)
        self._pinned_release_threshold.copy_(threshold)

    def pinned_object_mask(self) -> Any:
        return self._pinned_mask

    def set_object_catalogs(self, catalog_ids: Any) -> None:
        torch = self.torch
        ids = torch.as_tensor(
            catalog_ids, dtype=torch.int64, device=self._device
        )
        if tuple(ids.shape) != (self.worlds_per_rank, 4):
            raise ValueError(
                f"catalog_ids must have shape ({self.worlds_per_rank}, 4), "
                f"got {tuple(ids.shape)}."
            )
        if bool((ids < INACTIVE_CATALOG_ID).any()) or bool(
            (ids >= len(ACTIVE_CDPR_CATALOGS)).any()
        ):
            raise KeyError(
                f"Catalog ids must be -1 or [0, {len(ACTIVE_CDPR_CATALOGS)})."
            )
        table_index = ids + 1
        flat_geoms = self._slot_geom_ids.reshape(-1)
        self._model_geom_size[:, flat_geoms] = self._catalog_geom_size[
            table_index
        ].reshape(self.worlds_per_rank, -1, 3)
        self._model_geom_pos[:, flat_geoms] = self._catalog_geom_pos[
            table_index
        ].reshape(self.worlds_per_rank, -1, 3)
        self._model_geom_quat[:, flat_geoms] = self._catalog_geom_quat[
            table_index
        ].reshape(self.worlds_per_rank, -1, 4)
        self._model_geom_rgba[:, flat_geoms] = self._catalog_geom_rgba[
            table_index
        ].reshape(self.worlds_per_rank, -1, 4)

        size = self._model_geom_size[:, flat_geoms]
        position = self._model_geom_pos[:, flat_geoms]
        rbound = torch.linalg.vector_norm(size, dim=-1)
        capsule_index = PRIMITIVE_NAMES.index("capsule_0")
        local_capsule_columns = (
            torch.arange(4, dtype=torch.int64, device=self._device)
            * len(PRIMITIVE_NAMES)
            + capsule_index
        )
        rbound[:, local_capsule_columns] = (
            size[:, local_capsule_columns, 0]
            + size[:, local_capsule_columns, 1]
        )
        self._model_geom_rbound[:, flat_geoms] = rbound
        self._model_geom_aabb[:, flat_geoms, 0] = position
        self._model_geom_aabb[:, flat_geoms, 1] = rbound[..., None].expand(
            -1, -1, 3
        )

        self._model_body_mass[:, self._object_body_ids_tensor] = (
            self._catalog_body_mass[table_index]
        )
        self._model_body_inertia[:, self._object_body_ids_tensor] = (
            self._catalog_body_inertia[table_index]
        )
        self._object_rest_height.copy_(self._catalog_rest_height[table_index])
        self._catalog_ids.copy_(ids)
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.set_const(self.model, self.data)
            self.mjw.forward(self.model, self.data)

    def set_visual_variants(
        self,
        texture_variant_ids: Any,
        background_rgba: Any,
        gripper_shade: Any,
    ) -> None:
        torch = self.torch
        variants = torch.as_tensor(
            texture_variant_ids, dtype=torch.int64, device=self._device
        ).reshape(-1)
        background = torch.as_tensor(
            background_rgba, dtype=torch.float32, device=self._device
        )
        shade = torch.as_tensor(
            gripper_shade, dtype=torch.float32, device=self._device
        ).reshape(-1)
        if tuple(variants.shape) != (self.worlds_per_rank,):
            raise ValueError("One desk texture variant is required per world.")
        if tuple(background.shape) != (self.worlds_per_rank, 4):
            raise ValueError("background_rgba must have shape [world, 4].")
        if tuple(shade.shape) != (self.worlds_per_rank,):
            raise ValueError("One gripper shade is required per world.")
        if bool((variants < 0).any()) or bool((variants >= 7).any()):
            raise ValueError("Desk texture variants must be in [0, 7).")
        self._model_geom_matid[:, self.desk_geom_id] = (
            self._desk_material_ids_tensor.index_select(0, variants)
        )
        self._model_geom_rgba[:, self.background_geom_id] = background.clamp(
            0.0, 1.0
        )
        shade_rgb = shade.clamp(0.0, 1.0)[:, None, None].expand(
            -1, len(self.gripper_surface_geom_ids), 3
        )
        self._model_geom_rgba[
            :, self._gripper_surface_geom_ids_tensor, :3
        ] = shade_rgb

    def set_end_effector_poses(
        self,
        positions: Any,
        yaws: Any,
        *,
        zero_velocity: bool = True,
    ) -> None:
        """Tensorized equivalent of CPU teleport + hold_current_pose."""

        torch = self.torch
        position = torch.as_tensor(
            positions, dtype=torch.float32, device=self._device
        )
        yaw = torch.as_tensor(
            yaws, dtype=torch.float32, device=self._device
        ).reshape(-1)
        if tuple(position.shape) != (self.worlds_per_rank, 3):
            raise ValueError("End-effector reset positions must have shape [world, 3].")
        if tuple(yaw.shape) != (self.worlds_per_rank,):
            raise ValueError("End-effector reset yaws must have shape [world].")
        position = torch.maximum(
            torch.minimum(position, self._workspace_max), self._workspace_min
        )
        self._qpos[:, self.ee_free_qadr : self.ee_free_qadr + 3] = position
        self._qpos[:, self.yaw_qadr] = yaw.clamp(
            self._yaw_limits[0], self._yaw_limits[1]
        )
        if zero_velocity:
            self._qvel[
                :, self.ee_free_dofadr : self.ee_free_dofadr + 6
            ] = 0.0
            yaw_dofadr = int(self.host_model.jnt_dofadr[self.yaw_joint_id])
            self._qvel[:, yaw_dofadr] = 0.0

        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.forward(self.model, self.data)
            for _ in range(8):
                current_q = self._qpos.index_select(
                    1, self._slider_qadr_tensor
                )
                current_lengths = self._ten_length.index_select(
                    1, self._tendon_ids_tensor
                )
                targets = current_q + (
                    self._tendon_upper[None, :] - current_lengths
                ) / self._dlength_dq
                targets = torch.maximum(
                    torch.minimum(targets, self._slider_ctrl_limits[:, 1]),
                    self._slider_ctrl_limits[:, 0],
                )
                self._qpos[:, self._slider_qadr_tensor] = targets
                self._qvel[:, list(self.slider_dofadr)] = 0.0
                self._ctrl[:, self._slider_actuator_ids_tensor] = targets
                self.mjw.forward(self.model, self.data)

        self._controller_target.copy_(self._xpos[:, self.ee_body_id])
        self._controller_yaw.copy_(self._qpos[:, self.yaw_qadr])
        self._controller_attach_offset.copy_(
            self._site_xpos[:, self.topcenter_site_id]
            - self._xpos[:, self.ee_body_id]
        )
        self._controller_prev_lengths.copy_(
            self._ten_length.index_select(1, self._tendon_ids_tensor)
        )

    def set_gripper_openings(self, openings: Any) -> None:
        opening = self.torch.as_tensor(
            openings, dtype=self.torch.float32, device=self._device
        ).reshape(-1)
        if tuple(opening.shape) != (self.worlds_per_rank,):
            raise ValueError("One normalized gripper reset opening is required per world.")
        opening = opening.clamp(0.0, 1.0)
        joint_min, joint_max = self._finger_limits
        self._qpos[:, self.finger_qadr] = joint_min + opening * (
            joint_max - joint_min
        )
        self._qvel[:, self.finger_dofadr] = 0.0
        ctrl_min, ctrl_max = self._gripper_ctrl_limits
        self._ctrl[:, self.gripper_actuator_id] = ctrl_min + opening * (
            ctrl_max - ctrl_min
        )
        self._controller_gripper.copy_(opening)
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.forward(self.model, self.data)

    def set_free_body_poses(
        self,
        body_ids: Any,
        positions: Any,
        quaternions: Any | None = None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        torch = self.torch
        ids = tuple(int(value) for value in body_ids)
        pos = torch.as_tensor(
            positions, dtype=torch.float32, device=self._device
        )
        if tuple(pos.shape) != (self.worlds_per_rank, len(ids), 3):
            raise ValueError(
                f"positions must have shape ({self.worlds_per_rank}, {len(ids)}, 3), "
                f"got {tuple(pos.shape)}."
            )
        if quaternions is None:
            quat = torch.zeros(
                (self.worlds_per_rank, len(ids), 4),
                dtype=torch.float32,
                device=self._device,
            )
            quat[..., 0] = 1.0
        else:
            quat = torch.as_tensor(
                quaternions, dtype=torch.float32, device=self._device
            )
            if tuple(quat.shape) != (self.worlds_per_rank, len(ids), 4):
                raise ValueError(
                    f"quaternions must have shape "
                    f"({self.worlds_per_rank}, {len(ids)}, 4)."
                )
            quat = quat / torch.linalg.vector_norm(
                quat, dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
        for column, body_id in enumerate(ids):
            if body_id not in self.object_body_ids:
                raise ValueError(
                    f"Body id {body_id} is not a fixed MJWarp object slot."
                )
            slot = self.object_body_ids.index(body_id)
            qadr = self.object_qadr[slot]
            self._qpos[:, qadr : qadr + 3] = pos[:, column]
            self._qpos[:, qadr + 3 : qadr + 7] = quat[:, column]
            if zero_velocity:
                dofadr = self.object_dofadr[slot]
                self._qvel[:, dofadr : dofadr + 6] = 0.0
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.forward(self.model, self.data)

    def set_target_body_positions(
        self,
        target_slots: Any,
        positions: Any,
        world_mask: Any,
        *,
        zero_velocity: bool = True,
    ) -> None:
        """Move only selected target bodies, leaving every other free body untouched."""
        torch = self.torch
        slots = torch.as_tensor(
            target_slots, dtype=torch.int64, device=self._device
        ).reshape(-1)
        pos = torch.as_tensor(
            positions, dtype=torch.float32, device=self._device
        )
        mask = torch.as_tensor(
            world_mask, dtype=torch.bool, device=self._device
        ).reshape(-1)
        expected = self.worlds_per_rank
        if tuple(slots.shape) != (expected,):
            raise ValueError("One target object slot is required per world.")
        if tuple(pos.shape) != (expected, 3):
            raise ValueError(f"positions must have shape ({expected}, 3).")
        if tuple(mask.shape) != (expected,):
            raise ValueError("world_mask must contain one boolean per world.")

        rows = torch.arange(expected, dtype=torch.int64, device=self._device)
        qadr = self._object_qadr_tensor.index_select(0, slots)
        for axis in range(3):
            current = self._qpos[rows, qadr + axis]
            self._qpos[rows, qadr + axis] = torch.where(
                mask, pos[:, axis], current
            )
        if zero_velocity:
            dofadr = self._object_dofadr_tensor.index_select(0, slots)
            for axis in range(6):
                current = self._qvel[rows, dofadr + axis]
                self._qvel[rows, dofadr + axis] = torch.where(
                    mask, torch.zeros_like(current), current
                )
        with self.wp.ScopedDevice(str(self._device)):
            self.mjw.forward(self.model, self.data)

    def export_worlds(self, world_indices: Sequence[int]) -> list[dict[str, Any]]:
        observations = self.low_dim_observations()
        output: list[dict[str, Any]] = []
        for index in world_indices:
            if index < 0 or index >= self.worlds_per_rank:
                raise IndexError(index)
            output.append(
                {
                    "world_index": int(index),
                    "qpos": self._qpos[index].detach().cpu().numpy().copy(),
                    "qvel": self._qvel[index].detach().cpu().numpy().copy(),
                    "ee_position": observations.ee_position[index]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy(),
                    "object_positions": observations.object_positions[index]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy(),
                    "tendon_lengths": observations.tendon_lengths[index]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy(),
                    "catalog_ids": self._catalog_ids[index]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy(),
                }
            )
        return output

    def capacity_status(self) -> dict[str, int]:
        """Update-level diagnostic; deliberately synchronizes one scalar."""

        contacts = int(self._nacon[0].detach().cpu().item())
        max_constraints = int(self._nefc.max().detach().cpu().item())
        contact_capacity = int(self.data.naconmax)
        constraint_capacity = int(self.data.njmax)
        return {
            "contacts": contacts,
            "max_constraints_per_world": max_constraints,
            "contact_capacity": int(self.data.naconmax),
            "constraint_capacity_per_world": int(self.data.njmax),
            "contact_overflow": int(contacts > contact_capacity),
            "constraint_overflow": int(max_constraints > constraint_capacity),
        }

    def metadata(self) -> dict[str, Any]:
        xml_path = Path(self.config.xml_path or "").resolve()
        return {
            "backend": "mjlab_mjwarp",
            "versions": package_versions(),
            "worlds_per_rank": self.worlds_per_rank,
            "groups_per_rank": int(self.config.groups_per_rank),
            "grpo_group_size": int(self.config.grpo_group_size),
            "physics_substeps_per_action": self.config.physics_substeps,
            "physics_dtype": "float32",
            "controller_implementation": "batched_torch_cdpr_v1",
            "action_step_xyz": float(self.config.action_step_xyz),
            "action_step_yaw": float(self.config.action_step_yaw),
            "action_step_gripper": float(self.config.action_step_gripper),
            "lock_non_commanded_axes": bool(
                self.config.lock_non_commanded_axes
            ),
            "lock_non_commanded_axes_threshold": float(
                self.config.lock_non_commanded_axes_threshold
            ),
            "device": str(self._device),
            "xml_path": str(xml_path),
            "xml_sha256": _mjcf_tree_sha256(xml_path),
            "nconmax_per_world": int(self.config.nconmax),
            "njmax_per_world": int(self.config.njmax),
            "nccdmax_per_world": self.config.nccdmax,
            "render_width": int(self.config.render_width),
            "render_height": int(self.config.render_height),
            "camera_order": ["overview", "ee_camera", "ee_camera"],
            "object_slots": 4,
            "object_catalogs": list(ACTIVE_CDPR_CATALOGS),
            "gpu_nondeterministic": True,
        }

    def close(self) -> None:
        self.render_context = None
        self._overview_rgb_wp = None
        self._wrist_rgb_wp = None
