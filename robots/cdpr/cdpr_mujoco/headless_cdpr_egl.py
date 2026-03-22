import os
import time
from datetime import datetime

import mujoco as mj
import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

# Try to import EGL for true headless rendering
try:
    from mujoco.egl import GLContext
    EGL_AVAILABLE = True
except ImportError:
    EGL_AVAILABLE = False
    print("EGL not available, falling back to software rendering")

def _id(model, objtype, name):
    return mj.mj_name2id(model, objtype, name)


def _solve_slider_preload_targets(
    current_slider_qpos,
    current_tendon_lengths,
    tendon_upper_limits,
    dlength_dq,
):
    current_slider_qpos = np.asarray(current_slider_qpos, dtype=float).reshape(4)
    current_tendon_lengths = np.asarray(current_tendon_lengths, dtype=float).reshape(4)
    tendon_upper_limits = np.asarray(tendon_upper_limits, dtype=float).reshape(4)
    dlength_dq = np.asarray(dlength_dq, dtype=float).reshape(4)

    if np.any(np.abs(dlength_dq) < 1e-8):
        raise ValueError("Cannot solve slider preload with singular tendon Jacobian.")

    return current_slider_qpos + (tendon_upper_limits - current_tendon_lengths) / dlength_dq


class HeadlessCDPRController:
    def __init__(
        self,
        frame_points,
        initial_pos=np.array([0, 0, 0.40]),
        attach_point_offset=np.array([0.0, 0.0, 0.08]),
    ):
        self.frame_points = frame_points
        self.pos = initial_pos.astype(float)
        self.attach_point_offset = np.asarray(attach_point_offset, dtype=float).reshape(3)
        self.Kp = 100
        self.Kd = 130
        self.threshold = 0.03
        self.prev_lengths = np.zeros(4)
        self.dt = 1.0/60.0
        self.dlength_dq = np.ones(4, dtype=float)
        self.slider_q_per_length = np.ones(4, dtype=float)
        self.has_tendon_model = False

    def set_attach_point_offset(self, attach_point_offset):
        self.attach_point_offset = np.asarray(attach_point_offset, dtype=float).reshape(3).copy()

    def _attach_point_position(self, pos):
        return np.asarray(pos, dtype=float).reshape(3) + self.attach_point_offset

    def inverse_kinematics(self, pos=None):
        if pos is None:
            pos = self.pos
        attach_pos = self._attach_point_position(pos)
        diffs = attach_pos[None, :] - self.frame_points
        return np.linalg.norm(diffs, axis=1)

    def update_position(self, new_pos):
        self.pos = new_pos.copy()

    def configure_tendon_model(self, dlength_dq):
        dlength_dq = np.asarray(dlength_dq, dtype=float).reshape(4)
        if np.any(np.abs(dlength_dq) < 1e-8):
            raise ValueError("Tendon calibration has singular dlength_dq values.")

        self.dlength_dq = dlength_dq.copy()
        self.slider_q_per_length = 1.0 / dlength_dq
        self.has_tendon_model = True

    def compute_control(
        self,
        target_pos,
        current_ee_pos,
        current_slider_qpos=None,
        current_tendon_lengths=None,
    ):
        """
        Compute desired slider target positions given a target end-effector position.
        Optionally keep current joint qpos if target ≈ current.
        """
        self.update_position(current_ee_pos)
        cur_lengths = self.inverse_kinematics(current_ee_pos)
        target_lengths = self.inverse_kinematics(target_pos)

        if current_slider_qpos is not None:
            current_slider_qpos = np.asarray(current_slider_qpos, dtype=float).reshape(4)
        if current_tendon_lengths is not None:
            current_tendon_lengths = np.asarray(current_tendon_lengths, dtype=float).reshape(4)

        # If target is basically current → hold steady
        if np.linalg.norm(target_pos - current_ee_pos) < 1e-6 and current_slider_qpos is not None:
            return current_slider_qpos.copy()

        # Preserve the current XML-defined cable pretension and only compensate
        # for the geometric change implied by the target EE position.
        if current_slider_qpos is not None and self.has_tendon_model:
            return current_slider_qpos + self.slider_q_per_length * (cur_lengths - target_lengths)

        return current_slider_qpos.copy() if current_slider_qpos is not None else np.zeros(4, dtype=float)



class HeadlessCDPRSimulation:
    def __init__(self, xml_path, output_dir="trajectory_videos"):
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.model = None
        self.data = None
        self.gl_context = None
        self._glfw_window = None

        # CDPR frame anchor points (must match XML)
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535,  0.755, 1.309],
            [-0.755, 0.525, 1.309],
        ], dtype=float)

        self.controller = HeadlessCDPRController(self.frame_points)
        self.target_pos = np.array([0, 0, 0.40], dtype=float)
        self.gripper_min = 0.0
        self.gripper_max = 0.03
        self.yaw_min = -np.pi
        self.yaw_max = np.pi
        self.jnt_finger_l_qadr = None

        # Recording
        self.overview_frames = []
        self.ee_camera_frames = []
        self.frame_capture_timestamps = []
        self.trajectory_data = []

        os.makedirs(output_dir, exist_ok=True)

    def initialize(self):
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        self.model.opt.timestep = self.controller.dt
        
        self.jnt_yaw = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ee_yaw")
        self.jnt_yaw_qadr = self.model.jnt_qposadr[self.jnt_yaw]  # index into qpos for yaw angle

        # IDs we’ll need
        self.body_ee   = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "ee_base")  # EE body
        self.site_topcenter = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "topcenter")
        self.cam_id    = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")

        # Sliders (cables)
        self.act_sliders = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_1_pos"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_2_pos"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_3_pos"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_4_pos"),
        ]
        self.slider_joint_ids = [
        mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"slider_{k}") for k in range(1,5)
        ]
        self.slider_qadr = [self.model.jnt_qposadr[jid] for jid in self.slider_joint_ids]
        self.tendon_idx = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_TENDON, f"rope_{k}") for k in range(1, 5)
        ]
   
        # Tool actuators
        self.act_yaw     = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_ee_yaw")
        self.act_gripper = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        self.jnt_finger_l = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "finger_l")
        if self.jnt_finger_l != -1:
            self.jnt_finger_l_qadr = int(self.model.jnt_qposadr[self.jnt_finger_l])

        if self.jnt_yaw != -1 and bool(self.model.jnt_limited[self.jnt_yaw]):
            y_lo, y_hi = self.model.jnt_range[self.jnt_yaw]
            self.yaw_min = float(min(y_lo, y_hi))
            self.yaw_max = float(max(y_lo, y_hi))
        else:
            self.yaw_min = float(-np.pi)
            self.yaw_max = float(np.pi)

        # Read gripper limits from the active model so wrappers and base XML stay consistent.
        if self.act_gripper != -1 and bool(self.model.actuator_ctrllimited[self.act_gripper]):
            g_lo, g_hi = self.model.actuator_ctrlrange[self.act_gripper]
        elif self.jnt_finger_l != -1 and bool(self.model.jnt_limited[self.jnt_finger_l]):
            g_lo, g_hi = self.model.jnt_range[self.jnt_finger_l]
        else:
            g_lo, g_hi = 0.0, 0.03
        self.gripper_min = float(min(g_lo, g_hi))
        self.gripper_max = float(max(g_lo, g_hi))
        if self.gripper_max <= self.gripper_min:
            self.gripper_min, self.gripper_max = 0.0, 0.03

        # Target object (body + geom). If the geom is unnamed, we’ll find it by body.
        # self.body_target = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "target_object")
        # --- Detect the object body robustly ---
                # --- Detect the object body robustly ---
                # --- Detect the object body robustly ---
        def _detect_object_body(model):
            """
            Prefer placed LIBERO objects:
              - bodies whose name looks like 'p0_<something>', 'p1_<something>', ...
            Fallbacks:
              - 'target_object', 'object'
              - any other FREE-joint non-robot body
            """
            robot_prefixes = ("rotor_", "slider_", "ee_", "camera_", "yaw_frame",
                              "ee_platform", "finger_")

            placed_candidates = []   # (bid, name) for p[0-9]_*
            free_candidates   = []   # (bid, name) all other free-jointed non-robot

            for bid in range(model.nbody):
                nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
                if not nm or nm == "world":
                    continue
                if any(nm.startswith(p) for p in robot_prefixes):
                    continue

                # does it have a FREE joint?
                jn, ja = model.body_jntnum[bid], model.body_jntadr[bid]
                has_free = any(model.jnt_type[ja + k] == mj.mjtJoint.mjJNT_FREE
                               for k in range(jn))
                if has_free:
                    free_candidates.append((bid, nm))

                # placed LIBERO objects from cdpr_scene_switcher:
                # names like 'p0_object', 'p0_ketchup', ...
                if len(nm) >= 3 and nm[0] == "p" and nm[1].isdigit() and nm[2] == "_":
                    placed_candidates.append((bid, nm))

            # 1) Prefer placed objects even if they *don’t* have a free joint
            if placed_candidates:
                return placed_candidates[0]

            # 2) If no placed objects, try explicit names
            for name in ("target_object", "object"):
                bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                if bid != -1:
                    return bid, name

            # 3) Last resort: any free-joint non-robot body
            if free_candidates:
                return free_candidates[0]

            return -1, None

        self.body_target, self.body_target_name = _detect_object_body(self.model)
        if self.body_target == -1:
            # helpful error
            # import mujoco as mj
            names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
            raise RuntimeError(f"Could not detect object body. Bodies: {names[:30]}{' ...' if len(names)>30 else ''}")

        try:
            self.geom_target = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "target_box")
        except Exception:
            # Fallback: first geom belonging to target body
            self.geom_target = int(np.where(self.model.geom_bodyid == self.body_target)[0][0])
        
        # DEBUG: check object orientation
        if self.body_target >= 0:
            xquat = self.data.xquat[self.body_target]  # [w, x, y, z] in world
            print(f"[debug] object body '{self.body_target_name}' world xquat={xquat}")
            
        self._setup_offscreen_rendering()
        mj.mj_forward(self.model, self.data)
        self._sync_controller_geometry_from_state()
        # Seed target to current EE pose (so your higher-level controller doesn’t pull elsewhere).
        self.target_pos = self.get_end_effector_position().copy()
        self._match_sliders_to_ee_lengths(max_iter=12, tol=1e-6)
        self.target_pos = self.get_end_effector_position().copy()
        self.controller.prev_lengths = self.get_cable_lengths().copy()
        print("Headless CDPR Simulation initialized successfully!")
        print(f"Using {'EGL' if EGL_AVAILABLE else 'software'} rendering")

    def _setup_offscreen_rendering(self):
        self.overview_cam = mj.MjvCamera()
        self.ee_cam = mj.MjvCamera()

        self.overview_cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.overview_cam.lookat[:] = np.array([0.0, 0.0, 0.10])
        self.overview_cam.distance = 1.5
        self.overview_cam.azimuth = 90
        self.overview_cam.elevation = -30

        self.ee_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.ee_cam.fixedcamid = self.cam_id

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)

        self.offwidth, self.offheight = 640, 480
        self.offviewport = mj.MjrRect(0, 0, self.offwidth, self.offheight)

        # --------- IMPORTANT: ensure an active GL context exists ----------
        if EGL_AVAILABLE:
            self.gl_context = GLContext(max_width=self.offwidth, max_height=self.offheight)
            self.gl_context.make_current()
        else:
            # GLFW fallback for macOS / no EGL
            from mujoco.glfw import glfw
            if not glfw.init():
                raise RuntimeError("GLFW init failed (needed for rendering on this machine).")

            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden window
            self._glfw_window = glfw.create_window(64, 64, "offscreen", None, None)
            if not self._glfw_window:
                raise RuntimeError("Failed to create GLFW window for GL context.")
            glfw.make_context_current(self._glfw_window)
            glfw.swap_interval(0)
        # ----------------------------------------------------------------

        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # If OFFSCREEN fails on some drivers, you can flip this to WINDOW as fallback.
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

    def capture_frame(self, camera, camera_name):
        try:
            if EGL_AVAILABLE and self.gl_context is not None:
                self.gl_context.make_current()
            mj.mjv_updateScene(self.model, self.data, self.opt, None, camera,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(self.offviewport, self.scene, self.context)
            rgb = np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
            depth = np.zeros((self.offheight, self.offwidth), dtype=np.float32)
            mj.mjr_readPixels(rgb, depth, self.offviewport, self.context)
            return np.flipud(rgb)
        except Exception as e:
            print(f"Error capturing frame from {camera_name}: {e}")
            return np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)

    def get_cable_attach_position(self):
        return self.data.site_xpos[self.site_topcenter].copy()

    def get_cable_lengths(self):
        return np.array([self.data.ten_length[idx] for idx in self.tendon_idx], dtype=float)

    def _sync_controller_geometry_from_state(self):
        ee_pos = self.get_end_effector_position()
        attach_pos = self.get_cable_attach_position()
        self.controller.set_attach_point_offset(attach_pos - ee_pos)

    def _estimate_slider_length_jacobian(self, dq=1e-4):
        mj.mj_forward(self.model, self.data)
        jac = np.zeros(4, dtype=float)
        for idx, (qadr, tendon_idx) in enumerate(zip(self.slider_qadr, self.tendon_idx)):
            q0 = float(self.data.qpos[qadr])
            L0 = float(self.data.ten_length[tendon_idx])
            self.data.qpos[qadr] = q0 + float(dq)
            mj.mj_forward(self.model, self.data)
            L1 = float(self.data.ten_length[tendon_idx])
            jac[idx] = (L1 - L0) / float(dq)
            self.data.qpos[qadr] = q0
            mj.mj_forward(self.model, self.data)
        return jac

    def _configure_controller_tendon_model(self, dlength_dq=None):
        if dlength_dq is None:
            dlength_dq = self._estimate_slider_length_jacobian()
        self.controller.configure_tendon_model(dlength_dq=dlength_dq)

    def _slider_ctrl_limits(self):
        limits = np.full((4, 2), np.array([-np.inf, np.inf], dtype=float), dtype=float)
        for idx, act_id in enumerate(self.act_sliders):
            if act_id < 0:
                continue
            if bool(self.model.actuator_ctrllimited[act_id]):
                limits[idx] = self.model.actuator_ctrlrange[act_id]
        return limits

    def _set_slider_targets(self, slider_targets, *, update_qpos=False, zero_velocity=False):
        targets = np.asarray(slider_targets, dtype=float).reshape(4).copy()
        ctrl_limits = self._slider_ctrl_limits()
        targets = np.clip(targets, ctrl_limits[:, 0], ctrl_limits[:, 1])

        for idx, (target, qadr, act_id, joint_id) in enumerate(
            zip(targets, self.slider_qadr, self.act_sliders, self.slider_joint_ids)
        ):
            if update_qpos:
                self.data.qpos[qadr] = float(target)
                if zero_velocity:
                    dofadr = int(self.model.jnt_dofadr[joint_id])
                    if 0 <= dofadr < self.model.nv:
                        self.data.qvel[dofadr] = 0.0
            if act_id >= 0:
                self.data.ctrl[act_id] = float(target)

        return targets

    def _tendon_upper_limits(self):
        return np.array([self.model.tendon_range[idx, 1] for idx in self.tendon_idx], dtype=float)

    def _calibrate_slider_preload(self, *, max_iter=8, tol=1e-6):
        last_targets = np.array([self.data.qpos[idx] for idx in self.slider_qadr], dtype=float)
        for _ in range(max(1, int(max_iter))):
            mj.mj_forward(self.model, self.data)
            current_slider_qpos = np.array([self.data.qpos[idx] for idx in self.slider_qadr], dtype=float)
            current_tendon_lengths = self.get_cable_lengths()
            dlength_dq = self._estimate_slider_length_jacobian()
            targets = _solve_slider_preload_targets(
                current_slider_qpos=current_slider_qpos,
                current_tendon_lengths=current_tendon_lengths,
                tendon_upper_limits=self._tendon_upper_limits(),
                dlength_dq=dlength_dq,
            )
            last_targets = self._set_slider_targets(targets, update_qpos=True, zero_velocity=True)
            if float(np.max(np.abs(last_targets - current_slider_qpos))) <= float(tol):
                break

        mj.mj_forward(self.model, self.data)
        return last_targets

    def hold_current_pose(self, warm_steps=0):
        """
        Re-solve slider preload for the CURRENT EE pose so the system starts from
        a static cable-supported state instead of preserving a transient.
        """
        ee_now = self.get_end_effector_position()
        self.target_pos = ee_now.copy()
        self._sync_controller_geometry_from_state()
        self._calibrate_slider_preload(max_iter=8, tol=1e-6)
        self.controller.prev_lengths = self.get_cable_lengths().copy()
        for _ in range(int(warm_steps)):
            mj.mj_step(self.model, self.data)
        settled_ee = self.get_end_effector_position()
        self.target_pos = settled_ee.copy()
        self._sync_controller_geometry_from_state()
        self._calibrate_slider_preload(max_iter=8, tol=1e-6)
        self.controller.prev_lengths = self.get_cable_lengths().copy()
        self._configure_controller_tendon_model()
            
    def _neutralize_position_actuators(self):
        """
        For every position actuator, set ctrl = current joint qpos so nothing
        moves at t=0. This is crucial for slider_* (the cable winches).
        """
        # map joint -> qpos index
        jnt_qposadr = self.model.jnt_qposadr
        for i in range(self.model.nu):  # actuators
            if self.model.actuator_trntype[i] != mj.mjtTrn.mjTRN_JOINT:
                continue
            # this actuator targets a joint position
            j = self.model.actuator_trnid[i, 0]           # joint id
            if j < 0: 
                continue
            qadr = jnt_qposadr[j]                         # index into qpos
            # only safe if the actuator is mjtGain::position (yours are <position .../>)
            self.data.ctrl[i] = float(self.data.qpos[qadr])

    def _match_sliders_to_ee_lengths(self, max_iter=12, tol=1e-6):
        """
        Solve the slider preload so the current EE pose sits at the tendon upper
        limits, then refresh the Jacobian used for incremental control.
        """
        self._calibrate_slider_preload(max_iter=max_iter, tol=tol)
        self._neutralize_position_actuators()
        self._configure_controller_tendon_model()


    def get_end_effector_position(self):
        return self.data.xpos[self.body_ee].copy()
    
    def get_object_body_name(self):
        return getattr(self, "body_target_name", None)

    def get_object_position(self):
        if getattr(self, "body_target", -1) == -1:
            raise RuntimeError("Object body not set")
        return self.data.xpos[self.body_target].copy()

    # Backward-compat alias (so old code still works)
    def get_target_position(self):
        return self.get_object_position()

    def get_target_position(self):
        return self.data.xpos[self.body_target].copy()
    
    def set_target_position(self, target_pos):
        target_pos = np.asarray(target_pos, dtype=float)
        if np.all((-1.309 <= target_pos) & (target_pos <= 1.309)):
            self.target_pos = target_pos
            ee_pos = self.get_end_effector_position()
            self._sync_controller_geometry_from_state()
            self.controller.prev_lengths = self.controller.inverse_kinematics(ee_pos)
            return True
        return False

    def check_success(self):
        ee_pos = self.get_end_effector_position()
        return np.linalg.norm(ee_pos - self.target_pos) < self.controller.threshold

    # === Gripper / yaw helpers ===
    def get_gripper_opening(self):
        if self.jnt_finger_l_qadr is not None:
            return float(self.data.qpos[self.jnt_finger_l_qadr])
        # Fallback if finger joint is absent.
        return float(self.data.ctrl[self.act_gripper])

    def set_gripper(self, opening_m):
        """Set desired opening for left finger (right follows)."""
        if self.act_gripper == -1:
            return
        opening = float(np.clip(opening_m, self.gripper_min, self.gripper_max))
        self.data.ctrl[self.act_gripper] = opening

    def open_gripper(self):
        self.set_gripper(self.gripper_max)

    def close_gripper(self):
        self.set_gripper(self.gripper_min)

    def get_yaw(self):
        return float(self.data.qpos[self.jnt_yaw_qadr])

    def set_yaw(self, yaw_rad):
        if self.act_yaw == -1:
            return
        yaw_cmd = float(np.clip(yaw_rad, self.yaw_min, self.yaw_max))
        self.data.ctrl[self.act_yaw] = yaw_cmd

    def record_trajectory_step(self):
        ee_pos = self.get_end_effector_position()
        slider_q = [self.data.qpos[idx] for idx in self.slider_qadr]
        cable_lengths = self.get_cable_lengths()
        self.trajectory_data.append({
            'timestamp': self.data.time,
            'ee_position': ee_pos.copy(),
            'target_position': self.target_pos.copy(),
            'slider_positions': slider_q.copy(),
            'cable_lengths': cable_lengths.copy(),
            'control_signals': self.data.ctrl.copy() if self.model.nu > 0 else np.zeros(0),
        })
        
    def goto(self, world_xyz, max_steps=900, tol=0.03, capture_every_n=3):
        """Drive EE to world_xyz with your cable controller."""
        self.set_target_position(np.asarray(world_xyz, dtype=float))
        steps, total = 0, 0
        while steps < max_steps:
            capture = (total % capture_every_n == 0)
            self.run_simulation_step(capture_frame=capture)
            total += 1; steps += 1
            if np.linalg.norm(self.get_end_effector_position() - self.target_pos) < tol:
                return True, steps
        return False, steps

    def has_finger_contact(self):
        # any contact involving target geom and a finger geom
        finger_geoms = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_l_link"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_r_link"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_l_tip"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_r_tip"),
        ]
        tgt = self.geom_target
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == tgt and c.geom2 in finger_geoms) or (c.geom2 == tgt and c.geom1 in finger_geoms):
                return True
        return False

    def lifted_enough(self, z_start, min_rise=0.02):
        z_now = self.get_target_position()[2]
        return (z_now - z_start) >= min_rise
    
    def _angle_wrap(self, a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2*np.pi) - np.pi

    def rotate_to(self, yaw_target, duration=1.0, hold_xyz=None, capture_every_n=2):
        """
        Smoothly rotate yaw to yaw_target [rad] over 'duration' seconds.
        If hold_xyz is provided, we keep position controller targeting that XYZ while rotating.
        """
        dt = float(self.controller.dt)
        steps = max(1, int(round(duration / dt)))

        # Read current yaw directly from joint qpos
        yaw_now = float(self.data.qpos[self.jnt_yaw_qadr])
        # shortest arc
        dyaw = self._angle_wrap(yaw_target - yaw_now)

        # Fix the translational target if requested
        if hold_xyz is not None:
            self.set_target_position(np.asarray(hold_xyz, dtype=float))

        for k in range(steps):
            alpha = (k + 1) / steps
            yaw_cmd = yaw_now + dyaw * alpha
            self.set_yaw(yaw_cmd)
            capture = (k % capture_every_n == 0)
            self.run_simulation_step(capture_frame=capture)


    def run_simulation_step(self, capture_frame=True):
        ee_pos = self.get_end_effector_position()
        # control_signals = self.controller.compute_control(self.target_pos, ee_pos)
        # slider_qpos = [self.data.qpos[i] for i in range(4)]  # assuming sliders are first 4
        # control_signals = self.controller.compute_control(
        #     self.target_pos, ee_pos, current_slider_qpos=slider_qpos
        # )

        # # Apply slider targets by actuator index
        # for j, act_id in enumerate(self.act_sliders):
        #     self.data.ctrl[act_id] = control_signals[j]

        # mj.mj_step(self.model, self.data)
        # ✅ read the actual slider qpos (not qpos[0:4])
        slider_qpos = [self.data.qpos[idx] for idx in self.slider_qadr]
        tendon_lengths = self.get_cable_lengths()

        control_signals = self.controller.compute_control(
            self.target_pos,
            ee_pos,
            current_slider_qpos=slider_qpos,
            current_tendon_lengths=tendon_lengths,
        )
        if not np.all(np.isfinite(control_signals)):
            control_signals = np.asarray(slider_qpos, dtype=float)
        self._set_slider_targets(control_signals)

        mj.mj_step(self.model, self.data)

        if capture_frame:
            self.overview_frames.append(self.capture_frame(self.overview_cam, "overview"))
            self.ee_camera_frames.append(self.capture_frame(self.ee_cam, "ee_camera"))
            self.frame_capture_timestamps.append(float(self.data.time))

        self.record_trajectory_step()

    def run_trajectory(self, target_positions, trajectory_name="trajectory",
                       max_steps_per_target=600, capture_every_n=3):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = os.path.join(self.output_dir, f"{trajectory_name}_{timestamp}")
        os.makedirs(trajectory_dir, exist_ok=True)
        print(f"Starting trajectory: {trajectory_name}")

        self.overview_frames, self.ee_camera_frames, self.frame_capture_timestamps, self.trajectory_data = [], [], [], []
        total_steps = 0
        trajectory_success = True

        # Demo: open before motion
        self.open_gripper()
        for i, target_pos in enumerate(target_positions):
            print(f"Moving to target {i+1}/{len(target_positions)}: {target_pos}")
            if not self.set_target_position(target_pos):
                print(f"Invalid target position: {target_pos}")
                trajectory_success = False
                break

            steps_for_target = 0
            target_reached = False
            while steps_for_target < max_steps_per_target:
                capture = (total_steps % capture_every_n == 0)
                self.run_simulation_step(capture_frame=capture)
                total_steps += 1
                steps_for_target += 1

                if self.check_success():
                    print(f"Target {i+1} reached in {steps_for_target} steps")
                    target_reached = True
                    break

            if not target_reached:
                print(f"Timeout reaching target {i+1} after {max_steps_per_target} steps")
                trajectory_success = False

        # Demo: close after motion
        self.close_gripper()
        # step a little to see closing motion in video
        for _ in range(20):
            self.run_simulation_step(capture_frame=True)

        self.save_trajectory_results(trajectory_dir, trajectory_name)
        return trajectory_success
    
    def run_grasp_demo(self, object_xy=(0.5, 0.5), hover_z=0.35, grasp_z=0.06,
                   lift_z=0.35, yaw=0.0, name="grasp_demo"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = os.path.join(self.output_dir, f"{name}_{timestamp}")
        os.makedirs(trajectory_dir, exist_ok=True)
        self.overview_frames, self.ee_camera_frames, self.frame_capture_timestamps, self.trajectory_data = [], [], [], []

        ox, oy = object_xy
        hold_hover = np.array([ox, oy, hover_z], dtype=float)

        # Start open, move above object
        self.open_gripper()
        print("➡️  Move above object (hover).")
        ok, _ = self.goto(hold_hover, max_steps=120);  print("  done" if ok else "  timeout")

        # --- YAW TEST 1: rotate to the requested yaw while hovering ---
        print(f"🔁  Yaw to {yaw:.2f} rad while hovering.")
        self.rotate_to(yaw_target=float(2*yaw), duration=2.5, hold_xyz=hold_hover)

        # Descend to grasp height (keep that yaw)
        print("➡️  Descend to grasp height.")
        ok, _ = self.goto([ox, oy, grasp_z], max_steps=120);  print("  done" if ok else "  timeout")

        for _ in range(20): self.run_simulation_step(capture_frame=True)

        # Close & squeeze
        print("➡️  Close gripper.")
        self.close_gripper()
        for _ in range(40): self.run_simulation_step(capture_frame=True)

        z_before = self.get_target_position()[2]

        # Lift up
        print("➡️  Lift.")
        ok, _ = self.goto([ox, oy, lift_z], max_steps=150);  print("  done" if ok else "  timeout")

        # --- YAW TEST 2: rotate 90° while lifted (tests stability while carrying) ---
        yaw2 = float(self._angle_wrap(yaw + np.pi/2.0))
        print(f"🔁  Yaw to {yaw2:.2f} rad while lifted.")
        self.rotate_to(yaw_target=-yaw2, duration=2.5, hold_xyz=[ox, oy, lift_z])

        # Optional: yaw back to original
        print(f"🔁  Yaw back to {yaw:.2f} rad.")
        # self.rotate_to(yaw_target=float(yaw), duration=1.0, hold_xyz=[ox, oy, lift_z])

        # Return to hover, open/close a couple times (exercise fingers under new yaw)
        print("➡️  Move above object again (hover).")
        ok, _ = self.goto(hold_hover, max_steps=120);  print("  done" if ok else "  timeout")

        self.open_gripper()
        for _ in range(30): self.run_simulation_step(capture_frame=True)
        self.close_gripper()
        for _ in range(30): self.run_simulation_step(capture_frame=True)

        # Check grasp success
        for _ in range(30): self.run_simulation_step(capture_frame=True)
        grasp_ok = self.lifted_enough(z_before, min_rise=0.02) or self.has_finger_contact()

        self.save_trajectory_results(trajectory_dir, name)
        print(f"✅ Grasp {'SUCCEEDED' if grasp_ok else 'FAILED'}")
        return grasp_ok

    def save_trajectory_results(self, trajectory_dir, trajectory_name):
        """Save all trajectory data and videos"""
        os.makedirs(trajectory_dir, exist_ok=True)
        print("Saving trajectory results...")
        video_fps = self._estimate_video_fps()
        
        # Save videos if we have frames
        if self.overview_frames:
            try:
                overview_video_path = os.path.join(trajectory_dir, "overview_video.mp4")
                self.save_video(self.overview_frames, overview_video_path, fps=video_fps)
                print(f"Overview video saved: {overview_video_path} (fps={video_fps:.3f})")
            except Exception as e:
                print(f"Error saving overview video: {e}")
        
        if self.ee_camera_frames:
            try:
                ee_video_path = os.path.join(trajectory_dir, "ee_camera_video.mp4")
                self.save_video(self.ee_camera_frames, ee_video_path, fps=video_fps)
                print(f"End-effector video saved: {ee_video_path} (fps={video_fps:.3f})")
            except Exception as e:
                print(f"Error saving EE camera video: {e}")
        
        # Save trajectory data
        try:
            trajectory_file = os.path.join(trajectory_dir, "trajectory_data.npz")
            self.save_trajectory_data(trajectory_file)
            print(f"Trajectory data saved: {trajectory_file}")
        except Exception as e:
            print(f"Error saving trajectory data: {e}")
        
        # Save summary
        try:
            self.save_summary(trajectory_dir, trajectory_name)
        except Exception as e:
            print(f"Error saving summary: {e}")
        
        print(f"Results saved to: {trajectory_dir}")
    
    # --- NEW: helper for normalized delta actions ---
    def _compute_normalized_delta_actions(self, actions_abs: np.ndarray) -> np.ndarray:
        """
        Convert absolute actions [x, y, z, yaw, grip] (T, 5) into
        normalized deltas in [-1, 1] with shape (T-1, 5).
        """

        actions_abs = np.asarray(actions_abs, dtype=np.float32)
        if actions_abs.ndim != 2 or actions_abs.shape[1] != 5:
            raise ValueError(f"Expected actions_abs shape (T, 5), got {actions_abs.shape}")

        # Scaling factors (tune as needed; keep consistent with your training config)
        k_xyz  = 0.05   # meters per normalized unit (x, y, z)
        k_yaw  = 0.25   # radians per normalized unit (yaw)
        k_grip = 1.0    # grip is assumed to already be in [-1, 1]

        scales = np.array([k_xyz, k_xyz, k_xyz, k_yaw, k_grip], dtype=np.float32)

        # raw deltas between consecutive timesteps: (T-1, 5)
        deltas = actions_abs[1:] - actions_abs[:-1]

        # normalize + clip to [-1, 1]
        deltas_norm = deltas / scales[None, :]
        deltas_norm = np.clip(deltas_norm, -1.0, 1.0)

        return deltas_norm.astype(np.float32)
    
    def save_video(self, frames, filepath, fps=30):
        if not frames:
            return
        if imageio is None:
            raise RuntimeError("imageio is required to save videos.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # ensure parent exists
        with imageio.get_writer(filepath, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

    def _estimate_video_fps(self, default_fps=20.0):
        times = np.asarray(getattr(self, "frame_capture_timestamps", []), dtype=np.float64)
        if times.size >= 2:
            diffs = np.diff(times)
            diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
            if diffs.size > 0:
                return float(1.0 / np.mean(diffs))

        frame_count = max(len(getattr(self, "overview_frames", [])), len(getattr(self, "ee_camera_frames", [])))
        total_time = float(getattr(getattr(self, "data", None), "time", 0.0) or 0.0)
        if frame_count > 1 and total_time > 1e-9:
            return float(frame_count / total_time)

        dt = float(getattr(getattr(self, "controller", None), "dt", 0.0) or 0.0)
        if dt > 1e-9:
            return float(1.0 / dt)
        return float(default_fps)
    
    def save_trajectory_data(self, filepath):
        """Save trajectory data (.npz) including:
        - original observables (timestamp, ee_position, ...)
        - absolute actions (5D)
        - normalized delta actions (5D in [-1,1])
        - task_description (language string per step)
        """

        if not self.trajectory_data:
            return

        import numpy as np

        # -----------------------------
        # 1) Collect base data as before
        # -----------------------------
        trajectory_dict = {}

        arrays_to_save = [
            'timestamp',
            'ee_position',
            'target_position',
            'slider_positions',
            'cable_lengths',
            'control_signals',
        ]

        for key in arrays_to_save:
            if key in self.trajectory_data[0]:
                trajectory_dict[key] = np.array(
                    [step[key] for step in self.trajectory_data]
                )

        T = len(self.trajectory_data)
        if T < 2:
            # Not enough steps to form deltas; still save what we have
            np.savez(filepath, **trajectory_dict)
            return

        # ----------------------------------------------------------
        # 2) Extract ABSOLUTE actions from control_signals
        # ----------------------------------------------------------
        control = trajectory_dict.get("control_signals")
        if control is None:
            raise RuntimeError("trajectory_data has no 'control_signals', cannot compute actions.")

        control = np.asarray(control, dtype=np.float32)

        if control.ndim != 2 or control.shape[1] < 5:
            raise ValueError(
                f"'control_signals' expected shape (T,>=5), got {control.shape}"
            )

        # Interpret first 5 dims as [x, y, z, yaw, grip]
        actions_abs = control[:, :5]        # (T,5)
        trajectory_dict["actions_abs"] = actions_abs

        # ----------------------------------------------------------
        # 3) Compute normalized delta actions
        # ----------------------------------------------------------
        # Scaling constants (tune if needed; keep consistent with training config)
        k_xyz  = 0.05   # meters per normalized unit
        k_yaw  = 0.25   # radians per normalized unit
        k_grip = 1.0    # grip assumed [-1,1]

        scales = np.array([k_xyz, k_xyz, k_xyz, k_yaw, k_grip], dtype=np.float32)

        deltas = actions_abs[1:] - actions_abs[:-1]   # (T-1, 5)

        deltas_norm = deltas / scales[None, :]
        deltas_norm = np.clip(deltas_norm, -1.0, 1.0).astype(np.float32)

        trajectory_dict["actions_delta_norm"] = deltas_norm  # (T-1, 5)

        T_delta = deltas_norm.shape[0]

        # ----------------------------------------------------------
        # 4) Align observation arrays to length T-1
        # ----------------------------------------------------------
        for key in arrays_to_save:
            if key in trajectory_dict:
                data = trajectory_dict[key]
                if len(data) == T:
                    trajectory_dict[key] = data[:T_delta]

        # ----------------------------------------------------------
        # 5) Add language as task_description per-step (for RLDS)
        # ----------------------------------------------------------
        # meta_dataset.json expects: "language": "observation/task_description"
        # Here we store 'task_description' so the RLDS builder can map it to that.
        if hasattr(self, "language_instruction"):
            lang = self.language_instruction
        else:
            lang = ""

        # one string per step (T-1)
        trajectory_dict["task_description"] = np.array(
            [lang] * T_delta,
            dtype=object
        )

        # ----------------------------------------------------------
        # 6) Save final npz
        # ----------------------------------------------------------
        np.savez(filepath, **trajectory_dict)
    
    def save_summary(self, trajectory_dir, trajectory_name):
        """Save a text summary of the trajectory"""
        summary_file = os.path.join(trajectory_dir, "summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"CDPR Trajectory Summary\n")
            f.write(f"=======================\n")
            f.write(f"Trajectory: {trajectory_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total frames captured: {len(self.overview_frames)}\n")
            f.write(f"Total simulation steps: {len(self.trajectory_data)}\n")
            f.write(f"Simulation time: {self.data.time:.2f} seconds\n")
            f.write(f"Video fps: {self._estimate_video_fps():.3f}\n")
            if hasattr(self, "language_instruction"):
                f.write(f"language_instruction: {self.language_instruction}\n")
            if self.trajectory_data:
                f.write(f"Final EE position: {self.trajectory_data[-1]['ee_position']}\n")
                f.write(f"Final target position: {self.trajectory_data[-1]['target_position']}\n")
    
    def cleanup(self):
        if hasattr(self, "context"):
            try:
                mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self.context)
            except:
                pass

        if getattr(self, "_glfw_window", None) is not None:
            try:
                from mujoco.glfw import glfw
                glfw.destroy_window(self._glfw_window)
                glfw.terminate()
            except:
                pass

        if self.gl_context:
            try:
                self.gl_context.free()
            except:
                pass
        print("Simulation cleanup completed")


def main():
    xml_path = "cdpr.xml"
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    sim = HeadlessCDPRSimulation(xml_path, output_dir="trajectory_results")
    try:
        sim.initialize()

        # Try a grasp at the red block's XY
        success = sim.run_grasp_demo(object_xy=(0.5, 0.5), hover_z=0.35, grasp_z=0.06, lift_z=0.35, yaw=0.0)
        print("✓ grasp flow finished:", success)

    except Exception as e:
        print("Error during simulation:", e)
        import traceback; traceback.print_exc()
    finally:
        sim.cleanup()

if __name__ == "__main__":
    main()
