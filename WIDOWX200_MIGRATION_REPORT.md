# CDPR → Interbotix WidowX-200: embodiment migration

Date: 2026-08-28

Status: **the robot, the controller, and the scene are built and verified on
CPU MuJoCo. The batched MJWarp training path is designed but not yet wired.**

Everything measured, every number that came out differently from the estimate,
and the ordered list of what is left. Written so a fresh session can continue
from here without re-deriving any of it.

---

## 1. Headline

**The action contract survives the embodiment change intact.** The policy still
emits `[x, y, z, yaw, gripper]` in `[-1, 1]` with world-frame end-effector
deltas. That is what makes this a migration rather than a restart: the reward
functions, success predicates, curriculum shells, action codec, GRPO collector,
and residual head all apply unchanged, and a phase-4/5 CDPR checkpoint is a
candidate warm start rather than automatically worthless.

**The arm is exact where it has to be.** Closed-form IK against MuJoCo's own
forward kinematics: **2.2e-15 m** position error and **8.9e-16 rad** yaw error
over 465 random reachable poses, with the tool's approach axis at `(0, 0, -1)`
to the same precision. There is no iterative solver, no Jacobian, no null
space, and therefore no per-world divergence to contain.

**Four things did not survive, and each is a real physical difference:**

| | CDPR | WidowX-200 | consequence |
|---|---|---|---|
| control period | 14 ms | **50 ms** | 71 Hz asks this arm for 1.07 m/s, past its joint-rate limit |
| target integration | from measured pose | **integrated + leashed** | `ee + delta` attenuates every action by 63–88% |
| reachable set | ±0.28 m box | **annular sector** | the CDPR object square has corners 0.49 m from any mount |
| gripper span | 35–95 mm | **8–52 mm** | **no object in the current catalog is graspable** |

The last one has the widest blast radius and is covered in §5.

![reach, descend, close, lift](assets/research/widowx200/wx200_grasp_sequence.png)

Overview and wrist cameras through a scripted grasp driven only through the
normalized five-channel action. Object lifts 77 mm; scripted grasp succeeds in
**29/30** trials at uniformly sampled positions across the spawn sector.

---

## 2. Why top-down task-space control, and what it costs

The WidowX-200 has five revolute joints: waist (z), shoulder, elbow,
wrist_angle (all y), wrist_rotate (x). Three share the +y axis, so they control
exactly three quantities in one vertical plane — in-plane position (two) and
tool pitch (one). The waist picks the plane; wrist_rotate spins the tool about
its own approach axis. Five joints against six task DoF: one has to go.

Pinning pitch to straight down makes `(x, y, z, yaw) → (q1..q5)` square and
closed-form. It is also how these arms are driven in practice — Interbotix's
`set_ee_pose_components` and the BridgeData WidowX stack both command Cartesian
poses with a fixed approach — so the simulated interface stays
deployment-faithful rather than becoming a simulator convenience.

**The cost, stated plainly: no side or angled approaches.** Reaching into a mug
sideways is not expressible. Making it expressible means promoting pitch to a
sixth action channel, which changes the action space and invalidates existing
checkpoints. `TOP_DOWN_PITCH` is a parameter of `top_down_ik` rather than a
constant baked into the algebra, so that stays a config change and not a
rewrite.

The alternative — a joint-space action space — was rejected. It would be more
faithful to a real arm's low-level API and would allow arbitrary orientations,
but it discards every checkpoint, changes the SmolVLA output dimension, and
re-poses the exploration problem in a space where none of the curriculum's
distances, shaping windows, or success radii mean anything. Against 345+ hours
of RL and five phases of hard-won curriculum, that is not a trade worth making
for orientation freedom no current task needs.

Geometry is transcribed from the official Interbotix URDF
(`interbotix_xsarm_descriptions/urdf/wx200.urdf.xacro`, BSD-3, Trossen); meshes
are the upstream STLs, unmodified, with the licence copied alongside them.

---

## 3. The servo lag, which was not obvious and is not a tuning problem

The first scripted grasp failed with the arm 14 mm short of the object. The
diagnosis matters because the obvious fix is wrong.

Measured on the compiled model:

| | value |
|---|---|
| steady-state hold error | **1.4 mm** |
| settling error after a move | **1.6 mm** |
| fraction of a 15 mm step recovered at 14 ms | **12%** |
| ...at 26 ms | 21% |
| ...at 50 ms | **37%** |
| peak joint rate during all of the above | 0.35–0.6 rad/s (limit is π) |

So the servo holds and settles well, and the arm is nowhere near its rate
limit. The lag is servo *bandwidth*, and raising `kp` does not help — at the
2 ms MJWarp timestep, a 3× gain increase already diverges.

The real cause is the CDPR's target rule. It builds each target from the
**measured** pose (`ee_position + delta`, see
`mjlab_mjwarp_backend.py` in `step`), which is exact for a light platform on
stiff cables that tracks a full step inside one window. On an arm that recovers
a third of a step, the same rule silently **attenuates every action**, and the
policy's effective step size becomes a function of how fast the arm happens to
be moving — a non-stationary action space, which is considerably worse than a
small one.

The fix is to integrate the target instead (`target + delta`) and leash it to
the measured pose. Uncapped, the target winds up ~0.18 m ahead and a reversal
then does nothing for a dozen steps. The leash makes the trade explicit
(`hold_steps=24`):

| leash | effective step | coast after commands cease |
|---|---|---|
| 0.020 | 7.2 mm | 12.7 mm |
| **0.030** | **10.4 mm** | **19.1 mm** |
| 0.045 | 13.1 mm | 26.5 mm |

0.030 is shipped: the effective step stays at 69% of the CDPR's 15 mm while the
coast stays inside the 20 mm `move_to` success radius, so a policy can still
stop on target. These numbers are pinned by test, so a gain or damping edit
cannot move them unnoticed.

**Downstream consequence:** at 69% of the CDPR's effective step, covering the
same distance takes ~1.45× the steps. Episode horizons and the curriculum's
start-distance ladder are in metres and steps respectively, so the ladder
transfers but the horizons need scaling. This is a config change, not a code
change, but it must not be forgotten — a horizon that is 45% too short looks
exactly like a policy that cannot solve the task.

---

## 4. Where the arm may work

The datasheet's "550 mm reach" is not used anywhere. That is the
fully-extended figure; folding the wrist to vertical for a top-down grasp costs
about 15 cm of it. `workspace.py` derives the envelope from the kinematics and
the joint box instead:

| tool height (world z) | top-down reach |
|---|---|
| 0.168 (pads clearing the desk) | 0.381 m |
| 0.19 (object centre at rest) | 0.376 m |
| 0.24 (lifted 5 cm) | 0.362 m |
| 0.27 | 0.350 m |
| 0.31 | 0.329 m |
| ≥ 0.41 | nothing |

The desk stays at z = 0.1502, unchanged, because every object rest height, the
push predicates' support clearance, the container-z threshold, and every
`*_grasp_height_offset` is written against it.

The arm mounts on the desk at `(0, 0.24, 0.1502)` facing −y, at the **back** of
the scene. Rendering settled two camera questions that were not obvious on
paper:

- With the arm in front of the objects it occludes most of them; behind them it
  is context and they are not.
- The CDPR overview pose (`0 -0.5412341 0.5125`, fovy 45) shows the objects fine
  and crops the arm to a gripper hanging in from the top edge. Worse, a camera
  on the arm's own axis sees it **end-on**: every link foreshortens onto the one
  behind it, so a correctly placed arm still renders as a floating gripper. The
  overview is now a three-quarter view at `(-0.45, -0.45, 0.55)`, fovy 55, which
  shows shoulder, upper arm, forearm, and wrist as separate things.

This does change the overview image distribution relative to the CDPR
checkpoints — unavoidable, since the scene now contains a different robot. The
**wrist** camera is held to the old contract (fovy 60, comparable standoff)
because it sees mostly gripper and object, which are the parts a warm start can
still recognise. Its mount was solved against the compiled model rather than
eyeballed: 11.9 cm standoff, optical axis on the grasp point to under 0.01°,
gripper occupying 8.5% of the frame (measured by segmentation render; on the
bar itself it is 15.2%).

**Object placement had to change shape, not just size.** The CDPR collector
draws four objects from a 3×3 Cartesian grid at 0.18 m spacing. That grid's
corners fall outside the arm's annulus, and a rectangle inscribed in a sector
wastes most of it. `sector_lattice()` returns a two-ring polar lattice instead:
six cells, **0.180 m** minimum pairwise separation — above the 0.16 m the
collector requires — all reachable through a lift and a carry. The usable area
drops from the CDPR's 0.168 m² to 0.0785 m². That is a real reduction in scene
variety and is the price of a 55 cm arm.

---

## 5. The object catalog does not fit this gripper

Measured on the compiled model: the pads span **8 mm closed and 52 mm open**,
and the band that actually grasps and lifts is **30–56 mm**. Against the
current catalog's minimum grasp width:

| object | width | fits |
|---|---:|:---:|
| bowl (rim) | 10 mm | no — too thin to hold |
| mug (handle) | 12 mm | no — too thin to hold |
| carrot | 24 mm | no |
| banana | 28 mm | marginal |
| tomato, orange, potato | 58 mm | no |
| apple | 69 mm | no |
| bell pepper | 74 mm | no |
| plate | 182 mm | no (receptacle) |

**Nothing in the catalog is comfortably graspable.** Every grasp-dependent
instruction — `pick_up`, `put_into_bowl`, `put_into_plate` — is blocked until
this is fixed.

The fix is to rescale the graspable objects to a 34–50 mm minor width, not to
drop them. A 7 cm apple is out of scale for a 200 g-payload arm in the first
place, so rescaling is the physically correct correction rather than a
workaround. Indicative scale factors: apple ×0.66, bell pepper ×0.62,
tomato/orange/potato ×0.79, banana ×1.3, carrot ×1.4.

Rescaling invalidates three fields per variant in `cdpr_object_catalog.py`:
`rest_height`, `inertia`, and `fitted_gripper_opening`. The last is
embodiment-specific and must be re-measured on this gripper regardless — its
own docstring records that deriving it from mesh bounds overstated it by
0.15–0.33 on every catalog and broke both the caught-object reset and the
reward's release test. Re-measure it, do not compute it.

A separate limit worth recording: a 20 mm sphere was **not** reliably grasped
even though it is inside the 8–52 mm span. Thin features (the mug handle, the
bowl rim) should not be assumed graspable just because they fit between the
pads.

---

## 6. What was built

```text
robots/widowx200/
  README.md
  widowx200_mujoco/
    wx200.xml               URDF-exact kinematics; visual meshes, primitive collision
    wx200_scene.xml         desk, lights, three-quarter overview camera
    meshes/                 upstream Interbotix STLs + LICENSE (BSD-3)
    kinematics.py           closed-form FK / top-down IK, NumPy and Torch
    controller.py           host controller + shared action-integration algebra
    batched_controller.py   MJWarp-side controller, pure tensor ops
    workspace.py            reach envelope, scene layout, lattice, mount placement
  widowx200_dataset/        (empty)
tests/test_widowx200_embodiment.py   34 tests, all passing
assets/research/widowx200/wx200_grasp_sequence.png
```

Full suite: **940 passed, 1 skipped**, with the same three pre-existing failures
recorded in the local test recipe (`_qpos`,
`predict_normalized_action_chunk`, `grpo_bootstraps_from_td3`). Nothing here
touches them.

The design decision that keeps this maintainable: **the host controller and the
batched controller call the same `joint_targets_from_task_targets`.** The CDPR
paid for a CPU/MJWarp controller divergence with an entire parity-report gate
(`docs/cdpr_mjwarp_compatibility.md` §"Remote acceptance gates"). Here parity is
structural rather than maintained, and a test enforces that it stays so.

`ee_base` is deliberately placed **at the point between the pads** rather than
at the arm flange 5 cm above it, so every `*_grasp_height_offset` stays near
zero. Tracking the flange would put all of them 5 cm out — which is the exact
shape of the failure that cost `pick_up` 10M steps on the CDPR, and it raises
nothing.

---

## 7. What is left, in order

1. **Rescale the object catalog** (§5). Blocks every grasp task; independent of
   everything else; do it first.

2. **Repoint the MJWarp backend.** `MJLabMJWarpCDPRBackend` is ~1830 lines of
   which the CDPR-specific part is a well-delimited block:
   `_calibrate_host_cdpr`, `_resolve_host_ids`, `_initialize_controller_state`,
   `_write_controller_controls`, the target integration in `step`,
   `set_end_effector_poses`, and `_update_cable_visuals`. Everything else —
   world allocation, partial reset, group broadcast, rendering, contacts,
   catalogs, visual randomization, export, metadata — is embodiment-independent
   and reusable as it stands. `batched_controller.py` already implements that
   block's replacement with a matching surface; the work is extracting a
   controller protocol and holding one instead of inlining the cable code.
   Note the arm needs **no** startup calibration and **no** preload restore
   after reset, so two of the CDPR's sharper edges simply disappear.

3. **Object placement in the collector.** Replace the 3×3 Cartesian grid with
   `workspace.sector_lattice()`. Same four-of-N random subset per group.

4. **Re-measure the gripper-dependent thresholds.** `release_opening` (0.55),
   the caught-object reset's `fitted - 0.033` seat, and
   `pick_grasp_height_offset` are all CDPR-gripper numbers. Build the WidowX
   analogue of `cdpr_gripper_geometry` — derive from the model, assert against
   the config, exactly as that module does.

5. **Scale the episode horizons** by ~1.45× for the smaller effective step
   (§3). The curriculum's start-distance ladder is in metres and transfers
   as-is; the step budgets are not.

6. **MJWarp preflight on the A40 box.** The compatibility doc's gate list
   applies unchanged. The arm is *easier* here than the CDPR: no spatial
   tendons, no pulley wraps, no ball joint, no free-body robot — `nq` drops from
   46 to 7 for the robot, and the constraint count should fall with it.

7. **Decide the warm start.** The action semantics are identical and the yaw
   convention was deliberately matched, so a phase-4 checkpoint's actions mean
   the same thing on this arm. The wrist camera is close to the old contract;
   the overview is not. Worth one probe run before committing to either a warm
   start or a scratch run — cheaper than either mistake.

Steps 1, 3, 4, and 5 are config- and data-level and can proceed in parallel with
step 2.

---

## 8. Things that were checked and came back differently than expected

- **"Raise the gains."** kp × 3 diverges at the 2 ms timestep. The lag is
  bandwidth, and the fix was the target rule, not the servo (§3).
- **"Keep the CDPR overview camera for image-statistics parity."** Rendered, the
  arm is cropped out of frame entirely; on-axis it is foreshortened to a
  floating gripper regardless of framing. Parity was not achievable and was not
  worth chasing (§4).
- **"The CDPR object envelope roughly fits."** Its corners are 0.49 m from any
  sensible mount against a 0.35 m top-down reach — not close (§4).
- **"The catalog mostly works, with the apple as an edge case."** Nothing in it
  is graspable (§5).
- **Gripper actuator kp = 60** closed straight through every object and dropped
  all of them. At 200 all sizes lift, and the settled opening tracks the object
  diameter to about a millimetre.
- **`reachable` was reporting on the clamped target**, which made it a tautology
  and would have removed the only signal that a policy is driving into the
  workspace wall. Fixed at the source in `top_down_ik`.
