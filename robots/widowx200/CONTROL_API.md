# WidowX-200: control API

How to drive this arm, what it exposes, and how to put SmolVLA on top of it
with an action expert re-headed for its five degrees of freedom.

The short version: **the arm speaks the same five-channel normalized action as
the CDPR**, so anything already written against that interface works. What
follows is what those five channels mean here, what you get back, and the two
entry points — one model for host-side work, `nworld` models for batched RL.

---

## 1. Controllable dimensions

The policy commands **five numbers in `[-1, 1]`**, every one a *delta* on the
controller's current target:

| # | channel | meaning | scale per unit action |
|---|---|---|---|
| 0 | `x` | world-frame end-effector step, +x | 0.015 m |
| 1 | `y` | world-frame end-effector step, +y | 0.015 m |
| 2 | `z` | world-frame end-effector step, +z | 0.015 m |
| 3 | `yaw` | tool rotation about world z | 0.08 rad |
| 4 | `gripper` | opening, −1 closes, +1 opens | 0.05 (normalized) |

Gripper `0.0` is closed (8 mm between the pads), `1.0` is fully open (52 mm).
The band that reliably grasps and lifts is **30–56 mm**, which is a constraint
on the object set — see the README.

**These are task-space, not joint-space.** The arm has five revolute joints
(`waist`, `shoulder`, `elbow`, `wrist_angle`, `wrist_rotate`) plus a coupled
two-finger gripper, and the controller solves closed-form inverse kinematics
every step to hit the commanded Cartesian target. You never address joints
directly, and you should not want to: the whole reward, curriculum, and
checkpoint stack is written in task space.

**Tool pitch is pinned to straight down** (`TOP_DOWN_PITCH = π`). Five joints
cannot serve six task DoF, and pitch is the one given up. That makes
`(x, y, z, yaw) → (q₁..q₅)` square and closed-form — no iteration, no Jacobian,
no null space. The cost is that **side and angled approaches are not
expressible**. Making them expressible means a sixth action channel, which
changes the action space and invalidates existing checkpoints; `top_down_ik`
takes `gamma` as a parameter so that stays a config change rather than a
rewrite.

### Timing

One policy action advances physics by `1 + hold_steps` steps. Defaults:

| | value | why |
|---|---|---|
| `hold_steps` | 24 | 50 ms per action, **20 Hz** — the rate a real WidowX runs at |
| `action_step_xyz` | 0.015 m | the CDPR's step, so shaping windows and success radii carry over |
| `target_leash` | 0.030 m | caps how far the target may run ahead of the measured arm |

The leash matters more than it looks. The CDPR builds each target from the
*measured* pose; on this arm that silently attenuates every action, because the
servo recovers only ~37% of a 15 mm step in 50 ms. The target is integrated
instead and leashed to the arm, which makes the commanded step exact and turns
the lag into a constant trailing distance. Net **effective step ≈ 10.4 mm**,
69% of the CDPR's — so episode horizons need roughly **1.45×** the steps to
cover the same ground.

### Where it may go

Commanded targets are clamped, in this order: the world workspace box, then a
keep-out cylinder around the waist axis, then the arm's own reach.

| bound | value |
|---|---|
| `workspace_x` | (−0.30, 0.30) m |
| `workspace_y` | (−0.12, 0.22) m |
| `workspace_z` | (0.168, 0.27) m |
| `min_reach_radius` | 0.15 m from the base axis |
| `reach_margin` | 0.02 m held back from full extension |

Clamping moves the *target*, never the joints, so the tool still goes where the
controller says it went. Whether the **request** was reachable is reported
separately (`reachable`) rather than swallowed — a rising unreachable fraction
means the policy is pressing into a wall, which on the CDPR would have looked
like a policy that had simply stopped improving.

---

## 2. What you get back

### Proprioception — 6 numbers, and deliberately nothing more

```
[ee_x, ee_y, ee_z, ee_yaw, gripper_opening, 0.0]
```

Column 5 is a neutral pad. It used to carry the XY distance to the target, and
that was removed on purpose: it is privileged task geometry a real robot never
observes, and a policy trained on it collapses the moment it is withheld. The
policy localizes from the cameras and the language instruction.

`ee_*` is the pose of the body named `ee_base`, which sits **between the finger
pads**, not at the arm flange 5 cm above them. That is why every
`*_grasp_height_offset` stays near zero on this arm.

### Cameras — two, 320×240 native, 256×256 into SmolVLA

| slot | camera | what it sees |
|---|---|---|
| 0 | `overview` | three-quarter view, 0.42 m out; robot covers ~14% of frame, all spawn cells visible |
| 1 | `ee_camera` | wrist, 7.5 cm standoff, aimed at the grasp point |
| 2 | — | the wrist tensor again, or a masked empty slot |

Both were re-sited against what they actually resolve rather than where they
point; the wrist camera in particular sees 15% finger per side and 13% object,
against 1% and 0.1% before. Gripper state is readable from the image because
both fingers sit symmetric about the optical axis, so the gap between them is
legible.

---

## 3. Driving it

### One arm, host side

`WidowX200TaskSpaceController` duck-types the method names the CDPR's
`policy_control` already calls, so existing tooling runs on it unbranched.

```python
import mujoco
from robots.widowx200.widowx200_mujoco import workspace as W
from robots.widowx200.widowx200_mujoco.controller import (
    WidowX200ControlSpec, WidowX200MountPose, WidowX200TaskSpaceController)

layout = W.DEFAULT_LAYOUT
model = mujoco.MjModel.from_xml_path(
    "robots/widowx200/widowx200_mujoco/wx200_scene.xml")
W.mount_widowx200(model, layout)        # stands the arm up; no recompile
data = mujoco.MjData(model)

controller = WidowX200TaskSpaceController(
    model, data,
    spec=WidowX200ControlSpec(
        mount=WidowX200MountPose(layout.base_position, layout.base_yaw),
        workspace_x=layout.workspace_x,
        workspace_y=layout.workspace_y,
        workspace_z=layout.workspace_z,
        min_reach_radius=layout.min_reach_radius,
    ),
)

controller.reset_to_pose([0.0, 0.0, 0.26], yaw=0.0, gripper=1.0)
info = controller.apply_normalized_action([0.0, 0.0, -1.0, 0.0, 0.0])  # descend
```

| call | does |
|---|---|
| `reset_to_pose(xyz, yaw, gripper)` | teleports; writes `qpos` **and** `ctrl` so the servo starts satisfied and the arm does not lurch on an episode's first step |
| `apply_normalized_action(a)` | the entry point: integrate, leash, solve IK, run the substeps |
| `get_end_effector_position()` | `ee_base` in world coordinates |
| `get_yaw()` / `get_gripper_opening()` | commanded yaw; measured opening in `[0, 1]` |
| `get_joint_positions()` | the five joint angles, for logging and sim-to-real |
| `set_end_effector_target(xyz)` | absolute-target primitive; **skips the leash**, so prefer `apply_normalized_action` |

### Many arms, batched (the RL path)

`WidowX200BatchedController` holds per-world target, yaw, and gripper as GPU
tensors. Every method is pure tensor arithmetic — nothing branches on a tensor
value, nothing calls `.item()` — because one host synchronization per substep
would serialize 128 worlds.

```python
from robots.widowx200.widowx200_mujoco.batched_controller import (
    WidowX200BatchedController, WidowX200ModelIndices)

controller = WidowX200BatchedController(
    torch=torch, device=device, nworld=128, spec=spec,
    indices=WidowX200ModelIndices.resolve(host_model, mujoco),
)

controller.set_end_effector_poses(qpos, ctrl, qvel, positions, yaws, openings)
controller.broadcast_group_state(base_world_indices, group_size=8)   # GRPO
for _ in range(spec.sim_steps_per_policy_action):
    controller.integrate_actions(actions, active_mask, ee_position)  # once/action
    controller.write_controls(ctrl)                                  # once/substep
    mjwarp.step(model, data)
```

Unreachable requests are clamped and flagged in `controller.reachable`, never
raised: one bad world must not take down a rank of 128.

`controller.metadata()` returns the control contract — step scales,
`hold_steps`, `target_leash`, pitch, mount pose, workspace. **Put it in the
checkpoint.** Those numbers change what an action *means*, so resuming across a
change to any of them silently trains a different problem.

### Pure functions, if you want neither

```python
from robots.widowx200.widowx200_mujoco.kinematics import top_down_ik, forward_kinematics
from robots.widowx200.widowx200_mujoco.controller import (
    integrate_task_targets, joint_targets_from_task_targets)
```

All four work on NumPy arrays or Torch tensors — the array module is inferred
from the input, so the host path and the batched path run the same algebra and
parity is structural rather than maintained.

---

## 4. SmolVLA with the action expert re-headed to 5 DOF

### What SmolVLA does by default, and why it is a poor fit here

SmolVLA is trained with `max_action_dim = 32`: every embodiment is zero-padded
into a 32-wide slot, and which columns mean what differs per dataset. Two
layers carry that width —

```
model.action_in_proj  : Linear(32 → expert_hidden)
model.action_out_proj : Linear(expert_hidden → 32)
```

— and the expert transformer between them is width-agnostic.

The established path in this repo keeps all 32 and slices five columns out
(`smolvla_cdpr._resolve_action_indices`, which for a 32-wide output picks
`(0, 1, 2, 3, 31)` — XYZ, yaw, and the *last* column as gripper). The frozen
model is a prior; a trainable residual head does the real work.

On this stack that prior is measurably nearly worthless: it behaves as a
near-constant action offset, per-step sd ≤ 0.04. So the 32-wide head is 27
columns of arithmetic that never reach an actuator, and the residual is the
policy.

### Re-heading

```python
from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime
from rl_vla_bootstrapping.policy.smolvla_action_expert import (
    reinitialize_action_expert, trainable_action_expert_parameters)

runtime = load_smolvla_runtime(
    "lerobot/smolvla_base",
    device="cuda", action_dim=5, chunk_size=8, state_dim=6,
)

report = reinitialize_action_expert(runtime.policy, action_dim=5, state_dim=6)
optimizer = torch.optim.AdamW(trainable_action_expert_parameters(runtime.policy), lr=1e-4)
```

That narrows both projections to the arm's own width, re-initializes them,
updates `config.max_action_dim` so LeRobot's padding and sampling shapes follow,
freezes everything else, and returns what changed.

The expert transformer keeps its pretrained weights — **this is re-heading, not
retraining from scratch.** `state_dim=6` is optional; leaving it `None` keeps
the 32-wide state projection, which costs nothing since LeRobot zero-pads the
state anyway, and keeps that part of the pretrained model intact.

Downstream nothing else has to change: with a 5-wide output,
`_resolve_action_indices(5, None)` returns `(0, 1, 2, 3, 4)`, so the adapter
becomes the identity and the five columns reach the controller in order.

### The one thing to be clear about

**Re-heading destroys the prior.** `action_out_proj` no longer decodes
anything; until it is trained, the policy emits noise. `report.prior_preserved`
is `False` and says so.

Anything that leans on the prior stops being meaningful at that moment:

- a KL or trust-region leash back to the frozen prior,
- a residual head initialized to be a *small correction* on top of it,
- a warm start from a checkpoint trained against the 32-wide head.

So this is a real fork, not a drop-in improvement:

| | 32-wide, frozen, sliced | 5-wide, re-headed |
|---|---|---|
| prior at init | near-constant offset, sd ≤ 0.04 | none — noise |
| who is the policy | the residual head | the action expert itself |
| KL leash to prior | available | meaningless |
| CDPR checkpoints | loadable | not loadable |
| columns reaching an actuator | 5 of 32 | 5 of 5 |

Re-heading is the right call when you intend to train the expert and the prior
is not buying anything — which is the situation here. It is the wrong call on a
stack where the prior works. Record `report.to_metadata()` in the checkpoint so
a resume cannot cross the two silently.

If you want the expert body to move as well, pass `freeze_rest=False` and
attach LoRA (`rl_vla_bootstrapping/policy/lora.py`; LeRobot's own default PEFT
targets already cover these projections plus the expert's `q/v` projections).

---

## See also

- [`README.md`](README.md) — what this robot is and how it differs from the CDPR
- [`../../docs/reports/side_tracks/WIDOWX200_MIGRATION_REPORT.md`](../../docs/reports/side_tracks/WIDOWX200_MIGRATION_REPORT.md) — every measurement, and what is still to do
- `tests/test_widowx200_embodiment.py`, `tests/test_smolvla_action_expert.py`
