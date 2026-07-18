# CDPR SmolVLA GRPO simulator architecture

## Scope

The existing CPU MuJoCo training path remains the default and is unchanged as
the production fallback. The new `mjlab_mjwarp` path is a separate trainer and
configuration for rank-local, GPU-resident batched rollout.

| Property | `mujoco_cpu` | `mjlab_mjwarp` |
|---|---|---|
| Entrypoint | `smolvla_grpo_finetune_cdpr.py` | `smolvla_grpo_mjwarp_cdpr.py` |
| Simulator allocation | one CPU environment/rank | one batched MJWarp `Model`/`Data` allocation/rank |
| GRPO group ownership | candidates round-robin across ranks | complete contiguous groups on one rank |
| Candidate execution | capture/restore, time-multiplexed | simultaneous fixed-shape worlds |
| Images | EGL readback through NumPy | normalized BCHW float32 tensors on rank-local CUDA |
| Candidate synchronization | per-group object gather | none |
| Update synchronization | existing CPU behavior | DDP gradients plus update-level schedule, curriculum, and metrics |

The backend is selected by a backward-compatible root block:

```yaml
simulator:
  backend: mjlab_mjwarp
  worlds_per_rank: 16
  groups_per_rank: 2
  nconmax: 256
  njmax: 512
  render_width: 320
  render_height: 240
  object_slots: 4
  fixed_scene_xml: ../../robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml
```

For MJWarp, the invariant is
`worlds_per_rank == groups_per_rank * grpo_group_size`. A group is contiguous
within a rank and never has a global identity.

## Backend boundary

`CDPRSimulatorBackend` owns simulator and controller state. Its contract covers
partial reset, broadcasting one base state within complete local groups,
fixed-shape stepping with completion masks, low-dimensional observations,
overview/wrist rendering, body pose and velocity queries, contacts, object and
visual variants, reset poses, validation export, and checkpoint metadata.

The CPU adapter exists for reference/parity tooling. The established CPU
trainer continues to use `CDPRLanguageRLEnv` directly, so the migration does
not insert MJWarp branches throughout that path.

The MJWarp implementation allocates one model and data object for all local
worlds. Torch views use Warp's zero-copy interop. There is no Python list of
environments in rollout. The following remain on the rank-local GPU:

- physics and controller state;
- actions and active/terminal masks;
- robot, tendon, object, and gripper observations;
- sparse rewards, outcomes, and GRPO advantages;
- both physical camera batches and all three SmolVLA image slots;
- residual-policy records and PPO padding masks.

Host transfer is limited to compact per-group task/catalog identifiers used to
construct language strings and populate the tokenizer cache at reset, one
scalar rollout horizon, update-level scalar logging, and explicit
validation/debug export. No image, action, physics, reward, mask, or
per-candidate trajectory tensor follows those transfers.

## CDPR/controller parity

The action remains `[x, y, z, yaw, gripper]` in `[-1, 1]`, with scales
`[0.015 m, 0.015 m, 0.015 m, 0.08 rad, 0.05]`. `hold_steps=6` means seven
MJWarp `step` calls per environment action.

Startup calibrates the four slider/tendon finite-difference terms from the
host MuJoCo model. The batched controller keeps target position, target yaw,
target gripper opening, prior tendon lengths, end-effector attachment
geometry, and length-to-slider conversion in reset/broadcast state. It retains
the four spatial tendons and pulley sidesites, four slider position actuators,
free end-effector, yaw, ball stabilizer, joint equality, and finger contacts.

The fixed MJCF has four free-body object slots. Each slot contains an identical
primitive superset. Per-world `geom_size`, `geom_pos`, `geom_quat`,
`geom_rgba`, mass, inertia, material, and pose select one of:
`ycb_apple`, `ycb_pear`, `ycb_peach`, `ycb_b_cups`, `ycb_baseball`, `plate`,
or `bowl`. Inactive primitives are transparent and placed far below the
object frame. No reset recompiles the model.

The primitive representations preserve the checked-in stable collision
dimensions and grasp widths. They intentionally do not reproduce unavailable
high-polygon external YCB visual meshes; camera parity must therefore be
validated against the actual remote dataset/checkpoint distribution before
promotion.

## Cameras and SmolVLA

The physical cameras are named `overview` and `ee_camera`. MJWarp renders both
at 320×240 only when a policy observation is requested. GPU preprocessing
resizes to 256×256 with bilinear interpolation and keeps RGB values as
float32 in `[0, 1]`; BF16 autocast applies inside SmolVLA inference.

The active three-slot contract is exactly:

1. overview;
2. wrist/end-effector;
3. the same wrist tensor when no true auxiliary camera exists.

SmolVLA returns `[B, chunk_size, 5]`. `smolvla_inference_microbatch_size`
splits only model inference when the complete local world batch does not fit;
the simulator and collector remain fixed-shape. `torch.compile` and eager
fallback remain available. One full frozen SmolVLA replica is loaded per rank.
The active pipeline has no trainable SmolVLA LoRA parameters; DDP therefore
wraps only the residual head and its trainable log standard deviation. If a
future LoRA mode is enabled, those parameters must be placed in the same DDP
module before it is considered supported.

## Rank-local GRPO and DDP safety

Every group shares one instruction, shell, scene, catalog mapping, object
poses, robot/controller base state, and horizon. Candidate residual actions
use independent deterministic seed streams. Groups and ranks use distinct
streams.

Completed candidates remain allocated and are masked. At group termination,
sparse outcomes and centered/normalized advantages are computed independently
for each eight-candidate local group. The active continuation semantics are
preserved: a group is terminal and is reset; selected simulator state is never
transferred between ranks.

Informative record counts can differ by rank. Once per optimizer update, an
`all_reduce(MAX)` selects the common padded record count. Every rank then uses
the same number of PPO epochs, minibatches, microbatches, optimizer steps, and
backward calls. A rank with no informative rows still traverses the DDP graph
with zero-loss padding. This guarantees matching gradient collectives.

Curriculum evidence is reduced once per update and rank zero broadcasts the
canonical shell state. Frontier outcomes accumulate until the configured
50-continuation validation quota is met for an instruction, then the active
promotion/demotion threshold, minimum-train-update, maximum-one-shell-jump,
and saturation gates run. Per-instruction train-update, last-promotion,
validation, and partially accumulated quota state are checkpointed. These
continuations are the independent on-policy candidates owned by the two
ranks, rather than a second serial CPU validation environment. Rollout/update
metrics use one packed update-level reduction. There is no per-group
`all_gather_object` in the MJWarp collector.

`sampled_environment_actions` counts active action steps across all eight
candidates. `selected_environment_actions` applies the configured
uniform/best/softmax group selection and counts the corresponding one
terminal continuation per group, preserving the existing throughput
denominator and work-amplification definition. `global_step` is explicitly
the cumulative selected count summed across every rank. It is an
aggregated-work counter, not wall-clock decisions and not a per-rank count.
Per-component CUDA timing barriers are disabled in normal training. The
benchmark opts into `--mjwarp-profile-timers`, and emitted metrics record
whether component timers were synchronized, so profiling cannot silently
serialize the production hot path.

The active reverse-frontier config has
`lock_non_commanded_axes: false`; MJWarp therefore builds each XYZ target from
the measured end-effector pose exactly like the CPU environment. The optional
locked-axis mode is implemented in the backend and uses the same `0.05`
normalized-action threshold, but changing it is checkpoint-incompatible.

## Checkpoints

New checkpoints store backend, exact dependency versions, world/group counts,
group size, physics substeps and dtype, XML path, contact/constraint
capacities, render dimensions, camera ordering, object catalogs, curriculum,
and the global-step definition.

Residual-head and existing SmolVLA checkpoints remain loadable. A legacy
checkpoint without simulator metadata is rejected by default in the MJWarp
trainer. Use `--allow-legacy-simulator-checkpoint` only after reviewing
the old assumptions. A checkpoint with contradictory backend, group size,
world count, substeps, object slots, or camera contract is rejected rather
than silently resumed.
