# CDPR LC-HOL Implementation Notes

This note describes what was implemented for the OpenVLA-OFT CDPR complex-task experiment based on the LC-HOL / hindsight-option hypothesis.

Important framing: the final runnable training path uses **GRPO** as the RL algorithm, as requested. The LC-HOL contribution was implemented mainly in the CDPR task/environment layer: richer language-conditioned options, object-relation goals, sparse hindsight-friendly success predicates, a grab-first curriculum, and dense rollout diagnostics. I intentionally did not leave a separate off-policy LC-HOL trainer as the production entry point, because the requested algorithm was GRPO.

## Goal

The original CDPR policy checkpoint was trained on simpler move-to-object behavior:

```bash
--adapter-path /root/repo/RL_VLA_Bootstrapping/runs/step_390_to_490_movetoobj_20260420_153108/rl/step_0170000/vla_cdpr_adapter
--action-head-path /root/repo/RL_VLA_Bootstrapping/runs/step_390_to_490_movetoobj_20260420_153108/rl/step_0170000/action_head_cdpr.pt
```

The new experiment extends that checkpoint toward manipulation and object-relation tasks:

- `grab <obj>`
- `pick up <obj>`
- `push <obj> left`
- `push <obj> right`
- `put <obj> into plate`
- `move <obj1> to the left of <obj2>`
- `move <obj1> to the right of <obj2>`
- `move <obj1> between <obj2> and <obj3>`

The curriculum starts with grasping before introducing relation tasks, because the harder tasks are not meaningful until the policy has learned to close the gripper near the correct object and move it.

## Main Entry Points

The runnable GRPO experiment is configured here:

```text
configs/examples/cdpr_openvla_grpo_complex_tasks.yaml
```

The launcher is:

```text
scripts/train_cdpr_grpo_complex_tasks.sh
```

The complex-task reachability video script is:

```text
scripts/render_cdpr_complex_task_success.py
```

The GRPO wrapper that patches the external OpenVLA-OFT trainer is:

```text
rl_vla_bootstrapping/policy/grpo_finetune_cdpr_fast.py
```

The CDPR environment and instruction/reward logic are:

```text
robots/cdpr/cdpr_dataset/rl_cdpr_env.py
robots/cdpr/cdpr_dataset/rl_instruction_tasks.py
```

## LC-HOL Pieces Implemented

### 1. Language-Conditioned Option Vocabulary

I extended the CDPR instruction vocabulary from simple navigation/lift actions into manipulation options. The new instruction families are:

```text
grab_object
put_into_plate
move_left_of_object
move_right_of_object
move_between_objects
push_left
push_right
```

The instruction generator now supports not only a primary target object, but also one or two reference objects. For example:

- target: `ycb_apple`
- reference: `plate`
- instruction type: `put_into_plate`
- text: `put apple into plate`

For relation tasks, the reference objects become part of the option definition:

- `move apple to the left of mug`
- `move apple between mug and pear`

This is the language-conditioned part of LC-HOL: the same low-level action space is conditioned on a discrete language option and the option carries the target/reference object bindings.

### 2. Object Binding and Reference Tracking in the Environment

The CDPR environment now tracks:

```text
_target_catalog_name
_target_body_name
_reference_catalog_name
_reference_body_name
_second_reference_catalog_name
_second_reference_body_name
```

This was necessary because relation tasks cannot be evaluated from a single target object. The environment now chooses objects differently depending on the instruction type:

- `put_into_plate`: chooses a movable target and a plate-like reference.
- `move_left_of_object` / `move_right_of_object`: chooses a target and one distinct reference object.
- `move_between_objects`: chooses a target and two distinct references.
- push/grab/pick tasks only require the primary target.

The environment info dict now exposes the resolved target/reference object names and positions, which makes TensorBoard and validation much easier to interpret.

### 3. Relation-Aware Goal Computation

The environment now computes a live goal position differently for each option:

- `put_into_plate`: goal is the plate/reference center.
- `move_left_of_object`: goal is reference position minus `relation_left_right_offset` on X.
- `move_right_of_object`: goal is reference position plus `relation_left_right_offset` on X.
- `move_between_objects`: goal is midpoint between the two reference objects.
- `push_left/right`: goal is the initial object position shifted by the required push displacement.

This makes the same CDPR control stack usable for both direct object approach and relational manipulation objectives.

### 4. Sparse Success Predicates

The manipulation tasks use sparse success predicates, which are compatible with the hindsight-option idea: once a trajectory reaches a meaningful relation, the outcome can be interpreted as success for a corresponding option.

Current thresholds are configured in:

```text
configs/examples/cdpr_openvla_grpo_complex_tasks.yaml
```

Threshold table:

| Task | Success condition |
| --- | --- |
| `grab_object` | gripper closed and either target contact is detected or EE is within `0.045 m` XY of target |
| `push_left/right` | target moved at least `0.08 m` in commanded X direction |
| `put_into_plate` | target center within `0.08 m` XY and `0.10 m` Z of plate |
| `move_left/right_of_object` | target at least `0.08 m` left/right of reference, Y error <= `0.12 m`, target moved >= `0.02 m` |
| `move_between_objects` | target within `0.07 m` of midpoint, projected between references, target moved >= `0.02 m` |
| `pick_up` | lift height >= `0.05 m`, grasp XY threshold `0.04 m`, closed opening threshold `0.010` |

The sparse reward is:

```text
1.0 for success
0.0 for failure
minus optional action saturation penalty
```

Action saturation penalty is currently disabled in the complex-task config:

```yaml
action_saturation_penalty_weight: 0.0
```

### 5. Reward Info for Hindsight-Style Diagnostics

The reward code now emits additional per-step information:

```text
sparse_success
sparse_reward_mode
distance_ee_to_object
distance_ee_to_object_xy
target_motion_x
target_motion_y
target_motion_z
target_motion_xy
relation_error
signed_relation_offset
relation_motion_required
relation_motion_ok
gripper_closed
grasped
caught_object_score
caught_object_is_target
```

These fields are important because they let us see whether the policy is failing at:

- approaching the object,
- closing the gripper,
- actually catching the target,
- moving the object,
- satisfying the relation,
- or simply missing a threshold.

This is the practical monitoring layer for the LC-HOL hypothesis.

### 6. Grab-First Curriculum

The curriculum is implemented inside the CDPR environment. It filters the allowed instruction types by per-env episode index.

Current curriculum:

```yaml
instruction_curriculum:
  - until_episode: 80
    instruction_types: [grab_object]
  - until_episode: 160
    instruction_types: [grab_object, pick_up]
  - until_episode: 260
    instruction_types: [grab_object, pick_up, push_left, push_right]
  - until_episode: 420
    instruction_types: [grab_object, pick_up, push_left, push_right, put_into_plate]
  - instruction_types:
      [
        grab_object,
        pick_up,
        push_left,
        push_right,
        put_into_plate,
        move_left_of_object,
        move_right_of_object,
        move_between_objects,
      ]
```

This means each parallel GRPO environment first sees only `grab_object`, then gradually receives harder options. The goal is to let the checkpoint reuse move-to-object competence, then learn gripper closure, then learn object displacement and relation satisfaction.

### 7. GRPO Training Configuration

The final training config uses GRPO:

```yaml
algorithm: grpo
script_path: ../../rl_vla_bootstrapping/policy/grpo_finetune_cdpr_fast.py
launcher: torchrun
launcher_args:
  nproc_per_node: 2
```

Main GRPO parameters:

```yaml
grpo_group_size: 2
grpo_group_selection: uniform
grpo_normalize_group_advantage: true
grpo_clip_advantage_abs: 6.0
num_parallel_envs: 10
total_updates: 120
rollout_steps: 240
ppo_epochs: 4
minibatch_size: 16
microbatch_size: 16
max_env_steps: 120
gamma: 0.99
learning_rate: 5.0e-6
adam_eps: 1.0e-5
weight_decay: 0.0
normalize_advantage: false
init_log_std: -1.2
gradient_checkpointing: true
```

Control parameters:

```yaml
action_step_xyz: 0.015
action_step_yaw: 0.08
action_step_gripper: 0.05
hold_steps: 6
lock_non_commanded_axes: false
randomize_ee_start: true
ee_start_x_bounds: [-0.25, 0.25]
ee_start_y_bounds: [-0.25, 0.25]
ee_start_z: 0.15
```

`hold_steps: 6` means every policy action is applied for `1 + 6 = 7` simulation substeps.

Rendered frames are required by the external OpenVLA-OFT GRPO trainer, so the config uses:

```yaml
capture_frames: true
```

### 8. External Trainer Compatibility Fixes

Two remote-run compatibility issues were fixed after testing:

1. `--action_step_gripper` was rejected by the external OpenVLA-OFT trainer.

   Fix: the bootstrap OpenVLA plan now removes `action_step_gripper` from CLI args and passes it as:

   ```bash
   RLVLA_CDPR_ACTION_STEP_GRIPPER=0.05
   ```

   The CDPR env already reads this env var.

2. The external GRPO trainer requires rendered images.

   Fix: `capture_frames` was changed to `true`, so the generated command passes:

   ```bash
   --capture_frames
   ```

### 9. Frequent TensorBoard Logging

I extended the GRPO fast wrapper to log rollout-step metrics very frequently, not only after optimizer updates.

Config:

```yaml
tensorboard_rollout_every_global_steps: 8
tensorboard_every_updates: 1
```

The wrapper logs `rollout_step/*` scalars every 8 global environment steps. Added metric groups include:

- reward means,
- sparse success rate,
- target grasp/contact rates,
- object motion,
- relation error,
- distance to object,
- action saturation,
- instability/non-finite reward rates.

This is useful for complex CDPR tasks because optimizer-level metrics alone are too slow and too indirect. During early training, the most useful signals are often things like `gripper_closed`, `distance_ee_to_object_xy`, `target_motion_xy`, and `relation_error`.

### 10. Reachability Video

I added a separate video script:

```bash
python3 scripts/render_cdpr_complex_task_success.py
```

By default it renders a reliable schematic video of `put red cube into plate`. It writes a timestamped manifest and video under:

```text
runs/cdpr_complex_task_success_video/
```

The generated example video had:

```text
900x700 resolution
210 frames
10.5 seconds
success: true
plate_xy_error_m: 0.0
```

The script also supports a MuJoCo renderer:

```bash
python3 scripts/render_cdpr_complex_task_success.py --render-mode mujoco
```

I left schematic as the default because local macOS offscreen MuJoCo rendering entered an uninterruptible graphics wait. On the remote EGL machine, MuJoCo mode should be the preferred physical rendering path.

### 11. Tests Added or Updated

Tests cover:

- complex instruction text generation,
- canonical object names,
- sparse success/reward predicates,
- relation goal computation,
- instruction curriculum filtering,
- required-object scene filtering,
- GRPO TensorBoard rollout logging,
- bootstrap command/env generation for CDPR-specific env vars.

Relevant tests:

```text
tests/test_cdpr_instruction_tasks.py
tests/test_cdpr_reward.py
tests/test_cdpr_instruction_goals.py
tests/test_cdpr_env_instruction_sampling.py
tests/test_grpo_finetune_cdpr_fast.py
tests/test_pipeline.py
```

The focused suite passed locally after the implementation.

## How to Run on Remote

On the remote server:

```bash
cd /root/repo/RL_VLA_Bootstrapping/RL_VLA_Bootstrapping
git pull origin main

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

bash scripts/train_cdpr_grpo_complex_tasks.sh
```

Expected external paths:

```text
/root/repo/openvla-oft
/root/repo/openvla-oft/vla-scripts/grpo_finetune_cdpr.py
/root/repo/RL_VLA_Bootstrapping/runs/step_390_to_490_movetoobj_20260420_153108/rl/step_0170000/vla_cdpr_adapter
/root/repo/RL_VLA_Bootstrapping/runs/step_390_to_490_movetoobj_20260420_153108/rl/step_0170000/action_head_cdpr.pt
```

TensorBoard:

```bash
tensorboard --logdir runs
```

Look first at:

```text
rollout_step/sparse_success_mean
rollout_step/distance_ee_to_object_xy_mean
rollout_step/gripper_closed_mean
rollout_step/caught_object_is_target_mean
rollout_step/target_motion_xy_mean
rollout_step/relation_error_mean
```

## Limitations and Next Steps

1. The final training algorithm is GRPO, not a standalone off-policy LC-HOL/HER trainer.

   The LC-HOL hypothesis is represented in the task decomposition, option predicates, object-relation success functions, and curriculum. If we later want true hindsight relabeling of failed trajectories into alternative successful language options, that would require adding replay/relabeling support to the training loop or a separate data-generation stage.

2. The manipulation rewards are intentionally sparse.

   This keeps success definitions clean, but early training may be slow. If success rate stays near zero, the next step should be adding shaped auxiliary rewards for:

   - EE-to-object approach,
   - gripper close timing near target,
   - object displacement in the intended direction,
   - relation error reduction.

3. Relation tasks depend on reliable object placement and object/plate availability.

   The environment now tries to choose plate-like references for `put_into_plate`, but the asset/catalog setup on remote must actually provide `plate` or equivalent plate-like objects.

4. `capture_frames: true` is required by OpenVLA GRPO.

   This increases runtime cost, but the external trainer needs rendered image observations.

5. The current curriculum uses per-env episode index.

   In a vectorized run, each worker advances independently. This is simple and robust, but not a global success-based curriculum. If needed, a future version could promote curriculum stages based on measured success windows.

## Implementation Commits

The main implementation landed in these commits:

```text
8367fef Add GRPO complex CDPR task training
bed8d8b Pass CDPR gripper step through env
93d0f82 Enable CDPR frame capture for GRPO
```

