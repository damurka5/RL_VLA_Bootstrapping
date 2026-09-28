# Current PPO-to-GRPO OpenVLA CDPR Pipeline Report

## Abstract

This note reconstructs the current OpenVLA-based cable-driven parallel robot (CDPR) reinforcement-learning pipeline from the local repository state. The reconstruction uses the current PPO report, the PPO and GRPO trainer implementations from `openvla-oft`, the CDPR environment and reward code in this repository, and the currently available resume configurations. The goal is to describe the task, the reward, the PPO and GRPO formulations, the randomized scene generation, and the precise meaning of the training counters that appear during execution.

One practical caveat is important at the outset. The repository snapshot contains PPO example configurations for the intervals `0-60`, `120-180`, `180-280`, and `280-380`, and it contains both PPO and GRPO variants for the `380-480` move-to-object stage. The intermediate `60-120` PPO phase is referenced by checkpoint name in the later configs, but its example YAML is not present in this snapshot. Therefore, the historical statement "PPO was used for the first 280 updates and GRPO was started afterwards" is taken here as the experiment narrative supplied by the user, while the technical description of the active GRPO stage is taken from the current GRPO configuration file.

## 1. Task and control problem

The policy is trained for a 5-DoF MuJoCo CDPR embodiment with action channels

```text
a = [a_x, a_y, a_z, a_yaw, a_gripper] in [-1, 1]^5.
```

The current GRPO stage uses two camera views, namely an overview camera and an end-effector wrist camera, together with a language instruction. The OpenVLA policy itself is vision-language conditioned: the trainer feeds the two images and the text prompt to OpenVLA, and the reward/state logic uses the structured environment state only for reward computation, diagnostics, and reset bookkeeping.

The current move-to-object stage uses the instruction set

```text
{move_left, move_right, move_top, move_bottom, move_to_object}.
```

The active target-object pool is

```text
{ycb_apple, ycb_pear, ycb_peach, ycb_b_cups, ycb_mug, ycb_baseball, ycb_plate, ycb_bowl}.
```

The policy does not emit a single 5-dimensional action. Instead, it emits an open-loop chunk of `NUM_ACTIONS_CHUNK = 8` low-level actions. Since `ACTION_DIM = 5`, the policy head predicts a flattened 40-dimensional output:

```text
u in R^(5 x 8) = R^40.
```

After reshaping, one policy decision corresponds to an 8-step action chunk,

```text
u = [u_1, ..., u_8],  u_k in R^5.
```

Each low-level action is mapped to physical CDPR motion by

```text
Delta x   = 0.015 * a_x
Delta y   = 0.015 * a_y
Delta z   = 0.015 * a_z
Delta yaw = 0.08  * a_yaw.
```

The gripper is thresholded rather than regressed continuously: `a_gripper >= 0.2` closes the gripper and `a_gripper <= -0.2` opens it. Controller limits are `x,y in [-0.8, 0.8]`, `z in [0.08, 1.2]`, `yaw in [-pi, pi]`, and gripper opening in `[0.0, 0.03]`.

The temporal hierarchy is therefore:

```text
update
  -> rollout step (one policy decision per parallel environment)
     -> action chunk of length 8
        -> low-level 5D action
           -> 1 + hold_steps MuJoCo simulator ticks.
```

In the current GRPO config, `hold_steps = 2`, so each low-level action is held for `3` simulator ticks. Hence one policy decision can execute up to `8 x 3 = 24` MuJoCo ticks. In the earlier PPO phases, `hold_steps = 10`, so one policy decision could execute up to `8 x 11 = 88` MuJoCo ticks.

## 2. Task geometry and current curriculum stage

For the current GRPO move-to-object stage, the task metadata defines a workspace-centered curriculum with

```text
goal_center_xy = (0.0, 0.0)
goal_height_above_table = 0.10
lateral_goal_offset = 0.40
vertical_goal_offset = 0.10.
```

For directional instructions, the goal is a workspace-centered waypoint shifted by `0.40 m` in the commanded lateral direction. For `move_to_object`, the goal is the live position of the selected target object. The environment samples instructions with `instruction_sampling = uniform_cycle`, so the allowed instruction types are cycled approximately uniformly rather than drawn independently at every reset.

The local configs show a curriculum change across training. Earlier PPO phases used a broader directional instruction set, including `move_up`, `move_down`, and `move_center`, together with a larger object pool and wider random end-effector starts. The current move-to-object stage is therefore not merely a change of optimizer; it is also a task-stage refinement toward object-centric motion.

## 3. Reward definition and maximum values

### 3.1 Directional instructions

For the directional instructions used in the current stage (`move_left`, `move_right`, `move_top`, `move_bottom`), the reward has the form

```text
r_dir = r_dist + r_success - r_sat,
```

with

```text
r_dist = 1 / (1 + (d / 0.40)^2),
r_success = 1 if d <= 0.03 else 0,
```

where `d` is the Euclidean end-effector distance to the instruction goal. In the present configs, camera alignment reward is disabled, so there is no active orientation term in the directional reward.

The action-saturation penalty is computed on the non-gripper coordinates only:

```text
tau = 0.95,
e_i = clip((|a_i| - tau) / (1 - tau), 0, 1),
r_sat = mean_i(e_i^2),  i in {x, y, z, yaw}.
```

Because `r_dist <= 1`, `r_success <= 1`, and `r_sat >= 0`, the theoretical per-step maximum for the directional family is

```text
r_dir,max = 1 + 1 = 2.0.
```

The penalty itself is bounded above by `1.0`, so strong saturation can remove up to one reward unit from a directional step.

### 3.2 Move-to-object instruction

For `move_to_object`, the reward is explicitly decomposed into progress, proximity, success, and saturation terms:

```text
r_mto = r_prog + r_prox + r_above + r_success - r_sat.
```

With the current metadata,

```text
r_prog   = 1.25 * clip((d_xy,prev - d_xy) / 0.08, -1, 1),
r_prox   = 0.75 / (1 + (d_xy / 0.08)^2),
r_above  = 0.50 if d_xy <= 0.02 else 0,
r_success = 1.00 if d_xy <= 0.02 else 0,
```

where `d_xy` is the planar distance between end effector and target object. The same saturation penalty `r_sat` is subtracted.

The theoretical per-step maximum is therefore

```text
r_mto,max = 1.25 + 0.75 + 0.50 + 1.00 = 3.50.
```

This maximum is reached only in the idealized case of maximal positive clipped progress, zero planar distance, success, and zero saturation penalty.

### 3.3 Trainer-side reward handling

After the environment computes the reward, the trainer sanitizes it. Non-finite rewards are replaced by a fallback penalty, and excessively large rewards are clipped. In the current PPO and GRPO configs, the extra trainer-side distance-shaping coefficients are both zero:

```text
delta_closer_reward_coef = 0,
delta_farther_penalty_coef = 0.
```

Therefore, the reward used by PPO or GRPO is effectively the sanitized environment reward rather than an additional hand-shaped progress reward from the trainer.

## 4. PPO formulation in this pipeline

The PPO policy is a squashed Gaussian policy in pre-`tanh` latent space. Let `o` denote the multimodal observation, let `u_theta(o) in R^40` be the action-head mean output before `tanh`, and let `sigma_theta = exp(log_std)` be the learned exploration scale. Then

```text
z ~ N(u_theta(o), diag(sigma_theta^2)),
a = tanh(z).
```

Equivalently,

```text
z = u_theta(o) + sigma_theta ⊙ epsilon,   epsilon ~ N(0, I),
a = tanh(z).
```

The OpenVLA backbone contributes the multimodal hidden states, the action head predicts the 40-dimensional mean structure, and PPO also trains a value head `V_phi(o)` for generalized advantage estimation (GAE). The PPO advantage is

```text
delta_t = r_t + gamma * V(o_{t+1}) - V(o_t),
A_t = sum_{l >= 0} (gamma * lambda)^l * delta_{t+l},
```

with current PPO hyperparameters `gamma = 0.99` and `lambda = 0.95`.

The clipped PPO objective is the standard one:

```text
rho_t = pi_theta(a_t | o_t) / pi_theta_old(a_t | o_t),
L_policy = E[max(-rho_t A_t, -clip(rho_t, 1-eps, 1+eps) A_t)].
```

In this implementation, PPO additionally uses a value loss and an entropy bonus, so the optimized loss is of the form

```text
L_PPO = L_policy + c_v * L_value - c_e * H.
```

The PPO-specific trainable components are the LoRA adapters inside OpenVLA, the action head, the value head, and the learned `log_std` vector.

## 5. GRPO formulation in this pipeline

GRPO in this repository is implemented as a direct extension of the PPO code path. The key design change is that the learned value head is removed from optimization. The GRPO policy uses a zero value head placeholder and trains only the LoRA adapters, the action head, and `log_std`.

For each environment state inside a rollout step, GRPO first samples a group of candidate action chunks from the same policy distribution. In the current config,

```text
grpo_group_size = 2.
```

The trainer captures the full simulator snapshot, executes each candidate branch from the identical starting state, records the branch rewards, and then restores the simulator back to the same base snapshot before trying the next branch. If the two candidate rewards for environment `i` are `R_{i,1}` and `R_{i,2}`, then the group-relative advantages are

```text
A_{i,k}^grp = clip((R_{i,k} - mean_j R_{i,j}) / (std_j R_{i,j} + eps), -6, 6),
```

because the current config enables group normalization and uses absolute clipping:

```text
grpo_normalize_group_advantage = true,
grpo_clip_advantage_abs = 6.0.
```

One candidate is then selected uniformly to continue the real environment trajectory:

```text
grpo_group_selection = uniform.
```

This continuation rule preserves on-policy evolution of the actual simulator state, while the policy update still uses all candidates in the sampled group.

The policy objective remains PPO-like, but the advantage now comes from grouped relative rewards rather than from GAE:

```text
rho = pi_theta(a | o) / pi_theta_old(a | o),
L_GRPO = E[max(-rho A^grp, -clip(rho, 1-eps, 1+eps) A^grp)] - c_e * H.
```

There is no learned critic term in the GRPO loss. Conceptually, GRPO is therefore replacing critic-based credit assignment with within-group relative ranking at fixed observations.

## 6. Randomized scenes and randomized end-effector start

The effective domain randomization used by the RL pipeline comes from the CDPR environment and trainer, not merely from the high-level YAML labels. In the current GRPO stage:

1. Scene variants are generated from the catalog scene names and the target/distractor object pools.
2. Each scene contains between `1` and `3` objects.
3. The environment builds `128` scene variants for the current stage.
4. The trainer uses `scene_sampling = round_robin`, so resets cycle through available scene names rather than relying on unconstrained repeated random picks.
5. Desk textures are randomized through a prebuilt cache with `scene_pool_size = 32` and `texture_pool_size = 10`.

Object composition is randomized because the target-object pool and distractor-object pool are both sampled into scene variants. Physical placement is then randomized again inside the environment reset logic by non-overlapping placement within the tabletop workspace while keeping a safety margin from the end effector.

The end-effector start is also randomized, but only laterally in the current stage. The active GRPO config sets

```text
randomize_ee_start = true,
ee_start_x_bounds = [-0.03, 0.03],
ee_start_y_bounds = [-0.03, 0.03].
```

Although the config also passes `ee_start_z = 0.15`, the environment clamps this value to the minimum allowed spawn height `MIN_EE_START_Z = 0.40`. Consequently, the current random end-effector start is an `x-y` randomization around the default center, while the effective reset height remains at least `0.40 m`.

Earlier PPO phases used wider lateral start bounds, for example `[-0.25, 0.25]` in the initial fast config and `[-0.10, 0.10]` in later PPO resumes.

## 7. What one update means

An update is one complete collect-then-optimize cycle:

```text
1. Collect rollout data from the current policy.
2. Convert the collected data into PPO or GRPO training targets.
3. Run several optimization epochs over that collected batch.
4. Optionally validate and save a checkpoint.
```

Thus, an update is not a single episode and it is not a single gradient step. It is a full on-policy iteration.

## 8. What the "170 rollouts" mean

The config parameter `rollout_steps = 170` does not mean 170 complete episodes. It means 170 collection iterations per update, and at each collection iteration the trainer queries the policy once for every parallel environment.

In the current GRPO stage, the local per-rank collection volume is

```text
170 rollout_steps x 10 parallel envs = 1700 selected policy transitions per update per rank.
```

Because `nproc_per_node = 2`, the total selected policy transitions across both ranks are

```text
170 x 10 x 2 = 3400 selected policy transitions per update globally.
```

GRPO additionally evaluates two candidate branches at each state, so the local candidate volume is

```text
170 x 10 x 2 = 3400 candidate transitions per rank,
```

and the global candidate volume is

```text
170 x 10 x 2 x 2 ranks = 6800 candidate transitions per update.
```

This is why the GRPO trainer prints both a selected-step count and a candidate-sample count.

## 9. Why 170 rollout steps are not 170 episodes

Each policy transition is an 8-action chunk, and the base environment horizon is `max_env_steps = 32` low-level actions. Therefore, in the absence of earlier success or instability resets, one episode can last at most

```text
32 / 8 = 4
```

policy decisions.

Hence 170 rollout steps correspond to many short episodes distributed across the parallel environments, not to 170 episodes. With 10 parallel environments, one update can span on the order of hundreds of short resets.

## 10. What the "~836 additional rollouts" most likely are

The second counter that appears after rollout collection is not another set of environment rollouts. It is the optimizer minibatch loop. In GRPO, the number of stored training examples per rank is

```text
N_train = rollout_steps x num_parallel_envs x grpo_group_size
        = 170 x 10 x 2
        = 3400.
```

With

```text
ppo_epochs = 4,
minibatch_size = 16,
```

the number of optimizer minibatches per rank is

```text
N_mb = 4 x ceil(3400 / 16) = 4 x 213 = 852.
```

Therefore the post-rollout progress bar is expected to be roughly in the `8.5 x 10^2` range. If a run shows approximately `836`, that counter should still be interpreted as the PPO-style training minibatch count rather than as additional environment episodes. Small differences can arise if the actual launched run differs slightly in `minibatch_size`, active environment count, or other runtime details.

For comparison, the earlier PPO phases used `num_parallel_envs = 12`, so their local per-update transition count was

```text
170 x 12 = 2040,
```

and their local optimizer-minibatch count was

```text
4 x ceil(2040 / 16) = 512.
```

## 11. Checkpoint naming and the meaning of `step_0163200`

Checkpoint directories are named by `global_step`, not by update index. In the PPO trainer, `global_step` is incremented once for every selected policy decision in each local rank loop. For the PPO move-to-object stage with `170` rollout steps and `12` local parallel environments, one update advances local `global_step` by

```text
170 x 12 = 2040.
```

Therefore, the checkpoint name

```text
step_0163200
```

corresponds to

```text
163200 / 2040 = 80
```

local PPO updates inside that particular run, not necessarily to "update 380" in the global experiment narrative. The config filenames such as `380_to_480` are experiment labels, whereas the numeric checkpoint suffix records interaction-step count.

## 12. PPO-to-GRPO transition and saturation behavior

According to the current experiment description supplied by the user, PPO was used for the first 280 updates and reduced the saturation rate to approximately `0.25`, after which GRPO was introduced and reduced the same saturation indicator further to approximately `0.15` after 10 GRPO updates. Scientifically, this indicates that the GRPO phase is producing fewer near-boundary actions than the preceding PPO phase, which is consistent with better control smoothness and less actuator saturation.

However, the code exposes two different saturation-related measurements, so the comparison is meaningful only if the same metric source is used on both sides:

```text
1. Reward-side action_saturation_rate:
   threshold = 0.95, computed in the environment, excludes the gripper dimension.

2. Validation-side sat_frac_abs_ge_0_99:
   threshold = 0.99, computed from deterministic validation actions.
```

The qualitative conclusion is still the same: lower saturation means the policy spends less time at the action bounds.

## 13. Summary

The current pipeline is a vision-language, chunked-action, on-policy RL system for a MuJoCo CDPR robot. PPO models the policy as a squashed Gaussian over 40 chunked action coordinates and uses a learned value head with GAE. GRPO removes the learned critic, samples grouped candidate actions from identical simulator snapshots, converts within-group reward differences into relative advantages, and then optimizes a PPO-style clipped policy objective without a value loss. The current move-to-object stage uses randomized multi-object scenes, randomized desk textures, randomized lateral end-effector start, and short 8-action policy chunks with a 32-step low-level episode horizon. In this setting, `170 rollout_steps` means 170 policy-decision collection iterations per update, whereas the later `~8.5e2` counter is the minibatch optimization loop, not another set of environment rollouts.
