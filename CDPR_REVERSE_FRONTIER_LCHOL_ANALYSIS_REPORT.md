# CDPR Reverse-Frontier LC-HOL Training Implementation Analysis

Date: 2026-05-26

This report reviews the current CDPR/OpenVLA implementation against the intended idea:

> Use a minimal viable dense-reward policy for simple robot actions, then fine-tune it on unseen complex instructions with sparse binary reward using Reverse Curriculum Generation, frontier sampling, scheduler-probe validation, and LC-HOL++ compatibility.

The implementation mostly matches the curriculum and sparse-reward part of the idea. The main qualification is that the "HER" component is not full Hindsight Experience Replay in the classic sense. The current code uses LC-HOL-compatible hindsight relabeling as an auxiliary behavior-cloning replay loss, while explicitly keeping relabeled/replay data out of the GRPO policy-gradient batch.

## Files Reviewed

Primary training configs:

- `configs/examples/cdpr_openvla_bootstrap_fast.yaml`
- `configs/examples/cdpr_openvla_grpo_step_390_to_490_movetoobj.yaml`
- `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml`

Primary implementation files:

- `robots/cdpr/cdpr_dataset/rl_cdpr_env.py`
- `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py`
- `robots/cdpr/cdpr_dataset/cdpr_reverse_shells.py`
- `robots/cdpr/cdpr_dataset/cdpr_lchol_spec.py`
- `rl_vla_bootstrapping/lchol/grpo_runtime.py`
- `rl_vla_bootstrapping/lchol/frontier_scheduler.py`
- `rl_vla_bootstrapping/lchol/curriculum.py`
- `rl_vla_bootstrapping/lchol/replay_buffers.py`
- `rl_vla_bootstrapping/policy/grpo_finetune_cdpr_fast.py`
- `rl_vla_bootstrapping/policy/openvla_oft.py`
- local external trainer: `/Users/damirnurtdinov/Desktop/My Courses/Диплом/openvla-oft/vla-scripts/grpo_finetune_cdpr.py`

Existing reports were also reviewed. `CURRENT_CDPR_PPO_GRPO_PIPELINE_REPORT.md` and `CDPR_LC_HOL_IMPLEMENTATION.md` are useful background, but parts of them are stale relative to the current configs and reward code.

## Executive Verdict

The repo contains a working reverse-frontier curriculum implementation for complex sparse CDPR manipulation tasks. It has:

- a dense/proximity pretraining path for simple movement and object approach;
- sparse binary complex-task rewards;
- reverse-shell reset generation for success-near starts;
- frontier/rehearsal reset sampling;
- scheduler validation probes with promotion and demotion thresholds;
- LC-HOL-compatible hindsight records and replay BC;
- audits that prevent relabeled replay from contaminating the GRPO policy-gradient batch.

The biggest gaps are:

- the implementation is HER-inspired, not full HER;
- sparse GRPO currently uses `lchol_group_score: env_reward`, so many candidate groups can have all-zero reward and no useful group advantage;
- some configured LC-HOL++ fields are parsed but inactive;
- the reverse-curriculum metadata block in the YAML is not the active scheduler implementation;
- scheduler exposure accounting is per global update, not per instruction actually sampled;
- the complex stage starts from a move-to-object GRPO checkpoint, not directly from the dense primitive checkpoint.

## Training Flow: Step by Step

### 1. Configuration is loaded and exported to the trainer

`rl_vla_bootstrapping/policy/openvla_oft.py` reads the YAML config and exports task metadata, reward config, success config, and validation config through environment variables. The OpenVLA/OFT trainer receives those values as CLI arguments and environment variables.

Important exported controls include:

- instruction families and target object pools;
- `reward_mode`;
- success thresholds;
- scene variant count and object placement bounds;
- LC-HOL and reverse-frontier flags;
- checkpoint paths for adapter/action-head initialization.

### 2. Dense PPO primitive bootstrap trains simple movement

Config: `configs/examples/cdpr_openvla_bootstrap_fast.yaml`

This is the minimal-viable policy stage. It trains primitive commands such as:

- `move_up`
- `move_down`
- `move_left`
- `move_right`
- `move_top`
- `move_bottom`
- `move_center`

The reward is dense: distance-based shaping plus success bonus and action saturation penalty. Scene randomization exists even at this stage:

- `scene_variant_count: 64`
- `min_scene_objects: 1`
- `max_scene_objects: 3`
- `scene_pool_size: 16`
- `texture_pool_size: 10`
- `scene_sampling: round_robin`
- randomized end-effector starts in X/Y.

Main hyperparameters:

- policy learning rate: `1e-5`
- value learning rate: `1e-4`
- `num_parallel_envs: 12`
- `total_updates: 60`
- `rollout_steps: 170`
- `ppo_epochs: 4`
- `minibatch_size: 16`
- `max_env_steps: 32`
- `gamma: 0.99`
- `gae_lambda: 0.95`
- `init_log_std: -1.2`

Assessment: these are conservative and reasonable for LoRA/action-head fine-tuning. The value LR is higher than the policy LR, which is normal for actor-critic PPO when the critic must adapt quickly.

### 3. Move-to-object GRPO builds an object-approach checkpoint

Config: `configs/examples/cdpr_openvla_grpo_step_390_to_490_movetoobj.yaml`

This stage trains `move_to_object` using the OpenVLA GRPO path. It is not sparse binary in the current config. It uses a proximity/XY distance reward with an action saturation penalty:

- `move_to_object_xy_reward_scale: 0.10`
- `action_saturation_penalty_weight: 0.20`
- `success_bonus: 0.0`
- `move_to_object_z_penalty_weight: 0.0`
- XY success threshold: `0.02`

Main hyperparameters:

- learning rate: `1e-5`
- `num_parallel_envs: 10`
- `total_updates: 100`
- `rollout_steps: 170`
- `ppo_epochs: 4`
- `minibatch_size: 16`
- `microbatch_size: 16`
- `max_env_steps: 32`
- `hold_steps: 2`
- `lock_non_commanded_axes: true`
- `scene_variant_count: 64`
- `scene_pool_size: 32`
- `texture_pool_size: 10`

Assessment: this intermediate stage is consistent with the idea of making the policy capable of useful simple object-directed behavior before sparse manipulation. It is stronger than just primitive motion pretraining, so the final complex stage is not starting directly from the original dense primitive checkpoint.

### 4. Complex sparse GRPO starts from the move-to-object checkpoint

Config: `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml`

The complex stage loads:

`runs/step_390_to_490_movetoobj_20260420_153108/rl/step_0170000/...`

It trains these instruction types:

- `move_to_object`
- `grab_object`
- `pick_up`
- `push_left`
- `push_right`
- `put_into_plate`
- `move_left_of_object`
- `move_right_of_object`
- `put_in_front_of_object`
- `put_behind_object`
- `move_between_objects`

The task metadata sets:

- `reward_mode: sparse_binary`
- success reward: `1.0`
- failure reward: `0.0`
- `action_saturation_penalty_weight: 0.0`
- `scene_variant_count: 256`
- `min_scene_objects: 3`
- `max_scene_objects: 4`

Main GRPO hyperparameters:

- learning rate: `1.5e-5`
- LR scheduler: cosine
- LR warmup: `5` updates
- LR minimum factor: `0.25`, so final LR floor is `3.75e-6`
- `num_parallel_envs: 10`
- `total_updates: 120`
- `rollout_steps: 240`
- `max_env_steps: 120`
- `ppo_epochs: 2`
- `minibatch_size: 8`
- `microbatch_size: 8`
- `grpo_group_size: 2`
- `grpo_normalize_group_advantage: true`
- `grpo_clip_advantage_abs: 6.0`
- `hold_steps: 6`
- `lock_non_commanded_axes: false`

Assessment: the LR is plausible for LoRA/action-head fine-tuning, but the main risk is reward sparsity rather than LR magnitude. With `grpo_group_size: 2` and `lchol_group_score: env_reward`, many candidate pairs can both receive zero reward, producing zero relative advantage. If training stalls, switching the LC-HOL group score to phase-shaped scoring is likely more important than changing LR.

### 5. Scene variants are generated and sampled

`rl_cdpr_env.py` configures scene variants from task metadata. It selects target, distractor, required, container, and reference objects while filtering for instruction compatibility.

For the complex stage:

- target pool contains catchable YCB objects;
- required objects include receptacles such as plate and bowl;
- each scene has 3 to 4 objects;
- object placement is randomized with non-overlap constraints;
- the trainer uses cached scene wrappers and round-robin sampling;
- end-effector X/Y starts are randomized in `[-0.25, 0.25]`.

Important caveat: config values such as `ee_start_z: 0.15` are effectively clamped by the environment to at least `MIN_EE_START_Z = 0.40`. If low-Z starts were intended, the current environment does not use them directly.

Assessment: scene randomization is good enough for curriculum training. `scene_variant_count: 256` is a healthy variant count, and `scene_pool_size: 64` is a practical cache subset. Round-robin sampling is useful for coverage and reproducibility. The tradeoff is that each cached run may not cover all 256 variants before refresh.

### 6. Reverse-frontier reset options are sampled

`LCHOLGRPORuntime.sample_reset_options()` calls `FrontierScheduler.sample()` when `lchol_curriculum: reverse_frontier`.

The scheduler tracks one active shell per instruction type. In the complex config:

- `lchol_reverse_sample_frontier_probability: 0.80`
- `lchol_reverse_sample_rehearsal_probability: 0.20`
- `lchol_reverse_promotion_success: 0.50`
- `lchol_reverse_demotion_success: 0.20`
- `lchol_reverse_validation_rollouts_per_shell: 50`
- `lchol_reverse_min_train_updates_before_validation: 5`

Sampling behavior:

- about 80% of reset samples come from the active frontier shell;
- about 20% can rehearse easier shells when an instruction has already been promoted;
- instruction IDs are sampled uniformly from active frontier instructions.

Assessment: the curriculum does follow success-rate scheduling. The active shell is promoted/demoted based on validation success, not fixed episode counts.

### 7. Reverse shells place the environment near success states

`cdpr_reverse_shells.py` implements shell-specific reset shaping:

- `move_to_object`: places the end-effector near the target object at increasing distances.
- `grab_object` / `pick_up`: places the gripper near the object.
- `put_into_plate`: places a held target object near/above the receptacle.
- `push_left` / `push_right`: places the end-effector and object near a pushable state.
- relation tasks: place the held object near the requested relation.
- `move_between_objects`: positions near the midpoint/between relation.

The final shell for each instruction type falls back to normal reset, so promotion outward eventually reaches the original sparse task distribution.

Review note: gripper opening convention is `0 = closed`, `1 = open`. The grab/pick reverse shell currently forces `gripper = 1.0` in several shells, meaning the gripper starts open near the object. That can still be a valid near-success curriculum if the next action is closing, but it should be intentionally validated because sparse grab success requires a closed/caught condition.

### 8. GRPO samples candidate actions from the same base state

The external GRPO trainer creates candidate branches from the same saved environment state. It evaluates `grpo_group_size` candidates, records candidate rewards, restores the base state, and then advances the real environment with the selected candidate.

The current complex config uses `grpo_group_size: 2`, so each decision compares two candidate actions.

Assessment: this is compatible with sparse frontier learning because near-success shells can make binary reward differences appear in candidate groups. The weak point is still far-shell sparse exploration: if both candidates fail, relative advantage is zero.

### 9. Sparse reward and success are computed per instruction

`rl_instruction_tasks.py` dispatches to sparse binary rewards when `reward_mode: sparse_binary`.

For sparse mode:

- reward is `1.0` on success and `0.0` otherwise;
- action saturation penalty is forced off in the sparse metadata path;
- success predicates are instruction-specific.

Examples:

- grab requires object caught/held and gripper closed, depending on metadata thresholds;
- push tasks require signed object displacement;
- put tasks require the target object inside/near the receptacle plus release/motion conditions;
- relation tasks require the correct signed XY offset and alignment;
- between tasks require midpoint/projection constraints.

Assessment: sparse binary mode is implemented correctly for the complex stage. Saturation is still logged, but it does not affect sparse reward because the sparse path forces the penalty weight to zero.

### 10. LC-HOL hindsight records are created

`LCHOLGRPORuntime.capture_candidate()` stores policy-gradient candidate records and calls the CDPR LC-HOL spec to build hindsight records.

Important behavior:

- records are built from the candidate info dictionary;
- the runtime currently passes a single-step list, not a full episode trajectory;
- relabeled hindsight records feed an auxiliary BC replay loss;
- the replay ratio and capacity are configured by:
  - `lchol_hindsight_replay_ratio: 0.25`
  - `lchol_hindsight_replay_capacity: 20000`
  - `lchol_hindsight_bc_coef: 0.20`

Assessment: this is LC-HOL-compatible hindsight BC. It is not full HER because achieved goals are not inserted as relabeled RL transitions into the policy-gradient replay. The trainer also audits against that contamination.

### 11. Optimizer update combines GRPO and auxiliary hindsight BC

The GRPO update remains based on actual policy-gradient transitions. The LC-HOL runtime can add a sampled behavior-cloning loss from hindsight replay.

The wrapper audits sampled GRPO transitions:

- source must be `pg`;
- `was_relabelled` must be false;
- `from_replay` must be false.

Assessment: this is a clean LC-HOL++ compatibility choice. It avoids biased policy-gradient estimates from relabeled sparse rewards, while still allowing hindsight to guide action likelihood.

### 12. Scheduler validation probes promote or demote shells

The GRPO wrapper intercepts validation. For each active shell, it runs reverse-validation rollouts with the corresponding reset options, records success and saturation, then calls the scheduler update.

Promotion rule:

- active shell success must be at least `0.50`;
- saturation must be below the max saturation threshold;
- enough train updates must have elapsed since last promotion.

Demotion rule:

- active shell success at or below `0.20` can demote the instruction to an easier shell.

Assessment: this is the core success-rate scheduling mechanism, and it is implemented. Two caveats:

- `record_train_update()` increments counters for all instruction types each rollout update, not only for the instruction types actually sampled.
- scheduler state saving exists, but I did not find a load/resume call in the active runtime path, so resuming a run may reset frontier state unless added elsewhere.

### 13. Checkpoints, validation, and telemetry are written

Training logs reward, success, saturation, validation metrics, and LC-HOL/reverse-frontier state. The scheduler writes JSON snapshots under `lchol_reverse_frontier/state_latest.json` in the run directory.

The complex config validates frequently:

- `validate_every_updates: 5`
- `validation_episodes: 2`
- `validation_max_steps: 120`
- reverse validation rollouts per shell: `50`

Assessment: validation cadence is frequent enough to drive shell promotion, though full validation episodes are very small (`2`). The reverse scheduler probes are the more meaningful curriculum signal.

## Hyperparameter Review

### Learning rate

Dense PPO:

- `1e-5` policy LR and `1e-4` value LR are reasonable and conservative.

Move-to-object GRPO:

- `1e-5` is reasonable for adapter/action-head fine-tuning.

Complex sparse GRPO:

- `1.5e-5` with 5-update warmup and cosine decay is plausible.
- It is a little aggressive compared with the previous stage, but the LR floor of `3.75e-6` helps.
- The bigger risk is sparse all-zero advantages, not LR instability.

Recommendation: keep `1.5e-5` if KL, clip fraction, and saturation remain stable. If success oscillates after promotion, reduce to `1e-5`. If success stays flat at zero, prefer changing `lchol_group_score` to `phase_shaped` or increasing candidate diversity before changing LR.

### Scene randomization

Complex stage scene randomization is strong:

- 256 scene variants;
- 3 to 4 objects per scene;
- target/distractor/container/reference filtering;
- non-overlapping object placement;
- randomized end-effector starts;
- texture pool enabled;
- round-robin cached scene sampling.

Recommendation: keep this level for final training. For debugging, reduce scene count only temporarily. Also note the Z-start clamp to 0.40 if lower starts were expected.

### Curriculum scheduling

The reverse frontier curriculum is implemented and uses validation success rates for shell movement.

Good:

- shell promotion/demotion exists;
- frontier and rehearsal sampling exist;
- validation probes use shell-specific reset options;
- final shell returns to normal task reset.

Needs improvement:

- exposure counters should be instruction-specific rather than global-update based;
- scheduler state should be loaded on resume;
- active instruction sampling is uniform, not weighted toward weakest frontier tasks.

### Sparse reward

Sparse reward is implemented correctly as binary success/failure for the complex stage.

Main training risk:

- with `lchol_group_score: env_reward`, sparse failures create no relative GRPO signal;
- reverse shells make near-success reward differences more likely, but harder shells can still starve.

Recommendation: consider `lchol_group_score: phase_shaped` during early reverse-frontier training, then switch to `env_reward` for later sparse-only validation/fine-tuning.

### Rollout and horizon

Complex stage:

- `rollout_steps: 240`
- `max_env_steps: 120`
- `hold_steps: 6`
- `ppo_epochs: 2`
- batch sizes `8/8`

Assessment: the longer horizon is appropriate for manipulation and relation tasks. `ppo_epochs: 2` is conservative and helps avoid overfitting stale sparse rollouts.

### Action constraints

Complex stage uses:

- `lock_non_commanded_axes: false`

Assessment: this is appropriate for manipulation, because relation, put, grab, and push tasks need coupled 3D behavior. It makes exploration harder than the move-to-object stage, so reverse shells are especially important.

## LC-HOL++ Compatibility Review

Implemented:

- task-specific LC-HOL spec;
- hindsight record construction;
- auxiliary BC replay;
- GRPO transition audit that excludes relabeled/replay samples;
- optional phase score machinery;
- reverse-frontier scheduler state export.

Partially implemented or inactive:

- `lchol_group_score: phase_shaped` exists but is not used by the current complex config;
- `option_prior_*` fields are parsed but not actively used in the searched runtime path;
- `hindsight_done_weight` is parsed but not actively used in the searched runtime path;
- full-episode trajectory relabeling is not active because candidate capture builds records from single-step info.

Conclusion: the code is LC-HOL-compatible, but not all LC-HOL++ knobs in the config are currently functional.

## Important Gaps and Risks

1. HER is not full HER.

The repo does not replay relabeled achieved-goal transitions through GRPO. Hindsight is used as auxiliary BC. This is safer for GRPO but should be described accurately in the thesis/report.

2. Sparse all-zero candidate groups can stall GRPO.

With `grpo_group_size: 2` and `lchol_group_score: env_reward`, candidate advantages are only useful when one candidate succeeds and the other fails. Reverse shells help, but normal/far shells may still produce many zero-zero groups.

3. Current complex config starts from an intermediate move-to-object checkpoint.

The final stage does not start directly from the dense primitive checkpoint. The actual curriculum is:

`dense primitive PPO -> move_to_object GRPO/proximity -> complex sparse reverse-frontier GRPO`

This is reasonable, but the writeup should describe the intermediate stage explicitly.

4. YAML `reverse_curriculum` metadata is not the active scheduler.

The active scheduler comes from `lchol_reverse_*` trainer args and `FrontierScheduler`. The metadata block may be useful documentation, but changing it alone will not drive the runtime scheduler.

5. Scheduler exposure accounting is coarse.

`record_train_update()` increments all instruction states each rollout update. Promotion gating therefore measures global training time, not per-instruction exposure.

6. Resume support for scheduler state appears incomplete.

State is saved to `lchol_reverse_frontier/state_latest.json`, but I did not find a corresponding load call in the active training path.

7. Grab/pick shell gripper opening should be validated.

Grab/pick reverse shells force opening `1.0`, while sparse success uses closed/caught predicates. This may be intended as "near object, one close action away", but it should be checked with rollout videos or a focused unit test.

8. Existing documentation is stale in places.

Earlier reports describe an older move-to-object reward and a grab-first curriculum. The current code uses proximity move-to-object reward and reverse-frontier scheduling.

## Verification Performed

`pytest` was not installed in the local environment:

- `pytest -q ...` failed with `zsh:1: command not found: pytest`
- `python3 -m pytest ...` failed with `No module named pytest`

Fallback unit test command:

```bash
python3 -m unittest \
  tests.test_frontier_scheduler \
  tests.test_cdpr_reverse_shells \
  tests.test_lchol_curriculum_gates \
  tests.test_lchol_hindsight_relabeling \
  tests.test_lchol_replay_buffers \
  tests.test_cdpr_lchol_spec \
  tests.test_grpo_lchol_loss_mix \
  tests.test_cdpr_scene_sampling
```

Result:

```text
Ran 31 tests in 0.081s
OK
```

No full training smoke test was run because the full OpenVLA/MuJoCo/GPU checkpoint environment is outside the local lightweight test setup.

## Recommended Next Changes

Priority 1:

- Describe the method as "reverse-frontier sparse GRPO with LC-HOL hindsight BC", not full HER, unless full relabeled RL replay is added.
- Try `lchol_group_score: phase_shaped` for early sparse curriculum training, or compare it against the current `env_reward` setting.
- Add scheduler state loading on resume.

Priority 2:

- Make scheduler train-update counters instruction-specific.
- Add metrics for zero-advantage group rate, per-instruction shell ID, promotion/demotion count, and validation success by shell.
- Add a focused grab/pick reverse-shell test that checks whether shell 0 can reach sparse success after one close/hold action.

Priority 3:

- Remove or wire inactive config fields: `option_prior_*`, `hindsight_done_weight`, and metadata-only `reverse_curriculum`.
- Update existing reports so they match the current reward and curriculum code.
- Consider using all 256 scene variants during long final runs, while keeping smaller scene caches for iteration speed.

## Final Assessment

The implementation is strong enough to support the proposed training curriculum, provided the method is stated precisely. It implements dense pretraining, a move-to-object bridge checkpoint, complex sparse binary tasks, reverse success shells, frontier/rehearsal sampling, validation-driven promotion/demotion, and LC-HOL-compatible hindsight BC.

The main scientific risk is sparse signal starvation in GRPO, not the learning rate. If complex-task success does not improve, the first experiment should be phase-shaped LC-HOL group scoring or richer candidate sampling, followed by LR reduction only if optimization metrics show instability.
