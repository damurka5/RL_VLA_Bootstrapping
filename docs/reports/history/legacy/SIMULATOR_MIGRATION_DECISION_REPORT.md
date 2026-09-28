# Simulator Migration Decision Report

Date: 2026-06-06

Final recommendation: `STAY_ON_MUJOCO_REFACTOR_FIRST`

## 1. Executive summary

### Facts

- The project is already beyond "CDPR only" at the configuration/policy layer: YAML configs describe an embodiment, action keys, cameras, reward hooks, success hooks, and OpenVLA action dimensions. Example: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:55-99`, `rl_vla_bootstrapping/policy/openvla_oft.py:207-292`, and `rl_vla_bootstrapping/policy/openvla_actor_critic.py:80-122`.
- The current executable environment is still CDPR-specific. `CDPRLanguageRLEnv` imports `HeadlessCDPRSimulation`, hard-codes a 5-D action space, validates 5-D actions, and uses CDPR/body-name conventions: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1318-1326`, `1436-1457`, and `1663-1665`.
- The main MuJoCo inefficiency found in code is not "MuJoCo cannot randomize scenes"; object XY placement is already reset-time `qpos` randomization without recompiling: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1838-1848` and `robots/cdpr/cdpr_dataset/synthetic_tasks.py:886-908`. The inefficient part is that each episode closes the sim, builds/selects an XML wrapper, and initializes a fresh `MjModel` from XML: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1540`, `1776-1806`; `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py:257-263`.
- The current object/contact problems have direct asset/model evidence. Active YCB wrappers use visual meshes separated from collision meshes, but collision is still mesh-based COACD pieces with minimal active contact defaults: `robots/cdpr/cdpr_dataset/wrappers/desk__ycb_apple-ycb_baseball-ycb_pear_wrapper/placed_0_ycb_apple.xml:8-26`. The active CDPR XML lacks the aggressive solver/contact settings used in later physical tests and has a weak normalized gripper actuator: `robots/cdpr/cdpr_mujoco/cdpr.xml:1-6`, `136-151`, `204-212`.
- Existing physical gripper tests with stronger contact settings still show very large force spikes: apple max right normal force 14,854 N, pear 7,655 N, peach 1,635 N, baseball 28,358 N: `runs/ycb_gripper_physical_pick_release_videos/20260525_161149/manifest.json:31-36`, `93-98`, `155-160`, `217-222`.
- I did not run a live rollout profiler. `tools/audit/out/rollout_profile.csv` is explicitly marked `not_measured`. Therefore, there is insufficient evidence to claim that MuJoCo physics, rendering, OpenVLA forward pass, IPC, or optimizer update is the actual bottleneck.

### Recommendation

Stay on raw MuJoCo for the next decision stage and refactor first. Do not migrate because the current XML/asset/cache organization is rough. Migrate only after profiling and contact tests show that a properly refactored MuJoCo implementation still cannot meet throughput or stability needs.

The strongest near-term path is:

1. Replace the unstable object set with MuJoCo-friendly simple collision proxies.
2. Stop recompiling on every episode where the object set is unchanged.
3. Profile the rollout loop with timing for physics, render, image preprocessing, OpenVLA forward, reward, IPC, and optimizer.
4. Run one minimal ManiSkill/SAPIEN or Isaac Lab comparison task only as a benchmark, not as a full migration.

## 2. Current project state

### Training stages and checkpoints

Facts:

- Dense/proximity bootstrapping is represented by `configs/examples/cdpr_openvla_bootstrap_fast.yaml`. It uses PPO, 12 parallel envs, 170 rollout steps, 32 max env steps, 10 hold steps, scene cache prebuild, scene pool 16, texture pool 10: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:168-224`.
- Move-to-object GRPO continuation uses a resumed adapter/action head and `grpo_group_size: 2`, `num_parallel_envs: 10`, `rollout_steps: 170`, `hold_steps: 2`, `scene_pool_size: 32`, `texture_pool_size: 10`: `configs/examples/cdpr_openvla_grpo_step_390_to_490_movetoobj.yaml:217-281`.
- Complex sparse-task GRPO resumes from step `0072000` adapter/action head paths, uses LC-HOL, reverse-frontier scheduling, 10 parallel envs, 240 rollout steps, 120 max env steps, 2 hold steps, scene pool 64, texture pool 10: `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:300-387`.
- Existing reports reviewed: `CURRENT_CDPR_PPO_GRPO_PIPELINE_REPORT.md`, `CDPR_OPENVLA_FULL_PIPELINE_REPORT.md`, `CDPR_REVERSE_FRONTIER_LCHOL_ANALYSIS_REPORT.md`, and `CDPR_LC_HOL_IMPLEMENTATION.md`. The reverse-frontier report itself notes stale older docs and identifies the active complex config as the main source of truth.

### Robot embodiment interface

Facts:

- Config-level embodiment: `name: cdpr_5dof`, `kind: mujoco`, XML path `robots/cdpr/cdpr_mujoco/cdpr.xml`, controller class `HeadlessCDPRSimulation`, cameras `overview` and `ee_camera`: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:55-83`.
- Config-level action keys: `[x, y, z, yaw, gripper]` with per-channel scales and limits: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:84-99`.
- Runtime environment: `CDPRLanguageRLEnv` imports `HeadlessCDPRSimulation` and stores it as `_sim_cls`: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1318-1326`.
- Runtime action space is hard-coded `Box(-1, 1, shape=(5,))`: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1436-1441`.

Recommendation:

- Treat the current embodiment interface as a partial abstraction. It is enough to configure OpenVLA action dimensions, but not enough to swap in a non-CDPR robot without writing a new environment adapter.

### Action space

Facts:

- Low-level action is 5-D: `[dx, dy, dz, dyaw, gripper_cmd]`. The env validates shape 5: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1663-1665`.
- Actions are scaled by `action_step_xyz`, `action_step_yaw`, and `action_step_gripper`, then applied to the CDPR target, yaw, and gripper target: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:3667-3704`.
- OpenVLA action head predicts chunked actions. Config uses `chunk_size: 8` and action dim is inferred from action keys or dof: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:151-166`; `rl_vla_bootstrapping/policy/openvla_actor_critic.py:101-122`.

### Observation space and camera setup

Facts:

- Gym observation space is low-dimensional only: end-effector position, target object position, object positions/mask, instruction one-hot, and goal direction: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1442-1457`, `3711-3735`.
- Visual observations are produced by frame buffers/camera handles defined in config and the simulation class, then consumed by the external OpenVLA trainer: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:64-82`; `rl_vla_bootstrapping/policy/openvla_actor_critic.py:214-236`.
- Two MuJoCo cameras are configured: a free overview camera and fixed end-effector camera: `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py:448-459`.
- Rendering uses MuJoCo offscreen rendering and CPU readback via `mjr_readPixels`: `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py:494-505`.

### Instruction families

Facts:

- Current instruction family list includes movement primitives, move-to-object, pick, grab, receptacle placement, left/right/front/behind relations, between relation, and push directions: `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py:9-32`.
- Complex config enables: `move_to_object`, `grab_object`, `pick_up`, `push_left`, `push_right`, `push_forward`, `push_backward`, `put_into_plate`, `move_left_of_object`, `move_right_of_object`, `move_in_front_of_object`, `move_behind_object`, `put_in_front_of_object`, `put_behind_object`, `move_between_objects`: `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:92-110`.

### Reward modes and sparse binary success predicates

Facts:

- Dense bootstrapping has distance rewards and success bonus in metadata: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:101-126`.
- Complex config sets `reward_mode: sparse_binary`, `sparse_success_reward: 1.0`, `sparse_failure_reward: 0.0`: `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:176-178`.
- Reward dispatch enters `_compute_sparse_binary_reward` when sparse binary metadata is active: `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py:458-502`, `690-818`.
- Sparse manipulation predicates use object poses, gripper opening/closed state, caught-object proxy state, target displacement, spatial relations, container distance, release state, and between-object midpoint tests: `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py:939-1233`.
- Caught/grasp state is not direct MuJoCo contact. It is a heuristic using gripper closed, EE-object distance, and object motion following EE motion: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:3219-3268`.

### Scene randomization strategy

Facts:

- Scene variants are generated from object pools, target/distractor pools, required container pools, and scene counts. Object-set variants are generated in `_build_scene_object_variants` and `_build_scene_variants`: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:367-472`.
- `_configure_scene_sampling` builds variants from task metadata fields such as `scene_object_pool`, `target_object_pool`, `distractor_object_pool`, `required_scene_object_pool`, `container_object_pool`, and `scene_variant_count`: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:498-558`.
- Current configs request 64 scene variants for dense primitive training, 192 for move-to-object GRPO, and 256 for complex tasks: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:109-112`; `configs/examples/cdpr_openvla_grpo_step_390_to_490_movetoobj.yaml:103-175`; `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:132-170`.
- Current complex config uses 3-4 objects per scene and includes target/catchable/grippable pools plus required plate/bowl container objects: `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:132-170`.
- Object XY placement is randomized at reset after model compilation by writing freejoint `qpos`, then calling `mj_forward`: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1838-1848`; `robots/cdpr/cdpr_dataset/synthetic_tasks.py:886-908`.

### Scene cache, object count, texture pool

Facts from configs:

- Dense PPO: `scene_pool_size: 16`, `texture_pool_size: 10`, `num_parallel_envs: 12`, `hold_steps: 10`: `configs/examples/cdpr_openvla_bootstrap_fast.yaml:179-218`.
- Complex GRPO: `scene_pool_size: 64`, `texture_pool_size: 10`, `num_parallel_envs: 10`, `hold_steps: 2`: `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:334-382`.
- Local generated wrapper cache currently contains 14 main wrapper XMLs according to `tools/audit/out/mujoco_model_cache_report.csv`.
- Static inventory found 126 readable referenced/config files, 8 readable texture/image files totaling 11.541 MB, 31 readable mesh files totaling 9.107 MB, and no duplicate texture hash groups among readable referenced textures: `tools/audit/out/asset_inventory.csv`, `tools/audit/out/duplicate_textures.csv`.

### OpenVLA-OFT adapter/action-head training

Facts:

- `build_openvla_rl_plan` builds an external Python stage, injects base checkpoint, dataset/CDPR roots, catalog, desk textures, allowed objects, instruction types, action steps, image count, reward/success hooks, and action dimensions: `rl_vla_bootstrapping/policy/openvla_oft.py:207-292`.
- OpenVLA dimension env vars are `VLA_ROBOT`, `VLA_ACTION_DIM`, `VLA_NUM_ACTIONS_CHUNK`, and `VLA_PROPRIO_DIM`: `rl_vla_bootstrapping/policy/openvla_actor_critic.py:80-98`.
- Image preprocessing batches primary and wrist images in local helper code and in a GRPO patch: `rl_vla_bootstrapping/policy/openvla_actor_critic.py:214-236`; `rl_vla_bootstrapping/policy/grpo_finetune_cdpr_fast.py:713-749`.
- Complex GRPO resumes adapter/action head paths and trains the loaded adapter: `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:300-304`.

### Generality for non-CDPR embodiments

Facts:

- General at the orchestration layer: config fields describe action keys, dof, cameras, controller class, policy image count, and reward/success hooks.
- Not yet general at the simulator environment layer: runtime uses `CDPRLanguageRLEnv`, `HeadlessCDPRSimulation`, 5-D action validation, CDPR body naming, CDPR caught-object logic, CDPR reverse shells, and CDPR-specific object resolution.

Recommendation:

- Refactor toward a real embodiment adapter interface before migrating. A simulator change alone will not make the pipeline general to arbitrary new robot embodiments.

## 3. Current MuJoCo efficiency audit

### XML compilation and scene representation

Facts:

- Episode reset calls `self.close()` before sampling/initializing a scene: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1540`.
- `_initialize_episode_scene` builds/selects a wrapper XML, constructs a new `HeadlessCDPRSimulation(xml_path=...)`, and calls `sim.initialize()`: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:1776-1806`.
- `HeadlessCDPRSimulation.initialize()` compiles a fresh MuJoCo model from XML with `mj.MjModel.from_xml_path(self.xml_path)`: `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py:257-263`.
- Wrapper caching can reuse XML files, but it does not reuse compiled `MjModel`/`MjData`: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:3066-3135`.
- Existing wrappers are full MJCF worlds that include scene, CDPR, and placed objects. The scene switcher clones assets, prefixes names, strips joints, optionally adds a freejoint, and writes a placed object XML: `robots/cdpr/cdpr_mujoco/cdpr_scene_switcher.py:313-491`.

Recommendation:

- Separate "object set changed" from "object pose changed". Keep full XML rebuild for object-set changes, but reuse compiled models for repeated object sets and reset randomized object poses by state.
- Add a compiled-model pool keyed by `(scene_name, sorted_object_names, texture_variant, robot_xml_version)` or a process-local env pool. It should reuse `MjModel`, allocate new `MjData`, and reinitialize state.

### Texture/material loading and cache facts

Facts:

- Desk texture patching copies selected textures into a shared desk texture cache and emits XML/material variants: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:964-1057`.
- Static audit artifacts:
  - `asset_inventory.csv`: 126 rows, all exist after parser correction.
  - Readable textures/images: 8 files, 11.541 MB total.
  - Readable meshes: 31 files, 9.107 MB total.
  - `duplicate_textures.csv`: no duplicate texture hash groups among readable referenced textures.
  - `mujoco_model_cache_report.csv`: 14 wrapper rows, lower-bound referenced-file memory 7.170-15.892 MB per wrapper, 174.771 MB total lower bound across wrapper rows. This is not the real `MjModel` heap size; it is a file-size lower-bound.

Recommendation:

- Keep the shared desk texture cache idea, but do not multiply compiled models by texture variants unless the texture is required for the current rollout. For most RL training, downsample textures and render at the minimum resolution that preserves OpenVLA behavior.

### Rendering and camera cadence

Facts:

- `_apply_action` simulates `1 + hold_steps` physics substeps and captures frames only on the last substep: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:3705-3708`.
- `run_simulation_step` renders only when `capture_frame=True`: `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py:837-839`.
- Two images are read back from MuJoCo each capture through `mjr_readPixels` into NumPy arrays: `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py:494-505`.

Recommendation:

- The code is already avoiding rendering every physics substep. The next likely win is reducing image size, batching processor calls, and measuring CPU readback cost. If rendering dominates, consider GPU renderer stacks after measuring.

### Policy batching and CPU/GPU transfer

Facts:

- Local OpenVLA batch helpers process lists of observations/instructions and move tensors to device in batch: `rl_vla_bootstrapping/policy/openvla_actor_critic.py:214-236`.
- The GRPO wrapper patches external input preparation to batch PIL conversion/processor calls and move `pixel_values` to bfloat16 on CUDA: `rl_vla_bootstrapping/policy/grpo_finetune_cdpr_fast.py:713-749`.
- The actual external `/root/repo/openvla-oft` rollout loop was not fully audited in this repository snapshot.

Recommendation:

- Confirm that policy inference is batched across environment workers and GRPO candidates in the external trainer. If it is not, batching may matter more than simulator migration.

### Current bottleneck

Facts:

- No live profile was run. `tools/audit/out/rollout_profile.csv` is a marked placeholder with `not_measured` rows.

Evidence-based conclusion:

- The bottleneck is unknown. Based on code paths, plausible bottlenecks are: repeated MuJoCo compilation at reset, two-camera CPU readback, PIL/processor image preprocessing, OpenVLA forward pass, multiprocessing/IPC, or optimizer update. Physics alone is not yet proven to be the bottleneck.

## 4. Asset/contact stability audit

### Active object and gripper modeling facts

- Active CDPR XML has no explicit global `<option>` solver/timestep/contact settings beyond compiler/default site/tendon settings at the top of the file: `robots/cdpr/cdpr_mujoco/cdpr.xml:1-6`.
- Active gripper collision is simple box/capsule finger geometry with friction `1.5 0.02 0.01`: `robots/cdpr/cdpr_mujoco/cdpr.xml:136-151`.
- Active gripper actuator is a normalized position actuator with `kp="0.27"` and gear mapping to a 0-0.03 m slide range: `robots/cdpr/cdpr_mujoco/cdpr.xml:204-212`.
- YCB apple wrapper uses visual mesh `textured.obj`, collision meshes `textured_coacd_0..4.stl`, visual geom contact disabled, collision geoms contact enabled, inertial mass 0.068 kg, and one freejoint: `robots/cdpr/cdpr_dataset/wrappers/desk__ycb_apple-ycb_baseball-ycb_pear_wrapper/placed_0_ycb_apple.xml:8-26`.
- YCB cups wrapper uses five mesh collision pieces and mass 0.014 kg: `robots/cdpr/cdpr_dataset/wrappers/desk__bowl-ycb_b_cups-ycb_baseball/placed_2_ycb_b_cups.xml:8-26`.
- `make_placed_object_xml` copies object assets, absolutizes mesh/texture paths, strips joints, adds a freejoint if dynamic, and only adds a default geom density of 1200. It does not add robust per-object friction/solref/solimp/condim defaults for active RL wrappers: `robots/cdpr/cdpr_mujoco/cdpr_scene_switcher.py:313-491`.

### Existing physical test facts

- The later physical gripper test XML is much stronger than active training XML: timestep 0.001, Newton solver, 100 iterations, elliptic cone, noslip iterations, multiccd, condim/friction/solref/solimp defaults, dedicated finger pads/lips, and force-limited high-kp actuators: `runs/ycb_gripper_physical_pick_release_videos/20260525_161149/ycb_apple_physical_gripper.xml:3-10`, `43-75`, `85-89`.
- The physical placed apple uses high friction/contact settings on collision meshes: `runs/ycb_gripper_physical_pick_release_videos/20260525_161149/ycb_apple_placed.xml:4-25`.
- Despite that stronger setup, the manifest reports large force spikes during caught-lift tests:

| Object | Evidence | Contact result |
| --- | --- | --- |
| `ycb_apple` | `manifest.json:31-36` | max right normal force 14,854 N |
| `ycb_pear` | `manifest.json:93-98` | max left/right normal force 3,421/7,655 N |
| `ycb_peach` | `manifest.json:155-160` | max right normal force 1,635 N |
| `ycb_baseball` | `manifest.json:217-222` | max left/right normal force 2,511/28,358 N |

- `tools/audit/out/contact_stability_report.csv` marks those objects as force-spike failures and recommends primitive/convex proxy replacement plus mass/inertia/contact retuning.

### Proposed contact test battery

These tests were not run in this turn. They should be implemented as small deterministic MuJoCo scripts and recorded to `tools/audit/out/contact_stability_report.csv`.

| Test | Pass metric | Primary failure to detect |
| --- | --- | --- |
| Drop test | Object settles on table, no explosion, no sustained bounce | bad mass/inertia/contact stiffness |
| Rest test | After 5 s, max linear/angular velocity below threshold | persistent shaking |
| Push test | Lateral push produces bounded sliding, no tunneling | mesh penetration/tunneling |
| Grasp squeeze test | Gripper closes without deep penetration or huge force spikes | unsuitable collision proxy or gripper pads |
| Lift test | Object remains held when friction/geometry should allow it | slipping through fingers or false grasp |

Recommended metrics:

- max linear velocity after settling;
- max angular velocity after settling;
- max normal force per finger/object contact;
- min contact distance/penetration if available;
- number of lost-contact frames during lift;
- whether object pose crosses finger collision geometry.

### Contact recommendations

1. Replace the first training objects with a small curated set: cube/block, cylinder/can, sphere/ball, rounded fruit proxy, cup-like receptacle with simple collision, bowl/plate/receptacle with primitive collision.
2. Separate visual fidelity from collision fidelity. Keep visual OBJ/texture if useful for OpenVLA, but use primitive, convex, or very small convex-decomposition collision.
3. Fix mass and inertia manually for small objects. The 0.0033 kg peach and 0.014 kg cups are suspicious for stable grasp training unless intentionally modeling very light objects.
4. Tune active CDPR XML contact settings after object proxies are fixed: timestep, solver iterations, `condim`, friction, `solref`, `solimp`, `margin`, `gap`, and optional `multiccd`.
5. Add gripper pad/lip collision geometry to the active training gripper, not only the separate physical test XML.
6. Avoid solving contact instability by extreme friction alone. The physical test used very high friction/contact settings and still produced force spikes.

## 5. Binary reward and task predicate portability

### Key conclusion

Binary reward support is not primarily a simulator requirement. It is a task-state and predicate implementation requirement.

Any candidate simulator must expose enough state to implement the predicates cleanly:

- object poses and velocities;
- end-effector pose;
- gripper opening/command state;
- contact or grasp state, or enough contact force/distance data to build one;
- container/receptacle pose and relation;
- object displacement from reset;
- reset-time state capture/restore for GRPO candidate evaluation and reverse-frontier curricula;
- camera RGB frames for OpenVLA observations.

### Current predicate portability table

| Instruction family | Current predicate | Required state | MuJoCo-specific dependency | Port difficulty |
| --- | --- | --- | --- | --- |
| `move_to_object` | XY distance threshold to target object/goal; dense variant also proximity reward | EE pose, target pose | Mostly through env getters, portable | Low |
| `grab_object` | gripper closed plus caught-object-is-target or proximity fallback | gripper opening, EE pose, target pose, caught/grasp state | Current caught state is CDPR heuristic, not MuJoCo contact | Medium |
| `pick_up` | target grasped and target lift over support height | target pose, support z, gripper/caught state | Uses env body pose getter and caught heuristic | Medium |
| `push_left/right/forward/backward` | signed object displacement along axis exceeds threshold | object initial/current pose | Portable | Low |
| `put_into_plate` | target within container XY/Z tolerance, optional release, optional motion/grasp | target pose, container pose, gripper opening, target motion, grasp state | Portable if simulator exposes object poses; grasp heuristic needs replacement | Medium |
| `move_left/right/in_front/behind_object` | target located in relation zone around reference | target/reference poses | Portable | Low |
| `put_in_front/behind_object` | signed relation offset plus orthogonal tolerance, motion and grasp requirements | target/reference poses, target motion, grasp state | Portable except grasp | Medium |
| `move_between_objects` | target near midpoint, projection between two references, optional motion/grasp | target/reference/second-reference poses | Portable except grasp | Medium |

Evidence:

- Reward dispatch and sparse binary wrapper: `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py:458-502`, `690-818`.
- Pick-up sparse predicate: `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py:821-926`.
- Manipulation/relation/push predicates: `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py:939-1233`.
- Current caught-object heuristic: `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:3219-3268`.

Recommendation:

- Build a simulator-agnostic predicate API before migration:

```text
get_body_pose(name)
get_body_velocity(name)
get_ee_pose()
get_gripper_state()
get_contact_summary(body_a, body_b)
capture_state()
restore_state(state)
set_object_pose(name, pose)
render(camera_names)
```

Then porting predicates becomes mostly adapter work.

## 6. Training/rollout profiling results

### Facts

- No live profiling was run.
- `tools/audit/out/rollout_profile.csv` contains `not_measured` rows for env step time, render time, reward time, image preprocessing time, policy forward time, optimizer time, CPU RAM, GPU VRAM, and GPU utilization.

### Static bottleneck candidates

| Candidate bottleneck | Evidence | Current confidence |
| --- | --- | --- |
| MuJoCo model compilation/reset | Reset closes sim and recompiles XML via `MjModel.from_xml_path`: `rl_cdpr_env.py:1540`, `1776-1806`; `headless_cdpr_egl.py:257-263` | High as reset overhead |
| Rendering/readback | Two camera renders use `mjr_readPixels` into CPU arrays: `headless_cdpr_egl.py:494-505`, `837-839` | Medium |
| Image preprocessing | PIL/processor path for two views: `openvla_actor_critic.py:214-236`, `grpo_finetune_cdpr_fast.py:713-749` | Medium |
| OpenVLA forward | 7B VLA plus action head, external trainer | Unknown until measured |
| Reward predicates | Python state predicates, but simple vector math: `rl_instruction_tasks.py:939-1233` | Probably low, unmeasured |
| IPC/multiprocessing | External trainer not fully audited here | Unknown |
| Optimizer update | External trainer, LoRA/action head, 2 A40 GPUs | Unknown |

Recommendation:

- Treat any simulator migration decision as premature until the profile separates reset compilation, render, model forward, and optimizer time.

## 7. Candidate simulator decision matrix

Score meaning: 1 poor/high risk, 3 acceptable, 5 strong/low risk. "Migration effort" is scored as ease, so 5 means low effort.

External sources checked:

- robosuite docs describe MuJoCo-based sensors, cameras, observables, robots, controllers, and environment composition: [robosuite sensors](https://robosuite.ai/docs/modules/sensors.html), [robosuite overview](https://robosuite.ai/docs/modules/overview.html).
- RoboCasa is a MuJoCo household manipulation framework/benchmark: [RoboCasa overview](https://robocasa.ai/docs/introduction/overview.html).
- ManiSkill/SAPIEN documents GPU-parallel simulation/rendering, batched env data, heterogeneous parallel sub-scenes, and custom task building: [ManiSkill](https://maniskill.readthedocs.io/), [GPU simulation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/gpu_simulation.html), [quickstart](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html).
- Isaac Lab documents manager-based RL environments, parallel environments, RGB camera sensors, tiled cameras, multi-GPU training, and performance/memory benchmarks: [manager env](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html), [training guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/training_guide.html), [camera sensors](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html), [multi-GPU](https://isaac-sim.github.io/IsaacLab/develop/source/features/multi_gpu.html), [benchmarks](https://isaac-sim.github.io/IsaacLab/v1.4.1/source/overview/reinforcement-learning/performance_benchmarks.html).
- RoboTwin 2.0 is SAPIEN-based and focused on bimanual benchmarks/tasks/assets: [RoboTwin docs](https://robotwin-platform.github.io/doc/tasks/), [LeRobot RoboTwin summary](https://huggingface.co/docs/lerobot/main/robotwin).

| Criterion | A. Raw MuJoCo refactor | B. Higher-level MuJoCo | C. ManiSkill/SAPIEN | D. Isaac Lab/Sim | E. RoboTwin/other benchmark |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arbitrary new robot embodiments | 4 | 3 | 4 | 4 | 2 |
| CDPR-like/custom kinematics | 5 | 2 | 3 | 2 | 1 |
| Custom controllers | 5 | 3 | 4 | 4 | 2 |
| Visual observations for OpenVLA | 3 | 4 | 5 | 5 | 4 |
| Binary success predicates | 5 | 4 | 5 | 5 | 4 |
| Stable simple manipulation objects | 4 | 4 | 4 | 4 | 4 |
| Many randomized scenes | 3 | 4 | 5 | 5 | 4 |
| Rendering throughput | 2 | 2 | 5 | 5 | 4 |
| Parallel rollout throughput on 2x A40 | 2 | 2 | 5 | 5 | 3 |
| Ease of OpenVLA-OFT integration | 5 | 3 | 2 | 2 | 2 |
| Migration effort/ease | 5 | 3 | 2 | 1 | 2 |
| Low risk to scientific contribution | 5 | 3 | 3 | 2 | 3 |
| Low engineering distraction risk | 5 | 3 | 2 | 1 | 2 |
| Good simple object assets | 3 | 4 | 5 | 4 | 5 |
| Maintainability | 3 | 4 | 4 | 3 | 3 |
| Community/examples/docs | 4 | 4 | 4 | 4 | 3 |
| Ability to reproduce current experiments | 5 | 3 | 2 | 2 | 1 |

Interpretation:

- Raw MuJoCo refactor scores highest for preserving current science, CDPR custom mechanics, and integration.
- ManiSkill/SAPIEN and Isaac Lab score highest for GPU-parallel rollout/rendering, but the current evidence does not yet prove that rollout/rendering is the bottleneck.
- Higher-level MuJoCo stacks are attractive for infrastructure but may fight the CDPR/custom-embodiment goal.
- RoboTwin is useful as a secondary benchmark/asset inspiration, not as the primary CDPR simulator replacement.

## 8. Migration cost estimate

| Option | Files/modules to rewrite | Minimal prototype | Match current functionality | Comparable experiments | Main unknowns |
| --- | --- | ---: | ---: | ---: | --- |
| A. Raw MuJoCo refactor | `rl_cdpr_env.py`, `headless_cdpr_egl.py`, wrapper/cache code, object assets, contact tests, profiler | 3-5 days | 1-3 weeks | 1-3 weeks | true bottleneck; stable object set |
| B. robosuite/RoboCasa-style MuJoCo | New env class, custom CDPR robot model integration, controller bridge, task suite, camera adapter, reward adapter | 2-4 weeks | 6-10 weeks | 8-12 weeks | CDPR fit into robosuite robot/controller abstractions |
| C. ManiSkill/SAPIEN | Convert CDPR to SAPIEN/articulation or custom controller model, task env, object assets, camera/render API, state capture/restore, predicates, OpenVLA adapter | 3-6 weeks | 8-14 weeks | 10-16 weeks | CDPR cable/tendon modeling fidelity; GRPO state restore semantics |
| D. Isaac Lab/Isaac Sim | USD/assets, CDPR/custom mechanism, Direct/Manager RL env, cameras/tiled camera, predicates, reset/curriculum, OpenVLA adapter | 4-8 weeks | 12-20 weeks | 16-24 weeks | custom CDPR representation; setup complexity; visual RL memory |
| E. RoboTwin/benchmark hybrid | Benchmark task adapter, OpenVLA observation/action bridge, maybe no CDPR support | 2-4 weeks for benchmark only | Not suitable as primary CDPR match | 4-8 weeks for benchmark comparisons | benchmark embodiment mismatch |

Recommendation:

- Spend the first 1-3 weeks on option A. Only start C/D migration if the stop/go criteria in section 11 are triggered.

## 9. Recommended immediate fixes

1. Add a real profiler around rollout.
   - Instrument `reset`, `MjModel.from_xml_path`, `mj_step`, render/readback, image preprocessing, OpenVLA forward, reward, IPC, optimizer, CPU RSS, GPU VRAM/utilization.
   - Save to `tools/audit/out/rollout_profile.csv`.

2. Replace the first object curriculum.
   - Start with primitive collision objects that map to instruction families: cube/block, cylinder/can, sphere/ball, rounded fruit proxy, mug/cup proxy, bowl/plate/receptacle proxy.
   - Keep visual meshes optional; train first with stable physics.

3. Add active gripper contact geometry improvements.
   - Port the useful pad/lip concept from the physical test XML into active `robots/cdpr/cdpr_mujoco/cdpr.xml`, but tune against force spikes rather than copying extreme friction.

4. Stop recompiling on every reset when the XML model is unchanged.
   - Maintain a process-local compiled model/cache pool.
   - Randomize object pose, texture choice where possible, and EE start state at reset.

5. Reduce image/texture costs.
   - Downsample training textures and render sizes.
   - Keep high-resolution assets only for validation videos.

6. Make predicates simulator-agnostic.
   - Extract the state API described in section 5, so MuJoCo/SAPIEN/Isaac adapters can share predicates.

## 10. Recommended experiments before final decision

These are deliberately sized for 1-3 days.

| Experiment | Expected output | Decision value |
| --- | --- | --- |
| Profile current rollout pipeline | Filled `rollout_profile.csv` with timings/RAM/VRAM/utilization | Identifies true bottleneck |
| Replace 3-5 unstable objects with simple proxies | Contact test CSV and short smoke rollout | Tests whether assets solve instability |
| Reduce texture resolution and wrapper variants | Memory/time comparison before/after | Tests whether cache/resource handling is the issue |
| Reset-time randomization without recompiling | FPS/reset-time comparison | Tests whether raw MuJoCo is already sufficient |
| Short stable-object training smoke test | Reward/success curves, videos | Tests scientific viability before migration |
| Minimal candidate-sim task | One move-to-object or push task in ManiSkill or Isaac Lab | Measures real migration friction and FPS |

## 11. Final recommendation: stay, migrate, or hybrid

Final recommendation: `STAY_ON_MUJOCO_REFACTOR_FIRST`

Decision rules:

- Stay on MuJoCo if simplified collision assets and contact tuning fix object shaking/slipping, and if profiling shows OpenVLA forward/rendering/image preprocessing dominates more than MuJoCo physics.
- Stay on MuJoCo if reset-time scene randomization and compiled-model reuse reduce memory/reset overhead enough.
- Migrate only if, after refactor, simulation/rendering remains the dominant bottleneck, object stability remains poor with proper collision proxies/contact tuning, or a candidate simulator demonstrates a clear greater-than-3x rollout throughput improvement with acceptable porting cost.
- Do not migrate merely because XML/asset organization is inefficient. The evidence shows current inefficiency is largely implementation organization: wrapper files, recompilation, texture variants, and object assets.
- Consider a hybrid only after the MuJoCo refactor: keep MuJoCo for CDPR/custom embodiment experiments and add ManiSkill/RoboTwin/Isaac as external benchmark tracks.

## 12. Evidence appendix: file paths, functions, line numbers, commands run, logs

### Audit artifacts produced

- `tools/audit/simulator_migration_audit.py`
- `tools/audit/out/asset_inventory.csv`
- `tools/audit/out/duplicate_textures.csv`
- `tools/audit/out/mujoco_model_cache_report.csv`
- `tools/audit/out/contact_stability_report.csv`
- `tools/audit/out/rollout_profile.csv`
- `tools/audit/out/simulator_audit_summary.json`

### Static artifact summary

- `asset_inventory.csv`: 126 rows, 126 existing referenced/config files, 8 texture/image files, 31 mesh files.
- `duplicate_textures.csv`: header only; no duplicate texture hashes among readable referenced textures.
- `mujoco_model_cache_report.csv`: 14 main wrapper XML rows, lower-bound referenced-file memory 7.170-15.892 MB per wrapper, 174.771 MB total.
- `contact_stability_report.csv`: 7 objects, with force-spike failures for apple, pear, peach, baseball from existing physical gripper manifest.
- `rollout_profile.csv`: not measured; placeholder only.

### Commands run

```bash
python3 tools/audit/simulator_migration_audit.py
git status --short
awk 'NR>=... && NR<=... {printf "%6d %s\n", NR, $0}' <file>
python3 -c 'import csv, collections; ...'
wc -l tools/audit/out/*.csv
```

### Primary local evidence files

- `configs/examples/cdpr_openvla_bootstrap_fast.yaml:55-224`
- `configs/examples/cdpr_openvla_grpo_step_390_to_490_movetoobj.yaml:103-281`
- `configs/examples/cdpr_openvla_grpo_complex_tasks.yaml:92-249`, `300-387`
- `robots/cdpr/cdpr_dataset/rl_cdpr_env.py:367-472`, `498-558`, `641-744`, `1274-1457`, `1507-1855`, `3038-3135`, `3219-3268`, `3667-3736`
- `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py:257-505`, `808-852`
- `robots/cdpr/cdpr_mujoco/cdpr_scene_switcher.py:313-491`
- `robots/cdpr/cdpr_dataset/synthetic_tasks.py:820-908`
- `robots/cdpr/cdpr_mujoco/cdpr.xml:1-6`, `106-164`, `204-212`
- `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py:9-32`, `458-818`, `821-926`, `939-1233`
- `rl_vla_bootstrapping/policy/openvla_oft.py:207-292`
- `rl_vla_bootstrapping/policy/openvla_actor_critic.py:80-122`, `214-236`
- `rl_vla_bootstrapping/policy/grpo_finetune_cdpr_fast.py:713-749`, `872-930`
- `rl_vla_bootstrapping/lchol/grpo_runtime.py:44-160`
- `rl_vla_bootstrapping/lchol/frontier_scheduler.py:51-150`
- `robots/cdpr/cdpr_dataset/cdpr_reverse_shells.py:14-32`, `132-175`
- `runs/ycb_gripper_physical_pick_release_videos/20260525_161149/manifest.json:1-40`, `73-102`, `135-164`, `197-226`
- `runs/ycb_gripper_physical_pick_release_videos/20260525_161149/ycb_apple_physical_gripper.xml:3-10`, `43-90`
- `runs/ycb_gripper_physical_pick_release_videos/20260525_161149/ycb_apple_placed.xml:4-25`

