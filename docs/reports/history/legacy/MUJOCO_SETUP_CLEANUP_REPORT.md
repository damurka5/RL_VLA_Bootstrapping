# MuJoCo Setup Cleanup Report

Date: 2026-06-06

Final recommendation: `MUJOCO_NEEDS_RENDERING_OR_CACHE_REFACTOR`

This is not a recommendation to migrate yet. The cache/recompile path is now instrumented and behaves well for repeated same-topology resets, but the required two-camera render/readback measurement still fails in this local macOS/CoreGraphics context. There is also remaining gripper/object contact work for block/can grasp tests. The current evidence points to implementation, asset, and render-context issues rather than a proven MuJoCo limitation.

## Scope

No simulator migration was attempted. No OpenVLA run was required. The scientific training method was not changed.

The work focused on:

- compiled `MjModel` reuse and reset instrumentation;
- exact state continuation hooks;
- no-OpenVLA rollout profiling;
- stable primitive object assets;
- deterministic contact-stability tests;
- conservative gripper/contact audit;
- texture/render inventory;
- a small simulator-agnostic state/predicate API.

## Code Changes

### Compiled Model Cache

Added `robots/cdpr/cdpr_mujoco/model_cache.py`.

What changed:

- Added a process-local compiled-model cache around `mujoco.MjModel.from_xml_path`.
- Cache entries include the compiled `MjModel`, compile time, RSS at compile time, and a deterministic key.
- The key includes the resolved XML path, semantic model key, included XML/file signatures, timestep, and offscreen render config.
- Cache stats expose hits, misses, hit rate, last event, and cache size.

Why:

- Rebuilding XML and recompiling `MjModel` on same-topology resets was a major suspected reset-time cost.
- MuJoCo model topology must still be recompiled when XML topology changes, but object poses and task metadata do not require recompilation.

### Headless MuJoCo Wrapper

Modified `robots/cdpr/cdpr_mujoco/headless_cdpr_egl.py`.

What changed:

- `HeadlessCDPRSimulation` now accepts `use_model_cache` and `model_cache_key`.
- `initialize()` uses the compiled-model cache by default.
- Added `reset_data_state()` to reuse an existing compiled model and reset `mjData` with `mj_resetData`.
- Added optional logging with `RLVLA_CDPR_MJMODEL_CACHE_LOG=1`.
- Added `compiled_model_cache_stats()`.

Why:

- This separates model topology compilation from state reset.
- Cleanup still releases render/context state, but the process-local compiled model can survive wrapper close/recreate.

Feature flag:

- Disable cache with `RLVLA_CDPR_COMPILED_MODEL_CACHE=0`.

### RL Environment Instrumentation

Modified `robots/cdpr/cdpr_dataset/rl_cdpr_env.py`.

What changed:

- Added constructor flag `use_compiled_model_cache`.
- Added semantic cache key:
  `(robot_xml_version, scene_name, sorted_object_set, texture_variant, object_topology_version)`.
- Reset info now reports wrapper build time, simulator initialize time, scene setup time, model compile time, cache hit/miss, cache key, cache size, and RSS.
- Step info now reports reward and success predicate timing.
- Existing `capture_state()` / `restore_state()` support is preserved and surfaced through the simulator API.
- Added forwarding methods for pose, velocity, contact summary, object pose writes, state capture/restore, and rendering.

Why:

- GRPO candidate branching needs exact continuation without XML rebuilds.
- Training/debug code needs to distinguish reset state writes from topology changes.

### Simulator-Agnostic API

Added `robots/cdpr/cdpr_mujoco/sim_api.py`.

Prototype methods:

- `get_body_pose(name)`
- `get_body_velocity(name)`
- `get_ee_pose()`
- `get_gripper_state()`
- `get_contact_summary(body_a, body_b)`
- `capture_state()`
- `restore_state(state)`
- `set_object_pose(name, pose)`
- `render(camera_names)`

Modified `robots/cdpr/cdpr_dataset/rl_instruction_tasks.py` so success predicates first try `env.state_api().get_body_pose(...)`, then fall back to the previous private accessors.

Why:

- This proves the existing success predicates can move toward a backend-neutral state/predicate layer without a full rewrite.

### Stable Object Pack

Added primitive MuJoCo-friendly assets under `robots/cdpr/cdpr_mujoco/stable_objects/`:

- `stable_block.xml`
- `ycb_wood_block.xml`
- `stable_can.xml`
- `stable_sphere.xml`
- `ycb_baseball.xml`
- `ycb_apple.xml`
- `ycb_pear.xml`
- `ycb_peach.xml`
- `plate.xml`
- `bowl.xml`
- `ycb_b_cups.xml`
- `mug.xml`

Each asset uses simple primitive or convex collision proxies, separate visual/collision geoms where useful, explicit inertial parameters, and explicit friction/contact settings. Names are kept close to current instruction/object names.

Modified `robots/cdpr/cdpr_mujoco/cdpr_scene_switcher.py`.

What changed:

- Added stable-object lookup.
- Stable objects are used first when `RLVLA_CDPR_USE_STABLE_OBJECTS=1` or when the requested object name starts with `stable_`.
- Stable objects are also used as a final fallback when external YCB/LIBERO/RobotWin assets are unavailable.

Why:

- First-pass training/debug tests should not depend on unstable scanned collision meshes.

### Audit Scripts

Added `tools/audit/profile_mujoco_rollout.py`.

Outputs:

- `tools/audit/out/mujoco_rollout_profile.csv`
- `tools/audit/out/mujoco_rollout_profile_summary.json`
- comparison copies:
  `mujoco_rollout_profile_cached.*`
  and `mujoco_rollout_profile_no_cache.*`

Added `tools/audit/test_mujoco_contact_stability.py`.

Outputs:

- `tools/audit/out/mujoco_contact_stability.csv`
- `tools/audit/out/mujoco_contact_stability_summary.json`
- gripper comparison files:
  `mujoco_contact_stability_gripper_baseline.*`
  `mujoco_contact_stability_gripper_current.*`
  `mujoco_contact_stability_gripper_pad_experiment.*`

Added `tools/audit/test_mujoco_success_predicates.py`.

Outputs:

- `tools/audit/out/mujoco_success_predicate_smoke.csv`
- `tools/audit/out/mujoco_success_predicate_smoke_summary.json`

## Recompilation Audit

Places that still call or indirectly reach `MjModel.from_xml_path`:

- normal environment path through `HeadlessCDPRSimulation.initialize()`, now cache-backed;
- `robots/cdpr/cdpr_mujoco/model_cache.py`, the intended cache compilation point;
- standalone render/demo/doctor/audit scripts;
- contact/profile scripts under `tools/audit/`, by design;
- dataset/demo utilities that create one-off scenes.

The RL env still closes/recreates the wrapper around episode reset, but the compiled `MjModel` no longer has to be rebuilt when the model key is unchanged. The cache makes this path reversible through `RLVLA_CDPR_COMPILED_MODEL_CACHE=0`.

Recompilation is required when:

- robot XML or included XML topology changes;
- scene topology changes;
- sorted object set changes;
- object topology version changes;
- texture variant or compiled asset signature changes;
- offscreen model/render config changes in a way MuJoCo bakes into the model;
- cache is disabled.

Recompilation is not required for:

- restoring the last state;
- resetting qpos/qvel/act/ctrl/time;
- changing object poses for the same object set;
- changing end-effector target, gripper target, or task metadata;
- GRPO candidate branching from a captured state.

## Profiling Results

Profiler mode: `direct_mujoco_fallback`.

Reason: `CDPRLanguageRLEnv` could not be instantiated because `gym` is not installed in this environment. The profiler therefore used raw MuJoCo model/data reset and scripted stepping. This avoids OpenVLA and still measures compile/cache/physics/render-worker behavior.

Cached run:

- resets: 100
- cache hits observed: 99
- cache misses observed: 1
- cache hit rate: 0.9901
- reset mean: 10.66 ms, including first compile outlier
- reset p95: 4.60 ms
- compile mean: 6.69 ms, mostly zeros after first miss
- physics step mean: 0.270 ms
- physics step p95: 0.316 ms
- reward/success predicate mean: 0.137 ms
- no-render FPS: 3700.27
- CPU RAM: 319.67 MB
- GPU VRAM: not available

Disabled-cache baseline:

- resets: 20
- cache hits observed: 0
- reset mean: 104.21 ms
- reset p95: 117.52 ms
- compile mean: 99.95 ms
- compile p95: 113.77 ms
- physics step mean: 0.283 ms
- no-render FPS: 3533.74
- CPU RAM: 517.91 MB

Before/after interpretation:

- Same-topology reset improves from about 104 ms mean without cache to about 4 to 11 ms with cache, depending on whether the first compile miss is included.
- Physics step time is not materially affected by the cache, as expected.
- Memory in the cached run was lower in this process-level measurement, but this should be treated as an RSS observation rather than a precise heap accounting.

Rendering:

- Two-camera render/readback could not be measured locally.
- The render worker failed with `CGLError: invalid CoreGraphics connection`.
- `fps_with_render` is therefore 0 in the summary because no render frames completed.
- This is a render-context/readback evidence gap, not proof that MuJoCo physics is the problem.

## Contact Stability Results

Main output: `tools/audit/out/mujoco_contact_stability.csv`.

Final active XML results:

- rows: 35
- pass: 31
- fail: 4
- warn: 0
- error: 0

All selected objects passed:

- drop test;
- rest-on-table test;
- push test.

Remaining failures:

- `stable_block`: `gripper_squeeze`, `lift`
- `stable_can`: `gripper_squeeze`, `lift`

Failure mode:

- no gripper/object contact;
- recommendation from the audit script: check object placement and finger pad reach.

Objects passing gripper squeeze/lift in the final active XML:

- `stable_sphere`
- `ycb_apple`
- `plate`
- `bowl`
- `ycb_b_cups`

## Gripper Audit

I compared the active CDPR gripper with existing physical-gripper test XMLs and tried a conservative pad/lip experiment. The experiment was not retained.

Measured gripper-only comparison:

- baseline/current active XML: 10 pass, 4 fail across 14 gripper rows;
- temporary pad/lip experiment: 8 pass, 6 fail across 14 gripper rows.

The pad/lip experiment improved block contact but regressed plate/cup contact in the synthetic placement test. Because it did not reduce aggregate instability, the active `robots/cdpr/cdpr_mujoco/cdpr.xml` was left unchanged.

No global timestep, solver, or extreme friction settings were copied from physical test XMLs.

## Texture and Rendering Inventory

Static audit outputs:

- `tools/audit/out/asset_inventory.csv`: 126 rows
- `tools/audit/out/duplicate_textures.csv`: 0 duplicate texture groups
- `tools/audit/out/mujoco_model_cache_report.csv`: 14 wrapper models

Largest texture found:

- LIBERO plate texture: 4096x4096, 8.26 MB

Typical YCB object textures:

- 1024x1024, about 0.31 to 0.57 MB

The new stable object pack does not require high-resolution textures. External high-resolution textures were not rewritten in place because they live outside the repo and are shared assets. For training, prefer the stable pack or a downsampled texture variant. Keep high-resolution external textures for videos/evaluation only.

Rendering frequency:

- The environment captures at policy-step frequency, not every physics substep. The substep loop only captures on the final substep when frame capture is enabled.

Remaining render task:

- Run the profiler on the intended EGL/OSMesa training host and require a nonzero two-camera render/readback FPS before image-based training.

## Predicate Smoke Tests

Output: `tools/audit/out/mujoco_success_predicate_smoke.csv`.

Scripted predicate cases:

- move-to-object;
- push;
- relation;
- between;
- put-into-plate.

Result:

- cases: 5
- successes: 5
- all passed: true

## Smoke Tests Run

Commands run successfully:

- `python3 -m py_compile ...` for modified modules and audit scripts;
- `python3 -c "... stable_objects ..."` compiled all 12 stable object XMLs;
- `python3 -c "... cdpr.xml ..."` compiled active CDPR XML;
- `python3 tools/audit/profile_mujoco_rollout.py --resets 100 --steps 1000 --render-steps 1000`;
- `python3 tools/audit/profile_mujoco_rollout.py --disable-cache --resets 20 --steps 100 --render-steps 1`;
- `python3 tools/audit/test_mujoco_contact_stability.py --steps 240`;
- `python3 tools/audit/test_mujoco_success_predicates.py`;
- `python3 -m unittest tests.test_cdpr_env_state_snapshot tests.test_cdpr_instruction_tasks tests.test_cdpr_wrapper_bundle`.

Unittest result:

- 32 tests passed.

`pytest` was not available in this local Python environment, so the existing unittest suite was used.

## Is MuJoCo Good Enough For Short Training?

For short no-render physics/debug rollouts with selected stable objects: mostly yes.

For image-based short training: not yet, because two-camera render/readback could not be measured in this local context.

For grasp-heavy training over the full stable pack: not yet, because block and can fail the current gripper squeeze/lift tests due to reach/contact setup.

## Is Migration Comparison Justified Now?

No. The current evidence does not show that MuJoCo itself is the blocker.

The next work should stay in MuJoCo and focus on:

- validating two-camera render/readback on the actual training host;
- fixing gripper/object placement or finger reach for block/can;
- broadening contact tests after those two objects pass;
- optionally adding texture downsample variants inside the repo instead of mutating external asset packs.

Only after those items are measured should a simulator migration comparison be considered.
