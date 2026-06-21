# CDPR checkpoint 0048000 evaluation remediation

This report analyzes the downloaded evaluation:

`/Users/damirnurtdinov/Desktop/My Courses/Диплом/RL_VLA_Bootstrapping/remote_server/evaluation/step_0048000_20260621_185012`

The evaluated checkpoint was:

`/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_grpo_complex_tasks_20260618_221123/rl/step_0048000`

## What the old evaluation actually showed

- Overall metric result: 365/2820 = 12.94%.
- `move_to_object`: 179/600 = 29.83%, but this aggregate mixed easy reverse shells, the normal scene,
  canonical prompts, and synonyms.
- `move_to_object` reverse shells: 92%, 56%, 21%, then 3.33% on the final normal shell.
- Canonical `move_to_object` on the final normal shell: 0/100.
- Synonyms on the normal shell: 7/100 and 3/100.
- The success threshold was 0.10 m in XY. This is lenient rather than overly strict; the low normal
  scene score is model/generalization behavior, not a 2.5 cm benchmark artifact.
- The old evaluator did not record exact policy-call telemetry. From its `steps`, `chunk_length=8`,
  and `replan_every=8`, it made an estimated 37,351 OpenVLA generations in metric episodes and
  41,071 across all 3,068 attempts.
- All 40 downloaded MP4 files decode successfully with local `ffprobe`. They were falsely marked
  invalid on the server only because `ffprobe` was unavailable.
- The log contains 10 MuJoCo `Nan, Inf or huge value in QACC` instability warnings.
- Several saved summaries contain physically impossible post-reset positions, including object/goal
  coordinates outside the intended tabletop workspace. Reverse shells teleported the free joint
  without recalibrating cable preload, and grasp shells placed the long fingers too close to the
  table/object.
- Gripper-only object-edge tasks had 0% success. The curriculum did not provide clean language
  primitives for opening, closing, clockwise yaw, or counterclockwise yaw.
- The continuation configuration started directly in sparse mode while loading the same actor-stat
  file as both first- and second-stage statistics. No real exploration-distribution transition
  occurred, and Adam state for `log_std` was retained.

## Implemented fixes

### 1. OpenVLA action visibility

Every recorded frame now shows:

- `NEW OPENVLA OUTPUT` or `cached OpenVLA chunk`
- exact policy call number and chunk action index
- normalized `[x, y, z, yaw, gripper]` action
- scaled physical command
- current end-effector position/yaw and gripper actual/target opening

Every recorded video also receives a sibling `*_actions.csv` trace.

### 2. Exact OpenVLA generation counts

The evaluator now counts policy generations and applied action steps exactly per episode and in
aggregate. These values are written to `episode_results.csv`, video summary JSON files,
`validation_manifest.json`, and `validation_report.md`.

### 3. Honest `move_to_object` reporting

The report now separates canonical final-normal-scene scores from reverse-shell scores and adds a
2.5/5/10/15 cm threshold sweep using each trajectory's minimum XY distance. This prevents easy
shells from masking deployment-like performance and makes benchmark strictness measurable.

### 4. GRPO sparse-stage adaptation

Sparse entry now:

- requests a stage-specific actor-stat reset even when a continuation starts directly in sparse;
- initializes `log_std` from a dedicated second-stage checkpoint or `sparse_stage_init_log_std`;
- removes stale optimizer moments for `log_std`;
- writes before/after/source evidence to `grpo_stage_transition.jsonl`.

The recovery config resumes checkpoint 0048000, runs a dense actuator-calibration stage, then the
sparse reverse-frontier stage. It no longer points first- and second-stage actor stats at the same
file.

### 5. Reset and simulation stability

- End-effector and object targets are clamped to task workspace bounds.
- Reverse-shell teleportation zeros velocity and warm-start acceleration, recalibrates cable
  preload, and holds the current pose.
- Grasp-shell clearances account for finger length.
- Reset state and every post-action state are checked for non-finite/excessive dynamics and
  out-of-workspace end effectors/objects.
- Validation retries randomized invalid resets and records retry/instability counts.

### 6. Direct gripper/yaw language primitives

Added trainable/evaluable instructions:

- `open_gripper`: “open the gripper”
- `close_gripper`: “close the gripper”
- `rotate_gripper_clockwise`: “rotate the gripper clockwise”
- `rotate_gripper_counterclockwise`: “rotate the gripper counterclockwise”

Each has direct dense progress, command-direction feedback, explicit success thresholds, canonical
text, two synonyms, and arbitrary qualitative prompts. Open/close episodes start from the opposite
gripper state, and yaw episodes start at a neutral configured angle, so none of these tasks can
succeed trivially at reset.

## Verification

- 101 focused evaluator/environment/GRPO/reward tests pass.
- Full suite: 258/259 pass locally. The remaining test requires the external OpenVLA
  `prismatic` package, which is not installed in the local lightweight Python environment.
- Python compilation and `git diff --check` pass.
- All 40 downloaded evaluation MP4 files pass local `ffprobe`.
- Real MuJoCo normalized-action actuator check:
  - open gripper: 0.999999
  - close gripper: 0.000001
  - clockwise yaw delta: -1.0594 rad
  - counterclockwise yaw delta: +1.0594 rad
  - finite simulation state: yes

The actuator mechanics, reward criteria, telemetry, reports, video validation, reset guards, and
GRPO stage transition are confirmed. The old checkpoint's task success rates are historical and
cannot improve without running the recovery training and then re-evaluating the resulting
checkpoint on the remote GPU server.

## Remote commands

Run recovery training:

```bash
cd /root/repo/RL_VLA_Bootstrapping
bash scripts/train_cdpr_grpo_complex_tasks.sh
```

Then evaluate the new checkpoint:

```bash
CHECKPOINT_DIR=/root/repo/RL_VLA_Bootstrapping/runs/<new_run>/rl/<new_step> \
EVAL_OUTPUT_ROOT=/root/repo/RL_VLA_Bootstrapping/runs/<new_run>/evaluation \
bash scripts/evaluate_cdpr_complex_checkpoint_remote.sh
```

Run the standalone actuator check:

```bash
conda run -n openvla-oft python3 scripts/verify_cdpr_direct_actuators.py
```
