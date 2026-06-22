# CDPR recovery runbook

## Decision

Do not continue sparse training from checkpoint `step_0048000` yet.

Use the last dense-reward checkpoint as the recovery initialization:

`/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_grpo_complex_tasks_20260610_141107/rl/step_0216000`

Checkpoint `step_0048000` is still useful as an A/B baseline, but its downloaded evaluation showed
0/100 canonical `move_to_object` successes on the normal shell and 0% on the gripper manipulation
tasks. Resuming it is technically possible, but there is no evidence that its sparse-stage updates
are worth preserving.

## Stage A: infrastructure preflight

Pull the latest code, then run reset-only tests before loading OpenVLA:

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --rebase origin main

conda run --no-capture-output -n openvla-oft \
  python3 scripts/smoke_test_cdpr_resets.py \
  --episodes-per-case 10 \
  --max-reset-attempts 10
```

The required result is `failures=0`. Any failure is an environment/reset defect, not a model
failure. Inspect `reset_smoke_results.csv` before training.

The June 22 reset report (`episodes=110 failures=40`) was not caused by OpenVLA or by cached scene
positions. All 40 failures were the four direct-actuator cases. Their goal builder rejected
open/close/yaw instructions, while unsuccessful attempts also exposed the existing
`ee_outside_workspace` initialization retry. Direct-actuator instructions now keep a stationary
XYZ goal, so they no longer consume every valid reset attempt with
`Unsupported instruction type for goal generation`.

Verify the simulator response for every action dimension:

```bash
conda run --no-capture-output -n openvla-oft \
  python3 scripts/verify_cdpr_direct_actuators.py
```

The required result is `"passed": true`.

Before entering sparse training, also run:

```bash
conda run --no-capture-output -n openvla-oft \
  python3 scripts/smoke_test_cdpr_resets.py \
  --include-reverse-shells \
  --episodes-per-case 5 \
  --max-reset-attempts 10
```

Also exercise the training-specific prebuilt-wrapper path:

```bash
conda run --no-capture-output -n openvla-oft \
  python3 scripts/smoke_test_cdpr_prebuilt_randomized_resets.py \
  --episodes 40 \
  --max-reset-attempts 10
```

Require `failures=0`, `reset_retries=0`, and `passed=True` from both smoke tests.

Prebuilt wrappers are compatible with randomized EE starts only because reset now teleports the
MuJoCo `ee_free` joint to the sampled start and recalibrates cable preload before validation. Do
not remove that reset step: controller-only movement from the wrapper's baked pose caused repeated
`ee_outside_workspace` failures in training even when the reset-only smoke test passed.

Known reset failures quarantine the selected cached scene locally, force a fresh pose-matched
wrapper on the next attempt, and use a deterministic workspace-center start for the final two
attempts.

## Stage B: A/B checkpoint baseline

Evaluate the dense checkpoint:

```bash
CHECKPOINT_DIR=/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_grpo_complex_tasks_20260610_141107/rl/step_0216000 \
bash scripts/evaluate_cdpr_action_primitives_remote.sh
```

Evaluate the current sparse checkpoint:

```bash
CHECKPOINT_DIR=/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_grpo_complex_tasks_20260618_221123/rl/step_0048000 \
bash scripts/evaluate_cdpr_action_primitives_remote.sh
```

Compare canonical and synonym rates for every primitive. Prefer the dense checkpoint unless the
current checkpoint has a clearly higher minimum per-instruction rate without worse reset stability.

## Stage C: dense action-vector recovery

The dense stage now trains:

- X: `move_left`, `move_right`
- Y: `move_top`, `move_bottom`
- Z: `move_up`, `move_down`
- yaw: clockwise and counterclockwise gripper rotation
- gripper: open and close
- visual grounding: `move_to_object`

### Added paraphrases

- left: “move the end effector left”; “shift the gripper to the left”
- right: “move the end effector right”; “shift the gripper to the right”
- forward: “move the end effector forward”; “shift the gripper away from the robot”
- backward: “move the end effector backward”; “shift the gripper toward the robot”
- up: “raise the end effector”; “lift the gripper upward”
- down: “lower the end effector”; “bring the gripper downward”
- open: “spread the gripper fingers”; “open the fingers fully”
- close: “bring the gripper fingers together”; “close the fingers fully”
- clockwise: “turn the end effector clockwise”; “yaw the gripper clockwise”
- counterclockwise: “turn the end effector counterclockwise”; “yaw the gripper counterclockwise”

### Dense control rewards

For translation, progress is requested signed displacement divided by 5 cm, clipped to `[0, 1]`.
The reward adds positive step progress and correct-axis action, and subtracts wrong-direction
action, off-axis action, and orthogonal drift. Success requires 5 cm requested displacement with no
more than 5 cm orthogonal drift.

For opening/closing, progress is normalized movement toward opening 1 or 0. A correct gripper action
adds `0.10`; a wrong-direction action subtracts `0.10`. Mean absolute XYZ command is penalized with
weight `0.20`. Success is opening at least `0.80` or at most `0.20`.

For yaw, progress is signed rotation divided by `0.50` rad. Positive incremental rotation and a
correct yaw action are rewarded; wrong-direction yaw is penalized more strongly. Small physical
rotation is amplified with a `0.02` rad step scale, and mean absolute XYZ command is penalized with
weight `0.20`. Success requires at least `0.50` rad in the requested direction.

All dense success bonuses are explicitly `0.0`; success terminates the episode but does not add a
`+1` reward spike.

### Recovery learning rate

The recovery run loads adapter, action-head, and actor-stat weights from `step_0216000`, but creates
a new optimizer and restarts the scheduler. Use `5e-6` with five warmup updates and cosine decay to
`1.25e-6` (`lr_min_factor=0.25`). This is intentionally lower than the previous `1.5e-5` because
the run is preserving a useful dense checkpoint while adding stronger yaw/gripper signals.

### Scene freshness

`use_wrapper_cache` is disabled for recovery, while `prebuild_scene_cache` remains enabled. This
builds a fresh wrapper pool at the start of each run—so updated YCB assets cannot be shadowed by old
isolated wrapper bundles—then reuses that fresh pool for rollout speed. Object poses are still
randomized on every reset. Ten desk textures and a restrained six-color dark/neutral background
palette provide visual variation without making the image overly bright.

The OpenVLA-OFT prebuilder normally hardcodes `use_cache=True`. The fast GRPO wrapper overrides that
path when `--no-use_wrapper_cache` is set: every prebuilt base wrapper must have a run-local
`__rltmp_...` name, and startup prints:

```text
[env_cache] Built fresh run-local wrapper pool ...; old wrapper cache ignored.
```

Do not continue if startup instead prints `Using cached wrapper` during the scene-cache prebuild.
Known MuJoCo reset transients are retried up to `max_train_reset_attempts=10`; unrelated runtime
errors still fail immediately.

Sparse training cannot start until:

- every dense instruction has at least 10 validation episodes per round;
- every instruction reaches at least 70% success;
- the mean is at least 70%;
- the complete gate passes three consecutive validation rounds (at least 30 validation episodes per
  instruction across the passing rounds).

The 150-update dense limit is diagnostic only: `dense_stage_open_on_max_updates` is disabled, so it
cannot force a sparse transition.

Start recovery:

```bash
bash scripts/train_cdpr_grpo_complex_tasks.sh
```

Monitor:

- `stage/dense/success_rate/<instruction>`
- `stage/dense/consecutive_passes`
- `stage/dense/required_consecutive_passes`
- action-dimension means, standard deviations, and saturation
- reset failures or MuJoCo instability warnings

Stop and diagnose if any instruction remains below 50% for three validation rounds or if reset
failures are nonzero.

## Stage D: primitive acceptance evaluation

At candidate checkpoints, run:

```bash
CHECKPOINT_DIR=/root/repo/RL_VLA_Bootstrapping/runs/<recovery_run>/rl/<step> \
EVAL_OUTPUT_ROOT=/root/repo/RL_VLA_Bootstrapping/runs/<recovery_run>/primitive_evaluation \
bash scripts/evaluate_cdpr_action_primitives_remote.sh
```

Do not accept a checkpoint unless all canonical primitive rates are at least 70%, all ten requested
videos/action traces are valid, and reset failures are zero. Treat synonym rates as a separate
generalization metric; they should not be hidden inside the canonical score.

## Stage E: sparse reverse-frontier training

Only after Stage D passes should the automatic gate enter sparse training. Shell distances have
been smoothed:

- common positional shells: 1–2 cm, 2–5 cm, 5–10 cm, then normal reset;
- push approach shells: 0–0.5 cm, 0.5–2 cm, 2–5 cm, 5–10 cm, then normal reset;
- grasp shells now increase gradually in XY distance and safe vertical clearance.

Keep promotion at 70% if the purpose is stable competence. A 50% shell-promotion threshold is
appropriate only for exploration, not checkpoint acceptance.

### Sparse-stage episode outcome telemetry

LC-HOL writes the selected rollout episode outcome after every training update:

- `lchol_episode_stats/sparse_episode_outcomes.csv` contains one row per completed sparse-stage
  episode, including binary reward `0/1`, terminal reason, instruction, shell, episode returns,
  replay-record count, replay options, and whether the episode contributed hindsight records.
- `lchol_episode_stats/sparse_episode_outcome_summary.csv` contains refreshed global counts and
  reward `0/1` ratios for the whole run, each instruction, and each instruction/shell pair.

The CSV writer uses a process lock, so the files combine all DDP ranks. TensorBoard mirrors the
global summary under `stage/sparse/buffer_episode_outcomes/global/*` after all ranks finish each
update. Rank-local diagnostic metrics remain under
`stage/sparse/buffer_episode_outcomes/rank_local/*`.

## Experiment records

For each run, keep this tuple together:

1. Git commit.
2. Config file and any command-line overrides.
3. Initialization checkpoint.
4. Training run directory.
5. Candidate checkpoint.
6. Reset smoke report.
7. Primitive evaluation report.
8. Full shell/synonym evaluation report.
9. A short decision: reject, continue dense, or enter sparse.

Never compare headline aggregate rates without also checking canonical normal-scene rates,
per-instruction minima, synonym rates, reset failures, and action traces.
