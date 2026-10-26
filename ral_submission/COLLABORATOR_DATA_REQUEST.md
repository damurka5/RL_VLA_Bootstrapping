# Data request for the joint RA-L manuscript

26 October 2026. The manuscript now presents LIBERO reward experiments and CDPR embodiment adaptation as parts of one paper. Please recover existing files first; new experiments are listed separately below.

## Minimum package to request now

### 1. Exact code and resolved configurations

Please send the repository/commit and actual launched configurations for:

- Primary dense, filtered dense, phase-only with terminal coefficient 1, terminal-only with coefficient 5, and dense with coefficient 1.
- Both low-success runs and the shared initialization evaluation.

Include the reward/tracker implementation, training entry point, launch commands and any command-line overrides. For each run, identify the initial checkpoint, final evaluated checkpoint/step, and selection/stopping rule. Checkpoint hashes or immutable model revisions are useful; model weights themselves are not the first requirement.

The reward code must resolve:

- Exact reach, grasp, lift, move and place predicates, thresholds and progress functions; task-specific geometry or normalization.
- How the best-progress value is updated/reset, whether phases can regress or be skipped, and what happens on failure, timeout, success and environment reset.
- Whether reward is computed per simulator step, executed action or action chunk, and how it becomes the trajectory return used for GRPO. Include sum/mean/last-step aggregation, discount and terminal handling.
- Whether terminal reward is emitted once or repeatedly; where shaping clipping and terminal scaling happen; whether progress and phase bonuses share the same clip.
- Which tasks support the phase tracker and what reward is used for any unsupported task.
- All group-acceptance rules for every run: terminal-accuracy bounds, reward-variance thresholds, dynamic resampling/refill and invalid-trajectory handling. Batch size 32 alone does not specify the number of candidates sharing one scene.

**Why:** these details let us describe the method as a reproducible part of our paper and determine exactly how its phase logic aligns with CDPR milestones. A reward equation alone does not establish the implemented objective.

### 2. Raw evaluation outcomes behind all percentages

For the 61.0% initialization, all five aggregate configurations, and initial/intermediate/final low-success evaluations, please send per-episode CSV/JSON or the original evaluator output, containing:

- Run/configuration ID, training seed, checkpoint/step and evaluation seed.
- Canonical LIBERO task name and task ID, initial-state ID, binary outcome, episode length and failure/timeout status.
- Exact success counts and attempted-episode counts per task, including numerical failures/aborts and any excluded attempts.
- Evaluation action-sampling settings, horizon, simulator/benchmark version and split.

Confirm whether each percentage is a mean over task success rates or a pooled episode rate, and whether all conditions use the same initial-state list and number of trials. Provide every existing training seed; if there is only one, state that directly. Please identify the canonical task behind `task_1`.

**Why:** we need honest denominators, uncertainty and paired comparisons where possible. We cannot infer episode counts or independent training seeds from rounded percentages.

### 3. Training and throughput records

Please send full TensorBoard/W&B exports, JSONL/CSV logs, or console logs for those runs, preferably with wall-clock timestamps. Required counters are:

- Attempted rollout groups/episodes, accepted groups, rejected groups and rejection reasons.
- Optimizer updates and environment actions, distinguishing attempted/sampled interaction from accepted training data.
- Collection seconds, elapsed training time, evaluation time and available GPU-hours.
- Validation success versus elapsed time, optimizer updates and environment actions, with run resumption/interruption markers.
- Per-group terminal accuracy and dense-return spread where available.

State the budget or stopping rule actually shared by the reward comparisons. In the low-success experiment, dense training completes more updates, so equal elapsed time and equal optimizer-update count are different comparisons. Confirm whether all intended non-reward settings were actually identical.

**Why:** this distinguishes a practical throughput benefit from better learning per update or per interaction. Existing logs may already support several matched views without another training run.

### 4. Model and optimization specification

Please provide the base OpenVLA-OFT checkpoint/revision; the action-output and likelihood code; tokenizer, decoder and normalization details; the exact LoRA target-module list; rank/alpha; trainable/frozen modules; chunk length and executed prefix; group size; learning rate; PPO/GRPO settings; KL/entropy terms; epochs/minibatches; numerical precision; and GPU memory/offloading configuration.

**Why:** the current description combines an OFT checkpoint with discrete-token likelihood training. We need the actual code path to state this accurately and distinguish it from the CDPR continuous residual likelihood.

### 5. Editable experimental materials

Please send the original TeX, bibliography, plotting scripts and underlying figure data, plus representative successful and failed rollout videos with task/run IDs. Record publication/submission status and contributor roles in the internal project notes so the joint author list and final packaging can be completed. The anonymous manuscript will present this as one paper.

**Why:** editable sources avoid transcription errors, and videos help check whether phase predicates reflect the intended behavior. Videos do not replace evaluation counts.

## One high-value additional LIBERO experiment

If an additional run is feasible, compare the existing primary dense setting with a **phase-only run using the same terminal coefficient and all other settings**:

| Setting | Existing primary dense | Matched phase-only |
|---|---:|---:|
| Terminal coefficient | 5 | 5 |
| Phase coefficient | 0.3 | 0.3 |
| Within-phase progress coefficient | 0.2 | 0 |
| Shaping clip | 0.05 | 0.05 |
| Initialization, task states, filter, optimizer, budget | Same | Same |

Use the primary run's actual filter and stopping rule, to be recovered from the resolved config; do not substitute the filtered-dense condition. If the old run cannot be matched reproducibly, rerun both conditions together. Use all feasible independent training seeds and matched evaluation scenes.

The existing phase-only result changes terminal and shaping coefficients, so the 65.3% versus 62.3% gap does not isolate adding within-phase progress. The proposed comparison estimates the effect of adding that progress term under the shared clipping rule. Together with terminal-only, it would give the paper a much cleaner account of boundary feedback versus finer progress feedback.

This experiment strengthens the reward-construction claim. It does not validate the dense-to-binary switch or self-imitation schedule by itself.

## What remains on the CDPR side

These are project evidence gaps, not requests for collaborators to reconstruct our code:

- Exact selected-checkpoint lineage, per-skill dense-to-binary transition points and matched consolidation outcomes.
- Full-task comparison from the same initializer: terminal-only, milestone rewards with common trajectory credit, and implemented stage-specific credit.
- Completed hardware trial records and the final evaluation after checkpoint selection is frozen.

We can complete the narrative now while leaving these measurements pending. The paper must keep each experimental claim tied to what was actually compared.

## Short message ready to send

> We are integrating the LIBERO experiments into our joint RA-L paper as the reward-construction evaluation, alongside CDPR adaptation and task composition. Could you send (1) the exact reward/tracker code and resolved configs/launch commands for all five aggregate configurations and both low-success runs; (2) raw evaluation outcomes and per-task success/attempt counts, seeds, checkpoint IDs and the canonical `task_1` name; (3) training logs with elapsed time, environment actions, attempted/accepted groups and optimizer updates; and (4) the exact OpenVLA-OFT/action-likelihood/LoRA setup, plus editable TeX and plotting sources? Existing files are enough to start. If one more comparison is feasible, the most useful is phase-only versus primary dense with terminal coefficient 5 and identical phase coefficient, clipping, filter, initialization and budget. This would isolate adding within-phase progress, which the current phase-only row does not.

## Where to place files

Keep original collaborator files in `incoming/phase_aware_dense_rewards/`, retaining their names and checksums. Existing intake schemas are `data_templates/libero_eval_episodes.csv`, `training_timeseries.csv` and `run_manifest.json`; original exports are also acceptable and preferable to manually retyping results. Record unknown fields as unknown rather than zero.
