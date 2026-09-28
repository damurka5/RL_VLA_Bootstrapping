# SmolVLA complex-instruction GRPO comparison

## Common experimental contract

Both runs initialize the residual actor from:

`/root/repo/RL_VLA_Bootstrapping/runs/cdpr_smolvla_stage3_object_dense_complex_resume_step_5000000_to_10000000_20260710_193100/rl/step_6700000`

The old checkpoint is TD3-shaped, so only its `actor` state is imported. TD3 critics and optimizer state are intentionally not loaded into GRPO. A later GRPO checkpoint resumes the policy, optimizer, curriculum, and LC-HOL replay state normally.

The shared instruction set is:

- `move to <object>`
- `push <object> left/right`
- `put <object> into bowl`
- `put <object> on plate`
- `put <object> to the left/right of <other object>`
- `put <object> between <object 1> and <object 2>`

The relation tasks retain the existing internal environment IDs (`move_left_of_object`, `move_right_of_object`, and `move_between_objects`) for checkpoint compatibility, but the language given to SmolVLA uses the requested `put ...` wording.

Every policy-gradient candidate reward is exactly `0.0` or `1.0`. Dense manipulation terms, motion penalties, instability penalties, and output scaling cannot alter the GRPO reward in sparse-binary mode. `move to` now requires planar distance at most 3 cm and rejects an episode after a vertical excursion greater than 1.5 cm, even if the gripper later returns to its original height.

Both configurations use a GRPO batch/minibatch size of 16,384, a microbatch size of 2,048, groups of four candidates, and train from global step 6,700,000 to 10,000,000.

## Approach A: Reverse Frontier

Each instruction has exactly four independently tracked shells:

| Shell | Target simulator steps | Reset abstraction |
|---|---:|---|
| 0 | 1–5 | Very near completion |
| 1 | 5–10 | Near completion |
| 2 | 10–20 | Intermediate state |
| 3 | 20–40 | Full manipulation start; placement begins above the uncaught object |

The concrete reset is randomized inside every shell. Approach tasks randomize a planar direction and distance; push tasks randomize contact pose and remaining displacement; placement/relation tasks randomize a held-object pose around the desired goal in shells 0–2 and use an open gripper above the object in shell 3.

Frontiers are maintained per instruction, not globally. At each validation interval the active shell receives 50 deterministic rollouts. Promotion requires at least 80% success and can advance only one shell. Training samples the active frontier 80% of the time and previously passed shells 20% of the time to reduce forgetting. A separate 20-episode full-task validation is logged for a fair comparison with LC-HOL++.

Expected strengths:

- The sparse reward becomes observable from the first update.
- Progress and failure are easy to diagnose per instruction and shell.
- It explicitly solves the missing grasp-competence transition for placement.

Expected risks:

- Curriculum resets create a distribution shift from ordinary full-task starts.
- Hand-designed shell geometry can make some instructions easier than others.
- Promotion validation is comparatively expensive.

## Approach B: LC-HOL++

The design follows the goal-relabeling principle from [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495), adapted to an on-policy GRPO learner. When a trajectory fails its requested task but achieves an allowed alternative predicate, the prefix ending at the first achievement is relabeled with that achieved instruction.

Examples include direct Cartesian motion, approaching or grasping the target, picking it up, pushing either the target or another scene object, and placing the target near its reference. Relabel options are filtered by the original instruction, matching the requested allowlists.

Synthetic success records are not mixed into the GRPO advantage calculation. Doing that would make an action sampled under one language-conditioned policy appear on-policy under a different instruction. Instead:

1. Real environment candidates produce the exact binary GRPO update.
2. A relabeled instruction is encoded from the same pre-action state.
3. The transition enters a balanced, capacity-bounded per-option replay buffer.
4. A separate behavior-cloning negative-log-likelihood update trains the residual actor with coefficient 0.20.

This keeps the GRPO estimator honest while still learning from useful failed behavior.

Placement instructions have independent two-stage curricula:

- Stage 0: the target starts already held.
- Stage 1: the gripper starts open above the target, so the policy must acquire and manipulate it.

Each placement instruction advances after at least 30 same-stage episodes and 80% success in its recent 50-episode history.

Expected strengths:

- Failed rollouts can still teach useful language-action associations.
- Per-option balancing prevents common direct-motion relabels from dominating rarer grasp or near-reference events.
- Normal full-task validation directly measures whether relabel learning transfers.

Expected risks:

- Incorrect achievement predicates create incorrect supervision.
- LC-HOL++ can improve primitives without improving long-horizon composition.
- The auxiliary BC coefficient may need reduction if it suppresses GRPO exploration.

## TensorBoard analysis

Use the full-task validation curves as the primary comparison:

- `validation/success_rate`
- `validation/instruction_success_rate/<instruction>`
- `rollout/zero_advantage_group_rate`
- `rollout/candidate_binary_reward_rate` (should remain 1.0)

Reverse Frontier diagnostics:

- `reverse_frontier/<instruction>/active_shell`
- `reverse_frontier/<instruction>/validation_success`
- `reverse_frontier/<instruction>/train_updates`

LC-HOL++ diagnostics:

- `lchol/replay_size`
- `lchol/replay_size_by_option/<option>`
- `lchol/relabel_count/<option>`
- `lchol/put_stage/<instruction>/active_stage`
- `train/hindsight_bc_loss`
- `train/hindsight_records`

The preferable method is the one with the higher worst-instruction full-task success, not merely the higher mean. Also inspect time-to-first-success, zero-advantage group rate, and retention of `move_to_object` and push success. Reverse Frontier is favored if LC-HOL++ fills replay but full-task placement stays flat; LC-HOL++ is favored if it reaches comparable placement success with fewer environment steps and does not regress the simpler instructions.

## Running both experiments

```bash
bash scripts/train_cdpr_smolvla_complex_grpo_dual_remote.sh
```

By default the script launches Reverse Frontier on GPU 0 and LC-HOL++ on GPU 1, writes each experiment to its own timestamped directory, and prints both TensorBoard log directories. Important overrides include `REVERSE_GPU`, `LCHOL_GPU`, `CHECKPOINT`, `MAX_TRAIN_STEPS`, `RUN_REVERSE`, `RUN_LCHOL`, `WALLTIME`, and `DRY_RUN=1`.
