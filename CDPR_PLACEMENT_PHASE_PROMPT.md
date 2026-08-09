# Phase 2 briefing — `put_into_plate` / `put_into_bowl`

Paste this as the opening message of a fresh session.

---

## Task

Get `put_into_plate` and `put_into_bowl` working on the CDPR embodiment with RL
only. **No demonstrations, no behaviour cloning.** The contribution is that
manipulation is learnable by RL alone on a frozen VLA prior; proposals needing
demos are out of scope however effective.

## Setup

* **Robot**: cable-driven parallel robot. 5-D action `(x, y, z, yaw, gripper)` in
  `[-1,1]`, scaled 0.015 m / 0.08 rad / 0.05 opening per env step. The controller
  target is re-anchored to the measured EE pose every step. Workspace xy ±0.28 m,
  z [0.18, 0.60]; objects rest at z ≈ 0.19–0.21 on a desk.
* **Policy**: frozen SmolVLA prior + trainable residual MLP (1024 hidden) + LoRA
  on the action expert. `action = tanh(prior + residual)`, per-dim Gaussian
  exploration.
* **Residual input**: 6-d proprioception `[ee_xyz, yaw, gripper, 0]` + a frozen
  512-d random projection of SmolVLA connector tokens. No privileged target
  vector — it was removed as deployment-invalid.
* **Algorithm**: GRPO, groups of 8 sharing a start, advantage = group-centred
  reward / group std.
* **Sim**: MuJoCo Warp, 512 worlds/rank × 2 ranks, 26 decisions × 4 env steps.
* **Config**: `configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml`
* **Run**: remote 2×A40, `conda run -n cdpr-mjlab`, always with
  `RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl` — the box has no route to huggingface.co.
  I run things; you write and push. Always `git commit && git push` when done.

## Why phase 2 may be easier than phase 1, and why that is the interesting part

Phase 1 (`pick_up`) is **paused, not solved.** It is blocked on one measured
thing: the frozen SmolVLA encoder localizes objects to roughly **3–5 cm**, and
`pick_up` needs **~2 cm**. `CDPR_SMOLVLA_CAMPAIGN_REPORT.md` §4.12–§4.16 has the
full account.

Placement starts with the object **already held**, so it does not need the
approach phase 1 is stuck on. And its success radii are far larger — plate
**0.091 m**, bowl **0.057 m**, against pick_up's ~2 cm grasp tolerance.

**So the question this phase answers is: does 3–5 cm of localization suffice at a
9 cm target radius?** If yes, placement works and the campaign has a working
instruction. If no, the encoder limit is general rather than specific to grasping,
and that is a much more important result than either task.

## Established — measured, not argued. Do not re-litigate without new evidence.

* **The XY plant is healthy.** Realized gain 0.44–0.54, flat across every
  amplitude 0.05–0.60, symmetric in sign, same free and loaded, no dead zone,
  0.8 mm of uncommanded drift per episode. Note the gain: the effective step is
  **~0.0075 m**, half the nominal `action_step_xyz`. Any horizon arithmetic that
  assumes 0.015 is wrong.
* **The z plant IS dead-zoned when loaded**: sustained `a_z` 0.05→3 mm,
  0.20→34 mm, 0.30→83 mm. Per-step i.i.d. noise explores sustained bias with
  σ/√N, so the lifting region sat 3.3σ out. Fixed with a per-episode action
  offset scored against the marginal.
* **Vision is load-bearing and coarse.** Shuffling the residual's vision block
  across worlds takes success 0.053 → 0.000 and ever-grasped 0.162 → 0.035.
* **No reduction of that feature helps.** `flat_random`, `per_token_random` and
  both side by side were tried; the aiming cosine did not move in 5.2M steps.
* **RL gradient degrades the encoder.** Vision-tower LoRA attached (48 modules)
  and moved (`vla_lora/kl` 3.7e-5 → 1.16e-4), and the policy got monotonically
  worse — first curriculum demotion of the campaign. The tower feeds the prior
  AND the residual's feature, so adapting it moves the ground under both, and
  only the action is KL-constrained.
* **You cannot remove an input from a trained MLP.** Zeroing the residual's
  vision columns to swap the feature shifted every downstream activation:
  handed a *perfect* object position afterwards, ever-grasped fell 0.92 → 0.30.
  Adding zero-initialised columns is the only safe form.
* **The grasp detector, grasp-state observability and the reward ladder are all
  cleared.** Slip peaks at 5.7 mm against an 8 mm bound; proprioception alone
  decodes "holding" at 0.898.

## Traps this campaign actually fell into. Check for each in phase 2.

1. **Validation measured a different task for 52M steps.**
   `set_random_start_max_goal_distance` was called on the training resetter only,
   so validation ran full-workspace starts (≥0.10 m) while training ran ≤0.05 m.
   Fixed, with a regression test. **Verify realized start distance from the
   reset, never from the logged cap.**
2. **The horizon was coupled to the approach cap**, which starved the phase whose
   step cost does not shrink with distance. Now flat at 26 decisions.
3. **A third of every update was rollout noise.** Groups whose 8 candidates
   scored within 0.05 emit full-magnitude advantages after dividing by a tiny
   std. `grpo_min_group_reward_std` filters them per return stream. Watch
   `filtered_record_fraction` (~0.34).
4. **The promote gate fired on single threshold crossings** and landed below the
   line at the new rung. Now needs 5 consecutive updates, and reads the **grasp
   rate**, not full-task success — the approach curriculum must not wait on a
   skill the approach cannot influence.
5. **Metrics that look comparable and are not.** `policy_target_cosine_mean` is
   computed on the SAMPLED action; `prior_`/`residual_target_cosine_mean` on the
   mean. Pooling a cosine over an episode makes any sustained drift read
   negative. Decision 0 is the only comparable point.
6. **A success-vs-failure gap is not evidence of skill.** Selecting on success
   selects on alignment for any policy, including a blind one.

## Tools that already exist — extend rather than duplicate

* `tools/audit/xy_approach_probe.py` — builds the full training stack from a
  checkpoint and runs the trainer's own `validate_round` with the action or the
  residual's input substituted. Legs: `plant` (open-loop gain sweeps), `policy`
  (what the deterministic policy commands), `oracle` (hand over the true target
  position, and price how accurate it must be), `ablation` (destroy the vision
  input). Also `--start-distance-cap`, `--horizon-decisions`.
* `tools/audit/grasp_feature_probe.py` — linear and MLP probes on the frozen
  features with episode folds, hard negatives and permutation controls.
* `tools/audit/success_episode_videos.py` — renders successful episodes and
  reports aiming cosine stratified by start distance.
* `tools/audit/lift_barrier_probe.py` — scripted oracle to a real latched grasp,
  then sweeps sustained commands.

## What I want first

Do **not** start a long run yet. Placement has never been trained (phase 2 has
not started) and one consistency report already flags **F3: placement never
actually holds the object** — verify that before anything else.

Then: an analysis that establishes whether placement is reachable at all, using
the existing probes. The obvious first measurement is the oracle arm — hand the
policy the true receptacle position and see whether the task completes — because
that prices the localization requirement for a 9 cm radius the same way it did
for grasping, and it costs one probe run rather than a training run.

State up front what each measurement would show and what result would falsify it.

## How I work

Be sceptical of my framing and of your own conclusions. Over ~20 runs the
recurring failure has not been bad ideas — it has been measurements that were
technically correct but did not imply what we thought, and conclusions drawn from
a control that was not a null. Several confident interventions were reverted.
When a result contradicts a claim you made earlier, say so plainly and correct it.
