# SmolVLA pick-up training: run history and findings

CDPR · SmolVLA · GRPO · `pick_up` · MJWarp backend

- Prepared 2026-08-01
- Config: `configs/examples/cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml`
- Code: `ffe974d`
- Phase 0 (`move_to_object`) is covered by `SMOLVLA_MOVE_TO_TRAINING_REPORT.md`
  and is only summarized here.

Seven multi-million-step pick_up runs reached the same place: the policy learned
to grasp and refused to lift. This is what was actually wrong, what was
eliminated along the way, and where the current run stands at 2M steps.

---

## 1. The finding

**The reward paid nothing for a lift that happened and then settled.**

The GRPO return is the last active step's reward, and the lift term was read off
the *instantaneous* height. A policy that raised the object 40 mm at env step 40
and let it settle by step 128 scored exactly what a policy that never moved
scored. The reward was asking the policy to still be holding the object 5 cm up
at the final step — a strictly harder task than "raise it 5 cm", and not the task
that was asked for.

Measured over 4.2M steps under that term:

| | 0–0.5M | 3.0–4.3M |
|---|---:|---:|
| `physical_grasp_rate` | 0.380 | **0.447** |
| `bilateral_pad_contact_rate` | 0.416 | **0.482** |
| `post_grasp_rise_mean_m` | 0.0194 | **0.0099** |
| success \| pre-grasped | 0.239 | **0.036** |
| success \| normal start | 0.108 | **0.042** |
| `group_reward_std_mean` | 0.805 | **0.517** |

The policy got monotonically **better at grasping and worse at lifting**. It had
converged onto "close the gripper and hold still" — worth 2.75 with no risk —
and `group_reward_std` collapsing is the signature of the eight GRPO candidates
ceasing to differ. Handed a free grasp it raised the object 7 mm against a 50 mm
threshold.

GRPO was optimizing the reward correctly. The reward was wrong.

Fixed in `ffe974d`: `BatchedTaskState.peak_lift` ratchets the highest lift
reached *while grasped*. It cannot be earned by batting the object upward (it
only accumulates while `state.grasped`) and cannot be lost to a later drop.

Result over the first 2M steps of the next run:

| | best before | after |
|---|---:|---:|
| success \| pre-grasped | 0.336 | **0.60** |
| success \| normal start | 0.126 | **0.26** (peak, pre-promotion) |
| `post_grasp_rise_..._prelifted` | ~0.021 ceiling | **0.035** |
| `residual_target_cosine_mean` | flat / negative | **rising 0.018 → 0.063** |
| curriculum cap | 0.03 in every run | **0.05** |

---

## 2. What happened

### Prior state — three runs, same ending

Grasp rate reached ~0.30 and stopped; `post_grasp_rise_mean_m` peaked at 18 mm
and decayed to 7–10 mm against the 50 mm success height, with the first grasp
landing at env step ~27 of 64 — ample time remaining. Two interventions aimed at
it failed:

- `entropy_coef` 0 → 0.0002 slowed the decay, did not stop it.
- `e787c80` made the grasp bonus a ratchet on `ever_grasped`, so a failed lift no
  longer cost the grasp credit. Delayed the decay ~0.9M steps, did not stop it.

Both were betting on the policy *discovering* the lift from behind a
0.30-probability grasp. Neither questioned whether the reward paid for it.

### The verification harness could not run

`scripts/render_cdpr_task_reference_episodes.py` drives the production reset,
reward, success predicate and grasp detector under a scripted oracle, and the
config points at it as the verification of record. It aborted at `e787c80` with
`Reward breakdown disagrees with the training reward by 1.000000` — that commit
made the grasp bonus a ratchet and the overlay still read `state.grasped`.
Fixed in `894a516`.

It also hard-coded `backend="mujoco_cpu"`, so on the GPU box it verified a CPU
relative of the training physics and said so nowhere. `62e647b` added
`--physics {auto,mjlab_mjwarp,mujoco_cpu}`; MJWarp was missing `controller_state`
and `render_world`, which is why nobody had noticed it could not run there.

> **Rule this produced:** a verification harness that has not been run since the
> code it verifies changed is not evidence.

### The pre-grasped stage (`d610dcc`)

A configurable fraction of GRPO groups start with the object already grasped at
its rest height, so the lift gets dense signal from env step 0. Sampled per
*group* and broadcast with `repeat_interleave`, because GRPO normalizes advantage
within a group of eight and a mixed group would score the spawn rather than the
actions.

The A/B, both arms from the same checkpoint over 400k steps:

| normal-start worlds only | with the stage | control (fraction 0) |
|---|---:|---:|
| worlds / update | 780 | 1024 |
| ever-grasp rate | 0.593 | 0.609 |
| **grasp → success conversion** | **0.205** | **0.177** |
| **success rate** | **0.1215** | **0.1076** |

The control has 31% *more* normal-start worlds per update and still does worse,
and the whole gap sits in conversion at an identical ever-grasp rate — which is
exactly the transfer the stage was built for. Raised to 0.5 in `ffe974d`.

### The gate it quietly broke (`cd2f91d`)

The approach curriculum widens the start-distance cap when an instruction's pass
rate crosses 0.30, computed over every world. Pre-grasped worlds perform no
approach and succeed 0.320 against 0.121 — so the gate was reading **0.169**
where the approach-relevant number was **0.121**, and climbing 0.076 → 0.170
toward a threshold it had no business crossing.

`instruction_outcome_counts` now emits both pairs; the curriculum reads
`instruction_successes_normal_start/{name}`. Extracting it from the training loop
is what made it testable — six lines inline in a loop no test could reach is why
it stayed invisible.

> **Rule this produced:** when you add a population to the batch, audit every
> aggregate that feeds a control loop, not just the ones you added.

### The residual telemetry (`d3fa5d6`)

`policy_target_cosine_mean` minus `prior_target_cosine_mean` was negative in
every run measured — −0.005 over 1.5M, −0.005 on the probe, −0.022 on the
control and widening to −0.044. After 2.3M steps at `approx_kl` 0.11 per update,
the composed policy was aligned no better with the object than the frozen SmolVLA
prior it sits on.

Two very different failures produce that number and need opposite fixes: a
residual near zero, or a large residual pointed somewhere unrelated. Nothing
logged could tell them apart. `residual_action_norm_mean` against
`prior_action_norm_mean` sizes the trainable path;
`residual_target_cosine_mean` gives its own direction. Measured on the
deterministic mean action so σ ≈ 0.25 of exploration noise does not swamp it.

**Answer: the residual is roughly the same magnitude as the prior action
(1.30–1.39 against 1.32–1.59) and was aimed nowhere** — cosine 0.04, alignment
rate 0.40, below chance. The trainable path was learning hard and learning
nothing about reaching the object. That, not the reset distribution, was the
deepest problem.

It is the one metric now improving on its own (0.018 → 0.063 over 2M).

---

## 3. Eliminated, with evidence

Recording these because each cost a diagnostic cycle and none should be
re-litigated without new evidence.

| hypothesis | evidence against |
|---|---|
| MJWarp physics differs from the CPU reference in a way that blocks grasping | oracle 3/3 at reward 5.59–5.70 under production MJWarp |
| The grasp detector's 0.05 N threshold does not transfer to GPU contacts | measured pad forces 3–106 N |
| A failed lift is strongly −EV, so not trying is correct | computed break-even 0.077–0.206 against 0.165 achieved — roughly neutral |
| Degenerate GRPO groups amplify rollout noise | `group_reward_std_mean` 0.86; degenerate groups 7.4%, *identical* in both populations |
| Gripper geometry / pad offset | verified in `CDPR_MANIPULATION_TASK_CONSISTENCY_REPORT.md`; oracle succeeds |
| The pre-grasped stage transfers nothing | A/B: conversion 0.205 vs 0.177 |
| Peak-lift ratcheting just banks exploration noise | validation reward recovered 0.120 → 0.218 and residual cosine rose; noise-banking would show neither |

The fourth line is worth keeping: `torch_group_advantages` divides by the group
std floored at `1e-6`, and the dense-reward informative-group filter is
`reward_span > 1e-6`, which never fires — `informative_groups` has equalled
`groups_collected` on every update of every run. That *is* a real amplification
path, it is simply not this bug. ~7% of groups.

---

## 4. New metrics

All emitted per update; counts are global sums, `_mean`/`_rate` are rank means.

### Outcome by starting stage
- `successes_prelifted` / `worlds_prelifted`
- `successes_normal_start` / `worlds_normal_start`
- `prelifted_start_rate` — realized fraction, checks the sampler against config

A mean over both populations cannot distinguish "no world lifts" from "40% lift
fully and the rest not at all". That ambiguity cost one full diagnostic cycle.

### Post-grasp behaviour
- `post_grasp_rise_mean_m{,_prelifted}` — peak EE rise after the first grasp
- `post_grasp_first_env_step_mean{,_prelifted}` — how late the grasp lands
- `post_grasp_worlds{,_prelifted}`

Split rather than merged: pre-grasped worlds grasp at env step ~8 by
construction, so folding them in would drag the headline toward 0 as the fraction
rose and break comparability with earlier runs.

### Trainable-path health
- `residual_action_norm_mean` vs `prior_action_norm_mean` — is it doing anything
- `residual_target_cosine_mean`, `residual_target_alignment_rate` — is it aimed

### Advantage divisor
- `group_reward_std_mean{,_prelifted,_normal_start}`
- `degenerate_reward_groups{,_prelifted}` — groups with spread < 0.05

### Known measurement caveat
The `*_target_cosine_*` family is a **decision-0** probe against the direction to
the object. With `prelifted` at 0.5, half the worlds start with the end-effector
*at* the object, so that direction is undefined and the cosine is noise for them
— which is why `policy_target_alignment_rate` reads 0.29 at fraction 0.5, 0.45 at
0.25 and 0.58 at 0. **These need a prelifted split before the absolute values
mean anything.** The policy-minus-prior *difference* is robust to it.

---

## 5. The current run (2M of 16M)

Warm-started weights-only from `pick_up_16M_20260731_230423/rl/step_0806981`,
chosen because it was the peak of that run on all nine tracked metrics.

**The curriculum promoted for the first time in the project's history:**

```
1075k  cap=0.0300  ema=0.300  norm=0.400   <- EMA crosses the gate
1082k  cap=0.0500  ema=0.302  norm=0.320   <- cap promotes, start distance +67%
1089k  cap=0.0500  ema=0.180  norm=0.180   <- pass rate drops: harder task
```

`success | normal start` falling 0.264 → 0.170 after 1.2M is the harder task, not
a worse policy — `success | pre-grasped` held at ~0.60 through the same window,
as it must, since pre-grasped starts approach nothing.

Working: the lift. `success | pre-grasped` 0.60 against a 0.336 prior best;
`post_grasp_rise` 0.035 against a 0.021 ceiling; `validation/reward_mean`
0.19–0.22, the highest in any run; `residual_target_cosine` rising.

Stuck: the approach at the 5 cm cap. The EMA has been flat at 0.165–0.19 against
the 0.30 gate for ~900k steps. `post_grasp_rise` has also plateaued at 0.035
against the 0.05 needed for success.

---

## 6. Open

**`log_std_mean` is exactly −1.19375 across every update of the last three runs**
(549, 284 and 62 updates). In earlier runs it moved (−1.3893 → −1.3962). It is
interior to the [−1.45, −1.10] clamp, so it is not pinned at a bound. `approx_kl`
also fell 0.105 → 0.030 and `gradient_norm` 6.7 → 3.2 across the same boundary.
If the exploration parameter is frozen, the current results are being obtained
*despite* that rather than because of it. Unresolved; the check is a one-line
dump of the actor's `log_std` from any checkpoint.

**The train/validation gap.** Training normal-start success ~0.23 against
deterministic validation success 0.001–0.004. `validation/reward_mean` is
recovering, but the mean action still almost never completes the task. The
residual cosine trend is the thing to watch.

**Two untested levers**, deliberately not changed so the peak-lift result stays
interpretable: `vla_update_max_records: 128` against ~50k records per update
(`vla_lora/kl` has sat at ~0.0001 in every run), and `clip_range [0.20, 0.28]`
with `clip_fraction` ~0.23–0.35.

---

## 7. Reproducing

```bash
cd /root/repo/RL_VLA_Bootstrapping
conda run --no-capture-output -n cdpr-mjlab python3 -m pytest -q tests
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab PHYSICS=mjlab_mjwarp \
  bash scripts/render_cdpr_task_reference_episodes_remote.sh
```

The oracle harness is the check to run after touching reset shaping, the reward
ladder or the grasp geometry. Expect 3/3 at reward ~7.2 on the reference
catalogs (`robocasa_apple robocasa_tomato robocasa_orange robocasa_potato`); the
config's full pool includes a banana the scripted oracle cannot grip, so it
reports 2/3 there. Compare against a same-invocation baseline, never against a
number quoted in a config comment.

Commits, in order: `894a516`, `d610dcc`, `d4d693c`, `62e647b`, `857f3ea`,
`cd2f91d`, `d3fa5d6`, `ffe974d`.
