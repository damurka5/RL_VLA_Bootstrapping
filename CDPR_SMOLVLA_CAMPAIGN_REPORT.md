# CDPR SmolVLA manipulation campaign: move-to → pick-up

CDPR · SmolVLA · GRPO · MJWarp backend

- Prepared 2026-08-01, code `942bf52`
- Phase 0 `move_to_object`: **28M steps**, complete
- Phase 1 `pick_up`: **2M of 16M**, running
- Configs: `cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml`,
  `cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml`

Detail references, retained rather than folded in: metric-by-metric definitions
and the move-to framing/geometry forensics live in
`SMOLVLA_MOVE_TO_TRAINING_REPORT.md`; the pick-up geometry audit that preceded
training is `CDPR_MANIPULATION_TASK_CONSISTENCY_REPORT.md`.

---

## 1. The through-line

Both phases failed the same way, twice each, and it took multi-million-step runs
to see it both times:

> **The policy was optimizing the reward correctly. The objective was wrong in a
> way that every curriculum metric reported as healthy.**

- **move-to** — the action distribution inflated for 12M steps under a
  `max_log_std` ceiling of −0.30 that never bound. Cap climbing, gate promoting,
  demotions firing correctly, validation peaking at 7.3% — and underneath it a
  policy sampling too widely to servo.
- **pick-up** — the lift term was read off the *instantaneous* height while the
  GRPO return is the *terminal* reward, so a lift that happened and then settled
  paid nothing. Grasp rate climbed monotonically while the lift decayed, and
  every grasp-side metric looked like progress.

The second-order lesson is the same in both: the failure was invisible because
the thing being measured was not the thing that mattered. move-to needed
`log_std_mean` read as a *trend* rather than a level; pick-up needed the lift
outcome split by starting stage and the residual sized against the frozen prior.
Neither existed until the run had already failed.

---

## 2. Where it stands

| | move-to | pick-up |
|---|---:|---:|
| steps | 28M (complete) | 2M of 16M |
| best validation success | 7.3% (at 7.4M) | 0.1–0.4% |
| `validation/reward_mean` | — | 0.19–0.22 |
| success \| pre-grasped | — | 0.60 |
| success \| normal start | — | 0.17 at the 5 cm cap |
| curriculum cap | reached 0.23 | 0.05 (promoted at 1.08M) |

pick-up is where move-to was at roughly 3M: the mechanism works, the deterministic
policy does not yet generalize, and the curriculum has started to move.

---

## 3. Phase 0 — `move_to_object`, 28M steps

### 3.1 · 5M — baseline

Full-workspace starts, one object. Ended at 1–2% validation success. Became the
warm-start source for everything after, including phase 1.

Two runs were discarded before this line stabilized, both measurement bugs:

- **2M** — `BatchedRandomWorkspaceMoveToResetter.reset()` called the base class,
  which places the end-effector under the curriculum cap, then overwrote both
  pose and horizon from its own sampler with no maximum. Every curriculum signal
  was dropped. The tell: pass rate 0.044 at a 0.03 m cap and 0.037 at 0.34 m — a
  cap that changes nothing produces a pass rate that does not respond to it.
  Fixed in `c76cbb1`.
- **350k** — with the plumbing repaired, the 0.03/0.01 promote/demote thresholds
  were degenerate: they had been tuned while the previous bug pinned the pass
  rate under 0.045. The real range is 0.06–0.41, so promote fired every update.
  Now 0.30/0.12 with a 15-update cooldown and EMA re-seeding on cap change.
  Fixed in `17a83f7`.

> **Rule:** when a measurement bug is fixed, re-derive every threshold that was
> tuned against the broken numbers.

### 3.2 · 7M — productive phase, then the diffusion collapse

Best checkpoint `step_7405457`, validation 7.3%.

Everything curriculum-side behaved: the cap climbed under a real gate, demoted
twice when the policy fell behind, and the distractor unlock restarted it
cleanly. Underneath, `log_std_mean` bottomed at −1.227 at 3.6M and rose
monotonically thereafter. Nothing pushes it down, so a net-positive entropy bonus
wins on a long run, and the −0.30 ceiling never bound.

| global step | `log_std_mean` | validation |
|---:|---:|---:|
| 202k | −1.204 | 0.043 |
| 3.56M | **−1.227** (min) | 0.030 |
| **7.41M** | −1.162 | **0.073** (peak) |
| 11.29M | −1.100 | 0.051 |
| 13.73M | −1.000 | 0.033 |
| 16.00M | −0.895 | 0.007 |

Supporting: `entropy_mean` 0.83 → 4.15, `policy_target_cosine_mean` 0.25 → 0.14
(touching −0.007 at 12.3M, i.e. task-blind), `candidate_reward_mean` 0.79 → 0.47.

**Distractors were exonerated.** Split by phase, the two-object window scored the
highest mean reward of the whole run at unchanged grounding — 0.761 against 0.696
and 0.723 for the single-object windows.

Fixed in `7d99c3e`: `max_log_std` −0.30 → **−1.10**, just above the [−1.23,
−1.15] band the policy occupied while productive, so diffusion is impossible
rather than discouraged; `entropy_coef` 0.0005 → **0.0**, since lowering it from
0.002 slowed the drift but could not stop it.

### 3.3 · 8M + 8M — post-fix

Two runs with the diffusion closed off and grounding opened up: two objects from
step 0 (`64c1437`) and the `ee_workspace_z_bounds` ceiling 0.52 → 0.47, then a
second run at slightly different hyperparameters.

> **Gap in this document.** I have not analysed the TensorBoard for these two
> runs — the numbers above come from the 16M forensics. Their per-run outcomes,
> the final cap, and whether the −1.10 ceiling held flat or pinned are not
> recorded here. Send the two event files and I will fill this section in
> properly; until then treat it as a structural description, not a result.

What is known: phase 1 warm-starts from
`cdpr_smolvla_move_to_scratch_mjwarp_w512_20260719_081705/rl/step_5000081`, and
the pick-up config carries the `max_log_std −1.10` ceiling forward — the report's
own note that these must match across phases or the weight hand-off reintroduces
the failure.

---

## 4. Phase 1 — `pick_up`

Warm-started **weights-only** from the move-to adapter. A full resume would also
restore the approach-curriculum state and optimizer moments; the cap move-to
reached was earned on a different task, and reusing it would drop pick-up
straight into far starts with no grasp skill.

### 4.1 · The prior three runs — grasp learned, lift refused

Grasp rate reached ~0.30 and stopped; `post_grasp_rise_mean_m` peaked at 18 mm
and decayed to 7–10 mm against a 50 mm success height, with the first grasp
landing at env step ~27 of 64 — ample time remaining. Two interventions failed:

- `entropy_coef` 0 → 0.0002 slowed the decay, did not stop it.
- `e787c80` made the grasp bonus a ratchet on `ever_grasped`, so a failed lift no
  longer cost the grasp credit. Delayed the decay ~0.9M steps, did not stop it.

Both bet on the policy *discovering* the lift from behind a 0.30-probability
grasp. Neither asked whether the reward paid for it.

### 4.2 · The verification harness could not run

`scripts/render_cdpr_task_reference_episodes.py` drives the production reset,
reward, success predicate and grasp detector under a scripted oracle, and the
config names it as the verification of record. It aborted at `e787c80` with
`Reward breakdown disagrees with the training reward by 1.000000` — that commit
made the grasp bonus a ratchet and the overlay still read `state.grasped`
(`894a516`).

It also hard-coded `backend="mujoco_cpu"`, so on the GPU box it verified a CPU
relative of the training physics and said so nowhere. `62e647b` added
`--physics {auto,mjlab_mjwarp,mujoco_cpu}`; MJWarp turned out to be missing
`controller_state` and `render_world`, which is why nobody had noticed it could
not run there. First MJWarp run: **3/3 at reward 5.59–5.70**, which exonerated
the physics.

> **Rule:** a verification harness that has not been run since the code it
> verifies changed is not evidence.

### 4.3 · The pre-grasped stage (`d610dcc`)

A configurable fraction of GRPO groups start with the object already grasped at
rest height, so the lift gets dense signal from env step 0. Sampled per *group*
and broadcast with `repeat_interleave` — GRPO normalizes advantage within a group
of eight, so a mixed group would score the spawn rather than the actions.

A/B from the same checkpoint, 400k steps each, normal-start worlds only:

| | with the stage | control (fraction 0) |
|---|---:|---:|
| worlds / update | 780 | 1024 |
| ever-grasp rate | 0.593 | 0.609 |
| **grasp → success conversion** | **0.205** | **0.177** |
| **success rate** | **0.1215** | **0.1076** |

The control has 31% *more* normal-start worlds and still does worse, and the
whole gap sits in conversion at an identical ever-grasp rate — the transfer the
stage was built for. Raised to 0.5 in `ffe974d`.

### 4.4 · The gate it quietly broke (`cd2f91d`)

The approach curriculum widens the cap when a pass rate crosses 0.30, computed
over every world. Pre-grasped worlds perform no approach and succeed 0.320
against 0.121 — so the gate read **0.169** where the approach-relevant number was
**0.121**, climbing 0.076 → 0.170 toward a threshold it had no business crossing.
`instruction_outcome_counts` now emits both pairs and the curriculum reads
`instruction_successes_normal_start/{name}`.

> **Rule:** when you add a population to the batch, audit every aggregate that
> feeds a control loop, not just the ones you added.

### 4.5 · What the trainable path was doing (`d3fa5d6`)

`policy_target_cosine_mean` minus `prior_target_cosine_mean` was negative in
every run measured — −0.005 over 1.5M, −0.005 on the probe, −0.022 on the control
and widening to −0.044. After 2.3M steps at `approx_kl` 0.11 per update, the
composed policy was aligned no better with the object than the frozen SmolVLA
prior it sits on.

Two very different failures produce that and need opposite fixes: a residual near
zero, or a large residual pointed somewhere unrelated. Nothing logged could tell
them apart. **Answer: the residual is roughly the magnitude of the prior action
(1.30–1.39 against 1.32–1.59) and was aimed nowhere** — cosine 0.04, alignment
0.40, below chance.

Also added `min_log_std` −5.0 → −1.45: `max_log_std` had stopped upward diffusion
since move-to, but nothing stopped collapse into a point and −5.0 is not a floor.

### 4.6 · The finding — terminal lift credit (`ffe974d`)

The GRPO return is the last active step's reward, and the lift term was read off
the instantaneous height. A policy that raised the object 40 mm at step 40 and let
it settle by step 128 scored exactly what one that never moved scored. The reward
was asking it to still be holding the object 5 cm up at the final step — strictly
harder than "raise it 5 cm", and not the task.

Measured over 4.2M steps under that term:

| | 0–0.5M | 3.0–4.3M |
|---|---:|---:|
| `physical_grasp_rate` | 0.380 | **0.447** |
| `post_grasp_rise_mean_m` | 0.0194 | **0.0099** |
| success \| pre-grasped | 0.239 | **0.036** |
| success \| normal start | 0.108 | **0.042** |
| `group_reward_std_mean` | 0.805 | **0.517** |

Monotonically better at grasping, worse at lifting. It had converged onto "close
the gripper and hold still" — the 2.75 rung — and `group_reward_std` collapsing is
the eight candidates ceasing to differ. Handed a free grasp it raised the object
7 mm.

`BatchedTaskState.peak_lift` ratchets the highest lift reached *while grasped*:
it cannot be earned by batting the object upward, and cannot be lost to a later
drop. Ladder retuned around it — grasp bonus 1.0 → 0.5 (the half already
learned), lift weight 1.0 → 2.0, success bonus 2.0 → 3.0.

| terminal state | before | after |
|---|---:|---:|
| grasps, holds still | 2.75 | 2.25 |
| lifts 20 mm, holds | 3.15 | 3.05 |
| **lifts 40 mm, then settles** | **2.75** | **3.60** |
| lifts 51 mm → success | 5.75 | 7.25 |

### 4.7 · The current run — 2M of 16M

Warm-started from `pick_up_16M_20260731_230423/rl/step_0806981`, the peak of that
run on all nine tracked metrics.

**The curriculum promoted for the first time in the project's history:**

```
1075k  cap=0.0300  ema=0.300  norm=0.400   <- EMA crosses the gate
1082k  cap=0.0500  ema=0.302  norm=0.320   <- cap promotes, start distance +67%
1089k  cap=0.0500  ema=0.180  norm=0.180   <- pass rate drops: harder task
```

`success | normal start` falling 0.264 → 0.170 after 1.2M is the harder task, not
a worse policy — `success | pre-grasped` held at ~0.60 through the same window,
as it must, since pre-grasped starts approach nothing.

| | best before | now |
|---|---:|---:|
| success \| pre-grasped | 0.336 | **0.60** |
| `post_grasp_rise_..._prelifted` | ~0.021 ceiling | **0.035** |
| `residual_target_cosine_mean` | flat / negative | **rising 0.018 → 0.063** |
| `validation/reward_mean` | ~0.20 plateau | 0.19–0.22 |
| curriculum cap | 0.03 in every run | **0.05** |

Stuck: the approach at the 5 cm cap — the EMA has been flat at 0.165–0.19 against
the 0.30 gate for ~900k steps. `post_grasp_rise` has plateaued at 0.035 against
the 0.05 needed.

---

## 5. Eliminated, with evidence

Each cost a diagnostic cycle. None should be re-litigated without new evidence.

| hypothesis | evidence against |
|---|---|
| Distractors broke the 16M move-to run | the two-object window was its best phase (reward 0.761 vs 0.696/0.723) |
| MJWarp physics blocks grasping | oracle 3/3 at reward 5.59–5.70 under production MJWarp |
| The grasp detector's 0.05 N threshold does not transfer to GPU contacts | measured pad forces 3–106 N |
| A failed lift is strongly −EV, so not trying is correct | break-even 0.077–0.206 against 0.165 achieved — roughly neutral |
| Degenerate GRPO groups amplify rollout noise | `group_reward_std_mean` 0.86; degenerate groups 7.4%, *identical* in both populations |
| Gripper geometry / pad offset | verified in the consistency report; oracle succeeds |
| The pre-grasped stage transfers nothing | A/B: conversion 0.205 vs 0.177 |
| Peak-lift ratcheting just banks exploration noise | validation reward recovered 0.120 → 0.218 and residual cosine rose; noise-banking shows neither |

The degenerate-group line is worth keeping: `torch_group_advantages` divides by
the group std floored at `1e-6`, and the informative-group filter is
`reward_span > 1e-6`, which never fires — `informative_groups` has equalled
`groups_collected` on every update of every run. That *is* a real amplification
path. It is simply not this bug, at ~7% of groups.

---

## 6. Open

**`log_std_mean` is exactly −1.19375** across every update of the last three
pick-up runs (549, 284, 62 updates). Earlier runs moved (−1.3893 → −1.3962). It
is interior to the [−1.45, −1.10] clamp, so it is not pinned at a bound.
`approx_kl` also fell 0.105 → 0.030 and `gradient_norm` 6.7 → 3.2 across the same
boundary. If the exploration parameter is frozen, the current results are being
obtained *despite* that. Unresolved; the check is a one-line dump of the actor's
`log_std` from any checkpoint.

**The train/validation gap.** Training normal-start success ~0.23 against
deterministic validation 0.001–0.004. The mean action still almost never
completes the task; the rising residual cosine is the thing to watch.

**The cosine metrics need a prelifted split.** They are a decision-0 probe against
the direction to the object, and at `prelifted: 0.5` half the worlds start with
the end-effector *at* the object, so that direction is undefined for them —
`policy_target_alignment_rate` reads 0.29 at fraction 0.5, 0.45 at 0.25, 0.58 at
0. The policy-minus-prior *difference* is robust; the absolute values are not.

**Two untested levers**, deliberately unchanged so the peak-lift result stays
interpretable: `vla_update_max_records: 128` against ~50k records per update
(`vla_lora/kl` ~0.0001 in every run), and `clip_range [0.20, 0.28]`.

**Phase 2** (`put_into_plate` / `put_into_bowl`) has not started. The consistency
report's F3 — placement never actually holds the object — is unresolved.

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

Phase 1 commits, in order: `894a516`, `d610dcc`, `d4d693c`, `62e647b`,
`857f3ea`, `cd2f91d`, `d3fa5d6`, `ffe974d`.
Phase 0 fixes referenced: `c76cbb1`, `17a83f7`, `7d99c3e`, `64c1437`.
