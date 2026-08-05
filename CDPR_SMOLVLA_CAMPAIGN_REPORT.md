# CDPR SmolVLA manipulation campaign: move-to → pick-up

CDPR · SmolVLA · GRPO · MJWarp backend

- Prepared 2026-08-01, updated 2026-08-05, code `97c2e05`
- Phase 0 `move_to_object`: **28M steps**, complete
- Phase 1 `pick_up`: **~52M steps over 16 runs**, running
- **Compute spent: ~99M GRPO environment steps across 26 runs**
- **Training depth in the current policy: ~20.1M** — the runs are a tree of warm
  starts, not one chain, so these are different numbers (see §2)
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

The sharpest instance of that arrived late. `log_std_mean` read **exactly
−1.193750** on every update of five consecutive runs, which looked like a frozen
parameter and was treated as a curiosity for most of the campaign. Dumping the
actor showed the raw tensor spanning −3.51 to −0.30 with **all twenty trained
entries outside the clamp band** — fifteen on the ceiling, five on the floor,
none inside — and `clamp()` has zero gradient outside its bounds, so every one of
them had been dead since the first time it was hit. Eight sat at −0.2996: the
move-to ceiling of −0.30 that `7d99c3e` replaced, fossilized before that fix and
carried through every warm start since. `entropy_coef` had therefore been inert
for the entire campaign, and every `min_log_std`/`max_log_std` change had been a
no-op. Fixed in `80f6c35`.

---

## 2. Where it stands

| | move-to | pick-up |
|---|---:|---:|
| steps | 28M (complete) | ~52M over 16 runs, running |
| best validation success | 7.3% (at 7.4M) | 1.2% |
| `validation/reward_mean` | — | 0.16 |
| **success \| pre-grasped** | — | **0.83 held** (`split_credit`) |
| success \| normal start | — | 0.21 at the 5 cm cap |
| `post_grasp_action_z_mean` | — | **+0.33**, above the +0.30 lift threshold |
| `post_grasp_rise` (pre-grasped) | — | 44–48 mm against a 50 mm bar |
| deterministic validation | — | **0.001**, and 0.40 m from the object |
| curriculum cap | reached 0.23 | 0.05 |

The pre-grasped lift is solved as far as sampled success goes, and §4.9 explains
why it took so long: the loaded z axis is dead-zoned below `a_z ≈ 0.2`, and
per-step Gaussian noise explores sustained bias with std `sigma/sqrt(N)`, so the
lifting region sat 3.3 sigma out. Correcting the exploration estimator raised
`post_grasp_action_z_mean` for the first time in the campaign.

Splitting the GRPO return at the latch (§4.10) then recovered the approach the
lift work had cost, without giving the lift back — the first time both phases
hold at once, and a stable plateau rather than a peak. On the full task it is a
wash against the baseline: normal-start success 0.206 against 0.202, ever-grasped
0.226 against 0.234.

**Which is the finding.** Deterministic validation has read 0.000–0.012 all
campaign, and the metric beside it says why: the policy ends every validation
episode **0.39–0.42 m from the grasp point**, uniformly across objects, against a
ceiling-minus-grasp-point of 0.405 m — pinned at the top of the workspace, from
starts capped at 5 cm. `policy_target_cosine_mean` 0.09 says the same thing
another way. **The mean policy has never learned to servo to the object**; every
success in this campaign is exploration noise finding it from a start already
close, which is also why the cap has never left 0.05 m. §4.10 has the evidence
and the one-line check that would confirm it directly.

### Step accounting

Two different quantities, and they are not interchangeable.

**Compute spent — 98.6M steps.** Every step actually executed, measured from the
TensorBoard event files and deduplicated by run (several directories are
successive dumps of the same run):

| | runs | steps |
|---|---:|---:|
| `pick_up` (identified) | 16 | 52,193,837 |
| `move_to_object` (identified) | 4 | 26,004,402 |
| pre-dating the per-instruction metrics | 6 | 20,417,827 |
| **total executed** | **26** | **98,616,066** |

The third row cannot be attributed automatically — `instruction_worlds/{name}`
did not exist when those ran. The three runs added since the last revision are
the exploration arc of §4.9, all resumed from the same `adaptivestd` checkpoint:
`offset_ungated` 3,558,059, `offset_gated` 2,590,256, `offset_marginal`
2,593,019, and `split_credit` 2,493,631 resumed from `offset_marginal`.

**Training depth in the current policy — ~20.1M steps.** The runs are a *tree*, not
a chain. Warm starts were frequently taken from a mid-run checkpoint rather than
a run's end, which discards everything after the branch point, and some runs
started from no checkpoint at all. Summing runs therefore overstates how much
training any single set of weights carries. The traceable lineage of the run in
flight:

| segment | steps at the branch point |
|---|---:|
| move-to scratch adapter | 5,000,081 |
| `pick_up_warmstart_20260731_100335` | 803,288 |
| prelifted run | ~1,400,000 |
| control `nopre` | 407,399 |
| `pick_up_16M` | 806,981 |
| `peaklift_16M` | 1,006,618 |
| `liftgate_16M` | 3,203,866 |
| `adaptivestd_16M` | 2,406,053 |
| `offset_marginal` | 2,593,019 |
| `split_credit` | 2,493,631 |
| **lineage total** | **~20.1M** |

The two failed offset runs are *not* in the lineage: they branch from the same
`adaptivestd` checkpoint and were abandoned, so their 6.1M steps are compute
spent without training depth.

Note the phase-0 contribution is the **5M scratch adapter**, not the 28M
campaign end — pick-up warm-started from an early move-to checkpoint. The
pre-session hops are as recorded in the run launchers and have not been verified
against the checkpoints themselves; the exact tree is recoverable with
`grep -h '^warmstart_checkpoint=' runs/*/train.log`, which the launcher writes
for every run.

Also note the 28M quoted for move-to is the *productive* decomposition (5M
baseline + 7M to the collapse + 8M + 8M), whereas the executed table sums whole
event files, so the 16M run contributes its full length rather than its
productive 7M.

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
the 0.05 needed. It went on to turn over at ~3M; see the ledger below.

### 4.8 · Run ledger

Every pick-up run for which telemetry was analysed, in order. "Peak" is the best
`success | pre-grasped` the run reached; the two 400k entries are the A/B pair.

| run | steps | what changed | peak | outcome |
|---|---:|---|---:|---|
| prelifted | 1.54M | pre-grasped stage at 0.25 | — | every metric decayed; grasp 0.41→0.30, reward 2.24→1.69 |
| degenerate probe | 407k | instrumentation only | 0.32 | A/B arm **with** the stage |
| control `nopre` | 407k | fraction 0 | — | A/B arm **without**: normal-start 0.1076 vs 0.1215 |
| `pick_up_16M` | 4.22M | ratchet + retuned ladder | 0.24 | collapsed: grasp 0.38→0.45, rise 19→10 mm, success 0.11→0.04 |
| `peaklift_16M` | 4.78M | **peak-lift ratchet** (`ffe974d`) | **0.59** | best yet, then turned over at ~2M |
| `liftgate_16M` | 5.95M | **lift-gated grasp bonus** (`03f9f20`) | **0.82** | best of campaign; turned over at ~3M |
| `adaptivestd_16M` | 2.34M | **log_std projection** (`80f6c35`) | 0.82 | **decay arrested**; stable plateau, no climb. Re-read later: flat on *every* metric across its full length — the campaign's true baseline |
| `ladder_16M` | 4.00M | fraction + cap curricula (`0a9bbe4`) | 0.75 | **regression** — ended below its own start; fraction curriculum reverted in `ccc7589` |
| `offset_ungated` | 3.56M | per-episode z offset 0.20, whole episode (`f9dc94f`) | 0.71 | collapsed: `a_z` +0.20 → −0.37, rise 42.5 → 8.6 mm. Estimator bug, see §4.9 |
| `offset_gated` | 2.59M | same offset gated on holding (`1e4a401`) | 0.66 | collapsed **indistinguishably** — which is what identified the estimator rather than the gating |
| `offset_marginal` | 2.59M | **marginal log-prob** (`a4bf902`, on in `88b0b30`) | **0.83** | **first run to raise `post_grasp_action_z_mean`** (+0.24 → +0.32); pre-grasped record, held. Grasp rate and normal-start success fell |
| `split_credit` | 2.49M | **separate returns for approach and lift** (`a38dfb7`) | **0.83** | **first run to hold both phases**: approach recovered (ever-grasped 0.198 → 0.226) with the lift kept (+0.325). Full-task numbers level with the baseline; exposed the servoing failure in §4.10 |

Two results carried by that table:

**`03f9f20` produced the campaign's best numbers.** Grasp quality and lift were
measured anti-correlated — `corr(grasp rate, prelifted rise) = −0.786` and
`corr(pad force, rise) = −0.696` over 451 updates, with pad force rising
6.4 → 8.0 N and slip *falling* 3.86 → 2.97 mm as the lift died 35 → 15 mm. Gating
the grasp bonus on the object having left the desk reversed all four contact
metrics and took pre-grasped success 0.59 → 0.82.

**`80f6c35` stopped the decay.** Every run before it peaked and then collapsed;
this one held pre-grasped success at 0.79–0.82 across 2.3M steps with pad force,
slip and grasp rate all flat, and `log_std_saturated_fraction` fell 0.21 → 0.09
with `entropy_mean` moving for the first time in the campaign. It did not,
however, produce any *climb* — the system settled into a stable plateau.

**The pre-grasped stage is load-bearing, not a scaffold.** The clearest negative
result of the campaign, and it came from the curriculum that was built to
exploit the opposite assumption. `PreliftedStageCurriculum` annealed the
fraction on the stage's own success, and inside a single 4M-step run it ran the
experiment in both directions:

```
fraction 0.50 -> 0.20 over 602k    success | pre-grasped  0.753 -> 0.410
fraction 0.20 -> 0.50 over 618k    success | pre-grasped  0.410 -> 0.552
```

0.82 pre-grasped success never meant the lift had been learned and the practice
could be withdrawn — it meant the practice was holding the lift up. Nor is the
cost local: moving batch onto the approach made the **approach** worse too,
`success | normal start` 0.245 → 0.098, because a degraded lift means the
normal-start episodes that do reach a grasp can no longer finish it. The damage
is partly hysteretic — 2.4M steps back at 0.50 recovered pre-grasped success only
to 0.55 against the 0.79–0.82 it held before — so the run ended below the
checkpoint it started from. Reverted to a fixed 0.5.

The cap ladder from the same commit was never exercised: the EMA stayed at
0.10–0.22 against the 0.30 gate for the whole run, so no promotion occurred. It
is untested rather than implicated, and is retained.

> **Caveat on the pressing account.** `object_press_depth_mean_m` was added to
> confirm the mechanism and reads a flat ~1.4 mm — the object is not being pushed
> into the desk. The gate was active from step 0, so the metric cannot separate
> "pressing was never the cause" from "the gate removed it before it could be
> observed", and the later run reproduced the same anti-correlation *without*
> pressing. The four contact reversals are real; the mechanism behind them is
> not settled.

---

### 4.9 · The lift was an exploration problem, not a reward problem

Four reward interventions had each delayed the lift collapse without stopping
it. This arc stopped changing the reward and measured the plant instead.

**The leading hypothesis was wrong, and the metric behind it was measuring
something else.** The suspicion was that the grasp detector's 8 mm relative-pose
bound penalises the acceleration a lift requires — slip and lift had moved
together across every run. They do, but `relative_position_slip_mean_m` is not
gated on contact. It averages over every active grasp-eligible step, and most of
those are free-space approach steps where the "slip" is just how fast the gripper
is closing on the object. It falls whenever the policy moves less, which is the
behaviour under investigation rather than evidence about it. Conditioned on the
pads being loaded, on MJWarp, slip peaks at **5.74 mm against the 8 mm bound**
across every arm of the probe, and is *lowest* during the fastest lift — a held
object translates with the gripper, so the relative pose stops changing.
`pose_reject_rate_while_loaded` reads 0.013–0.024 in training. The detector never
rejects a loaded grasp.

**What the probe found instead: the loaded z axis is dead-zoned.** From an
identical latched grasp under production MJWarp, driving a sustained commanded
`a_z` (`tools/audit/lift_barrier_probe.py`, 15 episodes per arm):

| sustained `a_z` | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.60 |
|---|---:|---:|---:|---:|---:|---:|
| median lift (mm) | 3.1 | 3.1 | 33.8 | 82.7 | 134.8 | 238.6 |
| success / latched | 0/15 | 0/15 | 0/15 | 14/15 | 15/15 | 15/15 |

A lift is only reachable by a large **sustained** bias. Per-step i.i.d. Gaussian
noise explores sustained bias with std `sigma/sqrt(N)` — 0.09 over a 13-step
window at the −1.10 `log_std` ceiling — so the lifting region sits 3.3 sigma out.
A driftless Gaussian at the run's own sigma reached **0/15** successes from a
perfect grasp, with no episode exceeding 38 mm.

That made `post_grasp_action_z_mean` the control variable, and it is now logged
every update. It immediately cross-validated: at the resume point the policy's
own post-grasp command was **+0.20** and its pre-grasped rise **42.5 mm**,
against the probe's 33.8 mm at a sustained 0.20. Two instruments agreeing. The
target became a single number — move it from +0.20 to +0.30.

**Two runs then failed, on an estimator bug rather than the idea.** A
per-episode, per-world offset `eps` added to the action mean and held for the
episode should make the GRPO group — eight candidates sharing a start — a
finite-difference probe along sustained-bias directions. It did not, because the
log-prob was scored against `mu + eps`. That is the *conditional* density given
`eps`, and its score `(a - mu - eps)/sigma^2` is exactly the per-step noise,
independent of `eps` by construction. The gradient on `mu` learned nothing about
which offset paid. On synthetic data with advantage set equal to the offset
signal, that score reads **−0.015** where the correct form reads **+1.43**.

So the offset perturbed the rollouts and informed nothing: half the pre-grasped
worlds drew a −z offset into the dead zone and failed, and the runs paid that
variance without ever collecting the term the idea rested on. An ungated offset
and a post-grasp-gated one collapsed indistinguishably, which is what finally
identified the estimator rather than the gating as the fault.

`eps` and the per-step noise are independent Gaussians, so the marginal is
exactly `N(mu, sigma^2 + s^2)`. Scoring against that is equally valid importance
sampling and puts `eps` back into `(a - mu)`. Records carry the offset *std* in
effect rather than the realised offset — the marginal needs only the width.

**The corrected run moved the number for the first time in the campaign.**

| | baseline `adaptivestd` | ungated (bug) | gated (bug) | **marginal (fixed)** |
|---|---:|---:|---:|---:|
| `post_grasp_action_z_mean` | — | −0.285 | −0.420 | **+0.320** |
| `success \| pre-grasped` | 0.797 | 0.101 | 0.092 | **0.828** |
| `post_grasp_rise` pre-grasped | 46.0 mm | 10.9 mm | 9.9 mm | 43.9 mm |
| `success \| normal start` | 0.218 | 0.057 | 0.053 | 0.166 |
| ever-grasped worlds | 0.231 | — | — | 0.187 |
| `physical_grasp_rate` (step-avg) | 0.331 | 0.536 | 0.538 | 0.217 |
| pad force | 4.27 N | 7.62 N | 7.95 N | 3.26 N |
| curriculum cap | 0.05 | 0.03 | 0.03 | 0.05 |

`post_grasp_action_z_mean` rose monotonically +0.238 → +0.318, crossing the
plant's +0.30 threshold, and `success | pre-grasped` reached **0.834** — a
campaign record, and held for 2.6M steps rather than peaking before a collapse.
Both figures are 250k-step window means, as everywhere else in this report; the
best single updates were +0.389 and 0.915.

**But the trade reversed rather than resolved.** Worlds that ever grasped fell
0.231 → 0.187 and the first grasp arrived 29.4 → 34.1 env steps in, with
`success | normal start` 0.218 → 0.166 — below the baseline. (`physical_grasp_rate`
reads a larger 0.331 → 0.217, but it averages over active steps and successful
episodes terminate early, spending proportionally fewer steps holding — 6.18 →
5.65 selected actions per candidate. The ever-grasped count is the honest
figure; the step-averaged one roughly doubles the apparent loss.) The campaign's invariant has always been "better at grasping,
worse at lifting"; this run is the same invariant with the sign flipped. One
residual has to emit sustained −z for the descent and sustained +z for the lift,
and it still cannot hold both. Deterministic validation is unchanged at
0.000–0.002.

**A methodological note that cost two runs.** Neither offset run had a control,
on the argument that the `adaptivestd` plateau would serve as one. It does not —
that was continuous training, not a resume — and reading its logs afterwards
showed it was *flat across its whole 2.34M steps* (`success | pre-grasped`
0.820 → 0.802, rise 47.4 → 46.1 mm, grasp 0.339 → 0.327, pad force 4.28 → 4.15 N).
Nothing was decaying until the offset was added, so both collapses were caused,
not inherited. Two intermediate diagnoses — "the offset perturbs the approach and
GRPO cancels it", then "the offset is inert and this is the canonical decay" —
were both wrong, and both would have been caught by running the control first.

**Observability was ruled out along the way.** Whether the residual can even see
that it is holding the object was tested directly
(`tools/audit/grasp_feature_probe.py`), with hard negatives — episodes driven to
close on air at grasp height — and a between-episode label shuffle as the null,
since grasp state is near-constant within an episode and a step-level shuffle
lets episode identity pass as signal. On the matched subset (fingers closed):

| feature | balanced accuracy | margin over control |
|---|---:|---:|
| proprioception (6-d) | 0.898 | +0.317 |
| frozen vision projection (512-d) | 0.682 | +0.221 |
| un-projected connector (30720-d) | 0.904 | +0.349 |

The signal is there twice over — `gripper_opening` alone carries it, because a
hit stops the fingers at the object's width while a miss closes them fully. The
residual could always tell. Separately, the 512-d fixed random projection is the
weakest of the three and is discarding a real amount of what the connector
encodes; that is worth fixing on its own terms, but it is not what blocks the
lift.

### 4.10 · Splitting the credit — both phases held, and what that exposed

`split_credit_at_grasp` gives the approach the dense reward at the moment the
grasp latched and leaves the lift with the terminal reward, so neither segment's
gradient carries the other's outcome. Resumed from `offset_marginal`, 2.49M steps.

| tail-quarter mean | baseline | `offset_marginal` | **`split_credit`** |
|---|---:|---:|---:|
| `post_grasp_action_z_mean` | — | +0.316 | **+0.325** |
| `success \| pre-grasped` | 0.793 | 0.831 | **0.832** |
| ever-grasped worlds | 0.234 | 0.198 | **0.226** |
| `success \| normal start` | 0.202 | 0.176 | **0.206** |
| `candidate_reward_mean` | 4.111 | 4.000 | **4.133** |

**It did what it was designed to do.** The approach recovered — ever-grasped
0.198 → 0.226 and normal-start success 0.176 → 0.206 — *without* giving the lift
back: `post_grasp_action_z_mean` held at +0.325, above the plant's +0.30
threshold, for the whole run. This is the first time in the campaign that both
phases hold at once, and it is a stable plateau from ~6.2M rather than a peak
before a collapse.

**And it changed nothing about the task.** Against the true baseline the full-task
numbers are a wash: normal-start success 0.206 against 0.202, ever-grasped 0.226
against 0.234, reward 4.133 against 4.111. Three interventions and ~8M steps
bought a correct mechanism and no capability.

**What that finally exposes.** Deterministic validation has read 0.000–0.012 for
the entire campaign and was treated as a curiosity. The metric next to it is the
explanation. `validation/final_xy_distance_mean_m` — misnamed, it records
`dense_target_distance`, which for pick-up is the **3-D** EE→grasp-point distance
— reads **0.39–0.42 m** at the end of every validation episode, in all three
runs, uniformly across objects:

| object | apple | tomato | orange | potato | mug | banana |
|---|---:|---:|---:|---:|---:|---:|
| final distance (mm) | 403 | 433 | 340 | 429 | 336 | 416 |

The controller workspace is `z ∈ [0.18, 0.60]` and grasp points sit at
0.19–0.21 m, so ceiling-minus-grasp-point is **0.405 m**. Validation starts are
capped at the same 5 cm as training and validation runs the full task
(`allow_prelifted=False`), so the deterministic policy is not failing to close a
gap — it is opening one, and ending pinned near the top of the workspace.

That is consistent with everything the campaign has measured and never
assembled: `policy_target_cosine_mean` 0.09 and `residual_target_cosine_mean`
0.03–0.06 (the action is near-orthogonal to the direction to the object); the
frozen prior's documented +Z bias; the cap frozen at 5 cm since 171k, because at
5 cm the σ = 0.333 exploration noise can stumble onto the object and at anything
wider it cannot; and sampled success ~0.21 against deterministic ~0.001.

**The mean policy has never learned to servo to the object. Every success in this
campaign is exploration noise finding it from a start already 5 cm away.** The
lift work in §4.9 is real and the credit split is real, but they improved the
behaviour *after* the object is reached, which is not the part that was missing.

This is an inference from the distance metric and the geometry, not a direct
measurement: nothing logs the terminal end-effector height in validation. One
line adding `ee_z` to the validation diagnostics would settle it, and should be
the next thing run, before any further training.

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
| The pre-grasped stage is a scaffold that can be annealed away once its success is high | annealing 0.50 → 0.20 took pre-grasped success 0.753 → 0.410 **and** normal-start 0.245 → 0.098; restoring it recovered only to 0.55 |
| The grasp detector's 8 mm pose bound penalises the acceleration a lift needs | slip **conditioned on loaded pads** peaks at 5.74 mm against the bound on MJWarp, and is lowest during the fastest lift; `pose_reject_rate_while_loaded` 0.013–0.024 in training. The unconditioned slip mean that suggested this is dominated by free-space approach steps |
| The residual cannot tell whether it is holding the object, so its post-close z is an average over both cases | linear probe on matched negatives: proprioception alone decodes it at 0.898 (margin +0.317 over a between-episode shuffle), the un-projected connector at 0.904. `gripper_opening` carries it — a hit stops the fingers at the object's width |
| The lift is blocked by the reward | the reward pays 1.58 more for a 60 mm lift than for holding still, and nothing about the reward changed in `offset_marginal`, which raised the lift. The plant is dead-zoned and the estimator was not proposing sustained lifts to be paid for |

The degenerate-group line is worth keeping: `torch_group_advantages` divides by
the group std floored at `1e-6`, and the informative-group filter is
`reward_span > 1e-6`, which never fires — `informative_groups` has equalled
`groups_collected` on every update of every run. That *is* a real amplification
path. It is simply not this bug, at ~7% of groups.

---

## 6. Open

**~~`log_std_mean` is exactly −1.19375~~ — RESOLVED, see §1.** The parameter was
saturated on both clamp bounds, not frozen; `clamp()` has zero gradient outside
its range so every trained entry had been dead since the first time it left the
band. Fixed in `80f6c35`, which arrested the decay that had ended every previous
run. `log_std_saturated_fraction` now makes the state visible.

**~~The grasp detector's pose bound blocks the lift~~ — FALSIFIED, see §4.9.**
Conditioned on loaded pads, slip never exceeds 5.74 mm against an 8 mm bound and
is lowest during the fastest lift. `pose_reject_rate_while_loaded` and
`slip_mean_while_loaded_m` replace the unconditioned slip mean that suggested it.

**~~The lift cannot be learned~~ — RESOLVED, see §4.9.** The loaded z axis is
dead-zoned below `a_z ≈ 0.2`; per-step noise explores sustained bias with std
`sigma/sqrt(N)`, so the lifting region sat 3.3 sigma out. A per-episode offset
scored against the *marginal* raised `post_grasp_action_z_mean` +0.24 → +0.32 and
took `success | pre-grasped` to 0.83, held.

**The approach is the live bottleneck, and now it is also the price.**
`success | normal start` 0.17–0.22 and the gate EMA 0.17–0.24 against a 0.30
promote threshold, unmoved for ~14M steps across four runs, so the cap has sat at
0.05 since 171k. `offset_marginal` made it worse rather than better — ever-grasped
worlds 0.231 → 0.187 while the lift was being fixed. Raising the promote gate to 0.40
was considered and rejected — the EMA has peaked at 0.302 in the entire campaign,
so a 0.40 gate would freeze the cap at 3 cm permanently.

**~~One residual cannot hold both phases~~ — ADDRESSED, see §4.10.**
`split_credit_at_grasp` (`a38dfb7`) recovered ever-grasped worlds 0.198 → 0.226
while `post_grasp_action_z_mean` held at +0.325. Both phases coexist and the
state is stable. It bought no capability: the full-task numbers are level with
the baseline.

**The policy does not servo to the object — the live bottleneck, and the only
one that matters.** Deterministic validation ends 0.39–0.42 m from the grasp
point against a ceiling-minus-grasp-point of 0.405 m, uniformly across six
objects and identically in all three recent runs; `policy_target_cosine_mean` is
0.09 and `residual_target_cosine_mean` 0.03–0.06. Sampled success ~0.21 comes
from starts capped at 5 cm plus σ = 0.333 noise, which is also why the cap has
never promoted past 0.05 m — at anything wider, noise cannot find the object.
Everything else in this report improves what happens *after* the object is
reached. **Confirm it first** (§4.10: log the terminal end-effector height in
validation, one line) before spending another run on anything else.

**The 512-d vision projection is lossy.** The frozen random projection the
residual is fed decodes grasp state at 0.682 where the un-projected connector
manages 0.904. Independent of the lift question, and untested as a change.

**~~The train/validation gap~~ — EXPLAINED, see §4.10.** Training normal-start
success ~0.21 against deterministic 0.001 is not a generalization gap. The
deterministic policy drives away from the object and parks at the workspace
ceiling; the sampled policy stumbles onto it from 5 cm. Same task, same starts —
the difference is entirely the exploration noise.

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

Two audit tools were added during §4.9 and are the checks to run before
attributing a lift failure to the reward:

```bash
MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab python \
  tools/audit/lift_barrier_probe.py --physics mjlab_mjwarp --episodes 16
MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab python \
  tools/audit/grasp_feature_probe.py --physics mjlab_mjwarp --episodes 60
```

The first sweeps sustained `a_z` from a real latched grasp and reports slip
conditioned on loaded pads; the second linear-probes the frozen features for
grasp state against hard negatives. Both run without a checkpoint. Note that
`grasp_feature_probe` writes `features.npz` before probing, so `--from-features`
re-scores a capture without re-running SmolVLA.

The oracle harness is the check to run after touching reset shaping, the reward
ladder or the grasp geometry. Expect 3/3 at reward ~7.2 on the reference
catalogs (`robocasa_apple robocasa_tomato robocasa_orange robocasa_potato`); the
config's full pool includes a banana the scripted oracle cannot grip, so it
reports 2/3 there. Compare against a same-invocation baseline, never against a
number quoted in a config comment.

Phase 1 commits, in order:

| commit | change |
|---|---|
| `e787c80` | grasp bonus becomes a ratchet on `ever_grasped` |
| `894a516` | oracle overlay follows that ratchet (harness could not run) |
| `d610dcc` | pre-grasped start stage |
| `d4d693c` | remote runner for the oracle harness |
| `62e647b` | oracle harness gains the MJWarp physics path |
| `857f3ea` | outcome and advantage divisor split by start stage |
| `cd2f91d` | pre-grasped starts kept out of the approach gate |
| `d3fa5d6` | residual telemetry; `min_log_std` floor |
| `ffe974d` | **peak-lift ratchet** — lift credit survives a settle |
| `03f9f20` | **lift-gated grasp bonus** — no credit for a grasp that cannot lift |
| `80f6c35` | **`log_std` projection** — clamp bounds stop killing the gradient |
| `4d4eb06` | `min_log_std` widened to −2.5 now that it can bind |
| `0a9bbe4` | pre-grasped fraction curriculum; explicit cap ladder |
| `e3957ae` | pre-grasped fraction curriculum reverted — the stage is load-bearing |
| `5564139` | **plant probe** — the 8 mm pose bound is not what blocks the lift |
| `7e973cf` | per-episode exploration offset; contact-conditioned slip and `post_grasp_action_z_mean` telemetry |
| `f8465c1` | grasp-feature probe — can the residual see that it is holding? |
| `8a89aa6` | oracle harness: `--force-renderer` separates frames from video files |
| `f9dc94f` | offset on for z, whole episode + pick-up resume launcher — **failed** |
| `1e4a401` | offset gated on holding the object — **failed identically** |
| `45c0591` | offset back to 0 after both failures |
| `a4bf902` | **marginal log-prob** — the offset was invisible to the gradient |
| `88b0b30` | offset on again, now that it reaches the gradient |
| `a38dfb7` | **separate returns for the approach and the lift**, split at the latch |
| `97c2e05` | report: honest ever-grasped figure in place of the step-averaged one |

Phase 0 fixes referenced: `c76cbb1`, `17a83f7`, `7d99c3e`, `64c1437`.
