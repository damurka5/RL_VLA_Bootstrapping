# Four instructions above 70%: next campaign

Date: 2026-09-07. Source review: `26cddf6`, consolidated report §§1, 4, 7,
10 and the current resetter, collector, and Phase 7 config.

Status: pilot completed on the user's remote 2×A40 host; results supplied on
2026-09-08. The additional 3M-action continuation is running. Its uploaded
TensorBoard snapshot reaches 2,981,624 cumulative selected actions, with 97
continuation updates and 24 validation checkpoints. See
`CDPR_MANIPULATION_UPGRADE_PLAN.md` for the final-versus-two-peaks comparison
command and the proposed actor–critic/skill-curriculum upgrade. Later training
interventions remain proposed, not measured improvements.

Latest matched baseline → final results from
`release_recovery_pilot_20260907_193019`: move-to **279/400 → 285/400**
(0.6975 → 0.7125 at 0.08 m), pick-up **49/328 → 65/328** (0.1494 → 0.1982
at 0.06 m), composed plate **281/456 → 316/456** (0.6162 → 0.6930), bowl
**107/352 → 108/352** (0.3040 → 0.3068). See the 2026-09-08 entry in the
consolidated report for protocol, provenance and limits. Keep the final
checkpoint as a candidate. Its identity and learning trajectory have now been
supplied; confirmation remains pending. Do not increase difficulty on this
one evaluation.

The pilot ended at **527,307 selected actions / update 19**. Candidate:
`runs/release_recovery_pilot_20260907_193019/rl/step_0527307/smolvla_grpo_adapter.pt`,
SHA-256 `ee33b11d9a1a17c64ebe7251b614edb1706dbe69f83c793196a997f4596d329e`.
Its in-run validation trajectory is now recorded in §14 of the consolidated
report: bowl rose 0.2917 → 0.3295, plate recovered after an early dip, and
pick-up ended at 0.0950. The trajectory is mixed, not evidence that every
family improves monotonically or that the run has converged.

**User steering, 2026-09-08:** multi-million-step training is welcome; do not
keep the campaign at short pilot budgets. The running continuation is a **full
resume for 3,000,000 additional selected actions**, to cumulative 3,527,307,
with identical fixed task settings and validation/checkpoints every 100,000
actions. Use the existing configurable full-resume launcher
`scripts/train_cdpr_smolvla_pick_up_grpo_mjlab_dual_remote_resume.sh` with the
four-family pilot `CONFIG`, despite that launcher's historical filename.
The command builder was checked: it emits `--resume-checkpoint`, all four
instructions and the absolute stop target. Do not rerun the weights-only
pilot wrapper to continue this lineage. The full resume restores residual
and LoRA optimizer states, curriculum state and global step. Correction from
the continuation logs: the displayed update index restarts at 1 because it is
initialized from the disabled reverse curriculum's counter; it does not
resume at 20. This is not a claim of bit-identical RNG/simulator continuation.

## Objective and evaluation contract

The user selected: first exceed 70% at easier **fixed** difficulty, then expand
distances. Keep one shared checkpoint for all four instructions. Report each
instruction separately; a mean above 70% does not meet the objective.

- Initial candidate caps: move-to 0.08 m, pick-up 0.06 m. These are starting
  settings to measure, not claims that either is already easy enough.
- Pick-up acceptance starts uncaught on the desk, with no aligned/pre-grasped
  assistance. Keep the 5 cm lift predicate.
- Containers start uncaught on the desk; keep the existing radii, release,
  grasp-history, and settling requirements. Keep the current 40-decision
  composed budget for the first comparison.
- Do not choose an easier container spawn range until its reset geometry has
  been checked. The current default is 0.06–0.10 m object-to-receptacle distance.
- Record actual initial distances and poses, target catalogs, reset seed/group
  identities, caught fraction, per-instruction horizons, checkpoint checksum,
  resolved configuration and code revision. Fix them within each comparison.
- Use separate development and untouched confirmation seeds. Cluster uncertainty
  by reset scene; repeated candidates and repeated identical seeds are not new
  independent scenes. Reconfirm the same checkpoint on all four families.
- A point estimate above 0.70 is a milestone. A stronger final claim requires
  uncertainty bounds above 0.70 on all four, with simultaneous coverage if
  claiming joint confidence. Do not repeatedly select checkpoints on the final
  confirmation set.

## A crucial distinction in the reset code

`mjwarp_rank_local_collector.py` creates an uncaught container start by placing
the object 6–10 cm from the receptacle and putting the gripper directly above
that object, 1 cm above grasp height. `placement_grasp_object_min_distance` and
`placement_grasp_object_max_distance` control **carry distance**, not the
approach distance. The ordinary `--start-distance-cap` does not control this
branch. Workspace clipping and subsequent physical displacement can also make
the realized distance differ from the requested one.

Consequences:

1. Current composition is aligned grasp → lift → carry → release. It does not
   yet establish arbitrary-start approach → grasp → place.
2. The composed grasp rate cannot be directly transferred to ordinary pick-up
   resets. The composed prefixes remain potentially useful lift examples.
3. Shrinking 6–10 cm can put an object into contact with a receptacle. First
   inspect displacement, clearance and failures by catalog/location. A useful
   curriculum must simplify valid scenes, not create interpenetrating ones.
4. Eventually broadening container approach requires an explicit gripper-start
   offset curriculum, separately from object-to-receptacle carry distance.

## What the existing results imply

| Family | Reported reference | Next issue to resolve |
|---|---|---|
| move-to | 0.5775 at cap 0.14 | Measure easier fixed cap; separate wrong-object selection from final localization and insufficient travel |
| pick-up | 0.1465 at 0.06; 0.0488 at 0.10 | Recover lift without sacrificing descent, then raise grasp reliability |
| plate | 0.6272 on corrected composed protocol | Recompute the corrected composed funnel; test budget only if those episodes time out while still viable |
| bowl | 0.2841 on corrected composed protocol | Inspect reset validity and grasp failures, then carry/release accuracy |

At cap 0.10, perfect lift conditional on the current 0.4116 grasp rate would
only yield 0.4116 pick-up success. The same arithmetic applies to bowl's
previously measured 0.6051 grasp rate: perfect downstream behavior still falls
short. Recompute these conditional rates under the corrected protocol before
using them as current bounds. A practical example budget is 0.85 grasp × 0.90
completion given grasp = 0.765; this is a design target, not a prediction.

The scripted oracle's 0.6752 bowl grasp and 0.5229 success are **not physical
upper bounds**. An imperfect scripted controller can be outperformed. Its
failures are useful evidence for inspecting common reset/control problems.

## Experiment order

### Corrected composed audit returned by the user

Source: console output from `comp_fixed_020_next_audit`, produced from the
corrected composed recordings. JSON/CSV remain on the remote host. This is a
decomposition of existing episodes, not a new evaluation or training result.

| Quantity | Plate | Bowl |
|---|---:|---:|
| Success | 286/456 = 0.6272 | 100/352 = 0.2841 |
| Ever grasped | 417/456 = 0.9145 | 221/352 = 0.6278 |
| Release given grasp | 318/417 = 0.7626 | 134/221 = 0.6063 |
| Settle given release | 307/318 = 0.9654 | 127/134 = 0.9478 |
| XY valid given settled | 286/307 = 0.9316 | 100/127 = 0.7874 |
| No grasp | 39 | 131 |
| No release | 99 | 87 |
| No-release episodes using full budget | 64 | 41 |
| No-release episodes ending held | 28 | 29 |
| No-release episodes ending unheld after grasp loss | 71 | 58 |
| Released but not settled | 11 | 7 |
| Settled XY miss | 21 | 27 |

All reported no-release budgets are 160 env steps (40 decisions). Timeout and
ending-held counts are overlapping classifications; the printed marginals do
not provide their intersection. Nor does ending unheld prove a damaging drop:
the object may already be resting correctly with the gripper insufficiently
open. Inspect the joint final-state table before interpreting the timeout count
as recoverable success.

Plate needs 34 additional successes to exceed 70% (320/456). Converting only
all 28 ending-held failures gives 314/456 = 0.6886, still below target. These
are arithmetic scenarios, not predicted improvements or horizon upper bounds.
Bowl still requires better grasping and better completion after grasp.

The spawn-distance column is measured AFTER the first recorded physics step,
not from a saved pre-physics layout. Its distance alone cannot establish clamp
versus contact displacement. In particular, bowl's above-0.10 m subset grasps
in 35/56 = 0.6250 cases, versus 186/296 = 0.6284 among the rest. Crossing that
boundary does not identify the main grasp failure population in this sample.
Keep the reset-interference hypothesis unconfirmed. The much shorter median
distance among grasped bowls motivates a distance-stratified table, with counts;
it does not by itself prove a causal effect of distance or identify a valid
easier reset distribution.

The user then returned that joint CSV summary:

| No-release timeout final state: held / XY-valid / below settle-height bar | Plate | Bowl |
|---|---:|---:|
| false / true / true | 36 | 11 |
| false / true / false | 0 | 1 |
| true / false / false | 3 | 3 |
| true / false / true | 0 | 15 |
| true / true / false | 6 | 5 |
| true / true / true | 19 | 6 |

Thus plate has 55 no-release timeouts within the XY radius and below the
settle-height bar (36 unheld, 19 held). These are promising release-completion
candidates, not guaranteed successes: the table does not check the remaining
Z predicate or establish that the policy will open. Plate has 28 ending-held
timeouts; bowl has 29. The term “dropped” hid 36 unheld plate timeouts with
favorable XY/height. Their actual future behavior remains unmeasured.

| Observed object-to-receptacle distance | Plate episodes / grasps / successes | Bowl episodes / grasps / successes |
|---|---|---|
| [0, 0.03) m | 128 / 122 / 92 | 120 / 114 / 84 |
| [0.03, 0.06) m | 48 / 40 / 32 | 24 / 1 / 0 |
| [0.06, 0.08) m | 184 / 175 / 101 | 8 / 8 / 7 |
| [0.08, 0.10) m | 80 / 72 / 61 | 144 / 63 / 3 |
| [0.10, infinity) m | 16 / 8 / 0 | 56 / 35 / 6 |

Bowl's <3 cm bucket supplies 84/100 successes, at 84/120 = 70% within that
bucket. Those target centres already begin inside the 5.7 cm XY success
radius; this does not establish whether the objects are physically seated
inside the bowl. At >=8 cm, bowl grasps 98/200 and succeeds 9/200; even among
grasps, completion is only 9/98. A grasp-only intervention cannot fix this
regime. The 7/8 result at 6–8 cm is too thin to select a curriculum from:
eight candidates can belong to one reset group. The 1/24 grasp result at
3–6 cm motivates inspection of rim/contact geometry, catalog and scene
identity, rather than a claim that this interval is inherently impossible.

Next remote step: one 48-decision deterministic recording at the existing
spawn settings, with success measured cumulatively by decisions 40, 44 and 48
on the same trajectories. `sil_record --horizon-decisions` overrides the
budget AFTER reset generation, so it does not itself change the sampled
starting geometry. It overrides ALL instruction budgets; use only container
results here. Non-container worlds can consequently remain active longer
than in the original evaluation, and shared rollout computation can alter
trajectories. Treat the within-run completion curve as a diagnostic, not a
perfect counterfactual replay of the old 40-decision run or an improvement
in the unchanged policy. Any adopted 48-decision budget needs a new protocol
label and confirmation. Do not raise the budget repeatedly if the extra
window adds few successes; inspect final gripper commands instead.

The user returned the 48-decision probe:

| Completion time | Plate (456 episodes) | Bowl (352 episodes) |
|---|---:|---:|
| By decision 40 | 282 = 0.6184 | 98 = 0.2784 |
| By decision 44 | 285 = 0.6250 | 101 = 0.2869 |
| By decision 48 | 293 = 0.6425 | 102 = 0.2898 |
| Additional completions in decisions 41–48 | 11 (+0.0241) | 4 (+0.0114) |

This is a within-trajectory budget measurement on the same checkpoint, not
learning. A 20% increase in the maximum decision budget buys a modest gain;
the actual compute increase is not measured. It does not demonstrate that
arbitrarily longer horizons cannot help, but supplies no reason to prioritize
another horizon sweep. Keep 40 decisions as the primary campaign comparison
budget while developing better behavior. The new run's 40-decision count is
close to the earlier 286/100 result, without requiring the runs to be identical.

Next: use the existing recordings to measure each remaining no-release
timeout's gripper-command mean, fraction of opening commands, measured opening
change and gap to its own release threshold over the final eight decisions.
Separate held/unheld cases whose final XY, Z and settle-height terms pass.
Closing/near-zero commands point toward learning release completion under
the corrected task; sustained positive commands with little opening movement
point toward controller/contact behavior. A mixed result requires preserving
that distinction. Do not weaken the release threshold or substitute an oracle
release into the final policy score.

The candidate learning path remains joint RL from `step_2017690`, evaluated
under the corrected task. This checkpoint was trained before the termination
repair, so its training did not reward some continuations now allowed by the
environment. That motivates testing learning under the repaired termination;
it does not by itself prove why an individual current episode fails to open.

The user returned the release-tail probe, restricted to no-release timeouts
whose final XY, Z and settle-height terms pass:

| Family / final grasp state | Episodes | Median mean gripper command | Median remaining opening gap | Median opening change in final 8 decisions |
|---|---:|---:|---:|---:|
| Bowl / unheld | 9 | -0.0798 | 0.4559 | -0.0769 |
| Bowl / held | 14 | -0.1759 | 0.4143 | -0.0042 |
| Plate / unheld | 26 | -0.0947 | 0.4806 | -0.0212 |
| Plate / held | 12 | -0.1003 | 0.4394 | -0.0925 |

These are per-episode final-window summaries. Positive commands request
opening. Every cohort's median command is negative; held plate's command p90
is also negative (-0.0419). This supports learning release completion rather
than treating these episodes as uniformly slow physical openings. It does not
rule out individual controller/contact failures, nor establish that the object
was geometrically ready throughout each final window. The remaining opening
gaps are substantial, not tiny threshold misses. Keep the release predicate.

### Learning pilot (completed; full-resume continuation planned)

`scripts/run_cdpr_release_recovery_pilot.sh` uses
`configs/examples/cdpr_smolvla_release_recovery_pilot.yaml`:

- Warm-start weights from `step_2017690`; fresh optimizer/curriculum. No SFT.
- All four instructions jointly; one-rung approach ladders pin move-to at
  0.08 m and pick-up at 0.06 m. One-rung support is now implemented and tested.
  Keep the curriculum enabled: disabling it means uncapped starts. Preserve
  the shared 0.34 m normalization used by horizon coupling.
- All container starts uncaught, aligned above desk objects as in the current
  composed protocol, with original 0.06–0.10 m object-to-receptacle sampling
  and 40 decisions. No geometry, success-radius, lift-height or release changes.
- Preserve current Phase 7 exploration `[0,0,0,0,0.15]`, including its existing
  behavior scoring. No global post-grasp gating or split-credit toggle. This
  inherited exploration differs from the historical zero-offset winning run;
  the pilot tests a practical successor recipe, not the isolated effect of one
  change. Pick-up-only z exploration remains a separate later arm.
- Stop at 500,000 selected environment actions across both ranks, at the next
  completed update. Validate and retain checkpoints every 100,000 actions
  (also at update boundaries). There is no claim this budget yields convergence.
- Before and after training, record all four families at the fixed caps with
  three rounds × 512 worlds, identical round indices and torch seed. No scalar
  `--start-distance-cap` override: it would erase the per-family settings.
- Write checkpoint/config hashes, config snapshot, launch revision, baseline
  and final recordings, and `pilot_comparison.json`. Preserve all intermediate
  checkpoints; the last checkpoint is evaluated, not automatically promoted.
- CPU curriculum/predicate checks run before the baseline. The existing Phase 7
  launcher retains its argument checks and GPU preflight before training.

The next decision uses per-family validation trajectories and the final
comparison. A meaningful plate gain with retained other skills supports more
learning under the repaired task; flat results motivate scoped release
exploration, and skill regression motivates balancing/protection. Do not
declare success from pooled reward or automatically extend the run. The
already-separated bowl grasp/carry failures remain relevant even if release
improves.

### 1. Audit the corrected composed recordings on CPU

Use `comp_fixed_020`, not the misleadingly named mixed-start `..._fixed2` arm.
Run the existing placement decomposition with zero predicate disagreements
allowed. Inspect grasp, release, settle, XY, timeouts, spawn displacement, and
object-to-receptacle distance among grasped versus never-grasped episodes.
Do not pool the three scene-repeated cap runs as independent demonstrations.

If bowl failures cluster around displaced or overlapping resets, reproduce a
few scenes with the simulator before changing resets. If resets are valid,
test grasp alignment/control and visual localization. Neither overlap nor an
encoder ceiling is established by the current report alone.

### 2. Recover pick-up's lift with sparse RL first

Warm-start from `step_2017690`, with the corrected termination code, and retain
all four families in training. Establish a fixed easy baseline before updates.
First compare ordinary exploration against **pick-up-only post-grasp sustained
z exploration**, leaving the container sampler as in the control arm.

This needs a scoped implementation, not a global YAML toggle. The existing
`episode_offset_after_grasp` gate uses first-grasp history and affects every
instruction; it is not a current-contact-only gate. Apply the intended
instruction/phase mask consistently to action sampling and behavior scoring.
The empirical loaded-plant response motivates exploring sustained commands
around 0.3–0.4, but action-space values are not interchangeable with the
pre-tanh offset standard deviation. Calibrate actual commands in a short
collection probe before choosing a training amplitude.

Require the collection probe to produce real 5 cm lifts and mixed-outcome
pick-up groups without materially reducing grasp frequency. Then require
noise-free evaluation to improve: success caused only by exploration is not
learned lift. Track post-grasp commands, object rise, grasp rate, lift given
grasp, usable groups and valid gradient records **per instruction**.

If short-start exploration still supplies no usable lift learning, test a
small training-only mixture of pre-grasped-on-desk starts, with the genuine
desk-height lift baseline. The existing prelifted reset path supports this;
normal-start evaluation must demonstrate transfer before expanding it.

Do not blindly enable `split_credit_at_grasp`: it uses the reward at the latch,
which is normally zero with success-only pick-up reward. It is not an automatic
grasp bonus, and can remove useful approach records.

### 3. Change reward/returns only if the exploration test fails

Instruction-specific rewards do not mathematically require separate training
runs. Groups share an instruction and normalize within the group. A joint
multi-task objective can therefore use different rewards, although shaping,
gradient scale, sampling and shared-parameter interference still need control.
The earlier dense-reward campaign is evidence about those implementations,
not proof that only one binary reward permits joint learning.

The collector overwrites `candidate_rewards` each active step: it currently
optimizes the **last active reward**, not a summed discounted return. Adding
dense increments alone therefore does not implement a progress-return method.

A bounded height-progress or milestone auxiliary objective is a possible
separate ablation, conditional on genuine grasp and desk-relative object lift.
It changes the optimization objective; require sparse final success to rise
and anneal the auxiliary term away. Avoid a per-step payment for merely holding.

Potential shaping `gamma*Phi(s_next)-Phi(s)` has policy-invariance conditions,
but is not a free solution here. It requires correct returns and terminal
handling. With complete trajectories, constant terminal potential and the same
reset within a group, its summed contribution telescopes to a group-constant
offset, producing no extra terminal-return GRPO ranking. A useful denser credit
scheme needs additional design, such as time-local advantages with a critic.
Reference: [Ng, Harada and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

### 4. Improve plate and bowl according to their new funnels

- Plate: if corrected composed failures still hold a viable placement at the
  budget boundary, compare 40 versus 48 decisions on frozen reset scenes. A
  changed budget is a separate protocol; test both checkpoints again at the
  chosen common budget. The report's recent timeout evidence was mixed-start.
- Bowl: resolve any reset defect first. If valid-scene grasp is still below
  70%, prioritize it before release-only work. If release-position error then
  dominates, compare learned versus privileged target/receptacle localization
  in a diagnostic arm to isolate perception from control. Privileged inputs
  are diagnostics, not acceptable hidden inputs to the final visual policy.
- Move-to: train on the fixed easier cap while separating target selection
  from close-range servoing. If wrong-object choice dominates, audit explicit
  instruction information reaching the residual before increasing MLP size.

### 5. Retain all families and then expand difficulty

Uniform scene sampling does not guarantee uniform optimization: zero-variance
group filtering and different episode lengths can change which instructions
actually contribute records. Log those contributions before setting quotas.
Use measured per-family contributions to guide bounded collection budgets and
family-balanced losses; never oversample an all-failure family indefinitely.
Dynamic resampling recovers usable groups but cannot invent successes; see
the [DAPO authors' description](https://dapo-sia.github.io/).

Continue joint RL. Joint training reduces the need for alternating family
turns, but shared weights can still interfere. Evaluate all four at every
checkpoint and retain a rollback checkpoint. Select using the weakest-family
result plus explicit per-family retention checks, not composed mean alone.

Promote difficulty only after repeated held-out >70% at the current fixed
setting; use a margin (provisionally 80%) before stepping farther and keep a
slice of the previous setting in training. Measure both rungs after promotion.
Gate pick-up progress on full success as well as grasp, rather than letting
the approach gate certify a lift it cannot see. Tune rung spacing from measured
drops; historical 0.45/0.30 gates do not enforce a 70% competence target.

## Self-imitation remains a later, bounded experiment

Do not repeat the residual retention-SFT recipe that erased composition.
Its failure does not prove every form of imitation is incompatible with RL.
The current policy's successful unsmoothed trajectories are a different
candidate teacher, but using them requires a controlled experiment with
retention evaluation. Relabelled composed prefixes are off-policy under the
pick-up prompt: recompute prompt-conditioned state/prior and do not insert
them into ordinary on-policy GRPO as if generated under that prompt.

A selective auxiliary imitation loss on such prefixes can be tested later
with protected evaluation of all four families. First try the sparse-RL lift
route, which directly addresses the measured exploration problem.
