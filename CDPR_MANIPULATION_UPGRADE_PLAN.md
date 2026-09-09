# Improving pick-up and composed manipulation after the 3M continuation

2026-09-08; updated 2026-09-09. Status: comparison utility implemented and CPU-tested; training
architecture below is a proposed experiment, not an implemented trainer or a
measured improvement. The remote continuation has now completed at 3,540,208
steps; its final checkpoint still needs separate evaluation. Full-log results
and qualifications are recorded in §14 of the consolidated report.

## Compare final with the two placement peaks

After training exits, run from `/root/repo/RL_VLA_Bootstrapping`:

```bash
git pull --ff-only
conda run --no-capture-output -n cdpr-mjlab python3 \
  tools/audit/compare_release_recovery_checkpoints.py --plate-step 3416645
```

The helper resolves the run only when exactly one
`runs/release_recovery_continue_3m_*` directory exists. Otherwise supply
`--run-dir runs/<exact-continuation-directory>`. It selects the greatest
numeric step and requires it to be at least 3,527,307, then compares it with
step **1,505,251** (original plate peak) and **2,117,145** (bowl peak) by
default. The command above selects the newly observed **3,416,645** plate
tie instead: plate remains 0.6447, pick-up is 0.2100 versus 0.1050 at the
earlier tie, and bowl is lower (0.3220 versus 0.3939). This is a transparent
retention tradeoff, not a claim of dominance. Both peak choices can be
overridden with `--plate-step` / `--bowl-step`. It does not monitor completion;
invoke it once the training process has finished and released both GPUs.

Each checkpoint gets all four instructions, 3×512 worlds, groups of 8,
rounds 0–2, torch seed 0, and the fixed pilot configuration. There is no scalar
distance-cap or horizon override. Arms run sequentially, each sharded over
the two GPUs with the same round assignment. This reproduces the pilot's
development recording protocol, not the in-training validation set.

Outputs are under the continuation's `eval/final_vs_peaks_<timestamp>/`:
`comparison.md`, `comparison.json`, checkpoint/config SHA-256 manifest,
per-arm recordings and logs, and per-arm container failure decompositions.
Existing output directories are never overwritten. Scene identity mismatches
suppress episode-wise verdict comparisons and produce exit status 2; rates
remain available descriptively. The recorded scene check cannot certify every
hidden simulator/controller state. There is no automatic checkpoint promotion.

Use `--dry-run` to inspect the selected files and commands without GPU work.
Use `--round-index 100 --rounds 12` for a larger, different reset-seed range,
provided it has not already been used. Seed-range novelty must be tracked by
the experimenter. These settings compare the same three candidates again;
final confirmation should evaluate the chosen shared checkpoint on an
untouched set. Candidates in the same reset group are correlated.

## Recommended substantial upgrade: asymmetric actor–critic plus skill curriculum

The present GRPO implementation gives a trajectory's terminal outcome to its
action records and filters groups without outcome variation. Its clipped
policy loss and `ppo_epochs` parameter do not make it a value-based PPO+GAE
trainer. About half of collected groups are informative, but this says little
about which instruction and phase receive useful gradients.

My preferred substantial next branch is **PPO with a simulator-state critic,
decision-level returns/GAE, and a curriculum over physically valid manipulation
stages**. Keep one shared actor for all instructions. This is a recommendation
for this campaign, not a published guarantee for CDPR or a drop-in YAML option.

The critic learns how promising a state is and makes credit depend on the
decision's effect on future completion. During training it can see target and
receptacle pose, velocities, contacts, controller state, task identity, grasp
and release history, and remaining budget. Keep new simulator-only fields out
of the actor; audit the existing actor observation contract rather than
claiming the present residual is automatically suitable for real deployment.
The actor retains its current visual/proprioceptive inputs and SmolVLA prior.

This design is motivated by [asymmetric actor–critic for image-based robot
learning](https://arxiv.org/abs/1710.06542), which trains the critic from full
simulator state while the actor uses partial observations, and
[GAE](https://arxiv.org/abs/1506.02438), which uses learned values to construct
time-local advantage estimates. Neither establishes a success rate for this
robot. [ResiP](https://residual-assembly.github.io/) provides a related example
of PPO-trained residual corrections over a frozen chunked policy; its precise
assembly results are on different tasks, and its state-teacher/visual-student
experiments retain a performance gap.

### What the curriculum must teach

| Instruction | Physically valid training progression | Acceptance evaluation |
|---|---|---|
| pick_up | Real desk-level grasp → close but uncaught start → normal uncaught approach | Normal desk starts; grasp and full 5 cm lift |
| put_into_plate | Held above destination → short carry → uncaught grasp/carry/release | Existing composed start distribution, 40 decisions |
| put_into_bowl | Held and aligned near bowl → short carry → longer carry → uncaught composition | Existing composed distribution plus explicitly labelled easier fixed-carry evaluation |
| move_to_object | Retain existing fixed-cap scenes throughout joint training | 0.08 m cap until confirmation |

For initial exploration, a proposed mixture is 50% ordinary acceptance-like
starts and 50% valid states around the current bottleneck within each
manipulation family. These are initial experimental settings, not tuned
values. Keep a per-family success gate for assisted starts and separately for
ordinary starts. Reduce assistance only when the latter improves; never
advance from success on pre-lifted or already-in-radius starts alone.

Use successful policy prefixes to obtain realistic intermediate states where
possible. A prefix supplies a reset state, then the current policy generates
new on-policy actions. It does not supply stale actions labelled as on-policy
GRPO/PPO experience. Restoring qpos alone is insufficient: include qvel,
controller integrators/targets, grasp/contact persistence, task latches and
object dynamics; re-render observations and regenerate the SmolVLA prior.
For pick-up, retain a desk-referenced success baseline and ensure the restart
has not already satisfied the lift predicate. Existing synthetic pre-lifted
starts are not evidence that this transition has been learned.

The general curriculum principle has precedent in [reverse curriculum
generation](https://proceedings.mlr.press/v78/florensa17a.html), but this
proposal targets grasp/lift/carry/release states rather than merely re-enabling
the repository's old scalar reverse-distance curriculum.

### Give lift exploration temporal coherence

Pick-up's final-window training grasp rate is approximately 52%, with about 34%
completion conditional on ever grasping. Lift is a major bottleneck, but
perfect lift alone at that grasp frequency cannot produce >70% overall.
For example, 90% grasp × 85% completion conditional on grasp would give 76.5%.
Those are stage targets, not forecasts or validation measurements.

A bounded, pick-up-only z exploration process activated around an actual
stable grasp is the cheapest near-term intervention. Sustain a sampled
direction across several decisions so exploration can cross the measured
loaded-axis dead zone. Treat the observed executed commands around +0.3–0.4
as probe evidence, not a Gaussian offset standard deviation: tanh, the VLA
prior, and the residual mean change that conversion. Preserve grip closure
during this exploration and evaluate unassisted deterministic execution.

The current `episode_offset_after_grasp` switch is global and history-based;
turning it on with z noise would also alter container transport and release.
A task/phase-scoped implementation must make sampled actions, stored behavior
log probabilities, and optimization ratios agree. A correlated latent offset
requires consistent augmented-policy scoring or an explicitly derived
conditional distribution; a marginal Gaussian assertion is not sufficient.
Do not add a deterministic upward override and call the resulting success a
learned pick-up improvement.

### Rewards, perception, and retention

Start the actor–critic branch with sparse full success plus the valid-state
curriculum. If lift learning still stalls, test bounded progress shaping as a
separate arm. Compute true returns from per-transition rewards and handle
termination versus time-limit truncation correctly. For a chunk of executed
actions, aggregate rewards and discounts consistently at the decision boundary.
The present collector overwrites candidate reward at each active step;
inserting dense terms into that path does not implement accumulated return.

A grasp bonus must not make desk contact a successful policy. Repeated opening,
closing, lifting, and dropping must not farm milestone bonuses. Use explicit
history, finite auxiliary reward budgets, terminal failure handling, and
eventual sparse-only acceptance. Potential-based shaping has terminal and
discounting conditions; it is not an automatic source of useful within-group
terminal-return ranking.

Bowl also needs localization and carry control. Its 5.7 cm radius is comparable
to a previously measured visual-encoder error scale; that suggests a perception
test, not proof of the cause. Compare an otherwise matched diagnostic actor
given target/receptacle state with the visual actor. If only the privileged
actor improves, train and validate an object/receptacle localization head or
better visual representation. A privileged critic cannot give the actor
information missing from its observations. Report any privileged-actor result
as a diagnostic, not achievement of the visual-policy target.

Changing the container approach cap does not change the uncaught branch's
object-to-receptacle carry distance. A carry curriculum must modify
`placement_grasp_object_min_distance` and `placement_grasp_object_max_distance`,
with per-family controls if plate and bowl advance separately. Verify realized
distances and valid contact geometry after settling. Easier bowl results must
be labelled as a different fixed-carry benchmark; preserve the original as a
transfer test. Do not enlarge the bowl success radius to claim improvement.

Initialize the new actor from the selected shared checkpoint and first verify
that its outputs match before training. Initially freeze that checkpoint's
VLA/LoRA weights and train its residual plus a newly initialized critic, making
optimizer initialization explicit. Retain all four instructions in collection
and use per-family mean losses so episode length and success filtering do not
silently determine training weight. Log gradient-contributing records per
family and stage. Freezing SmolVLA alone does not prevent residual forgetting.
Optional anchor regularization requires its own measured retention/gain tradeoff.
Do not rerun the historical residual-SFT recipe that erased composition.

## Compute and decision schedule

1. Finish the current run and run the three-arm comparison above. Use the
   resulting pick-up and placement outcomes to select one shared starting
   checkpoint, with explicit per-family retention requirements.
2. Run a bounded mechanism test of scoped lift exploration against an unchanged
   joint-GRPO control from that checkpoint. Check actual lift success and grasp
   retention. A 0.5–1M selected-action block is a learning check, not a promised
   convergence budget.
3. Implement and validate the actor–critic path as a separate trainer branch.
   Run sparse+curriculum first, then an auxiliary-reward ablation if needed.
   Use multi-million-action budgets in roughly 1M reporting blocks, retain
   checkpoints every 100k–250k actions, and compare equal sampled interaction
   counts and GPU hours as well as optimizer updates. Removing GRPO's 8-way
   candidate selection changes the meaning of a selected-action counter.
4. Treat 5–10M additional actions as a possible campaign allocation, contingent
   on measured full-task gains and retention across blocks. Do not terminate
   useful learning merely because it outgrows a pilot; do not assume an
   unchanged plateau will resolve solely with a larger counter.
5. Confirm one shared checkpoint above 70% on all four at the stated easier
   fixed settings, with scene-group uncertainty. Only then expand distances.

The comparison helper is ready now. The actor–critic, scoped exploration, and
stage-reset implementation are the next engineering work, not commands that
the current trainer already supports.
