# GRPO with demonstrations of the missing manipulation transitions

2026-09-09. **User decision: keep GRPO.** The actor–critic recommendation in
`CDPR_MANIPULATION_UPGRADE_PLAN.md` is superseded as an active campaign proposal.
No alternative RL algorithm is being implemented or scheduled.

**Adopted by the user after the z-offset pilot review, 2026-09-09.** This is
the active implementation plan. The extractor and an inference-only handoff
probe exist; GPU validation and GRPO training integration are still pending. Full evidence is recorded in
`CDPR_CONSOLIDATED_PROGRESS_REPORT.md` §§4.2, 7.15 and the latest §14 entry.

`release_recovery_pilot_20260909_130003` regressed pick-up **91/328 → 84/328**
and bowl **122/352 → 103/352**, while move-to improved **321/400 → 331/400**
and configured plate **276/456 → 283/456**. Retain it as a negative experiment,
without promoting its candidate. The z-offset config also gates the existing
gripper offset, and the gate remains active after grasp loss. This does not
isolate z exploration or establish the cause of regression. The lift probe
uses a different baseline/opening rule from production success; reconcile
those definitions before using its conditional rate as a success funnel.

The implementation order is:

1. Reconcile recorded pick-up success with rescoring under the production and
   first-post-action height references. Capture true pre-action reset poses
   for future comparisons; retain legacy scores under their original protocol.
2. Establish an easier fixed transport benchmark with the object outside the
   receptacle's success region. Enforce realized distances after workspace
   handling; keep existing success radii. §7.14 shows why legacy plate scores
   dominated by inside-goal starts cannot certify transport.
3. Build a training-only demonstration bank with actual lift and transport
   transitions, then implement and verify decision-boundary replay/restore
   and handoff. No teacher-prefix action enters residual or LoRA GRPO losses.
4. Compare ordinary GRPO with demonstration-start GRPO from the same learner
   checkpoint under the same new evaluation contract. Use an initial transfer
   check before multi-million-step continuation, maintaining all four families.

The user proposes pick-up demonstrations with the plate or bowl present so the
grasp-and-lift behavior can also support placement. This is useful, provided we
separate a placement prefix from a completed placement and preserve the real
transition into carry/release.

**New matched comparison received during implementation:** final scores
move-to 0.7800, pick-up 0.2744, plate 0.5943, bowl 0.3381; the bowl-peak arm
scores 0.7650, 0.2195, 0.7018, 0.3778. Keep final as a pick-up donor/control
and bowl-peak as the placement donor/retention reference. They offer a real
tradeoff, not one dominant checkpoint. Do not average or merge their weights
as a shortcut. Choose the single learner initialization when the handoff
experiment is configured and evaluate it against both retained references.

For bowl-peak, 119/219 bowl failures are no-grasp and 66/219 no-release;
19/219 are XY misses. Its 233/352 = 0.6619 grasp frequency is itself below
70%, so downstream-only improvement cannot reach the overall target on these
episodes. Plate's largest remaining failure group is no-release, 73/136.
The demonstrations need the full chain and the GRPO curriculum must walk
back to grasp acquisition. Ended-unheld does not by itself prove an object
fell mid-transport or identify where it ended.

## Shared trajectories, separate task completion

Prefer complete composed trajectories with the receptacle in the scene:

```
desk → approach → grasp → lift → carry → release → settle
                         └ pick_up ends once held lift reaches 5 cm
                                                 └ put_into ends on its own predicate
```

One source can provide a complete placement demonstration and a truncated
pick-up demonstration. A placement failure after a valid lift still supplies a
pick-up success. A successful pick-up with a receptacle visible supplies a
placement PREFIX; it does not supply placement success or the missing carry
and release. The presence of a bowl alone does not make a plate-directed
trajectory a bowl success. Do not pool relabelled views as independent scenes.

The best starting material is the current policy's composed recordings. The
earlier measured lift-prefix yield was about 0.60, versus pick-up's own 0.1465
at its cap. That is historical yield, not a guarantee for the latest checkpoint.
The final-versus-peaks evaluation will also produce suitable source recordings.
Use one selected shared checkpoint's recordings first; do not merge all
checkpoints on repeated seeds and call the result independent demonstrations.

**Evaluation separation:** inspect/extract the existing evaluation recordings
as a demonstration prototype, but collect the actual training bank on new
training-only scenes if retaining rounds 0–2 as held-out evaluation. If those
evaluation trajectories are used for imitation or demonstration-start training,
retire their scene seeds from held-out claims and establish new confirmation
seeds. Different prompts or different checkpoint rollouts on the same scene do
not make that scene new. Keep all views/candidates of a scene in one split.

Existing composed resets align the gripper near the object. Those trajectories
are good lift/carry/release material but do not demonstrate the ordinary
pick-up approach distribution. Later collection needs both aligned starts for
the bottleneck and normal uncaught pick-up starts with receptacles present.
Measure object-to-receptacle distance at the end of the lift: this addresses
the previous failure to connect free-scene grasps to the carry distribution.

## Ready now: extract the real action demonstrations

`tools/audit/extract_cdpr_transition_demonstrations.py` reads existing NPZ
recordings on CPU and exports:

- successful grasp-and-lift action clips under a correctly rewritten pick-up
  prompt, cut at the first production-predicate success;
- the full placement clip from the SAME source episode, when placement also
  succeeded;
- source hashes, shared episode IDs, grasp/lift landmarks and distances,
  original/derived task labels, and rejection reasons.

The extractor rescores pick-up with `evaluate_active_sparse_tasks`, using the
first recorded post-action pose as the composed lift reference and reporting
its discrepancy from the stored production datum. It rejects caught or
visibly raised starts, missing receptacles, unresolved object labels/prompt
disagreements and non-finite clips. Divergence is quarantined per world when
the recording identifies the affected worlds, otherwise per round. It keeps
valid lift prefixes even when placement failed. Complete
placements without a valid 5 cm held lift do not enter this shared-prefix bank;
their original recordings remain available for placement retention.

Run on one chosen checkpoint's recording directory; replace the path below
with the real directory printed by the comparison helper:

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only
RECORDING_DIR="runs/<continuation>/eval/<comparison>/<final-or-selected-peak>"
DEMO_DIR="runs/grpo_transition_demos_$(date +%Y%m%d_%H%M%S)"
conda run --no-capture-output -n cdpr-mjlab python3 \
  tools/audit/extract_cdpr_transition_demonstrations.py \
  --recordings "$RECORDING_DIR/record_*.npz" --output "$DEMO_DIR"
```

This extracts measured action clips; it does not create new simulated episodes
or train a policy. The output is intentionally not the legacy SIL dataset
schema: relabelled actions still need the correct images and current
prompt-conditioned prior before an imitation loss is meaningful.

The now-known placement prototype directory is
`runs/release_recovery_continue_3m_20260908_102004/eval/final_vs_peaks_20260909_083221_340502/bowl_peak`.
It has 453 successful placement episodes before the extractor's additional
lift/start/quality checks, not 453 independent scenes or guaranteed eligible
pick-up demonstrations. Failed placements can add valid lift prefixes.

The original comparison's exit status 2 is the helper's pairing gate, not
evidence that its three evaluations/decompositions failed. In these legacy
recordings the earliest stored object poses are after the first action;
differences can be caused by the different policies. Use the comparison
helper's `--inspect-existing <comparison-directory>` to inspect the actual
mismatching fields without rerunning GPU work. Full pre-action snapshot
capture is needed to certify exact reset identity for future recordings.
The corrected helper returns success for a completed job whose only mismatch
is post-action pose, while still withholding paired verdicts. Metadata
mismatches continue to return status 2.

If more data are needed, collect fresh policy rollouts with the existing
four-family pilot config, unchanged geometry/caps, and a new round range:

```bash
# CHECKPOINT must identify the checkpoint selected by the comparison.
HARVEST_DIR="runs/grpo_transition_harvest_$(date +%Y%m%d_%H%M%S)"
conda run --no-capture-output -n cdpr-mjlab python3 \
  tools/audit/sil_record.py --mode record --checkpoint "$CHECKPOINT" \
  --config configs/examples/cdpr_smolvla_release_recovery_pilot.yaml \
  --worlds 512 --group-size 8 --rounds 12 --round-index 100 \
  --devices cuda:0,cuda:1 --seed-torch 0 --output "$HARVEST_DIR"
```

Round 100 is an example fresh range, not a universal unused seed guarantee.
The extractor selects the container-source subset of that four-task harvest.
This collection remains policy self-imitation material. Scripted-oracle
demonstrations can fill uncovered transitions later, with separate provenance
and full-task predicate verification; do not automatically run the old Phase 6
harvest-plus-SFT script, which also trains and changes the policy.

## First use in GRPO: demonstration-guided starts

Keep the terminal success objective and the GRPO optimizer. Use a demonstrated
prefix to reach a useful physical state, then release control to the current
policy. The prefix is initialization; only the newly sampled continuation
contributes policy-gradient records.

For each group:

1. Select a source scene, task instruction and handoff boundary shared by all
   eight candidates. The handoff must precede that instruction's success.
2. Restore a complete validated state, or replay the source prefix and verify
   the resulting state. Re-render observations and reset/regenerate policy
   caches and prompt-conditioned priors as required.
3. Generate all eight continuations using the current policy and its actual
   behavior distribution. Compute the normal task success and GRPO advantages.
   Exclude all teacher-prefix actions from both residual and LoRA updates.
4. If outcomes are all failures, try a later, easier handoff. If they are all
   successes, move the handoff earlier. Cap probing cost rather than endlessly
   refilling a group with no useful outcome variation. Learn transitions where
   outcomes vary, then gradually return to ordinary desk starts.

Changing the initial-state curriculum does not require replacing GRPO.
The principle is motivated by [Backplay](https://arxiv.org/abs/1807.06919)
and [DemoStart](https://arxiv.org/abs/2409.06613). We are borrowing the use of
demonstrations to choose useful starts, not importing either paper's complete
RL algorithm or claiming their reported performance for CDPR.

Suggested initial experiment settings, to be measured rather than assumed:
keep all four instructions in training; within manipulation tasks allocate
50% ordinary starts and 50% transition starts. Pick-up progresses from
just-before-lift to before-grasp to the ordinary approach. Plate/bowl progress
from before-release to short carry to grasp/lift to ordinary composed starts.
Keep a separate per-family assisted-start score and ordinary-start score.
Reduce assistance based on ordinary-start transfer, not assisted success alone.

The latest z-offset pilot's reported grasp rates are 46.65% for pick-up and
59.09% for bowl. With those grasp outcomes fixed, improving downstream behavior
alone cannot yield 70% overall. The backwards progression must reach the approach/grasp phase.
Likewise, full container success requires grasp, carry, release and settling;
prefix demonstrations cannot replace training the rest of that sequence.

## Engineering required before training from these starts

**Prototype extraction received:** 293 lift prefixes (76 bowl / 217 plate),
156 complete placements, 137 prefixes from failed placements, and 192
desk-start rejections. See the newest report ledger entry for all denominators.
The manifest is `runs/grpo_demo_prototype_20260909_170619/manifest.json`.

**Runnable now: handoff verification, zero optimizer updates.**

```bash
cd /root/repo/RL_VLA_Bootstrapping && git pull --ff-only
bash scripts/run_cdpr_demo_handoff_probe.sh
```

The launcher defaults to that manifest and its pilot baseline, selects source
round 0, and runs pickup/placement independently on cuda:0/cuda:1. It resolves
the donor from `pilot_manifest.json` and checks source/config hashes; it does
not guess the latest checkpoint. `DRY_RUN=1` checks provenance and plans
landmarks on CPU. `SOURCE_ROUND=1` or `2` selects another recorded round;
`MAX_BOUNDARIES` and `MAX_GROUPS` default to 2 and 8 for each task. These are
limits on the audit, not training hyperparameters. Logs and `summary.json`
are written under a new `runs/demo_handoff_probe_<timestamp>/` directory.

Each batch uses one common prefix length, because masking actions does not
freeze MJWarp physics. A source group's selected live state and task/contact
history are broadcast to all eight candidates, then the existing collector
generates fresh suffixes. No optimizer is called; prefix actions precede all
residual and LoRA record capture. The default trace tolerances are 2 mm for
object/EE coordinates and 0.03 normalized opening, with exact grasp-history
agreement. Failures identify replay errors, lost grasp, contained divergence,
or already-terminal destination tasks; investigate rather than silently
loosening thresholds. Suffix budgets preserve the unused source horizon.
Report clean groups with reward variation separately from all-failure or
all-success groups. Assisted suffix success does not count toward the 70%
ordinary-start objective. The first probe is under legacy source geometry;
training-only scenes and the outside-goal transport benchmark still need work.

The existing NPZs have actions, observations/positions and task metadata, but
not complete qpos/qvel, object orientations, controller integrators/targets,
contact persistence, grasp/release history and simulator continuation state.
The exported clips are not restorable snapshots. Validate the new live
replay-and-handoff probe on MJWarp before connecting it to training. Match every
candidate's reset and freeze ordinary-start evaluation.

Handoff is at a policy-decision boundary; a grasp event inside a four-action
chunk is a landmark, not automatically an eligible reset. Preserve the actual
desk-height lift reference, original task latches and an explicit remaining
budget. Never count an already lifted pick-up reset as earned success. Mask
teacher actions in EVERY optimization path, including action-expert LoRA.
Recompute the prior under the current target instruction; an old placement
prior is not the current pick-up prior.

Replay can diverge. Verify physical grasp, lift baseline and scene identity at
handoff, measure replay survival, and quarantine invalid state reconstructions.
New full-state capture must also record non-finite reset events per episode;
aggregate counters in old files cannot prove every accepted clip was unaffected.
If frames are needed, the current replay frame filter keeps original successful
episodes; extend it to select the manifest's valid lift prefixes from failed
placements too. Merely passing `--record-frames` would omit those examples.

The current relabel helper only generates placement prompts from simpler
instructions; it does not implement validated placement-to-pick-up truncation.
The new extractor handles that direction explicitly. Do not feed old actions
or forced successful teacher candidates into ordinary on-policy GRPO groups.

## Optional later use: a small auxiliary imitation loss

The user permits demonstrations; that does not require replacing GRPO with
another RL algorithm. A prefix-only auxiliary imitation term can be a later
controlled arm alongside GRPO, with freshly inferred prompts/priors and
family retention checks. It is a different learning objective and must be
reported separately. Do not restart the long broad residual-SFT phase that
previously erased about 72% of composed plate capability. The first proposed
use of this bank is transition-start curriculum, which avoids that SFT pass.

## Measurement and budget

Compare ordinary joint GRPO with demonstration-start joint GRPO from the same
selected checkpoint. Record success, grasp, lift given grasp, release given
grasp, placement geometry, usable groups, gradient-contributing rows per
family/stage, prefix replay overhead, sampled actions and GPU time. Prefix
replay is work even though it does not enter the selected training counter.

Use 0.5–1M selected actions as an initial transfer check, then multi-million
blocks when unassisted pick-up/manipulation improves with retention. Continue
the user's objective: one shared checkpoint above 70% at declared easier fixed
settings before expanding distances. No demonstration count or curriculum
method guarantees that result.

Current deliverable: extraction tool, live replay/handoff suffix-collection
probe, two-GPU probe launcher, CPU tests and this adopted GRPO-only plan.
The training loop does not yet load demonstration starts; no local change
establishes a GPU handoff pass or demonstration-guided learning gain. No new
remote demos or training were run from the local machine.
