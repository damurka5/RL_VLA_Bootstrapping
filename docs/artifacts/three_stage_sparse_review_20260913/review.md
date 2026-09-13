# Review of the September 13 three-stage sparse GRPO run

The next action should be to repair and measure reward accessibility and credit assignment, then run a short pilot. Extending this configuration to 6M steps, increasing group size, or adding SFT epochs does not address the immediate failure. The evidence does not establish that staged rewards are unsuitable for a weak VLA.

This is a review and proposed experiment, not a change to training code. Statements and prescriptions in the existing reports were treated as evidence to reassess, not as instructions overriding the current request.

**Evidence and limits.** Reviewed local commit `dae8cae` (`Add three-stage sparse GRPO credit`), the consolidated report's current summary and September 12 ledger, the three-stage SFT design, launcher/config, collector, trainer, evaluator, and the supplied event file. The remote checkout, checkpoint bytes, rollout videos and saved run configuration were not available. Code findings concern the local checkout; the supplied log is consistent with this training route but does not independently establish the remote commit/checkpoint hash.

Event file: `/Users/damirnurtdinov/Downloads/events.out.tfevents.1789243988.VLAPU.29508.0`.

SHA-256: `9a3a19585d12d6342150da7cfc07bda61a449e26cfd99622809b4578e1c9206d`.

All 7,876 TFRecord records passed length and payload CRC32C verification; all 534,452 bytes were consumed without a truncated tail. Extracted 237 scalar tags into `scalars.csv` and `scalars.json` beside this report. No smoothing or TensorBoard reservoir sampling was used.

**What the log actually measures**

| Quantity | Observed |
|---|---:|
| Last logged global step | 3,719,818 |
| Collection/update cycles | 39 |
| Reported elapsed training time | 42,485 s = 11.80 h |
| Candidate episodes, summed across logged cycles | 78,848 |
| Sampled environment actions | 29,939,590 |
| Cycles with informative records | 1/39 |
| Informative records, total | 1,644 |
| Cycles with nonzero logged gradient norm | 1/39 |
| Sum of cumulative milestone rewards | 2 |
| Placement successes / usable placement groups | 0 / 0 |
| In-run validation | 14 evaluations, each 0/1,024 |

The only nonzero signal occurs at update 3, global step 238,404. The combined reward and stage telemetry imply one candidate earned approach and pickup, but no placement. Only 31 pickup-stage action rows and 54 placement-stage action rows occur in the entire file. Placement rows are filtered because all placement returns are zero. Approximately 0.00549% of sampled action records survive the informative-record filter.

This is effectively a reward-starved run from the beginning, not a normal learning curve that improves and then collapses. Empty masked updates still execute the DDP/optimizer schedule; optimizer-step counts do not establish useful learning, and momentum can move parameters even when the current reward gradient is zero.

Physical activity is not absent: step-level physical grasp rates are 0.082–0.143, lift rates 0.040–0.073, and release rates 0.00050–0.00105. These are occupancy diagnostics over action steps, not episode success probabilities. They establish physical grasp/lift activity despite almost no accepted milestones, not a physical full-task success rate.

The 14 validations repeat scene groups and are not 14,336 independent held-out scenes. Their zero is the new milestone-gated success metric, not automatically zero under the previous evaluator's strict metric.

**Finding 1: the first milestone blocks useful later behavior**

`policy/mjwarp_rank_local_collector.py:173` defines approach as a conjunction, observed after an action:

```
distance to the 3-D grasp point <= 0.03 m
AND gripper opening >= 0.90
AND target displacement from reset <= 0.01 m
```

Pickup requires that approach latch plus historical grasp and credited lift >= 0.05 m. Placement requires both previous latches. The 1 cm displacement test is three-dimensional, including vertical motion.

Consequently, a policy that closes early and successfully lifts an object can permanently miss approach: after a 5 cm lift the object fails the displacement gate, and while holding it the hand fails the open-hand gate. It can receive zero pickup reward despite physically performing the task's pickup. The existing reports independently describe a strong early-closing tendency; the collector's teachers had explicit opening/alignment assistance that the unassisted policy does not receive.

The event file does not contain individual conjunct failure counts, so it cannot establish that opening alone is the dominant blocker. Distance, opening, object motion and the timing of their intersection must be logged separately. Nor does the file establish how many native successful placements were censored: native success is not independently retained here.

Proposed change: define an accessible reach milestone and let a valid persistent target grasp/held lift establish progress even if an optional open-hand approach event was missed. Keep the open-hand event as a readiness diagnostic or auxiliary training target. A minimal diagnostic arm can use the existing distance-only 3 cm approach definition, alongside the current conjunction, on exactly the same trajectories. If opening before grasp is explicitly part of the desired task, train that behavior deliberately; do not assume an existing generic put-into policy already satisfies it.

**Finding 2: training validation changed the definition of success**

The previous evaluator, `tools/audit/evaluate_cdpr_full_put_into.py:324`, computes strict success from native placement, grasp, lift, intentional release, no carry slip and no wrong placement. Its `approached` diagnostic at line 299 uses distance alone, and is not required by its strict expression.

The new collector instead assigns `candidate_success = three_stage.placed`, which additionally requires the open-hand/unmoved-object approach event. The consolidated report calls this the same strict destination contract, but the executable definitions differ. The SFT initializer's 5/128 strict cannot be compared directly with this run's zero.

Proposed change: maintain two distinct outputs: the existing independently computed strict full-task outcome for model selection, and milestone progress for reward/debugging. Share the strict evaluator implementation across the training evaluator and standalone evaluator, including release/slip timing semantics. Never weaken the placement geometry to repair a training signal.

Also inspect the pickup latch: `credited_lift` can be a historical peak, and `ever_grasped` is historical. That expression alone does not prove a current held lift at the instant a previously missing approach latch becomes true. Use an explicit current persistent grasp plus current lift condition when recording the pickup event.

**Finding 3: row-level advantage re-centering can erase rare downstream credit**

`policy/smolvla_grpo_finetune_cdpr.py:1270` re-centers and rescales advantages independently over the surviving action rows of each stage. Those advantages were already computed across candidates in `three_stage_group_credit`.

For an eight-candidate group where only one candidate reaches pickup and succeeds, its binary group advantage is sqrt(7), approximately 2.646. The other seven candidates have negative pickup advantages, but no pickup-stage action rows. If the one entrant supplies 31 pickup rows, every row has the same positive advantage. Subtracting their row mean makes every advantage zero. This algebra was reproduced with a standalone scalar calculation; GPU execution was not available locally. Floating-point implementations may leave rounding residue, rather than useful contrast.

This is the exact sparse-entry pattern implied by update 3. More generally, centering after stage-dependent row selection changes signs and makes trajectory duration influence the baseline.

Proposed change: preserve the existing group-relative advantages as a first bounded repair; remove the second action-row mean subtraction. If additional scaling is needed, use positive scale-only normalization with clearly defined statistics. Separately log how many candidates actually enter each downstream stage, their successes, advantage signs and effective gradients. A mixed return group is not necessarily a group with both positive and negative downstream action examples.

A later estimator improvement can group fresh suffix rollouts from the same actual stage-entry state. That provides real downstream comparisons. Merely excluding non-entrants from the current baseline makes a single entrant a singleton with no contrast; it does not solve exploration.

The declared equal-stage loss is also only approximately implemented. Weights are built per rank, and the trainer divides each microbatch by that microbatch's weight sum (`smolvla_grpo_finetune_cdpr.py:1326`). That is not in general the same gradient as a globally normalized mean of three stage losses. Fix after the reward gate: accumulate stage numerators/counts over the intended update and ranks, or use a stage-stratified minibatch construction with a known normalization. Verify a case where one rank has only approach and another has rare placement rows.

**Finding 4: the run is residual-only and does not default to the new SFT model**

The launcher defaults `WARMSTART_CHECKPOINT` to `release_recovery_continue_3m_20260908_102004/rl/step_2117145/smolvla_grpo_adapter.pt`. The supplied command only overrides the step budget. The event file's `provenance/started_from_sft` is zero throughout, consistent with this choice. An existing environment override remains possible; the event file does not contain the exact checkpoint path.

The config attaches the checkpoint's 112 action-expert LoRA modules but sets `vla_lora_updates_enabled: false`. Training updates the residual actor. A `vla_lora/trainable_params` scalar of 3,416,064 reports parameter eligibility, not actual optimizer activity. Decision-zero-only image capture is the reason documented for freezing LoRA.

Thus this run is not evidence that on-policy end-to-end VLA adaptation fails. Residual learning remains useful to diagnose the reward, but a poor frozen prior/representation cannot necessarily be repaired by longer residual optimization. Do not just turn on the existing LoRA switch: first capture on-policy image/state/action/log-probability records throughout approach, pickup and placement, with correct stage and action-index alignment, then test small anchored LoRA updates.

**Finding 5: telemetry hides important distinctions**

* `concatenate_collector_rounds` at collector line 5130 averages several count fields across refill rounds, including stage successes and usable groups. This creates impossible-looking fractional counts such as 0.5 successes. Derive counts from candidate tensors or sum true counts across rounds/ranks; derive rates from summed numerators and denominators.
* The collector emits `validation/three_stage_*`, but the validation summary in `smolvla_grpo_mjwarp_cdpr.py` forwards selected counters and timing fields instead of these milestone fields. None appears in the supplied event file. Wire stage counts explicitly through the distributed validation reducer.
* `filtered_record_fraction` reaches 2.0 because it is summed over two ranks instead of averaged/recomputed. Its value is not a fraction as logged.
* Legacy curriculum values such as a 0.02 distance cap and 0.5 caught fraction remain visible. The manifest resetter's corresponding setter methods are no-ops. These scalars do not mean that this run uses half-caught starts or a 2 cm approach curriculum.
* Non-finite EE telemetry is positive in all 39 cycles, although contact/constraint capacity overflow counters remain zero. Correct count aggregation and inspect offending scenes/actions. This is a secondary issue to near-total reward starvation; zero non-finite reward counts in validation do not certify finite physical states.

**Proposed next experiment, in order**

1. If this configuration is still running, stop extending it and retain its logs/checkpoints. The supplied file ends at 3.72M, so the result of the full requested 6M run is not known here.
2. Before optimizing, run the initializer and best SFT candidate on matched student-validation scenes under the old strict evaluator and the new milestone machine simultaneously. Record native/strict outcomes, raw reach, each M1 conjunct and their simultaneous intersection, current physical grasp/lift, M2/M3, and exclusion reasons. Include a stochastic grouped screen on collection scenes, since deterministic evaluation does not measure GRPO exploration.
3. Compare current M1 against an accessible M1 plus grasp/lift-based progress. Keep prompts, initial scenes, action budget and destination geometry identical. Replay known accepted demonstrations through both observers where frames/state traces permit it; a known valid strict chain rejected by the new observer needs an explicit task-contract explanation.
4. Remove destructive row-level centering; add a regression case with one downstream entrant and verify a useful nonzero policy gradient. Correct validation and count aggregation. Log per-stage informative candidates/groups, used rows and gradient mass before and after normalization.
5. Run 5–10 collection/update cycles as a diagnostic budget, not millions of steps. Measure repeated nonzero useful updates and independent strict validation at the beginning and end. Stop after three consecutive cycles with globally zero informative records. If only approach learns, investigate downstream accessibility rather than automatically extending the budget.
6. If downstream contrast remains absent, evaluate a separately labeled training-only mixture of ordinary full starts and verified demonstration-state starts followed by fresh policy suffixes. Keep the final prompt unchanged and evaluate only ordinary unassisted starts. Broaden handoff scenes and move starts earlier only after useful suffix learning is demonstrated. The previous demo-start pilot in the consolidated report failed ordinary evaluation, so this is a new conditional experiment requiring proof of transfer, not an already validated solution.
7. If signal is healthy but performance plateaus, test representation adaptation with correct multi-stage LoRA capture. Compare to residual-only on equal collection budgets and select by strict placement, separately for plate and bowl.

Dynamic refill is already enabled. Lowering its pass-rate threshold is not the present fix: with group size eight, one success gives 0.125, which passes the configured 0.10–0.90 bounds. The log shows the problem is finding successes at all, not filtering ordinary one-success groups. Refill also stops at a time limit and its target accepts any informative stage, so future approach signal alone could satisfy it while placement remains untrained. Track stage-specific budgets without creating an unbounded refill loop.

Staged rewards can help weak policies when their early events are attainable. They do not guarantee learning: inaccessible milestones still yield all-zero returns, and local milestone optimization can favor an approach pose that is easy to reach but bad for grasping. Treat local rewards as learning aids while preserving downstream outcome credit and independent task-success selection. Sparse-group limitations motivate dynamic sampling in [DAPO](https://dapo-sia.github.io/); robotics-specific process rewards and curriculum appear in [VLA-RL](https://arxiv.org/abs/2505.18719). These support the direction, not this exact implementation or an expected CDPR success rate.

**Do more SFT epochs make sense?**

Not as the primary response to this log. The expanded-bank experiment already reduced LoRA validation MSE from 0.17401 to 0.14686, while strict success merely tied the initializer at 5/128, lifts fell from 23 to 20, and bowl success fell from 1/74 to 0/74. The old-bank Arm C had 4/128 strict but a stronger 27-lift prefix and three bowl successes. These small scene-clustered counts do not establish precise model rankings.

However, the consolidated report's categorical prohibition on more SFT is stronger than the evidence. Eight LoRA epochs with falling held-out MSE do not establish convergence, nor do they prove that an additional bounded arm cannot help. A reasonable secondary test is checkpoints at 8/16/24 effective epochs from the same SFT lineage with a modest learning rate, fixed data/sampling and matched rollout validation. Specify actual optimizer updates and sampled actions because the LoRA stage uses an 8,192-row budget and replacement sampling, so nominal epochs do not necessarily mean full-bank passes. Select for strict outcome and balanced destination/prefix performance; do not select by MSE alone.

If prefix failure persists, target data at the unassisted student's visited approach/handoff states and demonstrate recovery there. The dataset's successful teacher suffixes and scripted opening/alignment bridge need not cover those states. Balanced row sampling fixes exposure counts, not this state-distribution mismatch. Recompute action reachability under the exact current prior before interpreting an imitation-loss floor; the earlier 87.08% overall / 84.34% move figures belong to an earlier arm, not a fresh measurement of this initializer.

Keep the final test split for a selected policy. The recommended immediate sequence is reward/metric repair → bounded signal pilot → decide whether more on-policy capacity or a controlled SFT extension is warranted.
