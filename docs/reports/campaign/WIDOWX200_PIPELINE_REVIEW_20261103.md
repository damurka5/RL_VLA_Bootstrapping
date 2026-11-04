# WidowX-200 pipeline: completed run, weak composition and lost retention

**Update 2026-11-04:** the requested diagnostic archive has arrived. The
[snapshot review](WIDOWX200_SNAPSHOT_REVIEW_20261104.md) verifies all 435 source
hashes and locates the first reach/pickup losses before composition SFT. The
following preserves the November 3 interpretation based on final summaries.

Reviewed **2026-11-03**, Europe/Moscow. Run
`widowx200_pipeline_20261024_131201` completed at the console's
`2026-11-01 17:00:09` (server timezone unspecified). It produced a single
adapter, but the returned scores do not meet the skill or composition targets.
Preserve the experiment; do not promote it as a successful four-skill policy
or repeat the entire pipeline unchanged.

## Evidence and scope

The supplied [final_report.json](../../artifacts/widowx200_pipeline_20261103/final_report.json)
and [final_report.md](../../artifacts/widowx200_pipeline_20261103/final_report.md)
are retained byte-for-byte. [review_summary.json](../../artifacts/widowx200_pipeline_20261103/review_summary.json)
records SHA-256 digests, exact pooled counts and reference differences. The
JSON counts agree with every reported rate; its composed totals agree with
the console's pooled two-GPU counts, 32 bowl and 41 plate successes.

Final policy on the server:

```text
runs/widowx200_pipeline_20261024_131201__s7_joint_staged/rl/step_7504473/smolvla_grpo_adapter.pt
```

Only the two summaries and the pasted console tail were supplied. Checkpoint
bytes/hashes, launch-time configuration, pipeline state, validation/training
curves, SFT reports, scene identities, videos and failure funnels are absent.
The configuration explanations below describe the current repository; the
remote snapshots are needed to verify which revisions this run used. This is
a simulation result for the WidowX arm, separate from the CDPR campaign.

## Final policy measurements

| Protocol | Instruction | Successes / episodes | Rate |
|---|---|---:|---:|
| Ordinary reaching | move_to_object | 341/1024 | 33.30% |
| Ordinary pickup | pick_up | 246/1024 | 24.02% |
| Already holding object | put_into_bowl | 72/520 | 13.85% |
| Already holding object | put_into_plate | 293/504 | 58.13% |
| Empty hand, aligned over object | put_into_bowl | 32/520 | 6.15% |
| Empty hand, aligned over object | put_into_plate | 41/504 | 8.13% |

Pooled caught placement is **365/1024 = 35.64%**; composed placement is
**73/1024 = 7.13%**. These use summed episode counts, not an unweighted mean
of destination rates. Destination episode counts are unequal and must remain
520 and 504.

The current driver's `eval_config` makes `put_into` a caught-start
carry/release skill. The composed leg sets caught fraction to zero, leaves
the object on its scene cell and starts the gripper aligned above it, with
a 40-decision minimum. It includes grasp, lift, carry and release under the
destination instruction, but does **not** establish approach from a distant
empty-hand start. Neither score is interchangeable with the CDPR strict
full-task evaluator or its 60M panel result.

The 50.00 pp plate gap and 7.69 pp bowl gap between caught and composed
scores make the grasp-to-placement sequence a clear bottleneck. The bowl
also fails badly when already caught, so improving pickup alone cannot
resolve all observed failures. The summaries do not establish whether grasp,
carry loss, release precision, reset validity or timeouts dominate.

## Retention against the supplied references

| Instruction | Reference | Final | Difference | Current degradation rule |
|---|---:|---:|---:|---|
| move_to_object | 54.20% | 33.30% | −20.90 pp | Fails |
| pick_up | 9.77% | 24.02% | +14.26 pp | Passes |
| put_into_bowl, caught | 54.87% | 13.85% | −41.03 pp | Fails |
| put_into_plate, caught | 74.38% | 58.13% | −16.24 pp | Fails |

The current rule flags a loss greater than
`max(0.08, 0.20 * reference_rate)`. All three losses exceed their own
tolerances. These are descriptive comparisons with the saved references,
some established on harvest rounds, not a paired statistical test. The
reference summaries omit their original episode counts and scene IDs.
The earlier stage validation maxima are a third measurement and must not be
substituted for these reference rates.

## What the stage summaries mean

- `s1_move_to_dense`: best 48.54%; `s4_move_to_binary`: best 56.93%.
  Final reaching at 33.30% has not preserved that capability.
- `s2_pick_up_dense`: best 12.11%; `s5_pick_up_binary`: best RL validation
  0%, followed by two attempted SIL cycles and a handoff at
  `s5_pick_up_binary_sil2/sft/sil_sft_adapter.pt`. The 0% describes the RL
  selection, not that SFT adapter or the final joint policy. It is a serious
  diagnostic signal, not evidence that the final policy cannot pick up.
- `s3_put_into_dense`: best worst-destination validation 71.875%;
  `s6_put_into_binary`: best 65.82%. These are caught-start skill metrics,
  not complete pick-and-place success.
- `s6` explicitly reports `user_checkpoint`: the handoff uses step 1,764,679
  although the validation maximum was step 1,016,667. The driver supports
  this deliberate override. It is not a checkpoint-selection bug; its effect
  needs matched evaluation rather than assuming the earlier maximum wins.
- `s7_joint_staged`: best worst-instruction composed score 6.696% at step
  7,504,473, against a configured 60% target. The checkpoint step identifies
  the selected best policy, not the end of training. The summaries do not
  contain total actions, actual stopping steps or run durations.

Six stages stopped on `plateau`, one on the user checkpoint. A plateau is an
allowed stopping condition, not a success criterion. The driver advances to
the next stage even below the stage target. Joint training explicitly has
`sil: false`; zero joint SIL cycles is expected, and this skips the automatic
retention evaluation/repair loop. Re-enabling the existing SFT loop without
checking composed outcomes is not a demonstrated repair: its acceptance
checks currently evaluate ordinary skills, including caught placement.

## Repairs and next experiment

The local driver now reports final retention status, exact success counts,
the reset protocols, accepted/attempted SIL cycles, and separate best RL,
selected RL and handed-off policies in both Markdown and JSON. This makes
the observed failure visible on future runs. It does not change this run's
weights, success predicates, training recipe, stage gates or old reports.

Before spending another full run, export the existing evidence on the server:

```bash
cd /root/repo/RL_VLA_Bootstrapping
conda run --no-capture-output -n cdpr-mjlab python3 tools/audit/export_widowx200_pipeline_results.py \
  --pipeline-dir runs/widowx200_pipeline_20261024_131201
```

Return the `Archive:` ZIP printed by the exporter. It contains pipeline state,
per-stage configs and metrics, retention and composition evaluations, SFT
reports and hashes, without models, images or trajectory arrays.

Use those records to locate the first failed handoff: before versus after
composition SFT, then SFT versus selected joint RL. Compare reaching, pickup,
caught placement and composed placement separately. Inspect `s5`'s actual
binary config, informative-group counts and advantages: the current config
sets `split_credit_at_grasp: false` for binary reward because first-grasp
credit would otherwise be zero before lift success. The returned summaries
cannot tell whether this setting reached the remote run or whether sparse
training simply received no useful groups.

Also audit the likelihood and optimizer settings before extending this branch.
The current WidowX YAML does not select `action_likelihood`, so the trainer
defaults to historical `clipped_action_v0`; it enables LoRA updates and uses
LR 1e-4 with four PPO epochs. The promoted CDPR continuation uses
`latent_gaussian_conditional_offset_v1`, frozen LoRA, LR 1e-5 and one PPO
epoch. The legacy likelihood scores clipped actions and widens the Gaussian
standard deviation for the episode offset, whereas the repaired mode scores
unclipped samples conditioned on the realized offset. This known implementation
difference is worth a controlled WidowX pilot; the summaries do not establish
that it caused this run's failure or that CDPR's gain transfers to the arm.
The latent mode requires frozen RL LoRA updates and explicit legacy-checkpoint
conversion, so changing one YAML flag or using ordinary full-state resume is
insufficient. Preserve the existing result and verify initialization/action
equivalence in a separate pilot before making that switch.

Then use a bounded restart from the last checkpoint that preserves the needed
skills, with all four evaluations and composed grasp/lift/carry/release
diagnostics before and after. If composition SFT damages the starting policy,
reject that handoff; if joint RL damages it, address joint retention and
selection. Require composed performance as well as ordinary-skill retention
when accepting a repair. Do not silently overwrite the completed experiment
or interpret a lower offline imitation loss as a successful policy repair.

Local verification: **61 checks passed**, covering the new final-report
regressions and the existing pipeline, manual finish, artifact safety and
export tests. No CUDA simulation was run.

## Separate CDPR work

The 60M CDPR endpoint is `step_60025885` (exact integer TensorBoard event
step), with `step_59268127` the panel peak. Its 70.61% panel result remains
unconfirmed by separate evaluation. The [60M evaluation runbook](CDPR_60M_CONTROL_EVALUATION_20261103.md)
compares the endpoint with the promoted 40M baseline on matched development
scenes. No remote evaluation or training was executed during this review.
