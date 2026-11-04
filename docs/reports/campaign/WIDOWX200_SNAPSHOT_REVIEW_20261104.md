# WidowX-200 snapshot: failures precede composition SFT

Reviewed **2026-11-04**, Europe/Moscow. This extends the
[November 3 final-summary review](WIDOWX200_PIPELINE_REVIEW_20261103.md).
The archive confirms repeated loss of earlier skills, weak ordinary pickup
signal, an ineffective composition-SFT transition, and a weak harvested prefix
bank. Joint RL subsequently recovers some skills and produces the measured
7.13% composed result; it is not where all of the earlier skill loss began.

## Retained evidence

The [original archive](../../artifacts/widowx200_snapshot_20261104/results_snapshot.zip)
is retained byte-for-byte, with [source digest](../../artifacts/widowx200_snapshot_20261104/source.json).
The [CPU audit script](../../artifacts/widowx200_snapshot_20261104/audit_snapshot.py)
verified **all 435 manifest-listed source files**, checking bytes and SHA-256,
and reproduced [audit_summary.json](../../artifacts/widowx200_snapshot_20261104/audit_summary.json).
Readable [state](../../artifacts/widowx200_snapshot_20261104/pipeline_state.json),
[stage table](../../artifacts/widowx200_snapshot_20261104/stages.csv),
[retention table](../../artifacts/widowx200_snapshot_20261104/retention.csv) and
[exported report](../../artifacts/widowx200_snapshot_20261104/report.md)
are retained alongside it. The archive has 445 members, including 3,544
training and 161 validation readings across seven stages. It contains no
checkpoint bytes, frame arrays or trajectory arrays.

Pipeline creation is logged October 24; completion November 1. The start
commit is `dbcb5c241dab259bb520ad2d936b79e776ff96d0`; the exporter and
composition-SFT commit are `3c6486dbd9fe69bc8897a5fc84f64286e6f16ba5`.
The source snapshots span code revisions. Export is explicitly a live,
non-atomic snapshot, and current source configs may differ from launch.
Matching manifest hashes prove internal archive integrity, not unchanged
simulator assets over the entire run. The server log timestamps do not
declare a timezone.

## Locate the first failed handoffs

| Policy/evaluation | Reach | Pickup | Caught bowl | Caught plate |
|---|---:|---:|---:|---:|
| After s2 dense pickup | 405/1024 | 140/1024 | — | — |
| After s3 dense placement, repair rejected | 55/1024 | 0/1024 | 316/520 | 372/504 |
| After s4 binary reach, repair rejected | 551/1024 | 0/1024 | 103/520 | 217/504 |
| After s5 binary pickup, before SIL | 0/1024 | 0/1024 | 62/520 | 122/504 |
| After s5's two accepted SIL cycles | 372/1024 | **1/1024** | 204/520 | 329/504 |
| After s6 selected checkpoint, repair rejected | 9/1024 | **0/1024** | 165/520 | 267/504 |
| Before composition SFT, fresh evaluation | 6/1024 | 0/1024 | 172/520 | 264/504 |
| After composition SFT | 26/1024 | 0/1024 | 191/520 | 258/504 |
| Final selected joint policy | **341/1024** | **246/1024** | 72/520 | 293/504 |

Reaching and pickup were already lost in s3. Later instruction-specific RL
repeatedly recovers its own family while damaging the others. SIL3 is rejected
because it damages placement; SIL4 because it damages reach; SIL6 because it
adds no mean gain. The controller nevertheless advances after these rejected
repairs. Its completion status therefore describes executed stages rather
than a retained four-family controller.

Both SIL5 candidates are accepted, but they leave ordinary pickup at zero
then one success. The acceptance rule uses mean gain over degraded families;
large gains in reaching/placement can compensate for no pickup recovery.
"Accepted repair" is not proof that every degraded instruction recovered.
Future handoff criteria need explicit per-family floors and preservation of
the stage's newly trained skill. These criteria should pause an unsuccessful
handoff for diagnosis, rather than start the next long stage automatically.

## Binary pickup had almost no ordinary-start signal

The archived s5 trainer config confirms **`split_credit_at_grasp: false`**.
That proposed configuration fix was already present. All seven archived
configs use historical **`clipped_action_v0`**, active LoRA updates, LR 1e-4
and four PPO epochs; this is now observed provenance, not an inference from
the current YAML.

S5 logged **57 updates**, ending at 2,258,951 selected actions, with eight
recorded training hours. Its curriculum's actual-approach success count is
zero in **55/57 updates**, with a maximum of seven in the other two. The
overall group pass rate nevertheless has a **34.38% median**, because assisted
starts solve a different, easier suffix. First update: 1,009 training
successes, all in prelifted starts; last: 1,422 successes, of which 1,397 are
prelifted, while actual-approach pickup success is zero. Other non-prelifted
starts include aligned groups and are not ordinary approach successes.

Median informative groups are **14**, with median usable fraction **2.73%**;
the last update has seven informative groups and 1.37% usable. Non-finite
live-episode rates have median zero and maximum 0.049%, so numerical reset
failures are not the dominant logged explanation for s5's zero validation.
The actual bottleneck is obtaining useful ordinary grasp/lift trajectories;
high assisted success cannot certify that learning signal. Split-credit or
optimizer changes alone are not demonstrated remedies.

## Composition SFT did not supply a working sequence

The source has **7,417 rows** from **452 source episodes**, including
314 final-prompt reach rows, 1,355 pickup rows, 4,059 placement rows and
1,689 original-label rows. The latter receive 20% sampling exposure. These
are relabelled skill samples: the dataset explicitly records **zero complete
chains**, no certified destination presence in prefix scenes and no
cross-teacher scene deduplication.

Reach-prefix diversity is particularly weak: bowl and plate each have only
**seven recorded source groups**, 18 and 12 source episodes respectively.
Original-label reach has only five episodes over four groups. Sampling 2,000
rows per requested cell does not create additional scenes or transitions.

Residual held-out MSE improves **0.201123→0.095522**, and LoRA-stage MSE
improves **0.094710→0.090353**. The provided scene split reports zero overlap.
However, ordinary pickup remains **0/1024**, and composed success changes
**11/1024→8/1024**, bowl zero in both evaluations. This small difference is
not a significant regression claim; the result shows no demonstrated rollout
benefit from the offline fit. Joint RL then reaches **73/1024**, recovering
ordinary reach/pickup while losing caught bowl capability relative to SFT.

Composition priors were refreshed for all 7,417 rows, with a 100% resolved
frame join. The refresh report flags a feature-integrity discrepancy at
5.062× its round-trip control, just over its heuristic cutoff; the later
LoRA diagnostic reads 1.01× and labels the difference consistent with batch
numerics. A positional join and full resolution do not independently prove
correct frame identity. The raw arrays are absent, so treat this as an audit
item, not proof of mismatched images.

Earlier SIL3–5 used **episode** splits, while SIL6 uses **scene** splits and
composition uses a provided scene split. Grouped GRPO copies can leak across
an episode split; those earlier imitation-validation scores do not establish
unseen-scene performance. The current driver already requests scene splitting.

## Reharvest and reference provenance need reconciliation

On October 30, move-to and placement banks were reharvested from their own
saved checkpoints using rounds 100–103 and scene-spread filming. The state
keeps the older high reference rates, but the rebuilt banks contain:

| Skill/checkpoint | Historical reference | New-round harvest |
|---|---:|---:|
| s4 reaching | 54.20% | **111/2048 = 5.42%** |
| s3 caught bowl | 54.87% | **369/992 = 37.20%** |
| s3 caught plate | 74.38% | **486/1056 = 46.02%** |

These are different reset rounds. The discrepancy can reflect poor reset
generalization, changed simulator/code/assets, or a combination; this archive
cannot isolate them. The old reference is not the success rate of the new
bank, and should not be relabelled as such. Pickup's bank retains older
round-0 filenames and lacks the new frame-selection marker. Earlier harvest
and candidate-evaluation rounds can overlap. Do not reuse those scores as a
clean held-out promotion test.

Across the 40 replay-survival entries, 4,974 of 5,398 source successes survive
smoothed replay, **92.15% conditional survival**. These entries include banks
from different stages and revisions; they are not 5,398 unique independent
scenes or an end-to-end task-success estimate.

## Next work

1. Run the already prepared **five-second MJWarp grip preflight** from the
   [object-scale audit](WIDOWX200_OBJECT_SCALE_AUDIT_20261103.md#server-verification).
   That separate local CPU audit found delayed grip loss, but this archive
   supplies no sustained GPU hold result. Treat its simulator parity check as
   pending rather than a demonstrated cause of this training failure.
2. Freeze simulator/task geometry and hash the catalog, XMLs, code and adapter.
   Evaluate the same s2/s3/s4 teachers on both old and new reset rounds with
   explicit scene IDs, identical settings, independent rollout repeats and
   grasp/lift/carry/release diagnostics. Resolve the reference/harvest gap
   before rebuilding the prefix bank.
3. Obtain diverse successful ordinary pickup and physically continuous
   final-prompt grasp/carry/release demonstrations. The current independent
   prefix samples do not establish transition coverage. Use scene-disjoint
   evaluation and require per-family and composed retention for accepted
   handoffs, with the initializer available as a candidate.
4. Only then run a bounded arm-specific pilot of the repaired latent
   likelihood/frozen-LoRA optimizer setup, preserving initialization and
   evaluating ordinary-start transfer. Do not start another full seven-stage
   pipeline on this evidence.

No new training or simulation was launched in this review; the existing
object-scale/preflight work was preserved without modification.
