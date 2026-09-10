# CDPR + SmolVLA: consolidated progress and achievement report

**Living report — current through 2026-09-10, Europe/Moscow**

**Repository state reviewed:** `e56b7fb` (local review; remote run commit not supplied)

**Scope:** simulated 5-DoF cable-driven parallel robot (CDPR), SmolVLA-conditioned control, GRPO reinforcement learning, self-imitation learning (SIL), and multi-instruction retention.

**Active user direction, 2026-09-10:** complete the entire `put_into` task:
empty gripper approaches an object, grasps it, carries it to a plate/bowl and
releases it there. The proposed self-imitation data concatenate pickup and
placement stages under the final `put_into` instruction. Those stages must be
physically continuous; a pickup prefix alone is not a completed placement.
The prior GRPO direction remains the campaign context. The demonstration-start
training integration now exists locally and the first remote 1M-action pilot
has completed, with lower final scores on all four families. It trains fresh
GRPO suffixes, not the proposed full-trajectory imitation dataset. See the
newest §14 entry for results, evaluation limitations and the next experiment.

**Three-stage collection/SFT specification, 2026-09-10:**
[`CDPR_THREE_STAGE_PUT_INTO_SFT_DESIGN.md`](CDPR_THREE_STAGE_PUT_INTO_SFT_DESIGN.md)
defines continuous `move_to → pick_up → put_into` demonstrations, with the
user-confirmed fixed calibrated pickup yaw, stage-balanced SFT sampling and
full-task evaluation. It includes a provisional teacher shortlist and the
repository change map.

**Implementation status, 2026-09-10 (same day, later):** the specification's
whole change map is now implemented and unit-tested locally — the scene/yaw
contract, the continuous three-stage recorder, teacher screening, the dataset
builder, the prior refresh and SFT changes, and the unassisted full-task
evaluation. See §7.12. **No GPU run has been executed against it:** no scene
manifest, no teacher screen, no bank, no SFT and no evaluation number exists
yet. The one measurement that does exist is the yaw calibration, which is
kinematics from the MJCF and needs no GPU: the fixed pickup yaw is
**0.000 rad** at the desk centre, and that single angle leaves a **mean 10.7°,
p90 19.8°, max 25.3°** camera-bearing residual across the ±0.19 m workspace,
with 18.4% of a 7×7 grid inside a 5° band. That number is the measured cost of
the user-confirmed fixed-yaw choice and is not a reason to change it; it is the
quantity a later per-position facing mode would be compared against. Teacher
screening on the new scene/handoff distribution remains the first GPU step.

This is the campaign's canonical high-level progress record. It consolidates the results that are still technically relevant, backed by retained evidence, or used by the current training loop. Failed branches and measurements later shown to be invalid are not presented as achievements. They are named only in §10 so they are not accidentally revived.

Filesystem creation times in §2 describe when a file appeared on this machine. They are not treated as experiment timestamps unless the report or artifact also identifies the run date.

---

## 1. Executive summary

The project has progressed from learning one language-conditioned reaching task to a single adapter that performs four instruction families:

1. `move_to_object`
2. `pick_up`
3. `put_into_plate`
4. `put_into_bowl`

The central idea is now demonstrated end to end:

> Start from a pretrained SmolVLA action prior, learn task-specific corrections with a compact residual policy and GRPO, harvest successful trajectories, preserve them in a retention bank, and alternate family-specific RL with balanced residual SFT so one adapter can recover old skills while adding or strengthening a new one.

### Current headline achievements

**Latest completed experiment, 2026-09-10: demonstration-start GRPO shows no
ordinary-evaluation improvement.** `demo_grpo_pilot_20260910_110040` completes
91 updates / 1,005,877 selected actions. Baseline → final: move-to
**314/400 → 311/400**, pickup **83/328 → 68/328**, configured plate
**274/456 → 263/456**, bowl **115/352 → 90/352**. All recorded pre-action
reset poses and compared metadata match exactly. These placement evaluations
still omit the approach: the gripper starts at the object's XY in every
container episode. **424/456 plate and 144/352 bowl starts are already inside
the XY goal radius.** Outside-radius success is **12/32 → 8/32 plate** and
**15/208 → 13/208 bowl**, over only four and 26 reset groups respectively.
Retain the donor and candidate as evidence; do not promote or extend this
configuration on these results. The full user task has not been demonstrated.
Per-rank demonstration logs, the training-bank manifest and the run's config
snapshot are still needed to establish assisted gradient coverage and remote
provenance. See the newest §14 entry and its linked reproducible CPU audit.

**Campaign decision, 2026-09-09: demonstration-guided GRPO is adopted as the
next implementation direction.** Keep one shared policy and first exceed 70%
on all four instructions at declared easier fixed settings, then expand
distances. Demonstrations initialize missing grasp/lift/carry/release
transitions; only fresh current-policy continuations enter GRPO. Keep ordinary
starts in training and evaluate transfer without assistance. The extractor is
implemented and the remote prototype now contains **293 lift prefixes**, of
which **156** also have complete placements and **137** come from failed
placements. A replay-and-handoff probe collecting fresh suffixes without
optimizer updates is implemented. The latest **2026-09-10 10:11 GPU probe**
verifies learning-signal collection for earlier pick-up starts (**52/56**,
two variable groups, **712** usable residual rows) and one bowl scene
(**1/8**, one variable group, **197** rows). Active LoRA capture is verified:
**24 nonzero-advantage rows** across those three groups. The previous probe
supplied four variable plate groups / 1,314 residual rows. These are assisted
suffix diagnostics, not trained success rates; no weights were updated.
At that probe stage, training-only demonstrations, broader scene coverage and
optimizer integration remained pending; the completed pilot above supersedes
that implementation status. See §7.15, the latest §14 entry and
`CDPR_GRPO_DEMONSTRATION_PLAN.md`.

**Latest z-offset pilot, diagnostic only:**
`release_recovery_pilot_20260909_130003` moved pick-up **0.2774 → 0.2561**
and bowl **0.3466 → 0.2926**, while move-to rose **0.8025 → 0.8275** and
configured composed plate **0.6053 → 0.6206**. Retain the experiment and
previous checkpoints; do not promote this candidate or extrapolate a benefit
from longer training with the same configuration. The gate also changes
gripper exploration, and the lift probe does not reproduce the task's lift
definition. Full counts, command diagnostics and provenance limits are in §14.

**Latest matched comparison, 2026-09-09:** final continuation checkpoint:
move-to **0.7800**, pick-up **0.2744**, composed plate **0.5943**, bowl
**0.3381**. The retained bowl-peak checkpoint is stronger for placement:
**0.7018 plate / 0.3778 bowl**, with move-to 0.7650 and pick-up 0.2195.
This is a checkpoint tradeoff, not a promoted four-family >70% policy.
**And plate's 0.7018 is not a placement result.** 92.6% of composed plate
episodes begin with the object already inside the 0.091 m success radius,
because the configured spawn range (0.06-0.10) is smaller than the plate and
the workspace clamp pulls more of it in; bowl starts inside only 42.7% of the
time. The two families are being asked different questions and their rates are
not comparable. See §7.14 and §10.
The helper's recorded-field pairing gate returned status 2; its earliest
object poses are post-action, so that check cannot establish different
resets by itself. Full evidence and limits are in the latest §14 entry.

**Previous pilot candidate, 2026-09-08:** the fixed-cap joint-RL release-recovery pilot
improved composed plate **0.6162 → 0.6930** and pick-up **0.1494 → 0.1982**
on matched baseline/final evaluation settings. Move-to reached **0.7125 at cap
0.08**, bowl **0.3068**. This is a retained candidate, not a promoted >70%
  four-family policy. A further 3M-action full resume has now completed at
  3,540,208 cumulative steps. Its 121 continuation updates show modest gains,
  with the last validation at 3,512,892; the final checkpoint's separate
  evaluation is recorded above. Fresh-seed confirmation remains pending. See §14 and
  `CDPR_NEXT_CAMPAIGN_PLAN.md`.

| Achievement | Strongest supported result | Evidence status |
|---|---|---|
| RL-only language-conditioned reaching | `move_to_object`: **0.6299** success, 645/1024 episodes, 128 independent scenes, 2 cm tolerance, cap 0.19 | Complete six-leg local evaluation; archive checksums pass |
| RL-only placement from a frozen VLA prior | Phase 2 final: plate **0.710**, bowl **0.320** after 15M steps; both started at 0 | Reported training/held-out validation result |
| Best historical placement validation | Phase 4: **0.633** overall, plate **0.791**, bowl **0.442** at `step_7753190` | Checkpoint retained and checksum-verified |
| Best later placement RL checkpoint | Phase 5 `step_2754052`: **0.6211** overall; plate **0.7982**; bowl **0.4073** at that checkpoint, with a separate bowl peak of **0.4655** | Reported as the checkpoint to retain/resume; adapter not in supplied archive |
| Retention across three instruction families | Cycle 1: move-to **0.311**, plate **0.523**, bowl **0.2765** | Adapter and SFT report retained and checksum-verified |
| Four-family single policy | Cycle 2: pick-up **0.1738**, move-to **0.4316**, plate **0.3463**, bowl **0.1702** | Adapter and SFT report supplied locally |
| Latest and strongest reported single policy | Cycle 3: pick-up **0.1191**, move-to **0.4779**, plate **0.7383**, bowl **0.5353** | Report and 20 placement videos supplied; Cycle 3 adapter not supplied locally |
| Data-scaling result | Raising the balanced slice from about 6k to 26k decisions per instruction moved move-to from **48.5% to 67.3%** of its reference and pick-up from **37.1% to 66.8%** of its source | Confirmed by Cycle 2 SFT artifact and Phase 5 evaluation |
| Self-imitation data pipeline | Record → smooth → replay → pool → rebalance → refresh current priors/state → residual SFT → multi-family evaluation | Implemented, tested in-project, and used for Cycles 1–3 |
| **Composed `put_into` is measurable for the first time** | Phase 6 seed `sft_phase6`: uncaught plate **0.0935** (80/856), uncaught bowl **0.0265** (18/680), against 0.0046 and 0.0114 for Cycle 3 under the identical protocol | Four evaluations run by `scripts/run_cdpr_phase6_compose_seed.sh`; adapter retained locally |
| Composed demonstrations come from a scripted oracle, not a policy | Oracle on the composed task: plate **1.000**, bowl **0.427** in smoke, plate **0.909** / bowl **0.455** pooled over 8192 worlds at cap 0.20 | 12 harvest rounds retained; replay survival 0.998–1.000 |
| Composition RL now anneals the pre-grasped start | `phase6_compose_iter0`: caught fraction **1.0 → 0.9 → 0.8**; validation peak **0.6240** overall (plate 0.7571, bowl 0.4634) at step 4 257 133 | TensorBoard event file; caught-dominated validation protocol, see §8.4 |
| Cross-instruction transfer | Composed demonstrations carry a grasp prefix under the `put_into` prompt, and `pick_up` rose **0.1191 → 0.1491** (+25%) with no pick_up data added | Same four-evaluation run as the seed row |
| **Composed pick-and-place, one sparse reward, four instructions** | Phase 7 `step_2017690`: composed plate **0.5000** (228/456), composed bowl **0.2159** (76/352), `pick_up` 0.1465 at its own cap, `move_to` 0.4200 — against the seed's 0.1203 and 0.0412 on the identical protocol, i.e. **4.2x and 5.2x** | `sil_record` at cap 0.20, 3 rounds x 512 worlds; decomposition retained |
| **The grasp gap to the scripted oracle is closed** | Plate grasp 0.9079 against the oracle's 0.9336 (97.2%); bowl 0.6051 against 0.6752 (89.6%), from 0.4054 and 0.2956 before | `placement_failure_decomposition`, 0 predicate disagreements |
| Residual SFT destroys the composed gain | The same bank that rebuilds forgotten families takes composed plate 0.5000 → 0.1285-0.1390 at every demonstration mix tested (0.5 and 0.8 composed) | Two-arm sweep, identical protocol |
| **The composed "drop" was a scoring boundary, not a physical one** | `wrong_place_settled` terminated episodes whose object was already at rest **inside** the receptacle, a median of zero env steps after the grasp latch broke, because it tested `~container_ok` and `container_ok` requires the release. Fixing it takes composed plate **0.5000 → 0.6272** and bowl **0.2159 → 0.2841**, with `move_to` moving +0.010 as a control — the campaign's best composed result, from a scoring correction rather than training | Protocol-matched repeat of the Phase 7 composed evaluation (cap 0.20, caught fraction forced to 0, containers 100% at the 40-decision floor); mechanism isolated on a scene-matched paired pair, `same_episodes=True`, step-0 actions identical in 512/512 worlds |
| The grasp detector is not rejecting real grasps | The 8 mm relative-pose stability test never fires on a held object: **0 of 12 496** genuinely-held policy steps crossed the bar, held slip p50 0.23 mm | `grasp_loss_forensics`, CPU-only, on recordings already on disk |

### Instruction success by phase and retention cycle

This table is the compact instruction-level result index. A dash means that the
instruction was not evaluated or not reported for that stage; it does **not**
mean zero success. Placement entries distinguish **caught** starts, where the
object begins held, from **composed** starts, where the policy must approach,
grasp, carry, and release from the desk.

| Phase / stage | Evaluation protocol | `move_to_object` | `pick_up` | `put_into_plate` | `put_into_bowl` |
|---|---|---:|---:|---:|---:|
| Phase 0 — dedicated reaching | Controlled six-leg validation, cap 0.19 | **0.6299** (645/1024) | — | — | — |
| Phase 1 — dedicated pick-up source | Dedicated/source reference, cap 0.06 | — | **0.2600**; direct re-harvest mean 0.279 | — | — |
| Phase 2 — RL-only placement, final at 15M | Held-out caught-placement validation | — | — | **0.7100**; run peak 0.778 | **0.3200**; run peak 0.387 |
| Phase 3 — SIL toolchain construction | No promoted comparable multi-instruction policy result | — | — | — | — |
| Phase 4 — before Cycle 1 SFT | Three-family evaluation at registered caps | 0.0800 | — | 0.5000 caught | 0.2305 caught |
| Phase 4 — Cycle 1 after retention SFT | Three-family evaluation at registered caps | **0.3110** | — | **0.5230** caught | **0.2765** caught |
| Phase 4 — placement RL historical peak | Caught-placement validation; 0.633 overall | — | — | **0.7910** | **0.4420** |
| Phase 5 — Cycle 2 four-family policy | Cycle evaluation at registered caps | **0.4316** | **0.1738** | **0.3463** caught | **0.1702** caught |
| Phase 5 — retained placement RL checkpoint `step_2754052` | Caught-placement validation; 0.6211 overall | — | — | **0.7982** | **0.4073**; separate bowl peak 0.4655 |
| Phase 5 — Cycle 3 four-family policy | Three-round `sil_record` evaluation | **0.4779** (734/1536) | **0.1191** (183/1536) | **0.7383** caught (632/856); 0.0046 composed (10/2168) | **0.5353** caught (364/680); 0.0114 composed (22/1928) |
| Phase 6 — composed SFT seed `sft_phase6` | Three-round `sil_record`; explicit caught and composed legs | **0.4798** (737/1536) | **0.1491** (229/1536) | 0.7150 caught (612/856); **0.0935 composed** (80/856) | 0.4794 caught (326/680); **0.0265 composed** (18/680) |
| Phase 6 — composition RL peak `step_4257133` | In-run validation, 80–90% caught starts; 0.6240 overall | — | — | **0.7571** mixed/caught-dominated | **0.4634** mixed/caught-dominated |

These rows are a chronology, not a single leaderboard. Dedicated validation,
bank-harvest evaluation, caught placement, composed placement, and mixed
caught-fraction validation use different reset distributions. Comparisons
should be made within a row or between rows with the same stated protocol.

The most important scientific result is not a single maximum score. It is that one adapter can carry non-zero competence in all four instruction families after repeated RL-induced forgetting, and that balanced self-imitation can recover multiple old skills from a durable bank. Cycle 3 contains three campaign-best results on its own evaluation protocol—move-to, plate, and bowl—while preserving non-zero pick-up.

The most important unresolved behavior is the **sawtooth**: family-specific RL strengthens the active family and erodes inactive ones; retention SFT rebuilds the inactive families only partially. The current loop works, but it alternates between peaks rather than holding every family at its dedicated-policy maximum simultaneously.

**Phase 7 removes the sawtooth's cause and exposes a different one.** A single
sparse binary reward is instruction-agnostic, so all four families train in one
GRPO run with one return stream — no alternation, and therefore no sawtooth to
compensate for. It produced the campaign's best composed pick-and-place and
closed the grasp gap to the scripted oracle. But the retention SFT that the
Phase 4/5 loop depends on is now measured to destroy ~72% of that composed
capability, at every demonstration mix tested. The loop's two halves have
become incompatible for this task: RL builds composition and SFT removes it.

The remaining loss is localised, and it turned out not to be a drop at all.
The policy grasps at 97% of the oracle's rate and places accurately once it
releases; what looked like losing the object mid-carry was
`wrong_place_settled` terminating episodes in which the object was **already
resting correctly inside the receptacle** and the gripper had not finished
opening — a terminal condition that tested a conjunct of success rather than
the placement it is named for. Fixing it takes composed plate **0.5000 → 0.6272** and
composed bowl **0.2159 → 0.2841** on a protocol-matched repeat, with `move_to`
moving +0.010 as a control, and it moves the binding constraint to the
horizon. See the 2026-09-07 entry in §14;
the earlier claim that closing this transition would reach **0.7622** was a
bound built on the assumption that every such episode would complete its
release, and about a third do.

---

## 2. Evidence base and chronology

### 2.1 Source reports

| Local file | Created | Last modified | Role in this report |
|---|---:|---:|---|
| [`CDPR_SMOLVLA_CAMPAIGN_REPORT.md`](CDPR_SMOLVLA_CAMPAIGN_REPORT.md) | 2026-08-05 11:02 | 2026-08-09 21:45 | Phase 0 reaching and Phase 1 grasp/lift foundations |
| [`CDPR_PLACEMENT_PHASE2_PREFLIGHT_REPORT.md`](CDPR_PLACEMENT_PHASE2_PREFLIGHT_REPORT.md) | 2026-08-13 20:10 | 2026-08-13 20:10 | Placement geometry corrections, localization ladder, 15M-step placement result |
| [`CDPR_PHASE3_SIL_REPORT.md`](CDPR_PHASE3_SIL_REPORT.md) | 2026-08-17 09:49 | 2026-08-17 09:49 | Recording, replay, smoothing, dataset construction, residual SFT, and forgetting evidence |
| [`CDPR_PHASE4_LOOP_DESIGN.md`](CDPR_PHASE4_LOOP_DESIGN.md) | 2026-08-17 21:33 | 2026-08-19 20:12 | Design and implementation of the alternating RL/SFT loop |
| [`CDPR_MOVE_TO_VALIDATION_REPORT.md`](CDPR_MOVE_TO_VALIDATION_REPORT.md) | 2026-08-21 09:46 | 2026-08-21 09:50 | Controlled six-leg move-to validation |
| [`CDPR_PHASE4_RETENTION_REPORT.md`](CDPR_PHASE4_RETENTION_REPORT.md) | 2026-08-24 21:45 | 2026-08-26 08:01 | Retention bank, Cycle 1, placement iteration 2, and preservation rules |
| [`CDPR_PHASE5_REPORT.md`](CDPR_PHASE5_REPORT.md) | 2026-08-28 22:17 | 2026-08-31 09:27 | Pick-up integration, Cycle 2, later placement runs, Cycle 3, and current result table |

The four reports originally identified for consolidation are preserved in this chain. The later Phase 5 report is included because it directly explains and quantifies the three supplied Downloads folders and supersedes the Phase 4 stopping point.

### 2.2 Supplied folders and retained archive

| Artifact | Local arrival/creation | Contents | Verified meaning |
|---|---:|---|---|
| `/Users/damirnurtdinov/Downloads/dataset_videos_28062026/` | 2026-08-29 09:06 | 40 MP4 + manifest; 666 represented decisions | Ten successful clips per instruction from the four-family bank used around Cycle 2 |
| `/Users/damirnurtdinov/Downloads/sft_cycle2/` | 2026-08-28 22:16 | `sil_sft_adapter.pt` + `sft_report.json`, 19 MiB total | Locally available four-family Cycle 2 adapter and its training report |
| `/Users/damirnurtdinov/Downloads/videos_put_into_cycle3/` | 2026-08-31 09:39 | 20 MP4 + manifest; 228 represented decisions | Ten bowl and ten plate examples from the refreshed placement slice used for Cycle 3 |
| `local_results/cdpr_phase4_retention_archive/` | 2026-08-26 08:17 | 82 files, 159 MiB | Three retained model files, Cycle 1 SFT report, complete move-to evaluation with 30 videos, project docs, and reports |

All entries in `local_results/cdpr_phase4_retention_archive/CHECKSUMS.sha256` pass SHA-256 verification as of this review.

Important local checksums:

| Artifact | SHA-256 |
|---|---|
| Cycle 2 adapter | `9024a9170d49d4183d805420d7c6eb0e97418022985f5dbee2798cfce617fc7a` |
| Cycle 2 SFT report | `de6eb206b3c74460a990bab19afdf37e1af225755f763151fc917b806ab0e741` |
| Four-family video manifest | `b28bd8db1c3e8d0069c640b6c2d13f8aa87d1ef1124ca9d8c1903efcf5449299` |
| Cycle 3 placement video manifest | `9cc77c9fc2a2a1ba3cf37e6f88d173a512036c4eb8ba9ff1824b7dab45a02c2c` |

### 2.3 Compact campaign timeline

| Date / phase | Retained progress |
|---|---|
| Phase 0 | `move_to_object` established the frozen-prior + residual + GRPO control stack and later reached 63.0% controlled validation at cap 0.19 |
| Phase 1 | `pick_up` was decomposed into approach/grasp and lift; the policy was shown able to descend, close, and lift, while object localization—not the plant or horizon—set the main approach limit |
| 2026-08-13, Phase 2 | RL alone learned plate and bowl placement; placement tolerance was shown compatible with the encoder's 3–5 cm localization regime |
| 2026-08-17, Phase 3 | The SIL recorder/replayer/dataset/SFT toolchain was built; moving-average smoothing and residual SFT were validated |
| 2026-08-21 | Move-to received a controlled 6,656-episode, six-leg validation with a proper curriculum-restored metric leg |
| 2026-08-24 to 26, Phase 4 | A three-family retention bank rebuilt move-to without sacrificing placement; placement RL reached its Phase 4 peak |
| 2026-08-28, Phase 5 Cycle 2 | Pick-up joined the single-policy bank; a 4.3× larger balanced slice broke the earlier retention ceiling |
| 2026-08-30 to 31, Cycle 3 | Refreshed placement data produced the strongest reported multi-instruction adapter: three family bests and non-zero pick-up |
| 2026-08-31, Phase 6 preparation | The project added a receptacle-present grasp scene and deterministic whole-episode relabelling to provide the missing `put_into_*` grasp prefix |

---

## 3. The current technical idea

### 3.1 Policy stack

The deployed controller combines:

1. **SmolVLA prior** — produces an eight-slot, five-action chunk from language, two cameras, and robot state.
2. **Residual actor** — observes a 518-dimensional state (6 proprioceptive values + a 512-dimensional pooled vision feature) and the flattened 40-value prior chunk, then predicts a correction through a 558 → 1024 → 1024 → 40 MLP.
3. **Bounded composition** — final action is `tanh(prior + residual_scale × residual)`, currently with `residual_scale = 1.0`.
4. **Chunk execution** — the model emits eight action slots but `replan_every = 4`; only slots 0–3 are executed and supervised.
5. **Current adapter policy** — vision-tower RL training is off in the active Phase 4/5 configs. Retention SFT trains the residual actor; the VLA LoRA is copied from the source checkpoint and is not changed when the LoRA stage fails to improve validation.

This architecture is intentionally asymmetric. The residual can learn quickly from banked action targets, but the 512-dimensional pooled vision feature is computed behind `no_grad` through a fixed `flat_random` projection. VLA adaptation can affect action through the prior, but it does not make the residual's visual input end-to-end trainable.

### 3.2 Training loop

The current loop is:

```text
family-specific GRPO
        ↓ successful trajectories
record executed actions + observations
        ↓
moving-average smoothing + same-checkpoint replay
        ↓
durable frames/actions bank
        ↓
rebalance by decisions per instruction
        ↓
refresh state and prior under the current checkpoint
        ↓
residual SFT with whole-episode train/validation split
        ↓
evaluate every instruction at its registered cap and seeds
        ↓
next family-specific GRPO turn
```

The loop is not joint RL. GRPO optimizes one family at a time because earlier mixed-task runs showed that merely rehearsing another instruction did not prevent forgetting. Cross-family competence is restored by a balanced, explicit bank rather than assumed to survive shared updates.

### 3.3 Retention bank contract

The durable bank stores what remains true when the network changes:

- rendered frames;
- executed actions;
- episode identity, instruction text, cap/source group, and success-truncated trajectory structure.

The bank does **not** treat recorded `prior` or vision-bearing `state` as permanent. Both depend on the adapter that produced them. Before each SFT pass, `sil_refresh_priors.py` recomputes the current SmolVLA prior and current 512-dimensional vision feature from the stored frames and instruction text.

The pooled `demonstrations.npz` is therefore a derived artifact. It must be rebuilt when the bank changes so instruction quotas and checkpoint-dependent values remain current.

---

## 4. Achievements by instruction family

### 4.1 `move_to_object`: instruction-conditioned reaching

The dedicated reference checkpoint at `step_11009573` is the cleanest standalone RL result:

- **0.6299 success** on the training distribution: 645/1024 episodes.
- 128 independent reset scenes; confidence interval clustered by reset: **±0.0731**.
- Start-distance cap restored to **0.19 m**, the top of the earned ladder.
- Success requires reaching within **2 cm XY** of the named object while the gripper is in the required height band.
- Independent replication on another scene seed: **0.5947**.
- Performance is 0.746 below 0.12 m, 0.506 at 0.15–0.18 m, 0.227 at 0.21–0.25 m, and 0 beyond 0.30 m.
- The first action points toward the target on successes, supporting aiming rather than random search.
- Language grounding is real but partial: when the named object is not the nearest object, success is 0.272.
- The wrist camera is load-bearing for localization in the measured regime.

This validates the base research claim: RL without demonstrations can learn language-conditioned reaching on the CDPR from a pretrained VLA prior plus a trainable residual.

### 4.2 `pick_up`: grasp and lift

The project established a reliable decomposition:

- **Approach/grasp** needs approximately 2 cm object localization for a high grasp rate.
- **Lift** needs a sustained positive z command; the measured plant crosses from ineffective to reliable between approximately `a_z = 0.20` and `0.30`.
- A perfect-object-XY oracle raises ever-grasped from roughly 0.49 to **0.92**, showing that localization—not the plant, detector, or horizon—is the binding approach variable.
- The dedicated Phase 1 source used for the bank has a normalized reference level of **0.260** at cap 0.06; a direct six-round re-harvest reported a mean of 0.279.
- SFT can install a non-zero grasp into a policy that has none: the seed pass moved pick-up from 0 to **0.0964** without reducing the other three families.
- Cycle 2 reached **0.1738**, or **66.8%** of the 0.260 reference level.
- Cycle 3 retains **0.1191** after placement's RL turn.

The remaining limitation is structural: one residual controls both descent and post-grasp lift. Pick-up RL develops an increasing upward bias that eventually harms the descent needed to re-grasp. This is an open architecture/control issue, not evidence that grasping was never learned.

**Superseded 2026-09-07: the bottleneck is the lift, not the grasp.** Phase 5's
conclusion — that `pick_up` fails at the grasp, 222 grasps in 102k worlds, and
should be seeded by SFT — described a checkpoint whose grasp rate has since
moved. On `step_2017690` the deterministic grasp rate is **0.4116 at cap 0.10
and 0.2835 at 0.13**, while `lift|grasp` is **0.1185 and 0.0323**. The lift is
the binding constraint by a factor of eight, and effort still aimed at the
grasp is aimed at a bottleneck that moved. See §7.13 for the mechanism: the
policy commands +0.02 mean `a_z` after grasping under `pick_up` and +0.40 under
`put_into`, on the same adapter.

**Current qualification, 2026-09-09:** the z-offset pilot does not support
treating lift as the only remaining bottleneck. Pick-up's reported grasp
frequency fell **0.5488 → 0.4665**, while its median held-step mean z command
was essentially unchanged (**+0.275 → +0.269**). The probe's conditional
lift rose **0.3556 → 0.4183**, but implies about 64 lifted episodes in each
arm, whereas production reports 91 and 84 successes. These use different
height references and grasp definitions; reconcile them before constructing
a success funnel. With the observed grasp outcomes fixed, perfect downstream
execution still cannot reach 70%. The adopted curriculum must therefore
progress back through grasp acquisition to ordinary approach, as well as
teaching lift. This updates the current diagnosis without changing the older
checkpoint-specific measurements above.

### 4.3 `put_into_plate` and `put_into_bowl`: placement

Phase 2 answered the original feasibility question positively. With no demonstrations and no behavior cloning, a frozen SmolVLA prior plus residual learned:

| Instruction | Start | Final at 15M | Peak in that run |
|---|---:|---:|---:|
| `put_into_plate` | 0.000 | **0.710** | 0.778 at 12.0M |
| `put_into_bowl` | 0.000 | **0.320** | 0.387 at 14.3M |
| Overall | 0.059 | **0.360** | 0.375 |

The plate/bowl gap is consistent with geometry: the success radii are approximately 0.091 m for plate and 0.057 m for bowl. A localization ladder showed graceful degradation at 3–5 cm error, unlike pick-up's much tighter grasp tolerance.

Later results strengthened the family:

- Phase 4 historical peak: **0.633 overall**, plate **0.791**, bowl **0.442**.
- Phase 5 retained RL checkpoint `step_2754052`: **0.6211 overall**, plate **0.7982**, bowl **0.4073** at that checkpoint; bowl separately peaked at **0.4655**.
- Cycle 3 single-policy harvest: plate **0.7383**, bowl **0.5353**.

Validation and `sil_record` harvest rates are different protocols and must not be plotted as one uninterrupted curve without a protocol marker.

Those historical placement episodes begin with the object already held, so
that bank teaches carry-and-release. Phase 6 in §8 adds an uncaught prefix.
The September 10 pilot uses uncaught but object-aligned container starts;
ordinary approach under the placement instruction remains untested there.

---

## 5. Self-imitation and retention achievements

### 5.1 Demonstration construction

The project built a production-aligned SIL path rather than a synthetic labeler:

- executed actions are captured at `backend.step`, so all four executed actions per replan are retained;
- production observations and success predicates are called through rather than reimplemented;
- trajectories stop at the first success, while the remainder of the final action chunk is masked;
- reset identity is tied to seed, rank, update, and round so replay begins from the correct scene;
- episode identifiers include source information so trajectories cannot collide across families or caps;
- train/validation splitting is by whole episode;
- a fixed `--seed-torch` is required because SmolVLA's sampled prior is otherwise stochastic.

Moving-average smoothing with window 5 became the retained default because it improved command smoothness while preserving successful replay better than the evaluated EMA and median alternatives. It is a dataset-construction operation, not a runtime filter.

### 5.2 Cycle 1: three-family retention

From a placement checkpoint that had nearly forgotten move-to, balanced retention SFT produced:

| Instruction | Before SFT | Cycle 1, 60 epochs |
|---|---:|---:|
| `move_to_object` @ 0.19 | 0.080 | **0.311** |
| `put_into_bowl` @ 0.20 | 0.2305 | **0.2765** |
| `put_into_plate` @ 0.20 | 0.500 | **0.523** |

Move-to improved by approximately 3.9× while both placement families also improved. This established that retention need not be paid for by reducing the new skill.

The retained Cycle 1 SFT report records:

- 1,352 episodes;
- 15,981 training rows and 1,520 validation rows;
- baseline validation MSE 0.049979 → best 0.017305 at epoch 58;
- 96.267% reachable target values;
- `headline_over_control = 1.024`, consistent with the expected image round-trip and batch-numeric floor.

### 5.3 Cycle 2: four families and the data-scaling result

Cycle 2 expanded to approximately 26,000 decisions per instruction, with pick-up as the binding slice. The supplied SFT artifact records:

- **6,839 episodes**;
- **92,071 train rows + 10,586 validation rows**;
- 518-dimensional state;
- baseline validation MSE **0.06971064**;
- best validation MSE **0.0230342** at epoch **38**;
- 91.398% reachable values;
- 5,437 seconds total wall time;
- residual actor trained; VLA LoRA copied but not updated;
- no LoRA epoch improved its baseline, so **no new LoRA was applied**.

Cycle 2 evaluation:

| Instruction | Success | Reference fraction where defined |
|---|---:|---:|
| `pick_up` @ 0.06 | **0.1738** | 66.8% of 0.260 |
| `move_to_object` @ 0.19 | **0.4316** | 67.3% of 0.641 |
| `put_into_plate` @ 0.20 | 0.3463 | Rebuilding after pick-up RL |
| `put_into_bowl` @ 0.20 | 0.1702 | Rebuilding after pick-up RL |

The main conclusion is causal and reusable: increasing the balanced data slice, not merely increasing epochs, broke the earlier retention ceiling. The larger dataset also produced a real validation minimum at epoch 38; later epochs overfit while the saved artifact remained at the best epoch.

### 5.4 Cycle 3: current reported single policy

Cycle 3 starts from the strongest Phase 5 placement RL checkpoint and refreshes placement demonstrations from that same checkpoint. Its three-round, 512-world-per-round `sil_record` evaluation reports:

| Instruction | Cycle 2 | Cycle 3 | Change |
|---|---:|---:|---:|
| `pick_up` @ 0.06 | 0.1738 | **0.1191** | −0.0547 |
| `move_to_object` @ 0.19 | 0.4316 | **0.4779** | +0.0463 |
| `put_into_plate` @ 0.20 | 0.3463 | **0.7383** | +0.3920 |
| `put_into_bowl` @ 0.20 | 0.1702 | **0.5353** | +0.3651 |

Raw Cycle 3 successes:

- pick-up: 183/1536;
- move-to: 734/1536;
- plate: 632/856;
- bowl: 364/680.

This is the latest reported all-family result. Move-to reaches 74.6% of its dedicated reference, plate and bowl set their strongest reported single-policy harvest rates, and pick-up remains non-zero after four million placement-RL steps have moved it away from its own optimum.

### 5.5 What the sawtooth means

Cycle 2 and Cycle 3 show the alternating loop from opposite sides:

- after pick-up RL, pick-up is relatively strong and placement must be rebuilt;
- after placement RL, placement is strong and pick-up must be rebuilt;
- move-to improves across the retention cycles without another dedicated RL turn.

The loop is therefore successful as a recovery mechanism but incomplete as a simultaneous optimum. “One policy for all instructions” is achieved in the practical sense of one adapter with measurable competence on all four tasks. It is not yet achieved in the stronger sense of one adapter matching every dedicated checkpoint at once.

---

## 6. Supplied dataset/video evidence

### 6.1 Four-family sample (`dataset_videos_28062026`)

| Instruction | Clips | Decisions represented | Caps represented | Prompts/objects represented |
|---|---:|---:|---|---|
| `move_to_object` | 10 | 205 | 0.19 | banana, bowl, orange, plate, potato, tomato |
| `pick_up` | 10 | 189 | 0.06 | orange, potato, tomato |
| `put_into_bowl` | 10 | 134 | 0.10, 0.15 | apple, orange, potato, tomato |
| `put_into_plate` | 10 | 138 | 0.10, 0.15, 0.20 | apple, orange, potato |

The videos are 640×240 dual views: overview camera on the left and wrist camera on the right. Representative frame inspection confirms that the supplied files are task-labelled successful trajectories with both views present. The manifest, rather than filenames alone, is the canonical link to episode UID, instruction text, cap/source group, and decision count.

### 6.2 Cycle 3 placement sample (`videos_put_into_cycle3`)

| Instruction | Clips | Decisions represented | Caps represented | Prompts/objects represented |
|---|---:|---:|---|---|
| `put_into_bowl` | 10 | 129 | 0.10, 0.15, 0.20 | apple, orange, potato, tomato |
| `put_into_plate` | 10 | 99 | 0.10, 0.15, 0.20 | apple, orange, tomato |

These are examples from the refreshed placement side of Cycle 3. They corroborate that the later bank spans both placement targets, multiple start-distance caps, and multiple object identities.

Two behavior observations from the video review remain relevant:

1. The policy frequently rotates the wrist camera toward a stable scene landmark/overview-camera direction. Because yaw is unconstrained for these instructions, this may be harmless or may stabilize the random-projected visual state. It is an observation-hacking hypothesis, not yet a measured result.
2. The learned placement behavior usually carries the object above the receptacle and releases it as a drop. The current success predicate accepts this terminal outcome. A release-height gate was implemented and shown capable of lowering the release, but the active config has the gate disabled because the tested setting greatly reduced success. Therefore the retained achievement is **successful put-into behavior under the current predicate**, not yet gentle physical placement.

---

## 7. Methodological and engineering achievements

### 7.1 Validation now measures the task actually trained

Curriculum state must be restored in validation. The move-to reference scores 0.630 with its earned 0.19 cap and 0.172 when the cap is omitted. This 3.7× difference established a general methodological rule: a held-out evaluator that ignores curriculum state evaluates a different reset distribution and can invert the conclusion.

### 7.2 Independent units are reset groups, not episodes

Eight GRPO candidates share a reset scene and start pose. Confidence intervals therefore cluster by reset group. Reporting all eight candidates as independent episodes would overstate statistical precision by roughly `sqrt(8)`.

### 7.3 Group-variance filtering is part of the active RL design

GRPO normalizes within-group advantage. Groups whose candidates have nearly identical reward can amplify rollout noise. `grpo_min_group_reward_std` masks those groups per return stream and is now present in the active task configs.

### 7.4 The bank is balanced in decisions, not episodes

Episode lengths differ strongly by instruction. Equal episode counts would give long-horizon instructions more gradient. `--rows-per-instruction` selects whole episodes until each family reaches an approximately equal decision budget, preserving episode-level train/validation separation.

### 7.5 Cross-checkpoint data are refreshed by inference, not physics

Replaying old actions in the simulator under a different checkpoint changes the closed loop and destroys trajectory survival. The current method runs stored frames through the current checkpoint to refresh prior and state without introducing reset, termination, or physics divergence.

### 7.6 Best-validation checkpointing prevents overfit from reaching artifacts

Both residual and optional LoRA stages compare against the untrained baseline and retain the best validation epoch. Cycle 2 demonstrates why this matters: the residual overfits after epoch 38, but the saved adapter remains at epoch 38; the LoRA stage never beats baseline and is not applied.

### 7.7 A curriculum-state check now travels with every evaluation

`--start-distance-cap` applies to every instruction in a config, but the
approach ladders are per instruction and end at different rungs. A composed
`put_into` evaluation at 0.20 is correct for the container families and
simultaneously scores `pick_up` — earned cap 0.080 — at a start distance it has
never trained at, where it reads 0.0000 against 0.1465 at its own 0.06.

This is §7.1 in a new place, and it cost two wrong conclusions before it was
found. `sil_record` now reads `extra_state["approach_curriculum"]` from the
checkpoint, compares it to the requested cap, warns before the rollout, and
writes a per-instruction verdict — `at_earned_cap`, `below_earned_cap`,
`above_earned_cap`, `unknown` — into `summary.json` as `cap_check`. The console
line is what gets missed; the JSON is what gets read later, so the caveat
travels with the number.

**A grasp-gated instruction's earned cap is an APPROACH cap, and its success
cap is a different number.** `_GRASP_GATED_INSTRUCTIONS` routes `pick_up`'s
ladder to `instruction_grasps_normal_start/`, not to
`instruction_successes_normal_start/`, because the approach curriculum measures
whether the policy can reach the object and gating the reach on the full lift
would stall it on a separate problem. So `pick_up`'s earned cap of 0.13 is the
distance at which it can reach and close, and its success there is **0.0091**:
0.1465 at 0.06, 0.0366 at 0.10, 0.0091 at 0.13, 0.0000 at 0.17. Reporting
0.0091 as "`pick_up` at its earned cap" is true and misleading in the same
breath, which is the exact failure this section exists to prevent. Measure the
success cap; never inherit it from the ladder.

### 7.8 Reset state and live state must not share a field

`BatchedReset.physical_grasp` is written by the grasp detector on every env
step and is additionally gated on `active_mask`. Two consumers read it after a
rollout and both silently got "was still running and still holding at the final
step" instead of "started holding": the recorder, which wrote it into every
recording's `physical_grasp_at_reset` column, and the collector's caught-start
mask, which made the uncaught-only approach gate a no-op that appeared to work.

The rule: a field describing the START must be snapshotted at reset or derived
from a per-step record. `_Recording.starts_grasped` uses `caught_target[0]`,
which is correct on every recording already written.

### 7.9 A curriculum ladder must not promote into a rung it cannot hold

Every promotion costs pass rate, because the next rung is harder — measured
over twelve promotions: median 0.091, p90 0.129, max 0.133. Against a 0.30
promote gate and a 0.20 demote gate, a family lands at 0.17-0.27 and parks:
too low to promote, too high to fall back. Zero of twelve promotions landed
back above their gate and three of four families ended parked.

The condition is on the level a family parks at, not the band width:
`promote - drop_p90 >= demote`. At 0.45/0.30 the next run took zero demotions,
held every family in or above the band, and rescued `pick_up` from five
consecutive validations at 0.0000.

### 7.10 Artifact integrity is now explicit

The Phase 4 archive has a checksum manifest, preserved model files, raw evaluation tables, manifests, logs, reports, and 30 evaluation videos. This is a substantial improvement over result-only reporting and should be continued for every promoted checkpoint.

### 7.11 A terminal condition must test what it is named for

`wrong_place_settled` ended a container episode on `~container_ok`, and
`container_ok` requires `released`. So an object carried into the receptacle
and set down while the gripper was still opening ended its own episode: the
surface takes the load, the pads unload, `~state.grasped` goes true,
`target_has_settled` is already true, and `released` is not true **yet**. The
median gap between the latch breaking and the termination was ZERO env steps.

This is §7.8's failure in a third place — a terminal condition sharing a
conjunct with success, firing because that conjunct is not satisfied yet — and
it is the more general rule. §7.8 was about a field describing the start being
read at the end; this is about a condition asking a question it was not named
to ask. An object settled inside the receptacle is not in the wrong PLACE,
whatever the gripper is doing.

The fix splits `container_ok` into `placement_geometry_ok` plus the
release-dependent terms and gives the terminal condition the geometry alone.
`container_ok` implies `placement_geometry_ok`, so termination strictly
narrows: no episode that used to run can start dying. That implication is the
property to assert when splitting any predicate this way, and
`tests/test_wrong_place_settled.py` asserts it over a grid rather than on a
chosen case.

It cost four hypotheses. Release height, horizon, grasp speed and more composed
demonstrations were all tested against the reading that the policy drops the
object mid-carry. That mechanism is 16% of plate's failures; the terminal
condition was 61%.

### 7.12 The horizon histogram tells you which task a run actually scored

Two evaluations of `step_2017690` at the same `--round-index`, `--worlds`,
`--start-distance-cap` and `--seed-torch` drew 808/1536 and 424/1536 long
horizons. That is not nondeterminism, and the seeded generator is not involved.
Horizons come from two places, and BOTH encode something about the task:

```
coupled       = curriculum_horizon_min + frac*(max-min)   # frac from the CAP
composed      = placement_grasp_horizon_min_decisions      # a FLOOR, 40 here
horizon_group = composed if the container start is uncaught else coupled
travel_group  = f(horizon_group * 4)                       # the START DISTANCE
```

So, per instruction, the histogram reads out the protocol:

- the **single** value on `move_to_object` and `pick_up` tracks the requested
  cap — 0.20 gives 25, 0.17 gives 23, 0.10 gives 20;
- the **split** on `put_into_*` is the caught/composed mix. Every episode at 40
  is a composed start; every episode at the coupled value began caught.

That is how a mislabelled evaluation was caught. The Phase 7 composed protocol
passes `--metadata-override placement_caught_object_fraction=0.0`; a later
series of evaluations omitted it, so roughly half of their container episodes
began with the object already held, and their rates were being read against
composed ones. The pooled histogram said 40/25 in both and looked like a
horizon difference; the PER-INSTRUCTION histogram said 128/128 at 40 against
72/128, which is the caught fraction and nothing else.

The rule: **print the horizon histogram per instruction before comparing any
two evaluations.**

```
np.unique(h[instruction_ids == i], return_counts=True)
```

A container split that is not 100% at the composed floor is not a composed
evaluation, whatever the directory is called. `sil_record --mode compare`
reports the pooled version as `same_horizons` in `reset_identity`; that check
was previously unreadable because `same_grasp_at_reset` beside it read
`physical_grasp_at_reset` — the §7.8 column — and therefore read False on every
honest comparison, training the reader to discount the whole block. It now uses
`starts_grasped` and prints each term separately.

### 7.13 A proxy gate that stops tracking its goal is invisible to its own thresholds

`pick_up`'s approach ladder promotes and demotes on the GRASP rate. That is a
deliberate and defensible proxy — the approach curriculum is about reaching the
object — and the gate was measured honest: at cap 0.13 the stored
`pass_rate_ema` is 0.2198 and the deterministic grasp rate is 0.2835, matching
in the direction exploration noise predicts. The three success-gated families
match too (`move_to` EMA 0.5874 against a measured 0.5775, bowl 0.2446 against
0.2756).

The failure is not in the gate. It is that the proxy stopped predicting the
goal and nothing in the ladder can say so:

```
cap 0.10   grasp 0.4116   lift|grasp 0.1185   success 0.0488
cap 0.13   grasp 0.2835   lift|grasp 0.0323   success 0.0091
cap 0.17   grasp 0.1707   lift|grasp 0.0000   success 0.0000
```

Grasping improved, the cap climbed on it, and success stayed at zero because
the lift never came. §7.9's promote and demote thresholds both read the grasp
rate, so no setting of them could have caught this — the ladder was working
exactly as designed while the instruction went nowhere.

The detector is cheap and should run beside every gate: **the gate metric
rising while the instruction's own success stays flat at zero.** That is the
same signature `_warn_on_structurally_dead_gate` already watches for in the
opposite direction — a gate reading zero while success is high — and it wants
the mirror case.

The cause here is measurable and specific. Over the steps the object is
actually held, the mean commanded `a_z` is:

```
pick_up          p10 -0.033   p50 +0.024   p90 +0.284   peak height 0.0027 m
put_into_plate   p10 +0.170   p50 +0.395   p90 +0.766   peak height 0.0896 m
put_into_bowl    p10 +0.083   p50 +0.257   p90 +0.442   peak height 0.0910 m
```

The same adapter, the same grasp detector, the same plant. It commands +0.37 to
+0.40 upward after grasping under `put_into` and lifts the object 9 cm; under
`pick_up` it commands +0.02 and lifts it 2.7 mm. Against the loaded plant's
dead zone of roughly 0.15-0.20, about 10% of `pick_up`'s grasped episodes clear
it — against a measured `lift|grasp` of 0.1185. The dead-zone model predicts
the lift rate.

So this is instruction conditioning, not the plant, not the grasp, and not
grasp quality. Under one sparse binary reward `pick_up` succeeds 0.003-0.05 of
the time, so its GRPO groups are near-degenerate and carry almost no advantage
— it has effectively not been trained while its ladder kept promoting it.

### 7.14 A spawn range smaller than the goal measures a different task

Composed `put_into_plate` reads 0.7018 and composed `put_into_bowl` 0.3778, and
the gap is not only manipulation difficulty. Measured over six 512-world
composed rounds, from recordings already on disk:

```
plate  radius 0.091   start xy p10 0.0134  p50 0.0667  p90 0.0898
       started ALREADY inside the radius   0.9259
       successes starting inside           0.9461   xy displacement p50 0.0430
bowl   radius 0.057   start xy p10 0.0136  p50 0.0809  p90 0.1011
       started ALREADY inside the radius   0.4271
       successes starting inside           0.8033   xy displacement p50 0.0389
```

**93% of composed plate episodes begin with the object inside the success
radius.** The family is therefore not measuring "bring the object to the
receptacle"; it is measuring approach, grasp, lift and release without ejecting
the object from a region it started in. `xy_ok|settle` of 0.9496 is that, and
`minimum_target_motion` is 0.0 in the dense branch, so no displacement is
required of the policy at all.

Two causes, and only one is the clamp:

- **plate: the configured range is smaller than the target.**
  `placement_grasp_object_min/max_distance` is 0.06-0.10 against a 0.091 m
  radius, so about 78% of the spawn range lies inside the goal before any
  clamp runs.
- **bowl: the clamp is the whole story.** Its range starts at 0.06, above the
  0.057 radius, so by construction nothing should start inside -- yet 42.7%
  does. The composed reset clamps the sampled position into the workspace
  bounds, and a receptacle near an edge has its object pulled back on top of
  it.

The tell is enrichment, not the base rate. On bowl, starting inside nearly
doubles success -- 42.7% of episodes against 80.3% of successes. On plate there
is nothing left to enrich, which is exactly what a free term looks like.

The rule: **check the start distribution against the success region before
reading a placement rate.** A spawn sampled inside the goal, or clamped into
it, turns a placement task into a retention task with the same name, the same
predicate and the same number. Neither the funnel in §8 nor `cap_check` can see
it; both describe what happened after the reset.

Fixing it means a minimum spawn distance above the larger receptacle radius and
a reset that resamples the direction rather than clamping the position. Both
change the task and require a new composed baseline. Historical numbers remain
valid for their recorded protocol; they do not establish performance under
the new outside-goal protocol.

### 7.15 Demonstration-guided GRPO after the z-offset pilot

**Adopted by the user on 2026-09-09; training integration pending.** Retain GRPO and
use demonstrations to select useful initial states for its current-policy
groups. The missing transitions are approached backwards: pick-up from
before lift to before grasp to ordinary approach; placement from before
release to short transport to grasp/lift and ordinary starts. Retain ordinary
starts and all four instructions throughout. Assisted success is a training
diagnostic, not achievement of the ordinary-start target.

The review of `356b497..5288a26` established these implementation constraints:

- `04929bd` adds z offset standard deviation 0.10 and enables
  `episode_offset_after_grasp`. The gate applies to the entire offset vector,
  including the existing gripper standard deviation 0.15. It therefore also
  removes pre-grasp gripper-offset exploration; this is not an isolated z
  ablation. In `mjwarp_rank_local_collector.py`, the gate tests
  `first_grasp_step >= 0`, so it remains enabled after grasp loss. It is an
  after-first-grasp gate, not a current-holding gate. Gating exploration also
  does not prevent shared policy updates from changing subsequent approach
  behavior. These confounds do not establish which caused the regression.
- `5288a26` measures lift from `object_xyz[0]`, a post-action pose, for every
  instruction. Production pick-up uses `initial_target_xyz` and requires
  `caught_target & (gripper_opening <= 0.94)`; the probe omits the opening
  check. Report recorded success and rescoring under both height references
  side by side. Held-step averages also select different episodes/states after
  training and cannot alone prove whether stochastic exploration engaged.
- `356b497` corrects the stale composed lift datum in extraction;
  `60757ee`/`b952704` provide per-world divergence attribution with whole-round
  fallback when attribution is unavailable; `fc019bf` uses object catalog
  labels instead of rejecting plate's different prompt template. `85f5c14`
  probes desk-height discrepancies, and `2e30a23` propagates cap reporting
  through sharding. These improve evidence and extraction; they do not yet
  implement a restorable demonstration state or a GRPO handoff.
- §7.14's start-distribution audit must carry into the new benchmark. Use
  genuine outside-goal starts for transport, enforce realized distances after
  workspace handling, and separately label legacy inside-goal scores.

For each transition group, restore a validated full simulator state or replay
and verify a prefix at a decision boundary, then generate eight fresh
current-policy suffixes from the same state. Recompute observations and
instruction-conditioned priors. Exclude every teacher-prefix action from
residual and action-expert LoRA losses. Preserve the desk lift reference and
remaining budget; never earn pick-up success by resetting already above its
lift threshold. A pick-up clip with a bowl/plate visible is a placement prefix,
not a completed placement demonstration.

Use training-only scenes for the demonstration bank; evaluation scenes used
for training lose their held-out status. Measure replay survival, ordinary
and assisted success separately, informative groups and actual gradient rows,
and count prefix replay in compute cost. Start with a transfer check against
ordinary GRPO from the same checkpoint; multi-million-step blocks are welcome
once ordinary-start manipulation improves with retention. This is a testable
training direction, not a guaranteed route to 70% or a return to broad residual
SFT, whose prior composition regression remains relevant.

The first implementation is `tools/audit/probe_cdpr_demonstration_handoff.py`,
launched by `scripts/run_cdpr_demo_handoff_probe.sh`. It resolves the donor and
config through the prototype's pilot manifest, verifies hashes, revalidates
the source episodes, and selects held decision boundaries before lift or
release. It replays full source rounds up to a COMMON boundary: MJWarp's
`active_mask` masks actions but does not freeze physics, so independently
stopping worlds at different times would let early handoffs drift. Verified
live states are broadcast within their original eight-candidate scene groups,
including backend/controller state and task/contact histories. Pick-up uses
the reconstructed pre-action desk datum and recomputed prompt; terminal
destination states are refused. The existing collector then writes only fresh
suffix records, including its LoRA capture path, with **zero optimizer updates**.
The probe reports prefix cost, replay errors, divergence, suffix success and
reward variation. This is a prototype on evaluation scenes under legacy
geometry, not an ordinary-start score or training launcher. The 19:58 GPU run
verified pickup/plate handoffs but saturated pickup. The 2026-09-10 10:11 run
moves pickup earlier and verifies two variable pickup groups plus one variable
bowl group. Five of seven pickup groups remain all-success; one bowl scene is
insufficient coverage. Review of the earlier run found LoRA capture selected
inactive worlds 0–127. The collector now selects whole positive-horizon groups;
the latest GPU reports confirm exact active-group capture and 24 nonzero
advantages. Ordinary all-active collection retains its previous indices and
cap. At the probe stage, actual optimization and transfer remained unverified.
The September 10 training pilot now completes optimization but its final
ordinary-evaluation scores decline; see the newest §14 entry. These container
starts are still object-aligned, and assisted gradient coverage cannot be
audited from the supplied progress log alone.

### 7.16 The three-stage `put_into` collection and SFT pipeline

**Implemented 2026-09-10; no GPU run yet.** `CDPR_THREE_STAGE_PUT_INTO_SFT_DESIGN.md`'s
change map is complete in the repository and unit-tested on CPU (1336 tests
pass; the three pre-existing local failures are unchanged). Nothing below is a
result about the policy. It is a description of what will produce one, and the
five design decisions inside it that are load-bearing.

| Artefact | What it does |
|---|---|
| `rl_vla_bootstrapping/simulation/cdpr_composition_scenes.py` | Generates and validates the full-task scene manifest; hash-derived disjoint splits |
| `mjwarp_rank_local_collector.FullTaskSceneResetter` | The explicit full-task reset route; one manifest scene per world, no curriculum, no forced EE alignment |
| `rl_vla_bootstrapping/policy/cdpr_staged_demonstrations.py` | Stage machine, yaw tail controller, teacher bank, the continuous rollout, the durable recording and its acceptance |
| `tools/audit/calibrate_cdpr_pickup_yaw.py` | Solves the fixed pickup yaw from the MJCF camera extrinsics; reports the workspace residual |
| `tools/audit/build_cdpr_composition_scenes.py` | Scene manifest CLI and audit |
| `tools/audit/select_cdpr_stage_teachers.py` | Sequential teacher screening on real upstream handoffs, scene-clustered intervals |
| `tools/audit/record_cdpr_staged_put_into.py` | Collection CLI; per-GPU shards, durable frames/actions/manifests, no optimizer |
| `tools/audit/build_cdpr_staged_sft_dataset.py` | Acceptance, row assembly, relabelling; writes the bank and its partial-pickup sibling |
| `tools/audit/sil_refresh_priors.py` | Extended: explicit frame ids, final-prompt check, complete-coverage requirement, clears the stale marker |
| `tools/audit/sil_sft.py` | Extended: scene-level split, stage/destination-balanced sampler in both stages, retention mixture, per-stage loss and reachability, stale-prior refusal |
| `tools/audit/evaluate_cdpr_full_put_into.py` | One student prompt from step 0, no stage machine and no servo; native and strict verdicts side by side |
| `configs/examples/cdpr_smolvla_three_stage_put_into.yaml` | The declared contract; predicate geometry unchanged from phase 7 |
| `scripts/run_cdpr_three_stage_{collection,sft}.sh` | Separate remote entry points, resumable by step |

**The pickup yaw is 0.000 rad, and the fixed choice costs 10.7° on average.**
Solved from the loaded model's camera extrinsics rather than guessed: the wrist
camera hangs at `(0, 0.05, 0.045)` under the yaw joint with a fixed −15° tilt,
so its horizontal bearing is `yaw − π/2`, and the overview camera sits at
`(0, −0.5412, 0.5125)`. At the desk centre those coincide at yaw 0 to
4×10⁻¹³ degrees. The same fixed angle over a 7×7 grid of the ±0.19 m workspace
leaves mean 10.7°, p90 19.8°, max 25.3° of bearing error, with 18.4% of the
grid inside a 5° band. **That is the measured price of the user-confirmed
fixed-yaw mode**, and it is stated so a later per-position facing mode has a
number to beat rather than an intuition.

**A release in progress is not a dropped object — again.** The chain's carry-loss
condition initially read "in placement and not holding it", which is §7.11 in a
new place: `physical_grasp` includes `~release_open`, so it goes False several
env steps before `container_ok` can latch, and at `action_step_gripper` 0.05 the
opening needs ~11 steps to cross the threshold. That rule would have terminated
every correct placement on the step the policy began letting go. A carry loss
now requires the hand to still be CLOSED. The unit test that pins this is named
after the failure.

**Stage observers are separately allocated.** `evaluate_active_sparse_tasks`
writes `ever_grasped`, `grasped`, `step_count`, `peak_lift` and
`release_clearance` in place. The chain runs three: a move-to and a pick_up
observer for the stage events, and the PERSISTENT placement observer that is
the reset's own task state and runs from env step zero. A shared
`ever_grasped` would have had the pickup evaluator rewriting the history
`container_ok` depends on.

**Balancing changes exposure, never length.** A carry is ~25 decisions against
a pickup's ~9, so a natural pass gives the carry three times the gradient from
duration alone. The sampler draws destination, then stage, then object
uniformly, with replacement; no episode is truncated and no action is invented,
and an empty stratum is an error rather than a silent substitution. The report
carries optimizer updates and sampled supervised actions because "epoch" is
ambiguous under replacement.

**The bank is marked unusable until its priors are refreshed.** Relabelling
rewrites the instruction on every row of a chain, including its move-to prefix,
but `state` and `prior` were computed under the teachers' prompts and adapters.
The dataset builder writes `priors_stale: true` and `sil_sft.py` refuses the
bank until `sil_refresh_priors.py` has cleared it. This is a mechanical guard
because nothing about the resulting loss curve would say otherwise.

**First teacher screen: 0 of 64 for every candidate, and the cause was the
harness.** Run 2026-09-10 on the `teacher_selection` split, 64 scenes, two
move-to candidates, one pickup, two placement. Every candidate of every role
scored `reached = 0`, with the conditional yields undefined because their
denominator was zero. Two move-to teachers that score 324/400 and 645/1024 on
their own protocols do not both go to exactly zero, so this was read as a
harness fault and not a teacher result.

The reach event is a conjunction — the production XY predicate, an open
gripper, no grasp, and a height band — and the height band was an absolute
[0.20, 0.34] m invented in the harness. The grasp point is
`object_z + pick_grasp_height_offset`, which is **0.185–0.192 m** for apple,
orange, potato and tomato, and the pickup teacher's own aligned start — the
pose its curriculum trains it to begin from — is one centimetre above that, so
**0.195–0.202 m**. The absolute floor therefore sat on top of the correct
answer, and below it for three of the four objects. Nothing in the reward
pushed the reach up to compensate: under `sparse_binary_reward` the move-to
reward's `z_penalty_weight` is zeroed and `distance_include_z` is off, so the
approach has no Z term at all and settles wherever its warm start puts it.

Two changes followed. Readiness is now expressed **relative to the grasp
point** (−0.005 to +0.12 m) with absolute rails at the controller floor and
ceiling, which is a rule derived from the pickup teacher's own reset rather
than invented. And `StagedRound.reach_diagnostics()` now decomposes the
conjunction on every round — how many worlds the raw predicate fired on, how
many passed readiness, the closest XY approach, the height above the grasp
point, and the share of predicate steps rejected by each individual gate.

**Second screen, same day: the teachers reach, and they arrive with the hand
closed.** With the diagnostic in place the same protocol reported the reach
predicate firing on **29–35 of 64 worlds** (closest-approach XY median
0.0145–0.0215 m against a 0.02 m window), height above the grasp point
comfortably inside the band (median 0.059–0.068 m), the absolute rails never
touched — and `gripper_closed` at **1.000 of 1075, 1276 and 1041 predicate
steps** for the three candidates. `premature_grasp_before_handoff` fired on
7–16 chains of 64, so the hand does not merely close, it grasps.

This is not a broken teacher. Under `sparse_binary_reward` the move-to reward
is `where(success, 1.0, 0.0)` with **no gripper term at all**, so that channel
is entirely unconstrained for `move_to`, and this is a shared four-instruction
policy whose `pick_up` and `put_into` experience is all about closing. Nothing
ever asked it to keep the hand open, and it does not. A closed hand cannot be
handed to the pickup teacher, whose aligned start is an open gripper
bracketing the object and which was never trained to open first.

So the approach now gets the same treatment the wrist does: a bounded,
recorded, **single-channel gripper hold** that can only open and can never
squeeze, with the raw teacher command stored beside the applied one and
`action_source` marking every substituted action. It is switchable off
(`--no-gripper-hold-before-pickup`) so the ablation exists. Like the yaw tail,
it makes these policy-plus-controller demonstrations, and that provenance
travels with the bank.

**Third screen: the chain runs, and the bottleneck is the grasp.** With the
gripper held open the chain produces its first end-to-end numbers on the
`teacher_selection` split, 64 scenes:

| stage | `step_3416645` | `step_11009573` |
|---|---|---|
| reach predicate met | 13/64 | **21/64** |
| pickup-ready | 12 | 21 |
| aligned (primary) | 9/64 = 0.141 | **15/64 = 0.234** |
| align given reach | 0.75 (12) | 0.714 (21) |

`step_11009573`, the dedicated long-distance reaching reference, leads — but
the two intervals overlap and the tool says so; neither is separated at 64
scenes. Downstream, with move-to fixed: **16 chains aligned, 1 picked up
(1/16), 0 placed.** Full-chain acceptance 0/64.

Two things this changes. First, holding the hand open is **not free**: the same
`step_3416645` went from 29/64 reaches to 13/64 once the hold was applied, and
its closest-approach XY median moved 0.0215 → 0.0619 m. `gripper_opening` is a
column of the residual's state vector and the fingers are in the wrist camera,
so the hold changes the observation the reach is conditioned on. The
intervention that made the handoff possible measurably degraded the approach,
and both halves belong in the record.

Second, the handoff pose is not the pickup teacher's trained pose. Median
height above the grasp point at the handoff is **0.0946 m**; the pickup
teacher's own aligned start is **0.01 m**. Whether that gap is what costs
15 of 16 chains is not yet established — and guessing a gate has already been
wrong once and right once in this sequence, so the next step is measurement,
not another intervention.

`StagedRound` now carries `pickup_diagnostics()` and
`placement_diagnostics()` alongside the reach one, reporting the ladders
(entered → descended → grasped → lifted; entered → geometry → released →
placed) with the distances underneath them. The pickup table includes
`mean_action_z_while_grasped`, which reproduces the retained measurement that
the same adapter commands **+0.40** mean `a_z` while holding under a
`put_into` prompt and **+0.02** under a `pick_up` one — so a lift failure can
be attributed to the prompt or exonerated in one read. `--pickup-prompt
destination` exists as the corresponding screening arm.

A flaw the third screen exposed: `reach_success` is the move-to predicate
evaluated on every step, so once a world is holding the object the predicate
keeps firing and the unscoped gate table attributed the pickup stage's
deliberate closure to the reach — `gripper_closed` 0.38 and `already_grasping`
0.15, both of them correct behaviour of a later stage. The reach table is now
scoped to the approach stages.

**Retained results:** at 0.06–0.10 m starts the reach predicate is met on
21/64 scenes by `step_11009573` and 13/64 by `step_3416645` with the gripper
hold on, and on 29–35/64 by `step_3416645` without it; alignment converts
71–75% of reaches; the grasp converts 1 of 16. **No teacher is ranked** —
neither move-to candidate is separated at this budget, and the pickup and
placement roles have not yet been compared on a chain that reaches them.

**What this pipeline still cannot tell you.** Whether any teacher triple
produces usable chains at a usable rate; whether the relabelled prefix is
inside the residual's bounded correction range under the final prompt; and
whether the student learns the alignment rather than needing the servo. The
first two are the first GPU steps; the third is the `--assisted-yaw` diagnostic
arm reported beside the unassisted headline.

---

## 8. Composing grasp with placement — measured

Phase 6 moved this from preparation to result. The composed task is
`put_into_*` with the object **on the desk**: approach, grasp, carry, release.

### 8.1 The relabelling route was measured and rejected

`cbd74a2` implemented the plan of recording grasps in a receptacle-bearing
scene and relabelling them onto `put_into_*`. The mechanism is sound — the
actions were executed by the plant, and instruction text is the only channel
the label travels down, so a relabelled episode is a real demonstration.

The **join** is what failed. A relabelled grasp ends wherever the object was;
a placement demonstration starts within its approach cap of the receptacle,
0.19–0.20 m. Measured over 347 grasp episodes in the Phase 6 scene:

| receptacle | median gripper-XY at grasp | within 0.20 m |
|---|---|---|
| nearest | 0.2471 m | 36.3% |
| farthest | 0.3967 m | 5.2% |

With targets split evenly between plate and bowl, roughly 20% of relabelled
episodes would end inside territory the bank demonstrates. At the measured
0.085 grasp rate in that scene, that is ~70 usable episodes. Not enough, and
biased toward scenes where the object happened to spawn near a receptacle.

**Retained conclusion:** relabelling is valid in principle and unusable here
because the *scene* geometry does not match the composed task's. The composed
task places the object 0.06–0.10 m from the receptacle
(`placement_grasp_object_min/max_distance`); a free grasp scene does not.

### 8.2 The seed comes from a scripted oracle

No policy in the campaign can seed the composed task. Under the identical
protocol, `sft_cycle3` scores plate **0.0046** (10/2168) and bowl **0.0114**
(22/1928).

`scripts/render_cdpr_task_reference_episodes.py` already defined the phase
chain and, for an ungrasped placement start, returns pick_up's approach and
close followed by the placement carry — eight phases. `tools/audit/sil_oracle.py`
drives it over a batch and `sil_record --mode oracle` records it in bank
format, with **only the five numbers handed to the plant** coming from the
oracle: states and priors still come from the loaded checkpoint, and the reset,
reward, grasp detector, horizon and success predicate are the trainer's own.

Oracle performance on the composed task, same reward and same predicate:

| | smoke (256 worlds) | pooled harvest at cap 0.20 (8192 worlds) |
|---|---|---|
| put_into_plate | 1.000 (160/160) | 0.909 (3890/4280) |
| put_into_bowl | 0.427 (41/96) | 0.455 (1781/3912) |

Twelve rounds at three caps; replay survival 0.998–1.000 unsmoothed.

### 8.3 Seed result — `sft_phase6`

Protocol: `sil_record --mode record`, 3 rounds × 512 worlds, caps as named.
"Composed" adds `placement_caught_object_fraction=0.0`, so every container
episode starts uncaught.

| instruction | protocol | `sft_cycle3` | **`sft_phase6`** | raw |
|---|---|---|---|---|
| put_into_plate | composed @0.20 | 0.0046 | **0.0935** | 80/856 |
| put_into_bowl | composed @0.20 | 0.0114 | **0.0265** | 18/680 |
| pick_up | @0.06 | 0.1191 | **0.1491** | 229/1536 |
| move_to_object | @0.19 | 0.4779 | 0.4798 | 737/1536 |
| put_into_plate | caught @0.20 | 0.7383 | 0.7150 | 612/856 |
| put_into_bowl | caught @0.20 | 0.5353 | 0.4794 | 326/680 |

Three things this supports:

* **The composed task is out of the noise.** Plate at 0.0935 puts a success in
  54% of GRPO groups of eight, which is the contrast `pick_up_iter0` lacked and
  almost exactly what the pick_up seed supplied (0.0964) before RL took its
  grasp rate to 0.621. Bowl at 0.0265 gives 19% and is thinner.
* **Cross-instruction transfer.** `pick_up` rose 25% with no pick_up data
  added: the composed demonstrations open with a grasp, and the motion
  transferred despite the prompt saying `put_into`.
* **Oracle actions are learnable by this residual.** `reachable` came in at
  0.91111, against 0.90494 and 0.91398 on policy demonstrations — the way this
  route could have failed quietly did not occur.

What it does **not** support: any claim of a working composed pick-and-place.
0.0935 is a seed, not a capability.

Cost: caught placement fell 2–6% relative, because composed and caught episodes
share the `put_into` quota. SFT `val_mse` was still falling at epoch 44 of 45,
so this mix wants more epochs than Cycle 2's turn at 38.

### 8.4 Composition RL — `phase6_compose_iter0`

222 updates, steps 2 763 249 → 5 010 151 (2.25 M new), from `sft_phase6` with
`cdpr_smolvla_phase5_compose_loop.yaml`.

**The curriculum annealed, which is the first time composition has been
trained at all.** `curriculum/placement_caught_fraction` went 1.0 → 0.90 at the
start → 0.80 at step 3 575 449, so 10–20% of container episodes now begin with
the object on the desk.

| step | overall | plate | bowl |
|---|---|---|---|
| 3 006 030 | 0.5371 | 0.6696 | 0.3772 |
| 4 002 765 | 0.5840 | 0.7518 | 0.3815 |
| **4 257 133** | **0.6240** | **0.7571** | **0.4634** |
| 4 750 563 | 0.4941 | 0.6054 | 0.3599 |
| 5 010 151 | 0.5254 | 0.6429 | 0.3836 |

**Protocol caveat, and it is the important one:** this validation runs against
the same resetter the training uses, whose caught fraction was 0.9–0.8 for the
whole run. It is therefore **dominated by caught starts and is not a composed-task
measurement**. The composed numbers in this report come only from §8.3's
explicit `placement_caught_object_fraction=0.0` evaluations. The peak of 0.6240
does exceed Phase 5 iter3's 0.6211 under a comparable caught-dominated
protocol, but the two runs differ in the caught fraction and the comparison is
therefore indicative rather than clean.

The run peaked at step 4 257 133 and the two following readings fell below it,
which is the campaign's stop rule. Supporting signals: `placement_caught_success_ema`
fell 0.5448 → 0.3590 against a 0.30 restore threshold, and the approach EMAs
fell to 0.433 (plate) and 0.284 (bowl) — bowl now below its 0.30 promote gate.
Entropy and `log_std` were flat throughout, so this is the cost of a harder
task rather than a collapse.

### 8.5 What remains

1. Evaluate `phase6_compose_iter0`'s peak checkpoint under the **composed**
   protocol. Nothing yet measures whether the RL improved on the 0.0935 seed.
2. Bowl is the weak family everywhere — 0.0265 composed, 0.284 approach EMA,
   0.455 oracle. Its release height (0.10 m above the reference against a
   0.042 m resting clearance) drops objects into a concave target.
3. The caught fraction has two annealing steps of eight. Reaching the 0.25 floor
   at 40 updates per step needs roughly 2.4 M further steps.

---

## 9. Glossary of current campaign terms

### System and policy

**CDPR** — Cable-Driven Parallel Robot. A robot whose end-effector is positioned by coordinated cable actuation; this project controls a simulated 5-DoF CDPR in MuJoCo Warp.

**SmolVLA** — The pretrained vision-language-action model providing a language- and image-conditioned action chunk before task-specific correction.

**VLA prior / prior chunk** — The eight-by-five action proposal emitted by SmolVLA. “Prior” means the residual is trained around this proposal rather than producing every action from scratch.

**Residual actor / residual policy** — The 1.66M-parameter MLP that corrects the VLA prior. It is the main trainable and retainable policy component in the current loop.

**Action chunk** — A planned sequence of actions. The model emits eight slots; the current controller executes four before replanning.

**`replan_every`** — Number of action slots executed before querying the policy again; currently four.

**Action-expert LoRA / `vla_lora`** — Low-rank adapter on SmolVLA's action-producing path. Current retention checkpoints carry it forward from the source checkpoint. Cycle 2 did not apply a new LoRA because no epoch improved validation.

**Vision-tower LoRA** — Low-rank adaptation inside the visual encoder. It is disabled in the current Phase 4/5 RL configs and should not be described as an active source of current gains.

**`flat_random` vision pooling** — Fixed random projection that compresses connector tokens to the 512-dimensional vision block used by the residual.

**Reachable action set / reachable fraction** — Because the final action is `tanh(prior + residual)`, some targets cannot be reached for a fixed prior. `reachable_fraction` reports the share of supervised action values that lie inside the residual's representable interval.

### RL and curriculum

**GRPO** — Group Relative Policy Optimization. Multiple policy candidates share a reset; their relative rewards define the policy update.

**GRPO group** — The eight candidate trajectories sharing one scene/start. It is the correct independent unit for grouped evaluation uncertainty.

**Group reward standard deviation filter** — `grpo_min_group_reward_std`; removes groups whose candidates do not differ enough to carry a useful learning signal.

**Curriculum cap / start-distance cap** — Maximum sampled distance between the controlled body and its task goal for normal-start episodes. Every reported success rate should name its cap.

**Ladder** — Ordered set of curriculum caps, from easier/nearer to harder/farther starts.

**Promote/demote gate** — Success or grasp-rate thresholds that advance or retreat the current ladder rung.

**Pass-rate EMA** — Exponential moving average used by the gate so one noisy update does not decide curriculum state.

**Promote dwell / cooldown** — Required persistence and minimum update spacing around a cap change.

**Normal start** — An episode that begins from the task's ordinary approach distribution and counts toward the approach/curriculum gate.

**Pre-grasped / caught start** — An easier training stage where the object begins between the gripper fingers. Placement currently uses caught starts; pick-up uses some pre-grasped groups to train lift.

**Caught-stage curriculum** — Existing mechanism that can anneal placement from caught to uncaught starts. It is implemented and checkpointed but has not yet produced a composed-task result.

**Split credit at grasp** — Separate return streams for approach/grasp and post-grasp behavior so a lift reward does not directly rewrite the approach in the same way as a monolithic return.

**`peak_lift` / terminal lift credit** — Maximum object rise achieved while grasped, ratcheted through the episode so a successful lift is not erased from the return if the object later settles.

**Episode offset / marginal scoring** — A per-episode action-mean perturbation held across time to explore sustained control biases, scored under the marginal action distribution so the perturbation contributes a valid policy gradient.

### Measurement

**Validation** — Trainer-aligned evaluation using the restored curriculum state and fixed protocol. It is distinct from a bank harvest.

**Harvest rate** — Success observed while recording bank data at specified caps, rounds, and seeds. It is useful as the pre-SFT baseline and data-yield measurement but is not automatically interchangeable with held-out validation.

**`policy_target_cosine` / `cos@d0`** — Alignment between the policy's first XY command and the direction to the task goal.

**`residual_target_cosine`** — Same alignment measured on the residual correction alone.

**`aim`** — Command-to-goal cosine minus a null formed by permuting commands across rows. It tests whether commands contain goal-conditioned direction beyond their marginal bias.

**`direction_concentration`** — Directional consistency of actions. It is interpreted only together with `aim`, because a genuine servo can also be highly concentrated.

**Success radius / tolerance** — Geometric threshold for task completion: approximately 0.02 m for move-to/grasp alignment, 0.057 m for bowl placement, and 0.091 m for plate placement.

**Release clearance** — Object height above its resting surface at the moment the gripper releases. It distinguishes lowering into a receptacle from dropping from above.

**Clustered confidence interval** — Interval computed using reset groups as independent units rather than treating all within-group candidates as independent.

### SIL and retention

**SIL / self-imitation learning** — Supervised training on successful trajectories generated by the policy itself or a retained source policy.

**Record** — Run the policy and save the actions actually executed, observations, task state, and success timing.

**Replay** — Re-run recorded actions from an identical seeded reset to test survival after smoothing. In the retained pipeline, physics replay uses the same checkpoint that produced the data.

**Survival** — Fraction of successful recorded episodes that remain successful after action transformation and replay.

**Moving-average smoothing (`w5`)** — Centered five-decision filter used offline to reduce action jitter while retaining successful trajectories.

**First-success truncation** — Keep a trajectory only through the decision that first achieves success; mask the dead remainder of that final chunk.

**Retention bank** — Durable collection of successful frames, executed actions, instruction metadata, and episode identities across task families.

**Frame join key** — Canonical episode identifier used to match demonstration rows to stored frame arrays.

**Resolved fraction** — Share of demonstration rows successfully matched to frames. The refresh step refuses a partially resolved bank below a configured threshold.

**Prior/state refresh** — Recompute checkpoint-dependent SmolVLA priors and residual vision state from durable frames immediately before SFT.

**Decision quota / `rows_per_instruction`** — Balances families by action decisions while keeping complete episodes.

**Slice** — The decisions assigned to one instruction inside a balanced SFT dataset.

**Consolidation** — SFT on demonstrations from a skill the source checkpoint still performs; typically easier than rebuilding.

**Rebuild** — SFT recovery of a skill that intervening RL has mostly erased from the source checkpoint.

**Gap closure** — Candidate success divided by a dedicated/source reference, or the recovered portion of the difference from the starting policy to that reference. The denominator must always be stated.

**Stable fixed point** — Repeated SFT cycles return a forgotten skill to approximately the same level after different amounts of RL erosion.

**Sawtooth** — Alternation in which the family most recently trained by RL is strongest while older families are partially rebuilt by SFT, then roles reverse on the next family turn.

**`headline_over_control`** — Integrity ratio comparing recomputed vision-state differences to a control difference caused by uint8 image round-trip and batch-size numerics. A value near 1 indicates consistency with the expected numeric floor.

**Residual-only checkpoint** — An SFT result in which the residual actor improved but no new LoRA epoch beat the untrained LoRA baseline; the source LoRA is copied unchanged.

### Current composition terms

**Grasp prefix** — The missing beginning of an uncaught `put_into_*` episode: approach the object, close the gripper, and establish the carry state.

**Instruction relabelling** — Reuse a physically valid grasp episode under a `put_into_plate` or `put_into_bowl` prompt, provided the target receptacle is visible in the scene. Only the instruction ID/text changes; actions remain real executed actions.

**Semantic scene correctness** — A relabelled prompt must name a receptacle actually present in the stored frames. Phase 6 forces plate and bowl into each grasp scene for this reason.

**Join gap** — Geometric gap between the final state of the grasp prefix and the start distribution covered by placement data. Measuring it is the precondition for claiming the two skills compose.

### Historical labels still encountered in the reports

**Phase 0–6** — Campaign stages, not software versions: Phase 0 reaching; Phase 1 pick-up; Phase 2 placement; Phase 3 self-imitation; Phase 4 alternating retention; Phase 5 four-family retention and renewed placement; Phase 6 grasp-prefix composition.

**Preflight** — A short, falsifiable check performed before committing to a long GPU training run. Preflights test reward reachability, reset geometry, action reachability, camera framing, or curriculum behavior.

**M0, M1, ... / P0, P1, ...** — Local measurement and preflight identifiers inside a particular phase report. They are not global metric names: for example, Phase 2's M1 is the placement-oracle arm, while Phase 4's M1 is the realized start-distance check. Always interpret them in the report where they appear.

**F3, F4, F6** — Finding identifiers inherited from the placement consistency audit. They label historical reset/contact/release failures, not current metrics. Their corrected preflight zeros are excluded from the achievement table.

**Oracle arm** — Diagnostic controller that replaces only a selected policy component with ground-truth control, such as true target XY. It measures a ceiling or isolates a bottleneck; it is not a trained-policy result.

**Localization ladder / oracle-error ladder** — Family of oracle arms with a fixed synthetic target-position error per episode. It converts centimeters of localization error into task success and established the contrast between pick-up and placement tolerances.

**Reverse sampling** — Phase 4's initial name for obtaining demonstrations from easier states or curriculum rungs. The active implementation did not require a separate reverse-sampling engine: it harvests successful episodes across the existing cap ladder, smooths/replays them, and banks the result.

**Reverse-Frontier shell / LCHOL** — Earlier reverse-curriculum and safety/exploration machinery on a legacy non-batched path. It is not connected to the current MJWarp SIL/retention loop and is not part of the active achievement claim.

---

## 10. Results and concepts intentionally excluded from the achievement record

The following should not be reused as current headline results:

- The z-offset pilot as a demonstrated pick-up improvement, its conditional
  probe lift as the production success funnel, or its gate as proof that
  approach/grasp behavior is unchanged. See §7.15 and the latest §14 entry.
- Demonstration-guided GRPO as an achieved result. It is the adopted next
  implementation direction; some handoffs and usable records are verified, but
  optimizer integration and ordinary-start learning gains remain pending.
- Preflight oracle zeros produced before placement reward/reset geometry was repaired.
- Validation numbers produced without restoring curriculum state.
- Unseeded single-round comparisons that treated SmolVLA's stochastic prior as deterministic.
- Phase 3 placement-only SFT as a retained single-policy solution; it improved near-cap bowl but erased pick-up because pick-up was absent from the mix.
- Hindsight relabelling of **caught-start** placement into pick-up; the required
  lift predicate was never reached because those episodes start already lifted.
  **Scoped to caught starts, and reopened for composed ones.** The composed
  stratum did not exist when this was written: a composed `put_into` episode
  starts on the desk, grasps, and lifts to a median 0.0896 m, past
  `pick_lift_success_height` of 0.05. Measured on three composed evaluations of
  `step_2017690`, **485-493 of 808** container episodes (0.60) grasp from the
  desk and clear the lift predicate, and about two thirds of those go on to
  place successfully. That is a 0.60 yield of valid `pick_up` prefixes against
  0.1465 from `pick_up`'s own rollouts at cap 0.06. Any use of them must
  reckon with the 2026-09-07 SFT result below. **Updated 2026-09-09:** verified
  prefixes are adopted for a demonstration-start GRPO curriculum (§7.15),
  with fresh policy suffixes and no teacher actions in the GRPO loss. Broad
  relabel-and-re-SFT remains unadopted; no new curriculum gain is claimed yet.
- Cross-checkpoint simulator replay as a way to refresh priors/state; it destroys trajectory survival and has been replaced by frame inference.
- Vision-tower LoRA as an active contributor to the current policy; it is disabled in active RL and no Cycle 2 LoRA epoch beat baseline.
- Phase 5 placement `iter4` and `iter5` as promoted checkpoints. Both are superseded by `step_2754052`; the active release-height gate is off and the attempted ladder extension was reverted.
- “Gentle placement” as an achieved behavior. The current success is predominantly carry-and-drop under the accepted terminal predicate.
- Composed pick-and-place as achieved. Only the missing-prefix data path is implemented.
- Legacy LCHOL-based relabelling on the MJWarp path; that implementation is not connected to the active batched trainer.

- Composed `put_into_plate` rates as evidence that the policy moves an object
  to a receptacle, and the **0.7018** figure as the 70% target reached on the
  hard protocol. 92.6% of those episodes begin with the object already inside
  the 0.091 m success radius, and 94.6% of the successes do; the median object
  displacement in a successful placement is 0.0430 m inside a 0.091 m region it
  never left. The plate spawn range (0.06-0.10) is smaller than the plate
  radius, and the workspace clamp pulls a further share of both families back
  onto their receptacle. The number is real for the task as configured -- grasp
  0.9211, and about 5% of episodes still eject the object -- but it is not a
  placement result and must not be quoted as one, nor compared against bowl,
  which starts inside only 42.7% of the time and is being asked the harder
  question. See §7.14.
- The reading of the composed `put_into` loss as the policy dropping the object
  mid-carry, and the **0.7622** plate figure derived from it. 61% of plate's
  `no_release` grasp losses were `wrong_place_settled` firing on an object
  already at rest inside the plate, a median of zero env steps after the latch
  broke; the fix at `1b78cbc` is worth +0.053, not +0.138. The four hypotheses
  tested against the drop reading — release height, horizon, grasp speed, more
  composed demonstrations — were aimed at a mechanism that accounts for 16% of
  plate's failures.
- Any comparison between evaluations of one checkpoint whose PER-INSTRUCTION
  horizon histograms differ. On `put_into_*` that split is the caught/composed
  mix — composed starts take the 40-decision floor, caught starts take the
  curriculum-coupled value — so a run that omits
  `--metadata-override placement_caught_object_fraction=0.0` is scoring a
  half-caught task and cannot be read against a composed number. See §7.12.
- The claim that the retention bank is ~98% composed by decision and that the
  composed fraction cannot be swept. That came from `physical_grasp_at_reset`,
  which stored the FINAL grasp state rather than the reset one; the bank is
  roughly balanced (plate 51% composed, bowl 40%) and the fraction spans
  [0, 1]. Fixed at `da8b834`; derive the stratum from `caught_target[0]`, which
  is correct on recordings already written.
- `pick_up` scored at a cap it never earned. A composed evaluation forces
  `--start-distance-cap 0.20` on every instruction, and `pick_up`'s ladder
  ended at 0.080; it read 0/328 there and 0.1465 at its own 0.06. Two separate
  conclusions were drawn from the zero before the cause was found. `sil_record`
  now emits a `cap_check` verdict per instruction into `summary.json`.
- More composed demonstrations as a route to better composition. Measured flat:
  0.5 and 0.8 composed slices differ by 0.021 on composed bowl, inside the
  ~0.04 noise floor, while 0.8 costs caught plate 0.075.

These exclusions do not erase the engineering lessons that produced active fixes. They prevent a superseded measurement or abandoned branch from appearing in the presentation as a current result.

---

## 11. Preservation status and recovery priorities

### 11.1 Present locally

- Phase 4 placement checkpoints `step_1504301` and `step_7753190`.
- Cycle 1 retention adapter and SFT report.
- Complete move-to evaluation outputs, including 30 MP4s.
- Cycle 2 adapter and SFT report in Downloads.
- Four-family and Cycle 3 placement video samples with manifests.
- Reports, current code, configs, and tests.

### 11.2 Still missing from the consolidated local evidence set

1. The actual move-to reference adapter at `step_11009573`; only its complete evaluation is archived.
2. The durable Phase 4/5 bank: all `replay_*.npz` and `frames_*.npz` files.
3. Run logs, resolved configs, validation summaries, trainer state, and source control identifiers for every promoted checkpoint.
4. The Phase 5 `step_2754052` adapter.
5. The Cycle 3 `sft_cycle3/sil_sft_adapter.pt` and its full `sft_report.json`.
6. Full Cycle 2 and Cycle 3 evaluation directories rather than only report tables/video samples.

Until items 4–5 are copied and checksummed, the latest reported results are not fully recoverable from this workstation even though their narrative and video evidence are present.

---

## 12. Presentation-ready conclusions

1. **RL without demonstrations learned language-conditioned CDPR reaching to 63.0% at a 2 cm tolerance.**
2. **RL without demonstrations learned placement from zero: 71% plate and 32% bowl at 15M steps.**
3. **Task tolerance determines whether the VLA's 3–5 cm localization is sufficient:** it is too coarse for a ~2 cm grasp but useful for 5.7–9.1 cm placement tolerances.
4. **Successful RL trajectories can be converted into a durable self-imitation bank** with verified recording, smoothing, replay, frame joining, checkpoint refresh, and whole-episode balancing.
5. **Balanced retention SFT can rebuild forgotten skills without sacrificing the active family.** Cycle 1 rebuilt move-to by 3.9× while improving both placement tasks.
6. **Dataset size, not more epochs, broke the first retention ceiling.** A 4.3× larger per-family slice brought both move-to and pick-up to about 67% of their references.
7. **One adapter now performs all four instructions.** Cycle 3 reports 0.478 move-to, 0.119 pick-up, 0.738 plate, and 0.535 bowl.
8. **The remaining problem is retention amplitude, not basic feasibility.** RL and SFT form a measured sawtooth: the newest family peaks while older skills are partially rebuilt.
9. **Full composition is achieved and measured.** Composed `put_into_plate`
   reaches **0.500** and `put_into_bowl` **0.216** from a single policy that
   also performs `move_to_object` and `pick_up` — grasp, carry and release,
   with the object starting on the desk.
10. **One sparse binary reward replaces four dense ones and removes the
    sawtooth's cause.** All four instruction families train simultaneously in
    one GRPO run. This is the SimpleVLA-RL two-stage pattern: an SFT cold start
    followed by outcome-only RL, and it is worth 4.2x on composed plate and
    5.2x on composed bowl over the SFT seed it began from.
11. **The binding constraint has moved from perception to grip retention.** The
    policy's grasp rate is 97% of a ground-truth oracle's; what it cannot do is
    hold the object through the carry. 140 of 147 remaining plate failures drop
    it. Closing that alone reaches 0.762.
12. **Retention SFT and composition RL are now in conflict.** The bank that
    rebuilds forgotten skills removes 72% of the composed capability RL builds,
    independent of the demonstration mix. Any future loop has to reconcile
    these or keep them apart.

---

## 13. How to extend this report

Add each new promoted result to the top of §1 and append one ledger entry below. Do not overwrite a historical best; mark whether a result is a new best, current recommended checkpoint, diagnostic-only result, or superseded branch.

### Result-entry template

```markdown
### YYYY-MM-DD — <short result name>

- Git commit:
- Run/config:
- Source checkpoint and lineage:
- Candidate checkpoint:
- Training steps / updates / wall time:
- Evaluation protocol:
- Caps, seeds, rounds, worlds, and independent reset groups:
- Instruction results (successes / denominator and rate):
- Comparison baseline under the same protocol:
- SFT dataset rows/episodes by instruction, if applicable:
- Best validation epoch and overfit behavior, if applicable:
- What this result supports:
- What it does not support:
- Status: promoted / retained historical best / diagnostic only / superseded
- Local artifact path:
- SHA-256:
- Missing provenance:
```

### Rules for future updates

- Never compare validation and harvest rates without labelling the protocol.
- Always record the per-instruction cap.
- Report raw successes and denominators, not only rounded rates.
- Use reset groups—not candidate episodes—as the independent count for uncertainty.
- Distinguish compute spent across branches from training depth in one checkpoint lineage.
- Preserve frames/actions as the durable bank and treat pooled/refreshed datasets as derived.
- Checksum every promoted adapter, SFT report, manifest, resolved config, and evaluation summary.
- Record negative experiments in their phase report, but promote only retained conclusions and active fixes into this document.

---

## 14. Result ledger

Newest first. Entries follow the §13 template.

### 2026-09-10 — First demonstration-start GRPO pilot completes; full approach and transport remain unproven

**Status: retained negative experiment, no candidate promotion.** This
supersedes the earlier statements that the training launcher/integration did
not exist. It does not invalidate physically continuous trajectory composition,
which this pilot did not train by imitation.

- Run: `runs/demo_grpo_pilot_20260910_110040`, launched remotely with
  `CUDA_VISIBLE_DEVICES=0,1 MAX_TRAIN_STEPS=1000000 DEMO_ROUNDS=12 nohup bash scripts/run_cdpr_demo_grpo_pilot.sh`.
- Source checkpoint:
  `runs/release_recovery_continue_3m_20260908_102004/rl/step_3540208/smolvla_grpo_adapter.pt`,
  identified by the baseline shard summaries. Candidate:
  `runs/demo_grpo_pilot_20260910_110040/rl/step_1005877/smolvla_grpo_adapter.pt`.
  Neither adapter was supplied for checksum verification.
- Local implementation reviewed: `e56b7fb`; remote git revision and dirty
  changes are absent. The launcher selects a warmstart with a fresh optimizer.
  Its intended bank partition is rounds 1000–1011, versus evaluations 0–2;
  verify this against the missing remote manifest/bank before certifying it.
- Training: **1,005,877 selected actions, 91 updates, 3:56:43** on the training
  progress bar. Two A40 GPUs, 512 worlds/rank, groups of eight, residual GRPO
  and action-expert LoRA; vision tower not adapted. Source collection, replay
  and external evaluation cost are additional. Selected actions are not total
  simulator steps and exclude teacher replay work.
- What the code trains: one ordinary rollout and one assisted attempt per
  update/rank, with assisted pickup:bowl:pickup:plate scheduling. Earlier
  handoffs are mixed in after half the budget. Replay actions initialize
  state; only fresh suffixes enter the residual and LoRA GRPO losses. This is
  **not full-trajectory SIL or direct imitation of the pickup prefix**.
  There is no matched ordinary-only training control.
- Evaluation: three rounds × 512 worlds = 1536 episodes/arm, **192 reset
  groups**: 50 move-to, 41 pickup, 57 plate, 44 bowl. Candidates within a
  group share a scene and are not independent trials. Config path:
  `configs/examples/cdpr_smolvla_demo_grpo_pilot.yaml`; shard summaries report
  Torch seed 0 and nondeterministic kernels. Logged validation seed: 2,000,000.
  Saved caps: move-to 0.08, pickup 0.06, containers 0.20; requested cap is null,
  so `cap_check=unknown`, not failure. Horizons: 19, 18, 40, 40 policy decisions,
  four actions/decision. Container spawn distance uses the separate 0.06–0.10 m
  configuration; cap 0.20 does not mean a 20 cm transport task.

| Instruction | Baseline | Final | Change, percentage points |
|---|---:|---:|---:|
| `move_to_object` | 314/400 = 78.50% | 311/400 = 77.75% | −0.75 |
| `pick_up` | 83/328 = 25.30% | 68/328 = 20.73% | −4.57 |
| configured `put_into_plate` | 274/456 = 60.09% | 263/456 = 57.68% | −2.41 |
| configured `put_into_bowl` | 115/352 = 32.67% | 90/352 = 25.57% | −7.10 |

**Reset identity and task mismatch.** True pre-action object/EE poses, stored
initial target positions, support/rest heights, release thresholds, task text,
instruction/catalog/slot identities, horizons and round indices match exactly.
Orientations and velocities are absent: this certifies the recorded fields,
not complete simulator-state identity. First recorded caught flags are all
false, but are post-action rather than serialized pre-action grasp flags.

The inspected resetter explicitly places uncaught container EE at object XY;
the recordings confirm **zero EE–object XY distance in all 808 container
starts**. Even the outside-goal subset therefore omits the ordinary approach
required by the user. Existing "composed" here means uncaught from alignment.

| Geometry from true pre-action positions | Plate | Bowl |
|---|---:|---:|
| Inside XY success radius at reset | 424/456 = 92.98% | 144/352 = 40.91% |
| Outside-radius episodes / reset groups | 32 / 4 | 208 / 26 |
| Outside-radius baseline success | 12/32 = 37.50% | 15/208 = 7.21% |
| Outside-radius final success | 8/32 = 25.00% | 13/208 = 6.25% |

These descriptive subsets use the existing 0.091/0.057 m XY radii with no
extra transport margin. Four plate scenes do not support a broad placement
claim. Inside-XY starts still need grasp/release and the other success terms,
but can succeed with little transport. Neither subset alone certifies a
sustained held carry, and neither tests approach from away from the object.

**Failure diagnosis:** the existing placement decomposition reproduces all
808 verdicts per arm with **zero predicate disagreements**. Bowl grasp falls
**223/352 → 194/352**. Final bowl failures: 158 no-grasp, 68 no-release,
20 not-settled, 16 XY-miss. Final plate failures: 68 no-grasp, 91 no-release,
21 not-settled, 13 XY-miss. Pickup ever-held frequency falls **183/328 →
177/328**, while successful lifts fall **83 → 68**. Among held episodes this
is **45.36% → 38.42%**, a descriptive comparison of different selected episodes,
not an isolated causal effect. "No release" names a predicate stage;
ended-unheld alone does not prove the object fell during transport.

The additional lift trace uses true pre-action target height and opening
≤0.94. It is a descriptive measurement, not a replacement success predicate.
Raw scores retain divergent episodes to match the supplied summaries:
32 baseline and 39 final unique worlds, including 27 and 31 pickup worlds.
Exclude these from demonstration construction. EPA warnings and omitted
non-finite contact metrics occur in the log; they do not establish the cause
of the lower scores.

**In-run validation is separate.** Ordinary overall validation peaks at
0.5039 at step 300,654 and ends at 0.4473. The printed `COMPOSED` overall
score peaks at 0.4922 at step 600,253 and ends at 0.4297, with final plate/bowl
components 0.4803/0.2992. These are not the external final scores above.
Retain intermediate checkpoints, but do not promote one solely from these
differently sampled validation maxima.

**Next experiment for the user's concatenation idea:**

1. Define full-task resets with a separate EE approach distance and an object
   outside the destination's success radius by a declared margin. Validate
   realized distances after workspace handling; changing only the cap is
   insufficient. Keep training and held-out scene groups separate.
2. Execute a pickup teacher with the destination present, then continue a
   placement teacher from that **exact live state**, at a valid policy decision
   boundary. Preserve object/receptacle poses, velocities, contacts, controller
   state, task history and the desk-height reference. Keep the full-task
   observer alive so pickup success does not terminate the composed episode.
   Regenerate placement caches/priors at the prompt switch and allow enough
   remaining actions. Arbitrary archived clips cannot simply be appended;
   archived suffixes require replay from the actual pickup result and verified
   continuity and completion.
3. Keep continuous successful placements as full-task positive demonstrations.
   Store original executed observations/actions and stage provenance; relabel
   the whole accepted trajectory to its actual destination's `put_into` prompt.
   Failed placement after valid pickup provides pickup/partial-prefix data,
   never a placement success. Sequential teacher success measures data yield,
   not the final learner's ability.
4. Rebuild prompt-dependent inputs under that final instruction. The current
   residual controller needs refreshed VLA priors, residual features/targets
   and correct chunk masks; replacing text while retaining pickup priors is
   insufficient. Keep every relabelled view of one scene in one data split.
5. Test a bounded imitation or auxiliary-imitation arm with GRPO and matched
   controls. Evaluate the learner with `put_into` **throughout**, from empty
   ordinary starts, with no teacher switch or demonstration reset. Also
   measure retention. Historical broad residual SFT erased composed capability
   (September 7 entry); this motivates a small controlled experiment, not a
   conclusion that continuous relabelled demonstrations cannot work.

**Local evidence:**
[`attachment_analysis.json`](docs/artifacts/demo_grpo_pilot_20260910/attachment_analysis.json),
its NumPy-only [`analyze.py`](docs/artifacts/demo_grpo_pilot_20260910/analyze.py),
and baseline/final decomposition JSON/CSV under the same directory. The audit
cross-checks CSV/NPZ success against both supplied summaries, verifies recorded
reset identity and stores all input SHA-256 values. Source attachments remain
under `/Users/damirnurtdinov/Downloads/{baseline,final_eval}` and `train.log`.
Training-log SHA-256:
`f0458e80f8fd150a4e5ca0c39fa203079e4f7fb330639b5020dec5416d9deeb6`.
No remote training or simulator rollout was executed by this local review.

**Missing evidence before attributing the regression:** `pilot_manifest.json`,
`config_snapshot.yaml`, `tracked_changes.patch`, `training_bank.json`,
`pilot_comparison.json`, `rl/demonstration_rank*.jsonl` and training metrics /
TensorBoard events. The supplied progress log establishes completion, not
per-family assisted acceptance, variable groups, actual gradient coverage or
replay cost. The launcher normally bundles these under the run directory as
`review_logs.tar.gz`.

### 2026-09-10 — Earlier pickup and bowl handoffs produce usable GRPO records; active LoRA capture verified

- Evidence: attached `demo_handoff_probe_20260910_101110/`, revision
  `fbfeb041b2ab5cfa8fdf3c918e85c09515dbd500`, empty tracked patch, both arms
  exit 0. The nine uploaded files, SHA-256 inventory and checked aggregates
  are archived locally under
  `runs/analysis/demo_handoff_probe_20260910_101110_review/`.
- Donor remains continuation `step_3540208`, SHA-256
  `69bc284949d0837469d5396d789805f2967582e704ceedf2ab7db797003b09bc`.
  Config and prototype-manifest hashes match the preceding probe. Pickup uses
  round 0 / boundary backoff 2; placement uses round 1 / backoff 0 / bowl-only
  sources. Group size remains 8, pose tolerance 2 mm, opening tolerance 0.03,
  lift-datum sanity ceiling 2 cm. Saved donor exploration is still
  `[0,0,0,0,0.15]`, after-grasp gate false; this is not another z-offset ablation.

| Destination / prefix env steps | Planned / accepted groups | Assisted success | Variable groups | Usable residual rows | LoRA rows / nonzero advantages |
|---|---:|---:|---:|---:|---:|
| pick_up / 20 | 6 / 4 | 31/32 | 1 | 255 | 32 / 8 |
| pick_up / 24 | 5 / 3 | 21/24 | 1 | 457 | 24 / 8 |
| bowl / 76 | 2 / 1 | 1/8 | 1 | 197 | 8 / 8 |
| bowl / 44 | 1 / 0 | unmeasured | 0 | 0 (no collection) | 0 / 0 |

- **Pickup now has a nonzero training signal:** 52/56 = 0.9286 assisted
  success across seven distinct source scene groups; two of seven groups
  have reward variation, supplying 712 residual rows and 16 nonzero LoRA
  advantages. The variable groups are source world 179 / group 22 (7/8,
  std 0.3307) and world 511 / group 63 (5/8, std 0.4841). Both are orange
  lift prefixes from failed bowl placements. Thus the failed-placement prefix
  pool contributes useful pickup records in this audit. The other five groups
  remain 8/8 and supply no GRPO advantage. All accepted pickup source traces
  have no previous live-datum lift event; no destination-terminal groups were
  dropped. This does not measure unassisted pickup or establish a learning gain.
- **Bowl is now measured:** source world 498 / group 62 supplies 1/8 successes,
  reward std 0.3307, 197 residual rows and eight nonzero LoRA advantages.
  This is one source scene, not eight independent scenes. Its object begins
  0.1011 m from the bowl center and is 0.1037 m away at the recorded lift,
  outside the 0.057 m radius. It is therefore not an inside-goal source of the
  kind dominating the earlier plate audit. However, the teacher runs for
  76 environment steps before handoff; the report does not establish how much
  transport remains at that boundary. It is not an outside-goal ordinary-start
  success estimate or proof the suffix learned transport.
- **LoRA capture fix verified on GPU:** recorded world indices exactly equal
  all accepted groups' eight candidates in each collected batch. The three
  variable groups contribute 24 nonzero-advantage rows, compared with inactive
  capture in the preceding producer revision. These are eligible records;
  optimizer updates and teacher-prefix loss rows remain exactly zero.
- Replay rejects four of eleven pickup trials: one pose/opening/grasp/lost-hold
  mismatch, two grasp mismatches and one pose mismatch (2.104 mm). Bowl rejects
  two of three trials: 4.095 mm pose plus opening mismatch at prefix 76, and
  30.177 mm pose/opening/grasp/lost-hold mismatch at prefix 44. Keep these
  thresholds; a near-boundary rejection is not grounds to relax them.
- All collected suffixes report zero divergence events and no diverged worlds.
  The 76-step bowl prefix reports seven divergence events across the replayed
  source batch; other prefixes report zero. Aggregate collector metrics also
  report three non-finite EE worlds per collected batch. These are full-batch
  diagnostics, not proof that an accepted source was affected or that all
  source worlds were clean; retain per-episode provenance requirements.
- Suffix collection costs 243.6 / 236.3 s for pickup and 146.7 s for bowl,
  with SmolVLA inference accounting for about 96% of each duration. Total
  active suffix actions are 1,233 versus 184,320 padded records; 909 residual
  rows are usable. Sparse 512-world inference remains a scaling cost. These
  timings describe the audit, not expected training throughput.
- **Decision:** the requested targeted audit is complete: earlier pickup has
  reward variation, bowl has a valid variable group, and active LoRA capture
  works. Do not request another identical probe as the default next action.
  Proceed to implementing a bounded demonstration-start GRPO training pilot:
  collect separate training-only source scenes; integrate replay/validated
  starts into fresh residual and LoRA updates with teacher exclusion; preserve
  ordinary-start groups and per-family evaluation; count replay/inference cost.
  Start with the planned 0.5–1M selected-action transfer check, then extend
  only when one shared policy improves unassisted manipulation with retention.
  Before-grasp boundaries and outside-goal carry stages are still required:
  a held-only lift curriculum cannot fix the ordinary-start grasp bottleneck.
- Validation of this review: recomputed group/candidate totals; checked exact
  LoRA index coverage and all zero-update/teacher-mask counters directly from
  JSON. Documentation-only update; no simulator or optimizer rerun locally.

### 2026-09-10 — Review of attached 19:58 handoffs: usable plate signal, saturated pick-up, no bowl suffixes

- Evidence: locally inspected attachment
  `/Users/damirnurtdinov/Downloads/demo_handoff_probe_20260909_195836/`.
  Run revision `793dde54ba7a2a08ef47b061e107c3b75ecaee7a`, empty tracked
  patch, both arms exit 0. This is a different run from the 19:43 results
  below. Archived files and SHA-256 inventory:
  `runs/analysis/demo_handoff_probe_20260909_195836_review/`.
- Donor: continuation `step_3540208/smolvla_grpo_adapter.pt`, SHA-256
  `69bc284949d0837469d5396d789805f2967582e704ceedf2ab7db797003b09bc`.
  Config hash `975e63f14fdb17a04b6fbe05a08b971dd43adee20bcc2460da8fdb58e1638c66`,
  prototype manifest hash `89c407085c38cbc354dc2f96455a37cf61b49e20ca7fe1f16870fed89d3e1ee3`.
  Source round 0, group size 8, boundary backoff 0; replay tolerance 2 mm,
  opening tolerance 0.03, lift-datum sanity ceiling 0.02 m. Although the scene
  config is named `zlift_offset`, suffix exploration uses the DONOR's saved
  settings: offset std `[0,0,0,0,0.15]`, after-grasp gate false. No z-offset
  experiment or weight update took place here.

| Destination / prefix env steps | Planned groups | Accepted groups | Assisted suffix success | Groups with reward variation | Usable residual rows |
|---|---:|---:|---:|---:|---:|
| pick_up / 40 | 6 | 1 | 8/8 | 0 | 0 |
| pick_up / 28 | 5 | 2 | 16/16 | 0 | 0 |
| plate / 44 | 6 (including a bowl candidate) | 3 | 23/24 | 1 | 218 |
| plate / 60 | 5 | 3 | 17/24 | 3 | 1,096 |

- Pick-up: three accepted group/boundary trials from **two** distinct source
  scene groups (22 and 42), all sourced from bowl trajectories. Group 22
  appears at both boundaries with different candidate demonstrations. All
  rewards equal 1, group std equals 0, and usable rows equal 0. At prefix 40,
  every candidate succeeds after one new env action (8 actions total); prefix
  28 uses 34 active actions across 16 candidates. The teacher has already done
  nearly all the work at these starts. This verifies continuation plumbing;
  it is not a learned 100% pick-up policy. Move handoffs earlier to seek
  nondegenerate current-policy outcomes.
- Placement: **40/48 = 0.8333**, six accepted group/boundary trials from
  **five** distinct scene groups (17, 24, 46, 60, 61). Group 60 is repeated.
  All six are **plate**, with group successes 8, 8, 7, 5, 6, 6. Four groups
  have reward std 0.331–0.484 and supply 1,314 masked-in residual rows. These
  are usable records, not measured gradients or completed optimizer updates.
  Every accepted plate source starts inside the 0.091 m XY radius (distances
  0.0069–0.0751 m), so this is release/retention evidence under legacy geometry,
  not proof of outside-goal transport. The planned bowl world 479 fails the
  replay pose check (5.10 mm), leaving **no bowl suffix measurement**.
- Across 22 planned group/boundary trials, 9 collect suffixes. Pick-up drops
  four already-terminal groups, three grasp-history mismatches and one pose
  mismatch. Placement drops five trials for pose and/or grasp mismatch.
  Accepted source traces remain inside the 2 mm pose bound with exact grasp
  history agreement. There are two contained prefix divergence events in the
  60-step placement batch, none in the other prefixes, and zero suffix
  divergence events. Do not describe the entire run as divergence-free.
- **Commit review:** `99098c7` correctly rejects a missing preflight file;
  `f6edc13` separates first-step Z motion from replay error and removes a
  placement gate on an unused lift Z datum; `793dde5` correctly drops terminal
  groups individually and adds earlier-boundary selection. The first-step
  delta is measurable, but its cause cannot be certified as settling alone
  from old NPZs. The 2 cm ceiling is a sanity check, not reset-identity proof.
  Zero horizons suppress actions/rewards, not simulator physics.
- **New capture issue found in review:** each batch reports 128 LoRA rows,
  while the collector at this revision always captures worlds 0–127. All
  accepted groups are later (minimum accepted source world 138), so none of
  those captured rows is an active handoff. This follows from the recorded
  accepted indices and the exact producer code; raw LoRA tensors were not
  attached. Fixed by selecting complete positive-horizon groups, retaining
  ordinary all-active behavior, and exposing captured indices plus nonzero
  advantage counts. GPU verification of this fix is pending.
- Cost: suffix collection takes 209.1 / 229.3 s for pickup and 202.1 / 173.8 s
  for placement. The collector still iterates the allocated horizon with
  512-world forwards after most/all candidates terminate: e.g. 8 active
  actions produce 61,440 padded rows. This audit is not a training-throughput
  benchmark. Stop empty batches and avoid inactive-world inference when
  building the training integration.
- Next probe: back off pickup by two validated boundaries on source round 0;
  request bowl-only placement sources on round 1. New per-arm launcher
  controls prevent changing the already-useful plate handoffs merely to move
  pickup earlier. Keep tolerances fixed. Acceptance requires reward variation
  and active LoRA coverage, not merely high assisted success. Training-only
  scenes, ordinary-start controls and outside-goal transport remain required
  before a manipulation learning claim. Validation: **149 CPU tests passed**,
  including 15 handoff/capture tests; shell syntax and whitespace checks pass.

### 2026-09-09 — First verified handoffs: placement collects and saturates, pick_up is terminal at its own planned boundary

- Evidence: user-supplied `runs/demo_handoff_probe_20260909_194319` at
  `git_head.txt` = `f6edc13`, empty `tracked_changes.patch`. Placement exit 0,
  pick_up exit 2. `optimizer_updates` 0 and `teacher_prefix_loss_rows` 0 in
  every batch; no promotion.
- **Placement collected fresh suffixes for the first time.** Prefix 44: 6
  planned, 4 clean groups, successes 8/8, 8/8, 8/8, 7/8. Prefix 60: 5 planned,
  2 clean groups, 7/8 and 5/8. Assisted suffix success **43/48 = 0.896**.
  Rewards are binary, so `reward_std` is 0 or ≈0.331/0.484.
- **Only 3 of those 6 groups can produce a gradient.** Three groups are 8/8
  with `reward_std` exactly 0, and `min_group_reward_std` is 0.05, so they are
  filtered. Gradient-contributing rows were 213 of 59 392 record rows at
  prefix 44 and 640 of 51 200 at prefix 60; `selected_environment_actions`
  51 and 111 against `sampled_environment_actions` 485 and 640, i.e.
  `trajectory_work_amplification` 9.51 and 5.77. Assisted success of 0.896 is
  not an ordinary-start number and does not count toward the 70% objective.
- Cost, for the budget line in §"Measurement and budget": suffix wall 202.1 s
  and 173.8 s, of which SmolVLA inference is 193.8 s and 167.0 s. Physics is
  6.5 s and 5.6 s. The handoff is inference-bound, not replay-bound.
- **pick_up returned `destination_task_already_terminal` on both batches, and
  the cause is a datum disagreement between planning and admission.** The
  extractor cuts pick_up at a success step rescored against the recording's
  first POST-action pose; the probe admits against the live pre-action reset
  pose. For w259 those are 0.19387 m and 0.18456 m. Lift is `z - datum`, so
  the lower live datum reaches the 0.05 m threshold about 9.3 mm of travel
  earlier, which can fall before the planned boundary. The landmark is late by
  construction, not by accident.
- It was a partial failure reported as a total one. At prefix 40, 5 of 6
  candidates were admitted and 3 of those groups (46, 57, 60) were already
  terminal; groups 22 and 39 were not. At prefix 28, 4 admitted and 2 terminal
  (24, 61). The batch aborted on the first terminal world and discarded the
  usable groups with it.
- Changes: already-terminal destinations are now dropped **per group** — the
  group's horizon goes to zero, which is the same inert state an unplanned
  group already has, and the batch proceeds — with `destination_terminal_groups`
  and a per-episode `destination_task_already_terminal` reason in the report.
  A batch returns that status only when nothing survives. Each episode also
  reports `live_datum_lift_env_step`, the replay step where the relabelled
  pick_up predicate first fires against the live datum, and a dropped batch
  reports `earliest_dropped_live_datum_lift_step`.
- New knob, matching plan step 4's "move the handoff earlier":
  `--boundary-backoff` / `BOUNDARY_BACKOFF` steps every episode back through
  its OWN validated boundary list, so a backed-off handoff is still held,
  active and unterminated. It never invents a boundary and clamps at the
  first one.
- **Replay is not bit-reproducible and grasp history flips at the margin.**
  Between the 19:34 and 19:43 runs, at identical source, seed and prefix, w374
  went from `grasp_mismatch` true to false while w259 went false to true, and
  w479's `object_m` moved 0.001985 → 0.002046. `_apply_determinism` runs with
  `deterministic_kernels=False`, so MJWarp contact ordering is free to differ.
  Roughly 1 in 6 candidates is decided near the tolerance and admission is
  therefore partly a coin flip. This does not change any success number above,
  but candidate sets should not be treated as reproducible.
- Status: local change only, 11 handoff tests and 17 extractor tests pass on
  CPU. Placement has a verified GPU handoff; **pick_up still has none**, and no
  demonstration-guided learning gain is claimed for either.
- Missing provenance: a pick_up rerun that clears the terminal groups; an
  ordinary-start control for the same checkpoint; training-only scenes.

### 2026-09-09 — First GPU handoff probe: replay is sound, the lift-datum gate rejected all 22 candidates

- Evidence: user-supplied `summary.json` from
  `runs/demo_handoff_probe_20260909_193440`. Both arms ran on GPU and both
  returned exit status 2, `no_verified_handoffs` on all four batches:
  pick_up at 40 env steps (6 planned) and 28 (5), placement at 44 (6) and
  60 (5). Zero suffixes were collected, so there is still **no** measured
  assisted-start success, reward variation or transfer number.
- Replay fidelity was in fact good. Over the 22 candidates the whole-prefix
  maxima were object ≤ 3.41 mm (median ≈ 0.9 mm), end-effector ≤ 1.02 mm and
  opening ≤ 0.0086; only 2 of 22 exceeded the 2 mm pose tolerance and 4 of 22
  disagreed on grasp history. No divergence and no early termination.
- Every one of the 22 was rejected by `reset_vs_first_post_action_lift_datum`,
  with `baseline_error_m` from 0.21 mm to 9.99 mm. That comparison is between
  two different instants, not two estimates of one state: `sil_record` appends
  a row per **predicate** call, which runs after `_original_step`, so
  `object_xyz[0]` is the pose after the first env step while the probe's
  `initial_target` is the pose at reset. The gap is the object's settle during
  that step, and it was being judged by the 2 mm replay-drift tolerance.
- Three independent checks confirm the reading. (1) `baseline_error_m` is
  constant per source scene group and independent of prefix length and of
  task — group 46 gives 0.008206292986869812 in both the pick_up and the
  placement arm, group 39 gives 0.00705774/0.00705801/0.00705786 across three
  worlds — so it is a deterministic per-scene datum offset, not stochastic
  replay error. (2) At step 0 the replayed pose matches the recorded pose to
  1.8 mm while the reset pose differs from it by 3.2 mm, so the replay
  reproduces the settle. (3) The offsets are the same size as the discrepancy
  the extractor already reports for these legacy NPZs, which carry no
  pre-action snapshot.
- The gate was also redundant with its own remedy. `run_job` overwrites
  `reset.task_state.initial_target_positions` with the live pre-action
  `initial_target` before collection, so the source's stale Z is not consumed
  afterwards. Only pick_up reads that Z at all: it enters `target_lift` and
  `pick_success` in `evaluate_active_sparse_tasks`, while container success
  reads `target_motion_xy` from the datum's **XY**, which the probe never
  overwrites. The placement arm was therefore being rejected on a quantity its
  success predicate does not use.
- Change, not a loosened threshold: the 2 mm figure is retained for replay
  drift, and the settle now has its own `--lift-datum-tolerance-m`, default
  0.02 m and rejected by argparse at or above the 0.05 m lift success height
  so a datum gap can never be most of an earned lift. It gates `pick_up` only.
  The report now also carries `reset_lift_datum_z_m` and
  `recorded_first_post_action_z_m` so the settle stays visible per episode.
- Protections that make this safe are unchanged and independent of the source
  datum: `peak_lift` is seeded from the live datum so an already-raised object
  cannot re-earn its lift, `prelifted` is set, and the admission re-score on a
  deep-copied task state still returns `destination_task_already_terminal`.
- Expected effect on this same source round, from the recorded rejection
  reasons alone: 17 of 22 candidates become eligible — pick_up 5/6 at 40 steps
  and 5/5 at 28, placement 4/6 at 44 and 3/5 at 60. The remaining 5 are the
  genuine `replay_grasp_mismatch` (4) and `replay_pose_mismatch` (2, one
  overlapping) rejections, which stay rejections.
- Status: local change only, 10 handoff tests and 17 extractor tests pass on
  CPU. **No GPU handoff has yet been verified and no demonstration-guided
  learning gain is claimed.** The next rerun produces the first
  `clean_groups_with_reward_variation` reading.
- Missing provenance: GPU rerun output; the settle magnitude is inferred from
  the datum gap, not from a recorded pre-action snapshot, which legacy NPZs
  still lack.

### 2026-09-09 — Handoff launcher preflight import repaired; GPU result still pending

- Remote attempt: `runs/demo_handoff_probe_20260909_172122`. CPU planning
  found placement handoffs at 44 env steps (6 groups) and 60 env steps
  (5 groups), with 34 incomplete-placement candidates excluded. These are
  proposed replay boundaries, not verified handoffs or suffix outcomes.
- The launcher then stopped at `python -m unittest
  tests.test_cdpr_demonstration_handoff`: `tests/` has no `__init__.py`, so
  importing it as a package is environment-dependent. Neither GPU worker had
  started; no training or suffix collection occurred.
- Fix: directory-based unittest discovery, and a self-contained handoff test
  fixture that removes the second `tests.*` dependency. The dotted-import
  failure was reproduced locally with a conflicting installed `tests` package;
  the corrected discovery command passed all **10** handoff tests in that
  environment. Shell syntax and whitespace checks pass. Rerun the same launcher;
  the existing demonstration manifest is still its input.

### 2026-09-09 — Demonstration prototype yields 293 lift prefixes; handoff probe prepared

- Evidence: user-supplied extractor output from
  `runs/grpo_demo_prototype_20260909_170619/manifest.json`, using baseline
  recordings in `runs/release_recovery_pilot_20260909_130003/`. The remote
  manifest/NPZ files have not been imported locally; hashes remain in those
  artifacts. Status: **usable extraction prototype, no demonstration-guided
  learning result**.
- Per round 0 / 1 / 2: **80 / 110 / 103** accepted lift prefixes and
  **46 / 58 / 52** complete placements. Total **293** lift prefixes, including
  **156** complete placements and **137** lift prefixes from failed placements.
  Source instruction split: **76 bowl / 217 plate**. Complete placement clips
  overlap the lift-prefix episodes; these are not 449 distinct demonstrations.
  The eight candidates in each source scene group are also not independent
  scenes. Unique accepted scene groups must be counted from the manifest.
- Rejections total **728 non-container sources**, **192 not-a-desk-start**
  (88 / 40 / 64), and **323 no-pickup-success** (104 / 114 / 105).
  Counts reconcile: 728 + 192 + 323 + 293 = 1,536. Among 808 container
  episodes, prefix yield is **293/808 = 0.3626**; among the 616 that passed
  desk-start selection it is **293/616 = 0.4756**. These filtered denominators
  are extraction statistics, not ordinary pick-up success. Keep the desk-start
  filter; investigate its 192 rejected episodes separately.
- The empty-bank issue is no longer blocking this prototype. The 137 usable
  lifts from failed placements establish why full-placement success must not
  be the only harvest filter. These data do not demonstrate ordinary approach
  or outside-goal transport coverage; retain evaluation-scene provenance.
- Next executable step: two-GPU handoff probe (§7.15), pickup on CUDA 0 and
  placement on CUDA 1. It verifies decision-boundary replay, copies live states
  into candidate groups and collects current-policy suffixes without training.
  Local validation includes 10 new CPU tests (including a mocked replay/handoff
  with the real task predicate), plus 111 extraction/comparison/recording tests.
  Shell syntax and whitespace checks pass. Real MJWarp replay and fresh-suffix
  results remain pending remote execution.

### 2026-09-09 — Z-offset pilot regresses manipulation; demonstration-guided GRPO adopted

- Status: **diagnostic-only negative pilot; next direction adopted, not yet
  trained**. Evidence is the user's pasted remote console output, reviewed
  against local commits through `5288a26`. No new GPU run was performed locally.
- Run: `runs/release_recovery_pilot_20260909_130003/`; config:
  `configs/examples/cdpr_smolvla_zlift_offset.yaml`, introduced by `04929bd`.
  Requested budget: 500,000 selected actions. Actual final step, update count,
  wall time and checkpoint SHA-256 have not been supplied.
- Source lineage: the first pasted command selects bowl-peak 2,117,145, but
  the next overwrites `CKPT` with the largest numbered continuation checkpoint,
  expected to be 3,540,208. Confirm the actual source/hash from this pilot's
  `pilot_manifest.json`; do not label it a bowl-peak warm start. The displayed
  `DRY_RUN=1` command itself only previews; completed console results establish
  that a real run was also executed, whose launch provenance is still pending.
- Launcher protocol: baseline and final, each 3×512 worlds, group size 8,
  rounds 0–2, seed-torch 0; fixed config caps move-to 0.08 m, pick-up 0.06 m,
  containers 0.20 m, uncaught composed starts and 40 decisions. Verify actual
  caps in remote artifacts. Nominal independent reset groups are 50 / 41 /
  57 / 44 for move-to / pick-up / plate / bowl, not 1,536 independent scenes.
  Matching settings alone do not certify exact pre-action reset identity.

| Instruction | Baseline | Final | Delta, percentage points |
|---|---:|---:|---:|
| move_to_object | 321/400 = 0.8025 | 331/400 = 0.8275 | +2.50 |
| pick_up | 91/328 = 0.2774 | 84/328 = 0.2561 | −2.13 |
| put_into_plate | 276/456 = 0.6053 | 283/456 = 0.6206 | +1.54 |
| put_into_bowl | 122/352 = 0.3466 | 103/352 = 0.2926 | −5.40 |

| Probe family | Grasp, before → after | Lift given grasp, before → after | Median held-step mean a_z, before → after | Median peak lift, m |
|---|---:|---:|---:|---:|
| pick_up | 0.5488 → 0.4665 | 0.3556 → 0.4183 | +0.275 → +0.269 | 0.0486 → 0.0493 |
| put_into_plate | 0.8947 → 0.8640 | 0.6471 → 0.6244 | +0.344 → +0.399 | 0.0682 → 0.0643 |
| put_into_bowl | 0.6307 → 0.5909 | 0.5360 → 0.5144 | +0.295 → +0.317 | 0.0565 → 0.0515 |

- Probe values are as supplied, not harmonized with production success. For
  pick-up, `328 × grasp × lift|grasp` implies about **64 → 64** lifted
  episodes, versus **91 → 84** recorded successes. Height-reference and
  opening-check differences are established in code; their individual
  contributions to this discrepancy require rescoring the remote recordings.
- This supports retaining the experiment as negative for its manipulation
  objective, with no promotion or automatic longer continuation. It does not
  prove statistical significance, identify the cause of regression, rule out
  useful z exploration, or establish that demonstrations are unnecessary.
  Pickup and bowl acquisition must improve too: holding the final observed
  grasp outcomes fixed caps downstream-only success at 46.65% and 59.09%.
- Review validation: **125 relevant CPU tests passed** across z-offset config,
  extraction, comparison, recording reporting and episode-offset exploration.
  They check implementation behavior, not GPU learning efficacy. Findings and
  the adopted demonstration-guided GRPO contract are in §7.15.
- Remote artifacts: `pilot_manifest.json`, `pilot_config_snapshot.yaml`,
  `pilot_comparison.json`, `baseline/record_*.npz`, `final_eval/record_*.npz`
  beneath the run above. No local artifact copy or file hash is claimed.

### 2026-09-09 — Matched final/peak evaluation: pick-up improves in final; bowl-peak retains strongest placement

- Evidence: user console output from
  `runs/release_recovery_continue_3m_20260908_102004/eval/final_vs_peaks_20260909_083221_340502/`.
  Run command included `--plate-step 3416645`; expected numbered checkpoints
  are final 3,540,208, plate 3,416,645, bowl 2,117,145. Remote manifest holds
  the file hashes; it has not been supplied locally. Counts/decompositions are
  console evidence, not independently imported remote artifacts.
- Protocol: all four families per checkpoint, 3×512 worlds, seed-torch 0,
  rounds 0–2, group size 8, fixed pilot config and uncaught composed starts
  with 40 decisions. Nominal independent scene groups by family are 50, 41,
  57 and 44, not the full candidate episode denominators below.

| Instruction | Final | Plate peak | Bowl peak |
|---|---:|---:|---:|
| move_to_object | 312/400 = 0.7800 | 324/400 = 0.8100 | 306/400 = 0.7650 |
| pick_up | **90/328 = 0.2744** | 80/328 = 0.2439 | 72/328 = 0.2195 |
| put_into_plate | 271/456 = 0.5943 | 287/456 = 0.6294 | **320/456 = 0.7018** |
| put_into_bowl | 119/352 = 0.3381 | 108/352 = 0.3068 | **133/352 = 0.3778** |

- **`put_into_plate` here is not a placement measurement.** 92.6% of composed
  plate episodes start with the object already inside the 0.091 m success
  radius against 42.7% for bowl, so the two columns answer different questions
  and 0.7018 must not be read against the 70% target as a placement rate. §7.14
  has the distribution and the two causes.
- The final checkpoint wins pick-up; bowl-peak wins both placements. Relative
  to final, bowl-peak has 49 more plate successes (+0.1075), 14 more bowl
  (+0.0398), 18 fewer pick-up (-0.0549), and six fewer move-to (-0.0150).
  These are descriptive differences. Plate has the minimum integer count
  needed to exceed 70% (320/456); no strong confidence claim follows.
- For final / plate-peak / bowl-peak, plate grasp counts are 404 / 407 / 420,
  release counts 299 / 317 / 347, settled counts 284 / 295 / 337. Bowl grasp
  counts are 219 / 218 / 233, release 145 / 137 / 167, settled 132 / 122 / 152.
  All recorded settled candidates satisfy the z term in these decompositions.
- Bowl-peak plate failures: no-grasp 36, no-release 73, not-settled 10,
  XY-miss 17. Bowl failures: no-grasp 119, no-release 66, not-settled 15,
  XY-miss 19. Bowl's grasp frequency is 233/352 = 0.6619, so improving only
  post-grasp behavior cannot exceed 70% with those grasp outcomes fixed.
  Plate's no-release group is the largest failure category. No-release plus
  ended-unheld is not sufficient to infer where/when an object was dropped.
- Thresholds: plate radius 0.091 m, bowl 0.057 m, container z tolerances
  0.12 m, settle margin 0.045 m, release opening 0.55; release-height gate
  off, nominal spawn interval 0.06–0.10 m. Realized clipping/spawn patterns
  remain visible and must not be treated as a clean uniform interval.
- All three evaluations and decompositions wrote their outputs. The helper
  printed its final comparison path, then returned nonzero. In that code path
  status 2 denotes its recorded-scene pairing gate. Subsequent CPU inspection
  supplied by the user found only `object_xyz_at_step_0` differences: maximum
  coordinate differences per round were 1.701 / 1.179 / 0.471 mm for plate-peak
  and 0.760 / 1.206 / 0.468 mm for bowl-peak. These poses are recorded AFTER
  the first policy action and cannot certify pre-action reset identity. No
  metadata mismatch was reported. The helper now explains this limitation
  through `--inspect-existing` and does not fail a completed job for this
  post-action-only mismatch; paired verdicts remain uncertified. Do not rerun
  all evaluations or relax tolerances merely to clear the status.
- Next direction remains **GRPO with demonstrations of shared transitions**.
  Bowl-peak is a placement donor/retention reference; final is a pick-up
  donor/control. No weight merging or automatic promotion. Existing evaluation
  clips can prototype extraction, but using their states/actions in training
  retires those scenes from held-out status. Collect training-only demo scenes
  to preserve the current evaluation set.
- Local transcription: `runs/analysis/release_recovery_matched_comparison_20260909/reported_results.json`.
  This supersedes the earlier “final evaluation pending” status below.

### 2026-09-09 — Additional 3M continuation completed; modest late gains, final evaluation pending at that time

- Source: user-uploaded complete TensorBoard event file
  `events.out.tfevents.1788852032.VLAPU.1071244.0_complete` (1,772,446 bytes),
  SHA-256 `5c884d6e87b5f027d6d6bac6cc625b57e0c1344499bca9476877a712fa31fc8d`.
  Its scalar histories exactly preserve the preceding incomplete upload.
- Lineage/config: full resume of release-recovery pilot step 527,307 with
  `cdpr_smolvla_release_recovery_pilot.yaml`; four fixed instruction settings,
  uncaught containers, 40-decision composed budget. No new SFT.
- End: **3,540,208 cumulative selected actions**, **3,012,901 additional**,
  **121 continuation updates**, **24,066,340 sampled candidate actions**.
  Logged training elapsed time 52,781 seconds (14h40m); last event
  2026-09-09 01:00:37 MSK. Nominal target 3,527,307 was crossed at an update
  boundary. The displayed update counter restarted after resume.
- There are **30 validation checkpoints**. The last is **3,512,892**:
  **do not assign these scores to final step 3,540,208**. Expected final
  checkpoint directory is `rl/step_3540208/`; its file and checksum have not
  been supplied or inspected locally. The comparison helper verifies them
  remotely when it runs.

| Step | move_to (`validation`) | pick_up (`validation`) | plate (`validation_composed`) | bowl (`validation_composed`) |
|---|---:|---:|---:|---:|
| 3003730 | 0.6836 | 0.1500 | 0.5691 | 0.3750 |
| 3122124 | **0.7422** | 0.1800 | 0.5987 | 0.3826 |
| 3211063 | 0.7070 | **0.2200** | 0.5822 | 0.3977 |
| 3318446 | 0.6602 | 0.1850 | 0.5987 | 0.3258 |
| 3416645 | 0.6914 | 0.2100 | **0.6447** | 0.3220 |
| 3512892 | 0.6836 | 0.1850 | 0.5921 | 0.3485 |

- Latest denominators are 256, 200, 304, 264 respectively: successes
  **175, 37, 180, 92**. Companion-leg latest scores are move-to 0.7969,
  pick-up 0.1600 (`validation_composed`) and plate 0.6546, bowl 0.3636
  (`validation`). Keep the series separate; these are not fresh independent
  confirmation scenes, and neither is the pilot's 3×512 recording protocol.
- Best plate remains **196/304 = 0.6447**, now tied at **1,505,251** and
  **3,416,645**. The late tie has stronger pick-up on this validation series
  (0.2100 versus 0.1050), but weaker bowl (0.3220 versus 0.3939). Bowl's best
  remains **112/264 = 0.4242 at 2,117,145**. Best pick-up is now **44/200 =
  0.2200 at 3,211,063**. These maxima do not constitute one shared policy.
- First-five → last-five primary validation means: move-to **0.6344 →
  0.6969**, pick-up **0.1530 → 0.1960**, plate **0.5513 → 0.6033**, bowl
  **0.3318 → 0.3553**. The late improvement qualifies the incomplete upload's
  plateau reading; it is not evidence of convergence or progress toward 70%
  at a predictable rate. Companion pick-up averages 0.1720 → 0.1880.
- First20 → last20 training-update pooled success: move-to **0.6849 →
  0.6958**, pick-up **0.1711 → 0.1784**, plate **0.5070 → 0.5409**, bowl
  **0.2197 → 0.2520**. Final-window pick-up grasp frequency is **0.5228**,
  with **0.3413** success conditional on ever grasping. Grasp frequency
  recovered from the earlier snapshot's last-window 0.4873; a large
  post-grasp completion gap persists. Training and deterministic evaluation
  remain different distributions.
- No obvious optimizer blow-up: KL 0.0656–0.0894, mean gradient norm
  3.981–4.647; logged contact/constraint overflows zero. Non-finite simulator
  reset-event metric remains nonzero (12–50 after round/rank aggregation).
  Zero validation final reward/distance non-finites do not prove no resets.
  Global usable-group fraction is 53.61% in the last20 updates, from counts;
  raw fraction tags can be summed across ranks and should not be read as
  global percentages.
- Next evaluation: final versus both placement peaks. Use the later plate
  tie via `--plate-step 3416645` to emphasize pick-up retention; the original
  peak is still retained and remains the helper's explicit default. No
  checkpoint is promoted. All candidates must be evaluated on all four
  instructions under the same protocol.
- Local evidence: `runs/analysis/release_recovery_complete_20260909/`
  contains the complete event copy, manifest, all scalars, validation CSV,
  and PNG/PDF plot. These large artifacts are ignored by Git. This analysis
  did not launch or alter remote training.

### 2026-09-08 — Fixed-cap joint RL: plate 0.6162 → 0.6930, pick-up 0.1494 → 0.1982

- Git commit: pilot implementation/config `fd0c73f`; actual host launch revision
  is saved in `pilot_manifest.json` and has not yet been supplied locally.
- Run/config: `release_recovery_pilot_20260907_193019`,
  `configs/examples/cdpr_smolvla_release_recovery_pilot.yaml`.
- Source: `runs/phase7_sparse_joint_20260904_212930/rl/step_2017690/smolvla_grpo_adapter.pt`,
  weights-only warm start, fresh optimizer and curriculum. No SFT.
- Candidate: `runs/release_recovery_pilot_20260907_193019/rl/step_0527307/smolvla_grpo_adapter.pt`.
  SHA-256 supplied by the user:
  `ee33b11d9a1a17c64ebe7251b614edb1706dbe69f83c793196a997f4596d329e`.
- Budget: 500,000 selected environment actions across both ranks, stopping at
  an update boundary. Actual total **527,307**, update index **19**. Elapsed
  time not supplied. These are training-loop updates, not individual optimizer
  minibatch steps.
- Settings: one shared policy, all four instructions, sparse binary reward,
  corrected `wrong_place_settled`. One-rung ladders fix move-to at 0.08 m and
  pick-up at 0.06 m. Containers are 100% uncaught, with aligned gripper starts,
  original 0.06–0.10 m object-to-receptacle spawn range and 40 decisions. Their
  nominal 0.20 cap does not set composed carry distance. Existing gripper
  episode-offset std 0.15 is on; no pick-up-specific z exploration.
- Evaluation: before/after `sil_record`, 3 rounds × 512 worlds, round indices
  0–2, group size 8, torch seed 0, same config. No scalar cap override. These
  are matched evaluation settings, not a demonstrated bit-identical rollout.
- Independent reset groups: 192 total, comprising move-to 50, pick-up 41,
  bowl 44 and plate 57. Candidate episodes within groups are not independent.

| Instruction | Baseline | Final | Change |
|---|---:|---:|---:|
| move_to_object @0.08 | 279/400 = 0.6975 | 285/400 = **0.7125** | +0.0150 |
| pick_up @0.06 | 49/328 = 0.1494 | 65/328 = **0.1982** | +0.0488 |
| put_into_plate, composed | 281/456 = 0.6162 | 316/456 = **0.6930** | +0.0768 |
| put_into_bowl, composed | 107/352 = 0.3040 | 108/352 = **0.3068** | +0.0028 |

- Supports: the successor joint-RL recipe improves the aggregate plate and
  pick-up estimates while the other two aggregate estimates do not regress.
  Plate requires four additional successes on this denominator to exceed
  70% (320/456); move-to exceeds 70% as a point estimate at its easier cap.
- Does not support: statistical significance without a paired group-level
  analysis; four-family >70%; harder-cap reaching; a claim that the final
  checkpoint is best or still improving; or that release specifically caused
  the plate gain. The final grasp/release funnel has not yet been supplied.
  Bowl is effectively unchanged in this comparison. The recipe differs from
  the historical run in training mixture, fixed caps and exploration as well
  as corrected termination, so this is not a single-variable causal ablation.
- In-run validation trajectory, supplied separately (different seeds/protocol
  from the recorded baseline/final comparison; do not splice the curves):

| Selected actions | Move-to | Pick-up | Composed plate | Composed bowl |
|---:|---:|---:|---:|---:|
| 112878 | 0.5664 | 0.1100 | 0.5263 | 0.2917 |
| 203598 | 0.5938 | 0.1350 | 0.4572 | 0.3068 |
| 313034 | 0.5469 | 0.1250 | 0.4737 | 0.3182 |
| 422092 | 0.6211 | 0.1400 | 0.4803 | 0.3295 |
| 527307 | 0.5859 | 0.0950 | 0.5164 | 0.3295 |

- Final training batch: 320 groups collected, 164 usable (51.25%),
  `rounds_collected=5`. These are globally reduced metrics; five is not a
  per-rank refill count. Pooled usable-group supply does not establish that
  every instruction receives sufficient gradient.
- Status: **retained candidate; no claim of convergence or four-family >70%**.
  The user explicitly authorized multi-million-step training. Plan a full
  resume for **3,000,000 additional selected actions**, to cumulative
  **3,527,307** at an update boundary, in a new run directory. Preserve
  optimizer/curriculum state and the fixed task settings. The mixed validation
  trajectory does not prove the final checkpoint is best, but 19 updates do
  not establish saturation. Continue per-family validation/checkpoint retention;
  do not impose a 70% early-stop or widen caps based on one recorded score.
- Evidence: user-supplied console output. Remote artifacts are under
  `runs/release_recovery_pilot_20260907_193019/`, including baseline/final NPZs,
  `pilot_comparison.json`, `pilot_manifest.json`, `rl/validation.jsonl`,
  `rl/metrics.jsonl` and all step checkpoints. Not copied into the local
  evidence set; candidate hash supplied, source/config hashes still only
  recorded remotely.

### 2026-09-07 — `wrong_place_settled` terminated correct placements; fixing it takes composed plate 0.5000 → 0.6272

- Git commit: `1b78cbc` (predicate fix and its tests); `0616cee`, `a9bbae9`,
  `5523a94`, `e450de2` (`tools/audit/grasp_loss_forensics.py`, the CPU-only
  audit that found it); `5c0870d` (reset-identity check in `--mode compare`)
- Run/config: `configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml`. Arms are
  `runs/phase4_bank/eval/phase7rl_composed_fixed` — **pre-fix, despite the
  directory name**; the host was at `e450de2` when it ran and never had the
  patch — and `runs/phase4_bank/eval/phase7rl_composed_fixed2`, post-fix
- Source checkpoint and lineage:
  `runs/phase7_sparse_joint_20260904_212930/rl/step_2017690/smolvla_grpo_adapter.pt`,
  byte-identical in both arms
- Candidate checkpoint: none. This changes the task predicate, not a policy
- Training steps / updates / wall time: none; both arms are inference-only
- Evaluation protocol: `sil_record --mode record`, 3 rounds x 512 worlds,
  `--seed-torch 0`, `--start-distance-cap 0.20`. **Scene-matched**: `--mode
  compare` reports `same_episodes=True` on all three rounds — identical
  instructions, slots and horizons, object layout within 1.2 mm — and step-0
  actions bit-identical in 512/512 worlds. The two arms differ in the predicate
  and in nothing else
- **Protocol correction.** These arms did NOT pass
  `--metadata-override placement_caught_object_fraction=0.0`, which the Phase 7
  protocol uses to force every container episode to a composed start. About
  half of their container episodes therefore begin CAUGHT, and caught starts
  are far easier. The container rates below are **mixed-start**, not composed,
  and must not be read against any composed number. Diagnosed from the horizon
  histogram: composed container starts take the
  `placement_grasp_horizon_min_decisions` floor of 40, caught starts take the
  curriculum-coupled value, so the split is a direct readout of the mix — 100%
  at 40 in the Phase 7 run below, roughly half here. The paired comparison
  itself is unaffected: both arms share the identical mix, episode for episode
- Caps, seeds, rounds, worlds, and independent reset groups: requested cap 0.20
  for every instruction. `cap_check` reports earned caps `move_to_object` 0.14,
  `pick_up` 0.13, `put_into_bowl` 0.10, `put_into_plate` 0.17 — **every family
  reads `above_earned_cap`**, and bowl is scored at twice the start distance it
  earned. 1536 episodes; at the default `--group-size 8` that is 64 reset groups
  per round, and per §13 the group is the independent unit, so the per-episode
  flip counts below are optimistic as a significance claim
- Instruction results (successes / denominator and rate). **The headline arm is
  protocol-matched**: `comp_fixed_020` repeats the Phase 7 composed protocol
  exactly — cap 0.20, `placement_caught_object_fraction=0.0`, containers 100% at
  the 40-decision floor — so it differs from the entry below in the predicate
  and nothing else:

| instruction | composed @0.20, pre-fix (Phase 7) | composed @0.20, post-fix | Δ |
|---|---|---|---|
| `put_into_plate` | 228/456 = 0.5000 | **286/456 = 0.6272** | **+0.127** |
| `put_into_bowl` | 76/352 = 0.2159 | **100/352 = 0.2841** | **+0.068** |
| `move_to_object` | 168/400 = 0.4200 | 172/400 = 0.4300 | +0.010 (control) |
| `pick_up` | 0/328, above its cap | 1/328, above its cap | — |

- `move_to_object` is again the control — `wrong_place_settled` is gated on
  `is_container` — and it moves +0.010, inside the world-coupling noise measured
  at 25:25 below. Two further composed runs at caps 0.17 and 0.10 give plate
  0.6404 and 0.6491 and bowl 0.3153 and 0.2756; `--start-distance-cap` is a
  **no-op for composed container starts**, whose spawn comes from
  `placement_grasp_object_min/max_distance`, so those are three draws of one
  task and pool to plate **874/1368 = 0.6389** and bowl **308/1056 = 0.2917**
- The scene-matched paired arms that isolate the mechanism ran WITHOUT the
  composed forcing, so their container rates are mixed-start and are reported
  as such. They are the causal evidence, not the headline:

| instruction | protocol | pre-fix | post-fix | flipped, pre:post |
|---|---|---|---|---|
| `put_into_plate` | **mixed-start** @0.20 | 216/456 = 0.4737 | **240/456 = 0.5263** | 84, 30:54 |
| `put_into_bowl` | **mixed-start** @0.20 | 69/352 = 0.1960 | **93/352 = 0.2642** | 36, 6:30 |
| `move_to_object` | @0.20 | 172/400 = 0.4300 | 172/400 = 0.4300 | 50, **25:25** |
| `pick_up` | @0.20, above its cap | 0/328 = 0.0000 | 1/328 = 0.0031 | 1, 0:1 |

- Comparison baseline under the same protocol: the pre-fix arm above.
  `move_to_object` is a **null control** — `wrong_place_settled` is gated on
  `is_container` and cannot reach it — and it flips 50 of 400 episodes at
  exactly 25:25, net zero. That is the world-coupled rollout noise, and both
  container families sit outside it
- What this result supports: **`wrong_place_settled` was terminating correct
  placements.** It tested `~container_ok`, and `container_ok` requires
  `released`; `state.grasped` is `caught_target & (opening <= 0.94)` over the
  live `physical_grasp`. So an object carried into the receptacle and set down
  while the gripper is still opening ended its own episode: the surface takes
  the load, the pads unload, `~grasped` goes true, `target_has_settled` is
  already true, and `released` is not true *yet*. Median **zero** env steps
  between the latch breaking and the termination. This is §7.8 in a third place
  — a terminal condition sharing a conjunct with success, firing because that
  conjunct is not satisfied yet — and the condition is named for the PLACE, so
  it now tests `~placement_geometry_ok` alone. `container_ok` implies
  `placement_geometry_ok`, so the change strictly *narrows* termination
- ...and the mechanism is confirmed, not only the outcome. Set-downs coming to
  rest **inside** the receptacle radius went 28 -> 0 on plate and 26 -> 0 on
  bowl; the survivors are genuine misses (plate's 5 sit at xy p50 0.092 against
  a 0.091 radius). Real drops are untouched: `separated_and_fell` 44 -> 42 and
  34 -> 32. In the funnel, `no_release` fell 104 -> 71 and 115 -> 88, success
  rose +24 on each, and **`xy_miss` stayed flat** (90 -> 89, 77 -> 79) — the
  freed episodes became successes rather than late misses. `release|grasp`
  0.7626 -> 0.8364 (plate) and 0.6007 -> 0.6890 (bowl); `xy_ok|settle` rose on
  both, 0.7059 -> 0.7295 and 0.4726 -> 0.5407
- ...and that **the horizon now binds on what remains.** `no_release` episodes
  that ran the whole budget went 12/104 (0.1154) -> 32/71 (0.4507) on plate and
  21/115 (0.1826) -> 34/88 (0.3864) on bowl. The bottleneck moved rather than
  vanishing, and for the first time it is a measured constraint rather than a
  guess. These arms drew the SHORT horizon mix (see below), so the lever is
  cheap to test
- ...and that four hypotheses about the composed loss were aimed at the wrong
  mechanism. The 8 mm relative-pose stability test never rejects a real grasp —
  **0 of 12 496 genuinely-held policy steps** crossed the bar, held slip p50
  0.23 mm against it — and detector chatter is 5%
- What it does not support: **any comparison with the Phase 7 entry below.**
  That entry is 100% composed starts; these arms are roughly half caught. Same
  checkpoint, same seed, same requested cap, different reset distribution.
  `0.5263` is therefore **not** an improvement on the `0.5000` recorded below,
  and the two must not be placed in one column. It also
  does not support any absolute claim about this checkpoint's ability, because
  every family was scored above its earned cap. And it does not support the
  bound this work was launched on: the predicted +0.138 on plate assumed every
  inside-radius set-down would complete its release, and about a third do
- Status: **active fix, and a new campaign best on the composed protocol.** The
  checkpoint is unchanged and nothing is promoted — `step_2017690` did not get
  better, the task it was scored against got correct. Part of the 0.5000
  recorded below was a measurement artifact, and composed plate 0.6272 is the
  closest this campaign has come to the 70% target on the hard protocol. The predicate change is landed and is a change to the TASK —
  episodes that used to end now keep running, horizon usage rises, and results
  before and after it are not commensurable
- Local artifact path: on the training host
- SHA-256: not recorded
- Missing provenance: neither arm's recordings are in the local evidence set;
  the pre-fix arm's directory name (`..._fixed`) misdescribes it and should be
  renamed before the two are archived together

### 2026-09-07 — Phase 7: one sparse binary reward, four instructions, one run

- Git commit: `994df48`
- Run/config: `phase7_sparse_joint_20260904_212930`, `configs/examples/cdpr_smolvla_phase7_sparse_joint.yaml`
- Source checkpoint and lineage: `runs/phase4_bank/sft_phase7/sil_sft_adapter.pt`, weights-only warm start, fresh optimizer and curriculum
- Candidate checkpoint: `rl/step_2017690`
- Training steps / updates: ~2.0 M to the promoted step, 2.51 M to the end of the run
- Evaluation protocol: `sil_record --mode record`, 3 rounds x 512 worlds, `--seed-torch 0`. Composed forces `placement_caught_object_fraction=0.0`; `pick_up` is scored at its OWN cap 0.06, never at the composed 0.20
- Caps: composed put_into 0.20, move_to 0.20, pick_up 0.06

| instruction | protocol | `sft_phase7` seed | **phase 7 RL** | raw |
|---|---|---|---|---|
| put_into_plate | composed @0.20 | 0.1203 | **0.5000** | 228/456 |
| put_into_bowl | composed @0.20 | 0.0412 | **0.2159** | 76/352 |
| move_to_object | @0.20 | — | 0.4200 | 168/400 |
| pick_up | @0.06 | 0.1562 | 0.1465 | 225/1536 |

- Comparison baseline under the same protocol: the `sft_phase7` seed above. Composed plate **4.2x**, composed bowl **5.2x**; `pick_up` and `move_to` unchanged within noise
- What this result supports: one binary outcome reward, instruction-agnostic, trains all four families in a single GRPO run and produces the campaign's best composed pick-and-place. The sawtooth's *cause* — per-family dense rewards forcing per-family RL turns — is removed, not compensated for
- What it does not support: 70% on any composed family, and no claim that the run is converged. A second run under a corrected ladder landed at composed plate 0.4145 / bowl 0.1364, i.e. ~0.08 lower on both, so the level is reproducible in band but the peak is not
- Status: **promoted — the current best four-family policy and the base for all downstream work**
- Local artifact path: on the training host
- SHA-256: not yet recorded
- Missing provenance: adapter not in the local evidence set

**The decomposition, which is where the remaining work is.** Conditional rates
against the scripted oracle on the identical protocol:

```
                grasp   release|grasp   settle|rel   xy_ok|settle   success
  plate RL     0.9079      0.6449         0.9401       0.9084       0.5000
  plate oracle 0.9336      0.9831         1.0000       1.0000       0.9178
  bowl  RL     0.6051      0.5305         0.9115       0.7379       0.2159
  bowl  oracle 0.6752      0.7778         0.9958       1.0000       0.5229
```

The grasp gap is closed: plate is 97.2% of the oracle's grasp rate and bowl
89.6%, from 0.4054 and 0.2956 before. The horizon is no longer binding — 4.8%
of plate's `no_release` timed out, against 100% two runs earlier — and the
policy now grasps FASTER than the oracle, first grasp p50 19-21 env steps
against 90-104.

**What remains is the drop.** 140 of plate's 147 `no_release` failures (95.2%)
lost the object mid-carry without ever opening the gripper. Give plate the
oracle's `release|grasp` and nothing else and it reaches **0.7622**, over the
70% target. Bowl reaches only 0.3166, because its grasp (0.605) and its
placement accuracy (`xy_ok|settle` 0.7379 against the oracle's 1.0) are also
short.

Three hypotheses about the drop have been tested and none isolated it: release
height (falsified, bounce is 0.18 mm under the oracle), horizon (fixed, now 5%
of failures), and grasp speed (backwards — dropped episodes grasp SLOWER,
median 19 vs 16 on plate, and the correlation does not transfer to the oracle,
which is five times slower and more reliable).

### 2026-09-07 — Phase 7 SFT collapses the RL gain, at every demonstration mix

- Git commit: `994df48`
- Run/config: `scripts/run_cdpr_phase7_composed_fraction_sweep.sh`, arms 0.5 and 0.8
- Source checkpoint: `phase7_sparse_joint_20260904_212930/rl/step_2017690` — composed plate 0.5000
- Evaluation protocol: identical to the entry above

| arm | realized | comp plate | comp bowl | caught plate | caught bowl | pick_up | move_to |
|---|---|---|---|---|---|---|---|
| base (RL) | — | **0.5000** | **0.2159** | — | — | 0.1465 | 0.4200 |
| knob off (SFT) | — | 0.1203 | 0.0412 | 0.6822 | 0.4985 | 0.1562 | 0.4668 |
| 0.5 (SFT) | 0.5 | 0.1390 | 0.0471 | 0.6857 | 0.5132 | 0.1712 | 0.5007 |
| 0.8 (SFT) | 0.8 | 0.1285 | 0.0676 | 0.6110 | 0.4500 | 0.1901 | 0.4616 |

- What this result supports: **residual SFT on the retention bank destroys ~72%
  of the composed capability RL built, and it does so at every mix tested.**
  0.5000 becomes 0.1285-0.1390 regardless of whether the demonstration slice is
  50% or 80% composed. The bank's demonstrations are a weaker teacher than the
  policy being retrained on them
- ...and that the mix itself is not a lever in this range. 0.8 buys composed
  bowl +0.021 (inside the ~0.04 noise floor) and costs caught plate -0.075 and
  caught bowl -0.063 (outside it). 0.5 is the better mix and is barely
  distinguishable from leaving the knob off
- What it does not support: any claim that more composed demonstrations help.
  That hypothesis is now tested and flat
- Status: **diagnostic — and it invalidates the alternating RL/SFT loop for the
  composed task specifically.** The loop remains the retention mechanism for
  families RL is not currently training
- Missing provenance: the arms' adapters are on the training host

### 2026-09-01 — Phase 6 composition RL, first annealed run

- Git commit: `c5251aa`
- Run/config: `phase6_compose_iter0`, `configs/examples/cdpr_smolvla_phase5_compose_loop.yaml`
- Source checkpoint and lineage: `runs/phase4_bank/sft_phase6/sil_sft_adapter.pt`, itself from `sft_cycle3` + oracle composed demonstrations
- Candidate checkpoint: `runs/phase6_compose_iter0_*/rl/step_4257133`
- Training steps / updates: 2 763 249 → 5 010 151 (2.25 M new), 222 updates
- Evaluation protocol: in-run validation against the training resetter, caught fraction 0.9–0.8 — **caught-dominated, not the composed task**
- Caps, seeds, rounds, worlds: plate cap 0.20, bowl cap 0.19 throughout; `validation_seed` from the placement config
- Instruction results: peak overall 0.6240; plate 0.7571; bowl 0.4634 at step 4 257 133
- Comparison baseline under the same protocol: Phase 5 `iter3` peak 0.6211 — indicative only, the caught fractions differ
- Best validation epoch and overfit behaviour: peak at 4 257 133, two subsequent readings below it; stop rule met
- What this result supports: the caught-fraction curriculum anneals under a seeded policy (1.0 → 0.9 → 0.8) and the policy absorbs the harder mix without entropy collapse
- What it does not support: any composed-task success rate; this protocol does not measure composition
- Status: promoted as the current placement lineage; composed evaluation outstanding
- Local artifact path: TensorBoard event file supplied; adapter on the training host
- SHA-256: not yet recorded
- Missing provenance: composed-protocol evaluation of `step_4257133`

### 2026-08-30 — Phase 6 seed: composed `put_into` from a scripted oracle

- Git commit: `cbd74a2` (oracle mode `afdbdae`, seed script `a6659c7`)
- Run/config: `scripts/run_cdpr_phase6_compose_seed.sh`, `cdpr_smolvla_phase5_compose_loop.yaml` with `placement_caught_object_fraction=0.0`
- Source checkpoint and lineage: `sft_cycle3` → 12 oracle harvest rounds → pooled bank → refresh → residual SFT
- Candidate checkpoint: `runs/phase4_bank/sft_phase6/sil_sft_adapter.pt`
- Training steps / updates: SFT only, 45 epochs on 91 936 train / 10 289 val rows, 5 867 episodes
- Evaluation protocol: `sil_record --mode record`, 3 rounds × 512 worlds; composed evaluations force every container episode uncaught
- Caps: pick_up 0.06, move_to 0.19, placement 0.20
- Instruction results: composed plate **80/856 = 0.0935**; composed bowl **18/680 = 0.0265**; pick_up 229/1536 = 0.1491; move_to 737/1536 = 0.4798; caught plate 612/856 = 0.7150; caught bowl 326/680 = 0.4794
- Comparison baseline under the same protocol: `sft_cycle3` composed plate 10/2168 = 0.0046, composed bowl 22/1928 = 0.0114
- SFT dataset rows by instruction: 25 550–25 563 decisions each across four instructions, quota bound by pick_up
- Best validation epoch: 44 of 45, `val_mse` still falling; `reachable` 0.91111
- What this result supports: the composed task is above noise for the first time; oracle demonstrations are learnable by this residual; a grasp prefix recorded under one instruction transfers to another
- What it does not support: a working composed pick-and-place, and no claim that the RL has improved on the seed
- Status: promoted as the composition seed
- Local artifact path: on the training host, `runs/phase4_bank/sft_phase6/`
- SHA-256: not yet recorded
- Missing provenance: adapter not in the local evidence set

### 2026-08-30 — Negative result: relabelled grasps cannot supply the composed prefix

- Git commit: `cbd74a2`
- Run/config: `runs/phase4_bank/g6_probe`, `cdpr_smolvla_phase6_grasp_in_place_scene.yaml`
- Evaluation protocol: 2 rounds × 2048 worlds at cap 0.06; join measured as gripper-XY distance to each occupied non-target slot at the first caught step
- Results: grasp rate 347/4096 = 0.085; nearest-receptacle distance median 0.2471 m with 36.3% inside 0.20 m; farthest median 0.3967 m with 5.2% inside
- What this result supports: the relabelling mechanism is sound but the free grasp scene's geometry does not meet placement's start distribution, leaving ~20% of episodes usable
- What it does not support: abandoning relabelling in general — it fails on scene geometry, not on principle
- Status: diagnostic only; superseded by the oracle route
- Missing provenance: none

### 2026-09-10 — Three-stage `put_into` pipeline implemented; yaw calibrated; no rollout yet

- Git commit: this change
- Run/config: `configs/examples/cdpr_smolvla_three_stage_put_into.yaml`, derived from `cdpr_smolvla_phase7_sparse_joint.yaml` with the scene/reset route, the yaw contract and the stage budgets changed and the predicate geometry untouched
- Source checkpoint and lineage: none — nothing was trained
- Training steps / updates: **zero**
- Evaluation protocol: none executed. The tools are `tools/audit/record_cdpr_staged_put_into.py`, `select_cdpr_stage_teachers.py` and `evaluate_cdpr_full_put_into.py`; all three run no optimizer and all three refuse a yaw calibration whose provenance is unset
- Measured result: the fixed pickup yaw is **0.000 rad** (residual 4×10⁻¹³ deg at the calibration pose, desk centre, EE z 0.26). Fixed-angle bearing residual over a 7×7 grid of the ±0.19 m workspace: mean **10.7°**, median 10.0°, p90 **19.8°**, max **25.3°**; **18.4%** of the grid inside a 5° band. This is kinematics from `robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml`, not a rollout
- Local verification: 1336 unit tests pass; the three pre-existing local failures (`_qpos`, `predict_normalized_action_chunk`, `grpo_bootstraps_from_td3`) are unchanged. New suites: `test_cdpr_staged_put_into.py` (21), `test_cdpr_staged_dataset.py` (17), `test_cdpr_composition_scenes.py` (22), `test_cdpr_staged_sft_sampling.py` (19)
- What this result supports: the yaw the user described is a solvable pose and not a guess, and the cost of pinning it to one angle across the workspace is now a number
- What it does not support: **any claim about the task**. No scene manifest, teacher screen, demonstration bank, SFT or full-task score exists. In particular the provisional teacher shortlist in the design remains a list of candidates to test, not a ranking
- Status: implementation landed; the first GPU step is scene generation followed by teacher screening on the `teacher_selection` split
- Local artifact path: none yet; the calibration is regenerated in one second by `tools/audit/calibrate_cdpr_pickup_yaw.py`
- SHA-256: not applicable
- Missing provenance: every rollout number in the pipeline
