# CDPR + SmolVLA: consolidated progress and achievement report

**Living report — current through 2026-09-02, Europe/Moscow**  
**Repository state reviewed:** `c5251aa`  
**Scope:** simulated 5-DoF cable-driven parallel robot (CDPR), SmolVLA-conditioned control, GRPO reinforcement learning, self-imitation learning (SIL), and multi-instruction retention.

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

The most important scientific result is not a single maximum score. It is that one adapter can carry non-zero competence in all four instruction families after repeated RL-induced forgetting, and that balanced self-imitation can recover multiple old skills from a durable bank. Cycle 3 contains three campaign-best results on its own evaluation protocol—move-to, plate, and bowl—while preserving non-zero pick-up.

The most important unresolved behavior is the **sawtooth**: family-specific RL strengthens the active family and erodes inactive ones; retention SFT rebuilds the inactive families only partially. The current loop works, but it alternates between peaks rather than holding every family at its dedicated-policy maximum simultaneously.

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

Placement episodes currently begin with the object already held. The current bank therefore teaches carry-and-release, not the preceding grasp. Phase 6 preparation in §8 addresses this missing prefix.

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

### 7.7 Artifact integrity is now explicit

The Phase 4 archive has a checksum manifest, preserved model files, raw evaluation tables, manifests, logs, reports, and 30 evaluation videos. This is a substantial improvement over result-only reporting and should be continued for every promoted checkpoint.

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

- Preflight oracle zeros produced before placement reward/reset geometry was repaired.
- Validation numbers produced without restoring curriculum state.
- Unseeded single-round comparisons that treated SmolVLA's stochastic prior as deterministic.
- Phase 3 placement-only SFT as a retained single-policy solution; it improved near-cap bowl but erased pick-up because pick-up was absent from the mix.
- Hindsight relabelling of caught-start placement into pick-up; the required lift predicate was never reached because placement starts already lifted.
- Cross-checkpoint simulator replay as a way to refresh priors/state; it destroys trajectory survival and has been replaced by frame inference.
- Vision-tower LoRA as an active contributor to the current policy; it is disabled in active RL and no Cycle 2 LoRA epoch beat baseline.
- Phase 5 placement `iter4` and `iter5` as promoted checkpoints. Both are superseded by `step_2754052`; the active release-height gate is off and the attempted ladder extension was reverted.
- “Gentle placement” as an achieved behavior. The current success is predominantly carry-and-drop under the accepted terminal predicate.
- Composed pick-and-place as achieved. Only the missing-prefix data path is implemented.
- Legacy LCHOL-based relabelling on the MJWarp path; that implementation is not connected to the active batched trainer.

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
9. **The next scientific step is full composition.** The grasp prefix for uncaught `put_into_*` is now representable in the bank, but the geometric join and end-to-end success are still unmeasured.

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


