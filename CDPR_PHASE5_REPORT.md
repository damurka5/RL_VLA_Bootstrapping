# Phase 5 — pick_up joins the single policy, and slice size breaks the 41% ceiling

Everything measured while adding `pick_up` to the one-policy set: a warm start
that could not learn, the seed-first inversion that fixed it, two RL runs, a
curriculum gate that was calibrated backwards, and a bank four times the size of
the one phase 4 built.

Read `CDPR_PHASE4_RETENTION_REPORT.md` first for the loop this continues.

**Headline: the retention ceiling was data, not epochs.** Quadrupling each
instruction's slice took `move_to` from 48% of its reference to **67%**, and
`pick_up` from 37% of its bank source to **67%** — after §6.1 had ruled out
training longer as the lever. That answers phase 4's open question #3.

---

## 1. The numbers, for plotting

### 1.1 Instruction success across the campaign

Every cell is `sil_record --mode record`, 3 rounds, 512 worlds, at the cap
named. Weighted by episodes, not averaged across rounds — placement rounds draw
uneven instruction counts and the naive mean misreads a small round.

| checkpoint | pick_up @0.06 | move_to @0.19 | plate @0.20 | bowl @0.20 |
|---|---|---|---|---|
| phase-4 `sft_retention_e60` | 0.000 | 0.311 | 0.523 | 0.2765 |
| `sft_pickup_seed` (cycle 1 of phase 5) | **0.0964** | 0.3203 | 0.5736 | 0.2721 |
| `step_1003315` (after 1.2M pick_up RL) | ~0.155 | eroded | eroded | eroded |
| **`sft_cycle2`** | **0.1738** | **0.4316** | 0.3463 | 0.1702 |

Raw counts for `sft_cycle2`: pick_up 267/1536, move_to 663/1536,
plate 302/872, bowl 113/664.

### 1.2 Reference levels, for normalising the plot

| instruction | reference | source |
|---|---|---|
| move_to_object | 0.641 | `step_11009573`, the dedicated move_to policy |
| pick_up | 0.260 | `step_7505256` harvest rate at cap 0.06 |
| put_into_plate | 0.490–0.648 | placement checkpoint harvest, by cap |
| put_into_bowl | 0.223–0.436 | placement checkpoint harvest, by cap |

### 1.3 Gap closure — the phase's actual result

| | cycle 1 (6k rows/instruction) | cycle 2 (26k rows/instruction) |
|---|---|---|
| move_to vs 0.641 | 0.311 → **48.5%** | 0.4316 → **67.3%** |
| pick_up vs 0.260 | 0.0964 → **37.1%** | 0.1738 → **66.8%** |

Two independent families, both at ~67% after a 4.3× slice, both under 50%
before. Phase 4 measured 41% and could not move it with epochs.

### 1.4 RL run timeline

| run | steps | updates | outcome |
|---|---|---|---|
| `phase4_pick_up_iter0` | 1.83 M | 191 | dead; 222 normal-start grasps in 102 016 worlds |
| `phase4_pick_up_iter1` | 1.15 M | 130 | peak val **0.1387** @ 206 k, then collapse |
| `phase4_pick_up_iter2` | 0.21→1.41 M | 140 | peak val **0.1328** @ 1.00 M, then decay |
| `phase5_placement_iter3` | 1.00→5.01 M | 525 | peak val **0.6211** @ 2.75 M; 73% of it at the ladder ceiling |
| `phase5_placement_iter4` | 2.76→8.01 M | 695 | release gate armed; val 0.6211 → peak **0.3320**, plateau |

### 1.5 pick_up validation series (for the graph)

`validation/by_instruction/pick_up/success_rate`, every 200 k steps.

* iter0: 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
* iter1: 0.1387, 0.0381, 0.0146, 0.0361, 0.0010
* iter2: 0.0156, 0.0371, 0.0312, **0.1328**, 0.0586, 0.0166

### 1.7 Placement validation series (for the graph)

`validation/success_rate` overall, every ~250 k steps.

* iter3 (gate off, old ladders): 0.5977, 0.5850, 0.5752, 0.5869, 0.6045,
  0.6104, **0.6211**, 0.5576, 0.6191, 0.5840, 0.6143, 0.6104, 0.6133, 0.6074,
  0.5547, 0.5117
* iter4 (gate armed, ladders extended): 0.2295, 0.2637, 0.2783, 0.2490,
  0.3291, 0.3174, **0.3320**, 0.2637, 0.2666, 0.2627, 0.2500, 0.2305, 0.2344,
  0.2617, 0.3057, 0.2373, 0.3174, 0.2822, 0.3018, 0.2832, 0.3027

`release_clearance_mean_m` across iter4: 0.0897, 0.0696, 0.0696, 0.0683,
0.0683, 0.0693, 0.0719, 0.0661, 0.0644, 0.0636 — the constraint being learned,
and converging above the bowl's threshold.

### 1.6 SFT curves

| | cycle 1 (`sft_pickup_seed`) | cycle 2 (`sft_cycle2`) |
|---|---|---|
| rows (train / val) | 21 261 / 2 161 | 92 071 / 10 586 |
| episodes | 1 757 | 6 839 |
| untrained baseline val_mse | 0.207699 | 0.069711 |
| best val_mse | 0.044725 | **0.023034** |
| best epoch | 59 (**cap, not convergence**) | **38**, then overfits |
| optimizer steps to best | ~2 490 | ~6 840 |
| reachable fraction | 0.90494 | 0.91398 |
| `headline_over_control` | 1.024 | 0.988 |
| LoRA stage | best e0, +0.5% | **no epoch beat baseline** |

---

## 2. A warm start with no grasp cannot bootstrap one

`phase4_pick_up_iter0` warm-started from the placement checkpoint
`step_7753190` and ran 1.83 M steps to nothing:

* **222 normal-start grasps in 102 016 normal-start worlds** — 0.22%. Peak rate
  0.0181 against a 0.30 promote gate; **119 of 191 updates recorded exactly
  zero**, the last 54 consecutively.
* Validation 0.0000 at all nine readings; validation reward *fell* 0.726 → 0.573.
* Update 1 already read 0.002, so nothing decayed — the skill was never there.
  Phase 3 had measured pick_up at exactly 0.000 after excluding it from the SFT
  mix, and that is what a placement checkpoint inherits.

Three mechanisms, all confirmed in the event file:

**The approach half trained on amplified noise.** Group reward std was ~0.10
for normal-start groups against 0.93–1.36 for pre-grasped ones. With
`grpo_normalize_group_advantage: true` the advantage is the centred reward over
the group std, so — as the collector states at
`mjwarp_rank_local_collector.py:2471` — a group that separated nothing
contributes the same gradient magnitude as one that separated a success from a
failure. `grpo_min_group_reward_std: 0.05` sits *below* that 0.10 noise floor
and never fired. The approach got worse, not slower: target cosine 0.140 →
0.089, pad force 1.67 → 0.16 N, grasp rate 0.156 → 0.031.

**The placement warm start's release was reinforced.** `physical_release_rate`
climbed 0.026 → 0.173. Validation `min_ee_z_mean_m` held at 0.213, never
descending to the 0.185–0.203 grasp height — a placement policy hovers at
release height and never visits the desk.

**The vision LoRA's own kill criterion fired.** `vla_lora/vision_modules` = 48,
so the tower *was* adapted, while `vla_lora/kl` sat at ~6e-5 for all 191 updates
and never moved. The config had written the test in advance. Turned off; it
costs a full backward through the tower per update and is what pins
`vla_microbatch_size` at 4.

---

## 3. Seed the grasp by SFT, then run RL

The loop's usual order — RL on the new instruction, then retention SFT — assumes
the new instruction is reachable from what the policy already does. pick_up is
the exception: it is the only instruction gated on a **grasp**, and the
accumulated policy has zero of it.

Inverted for this one instruction: harvest pick_up from the phase-1 reference
`step_7505256`, pool it into the bank, SFT, *then* RL.

The harvest settled the load-bearing assumption. `step_7505256` came from a run
trained under the broken 0.08 grasp offset, and its 0.289–0.328 was only ever
measured by later probes under the fixed 0.0075. Measured directly at cap 0.06:
**0.231 / 0.262 / 0.231 / 0.248 / 0.254 / 0.357**, mean 0.279. The grasp is real.

Result — `sft_pickup_seed`, 23 422 rows across four instructions:

| instruction | before | after seed SFT |
|---|---|---|
| pick_up @0.06 | 0.000 | **0.0964** |
| move_to @0.19 | 0.311 | 0.3203 |
| plate @0.20 | 0.523 | 0.5736 |
| bowl @0.20 | 0.2765 | 0.2721 |

pick_up installed from nothing at 37% of its source, no family paying for it.

**This also answered phase 4's open question #2.** move_to returned to 0.3203
against cycle 1's 0.311 — after **8.5 M steps** of erosion versus cycle 1's
495 k. Retention is a stable fixed point, not a sawtooth; no behaviour-cloning
anchor inside the RL objective is needed.

---

## 4. The seed worked, and then the curriculum gave it back

`phase4_pick_up_iter1`, from the seed:

* Normal-start grasp rate opened at **0.514** and peaked at **0.621** — against
  0.0181 for the whole of the unseeded run.
* `group_reward_std_mean_normal_start` went **0.10 → ~1.0**. The
  noise-amplification mechanism was gone, and `grpo_min_group_reward_std` never
  had to be touched: the seed fixed it.
* Validation reached **0.1387**, the first non-zero pick_up validation of the
  campaign.

Then the ladder unwound it. The three promotions, with the instantaneous rate
the lagging EMA hid:

| step | cap | EMA | graspN |
|---|---|---|---|
| 205 806 | 0.06 → 0.08 | 0.353 | 0.522 |
| 383 716 | 0.08 → 0.10 | 0.551 | 0.517 |
| 562 813 | 0.10 → 0.13 | 0.408 | **0.298** |

A 0.30 gate cannot bind a policy that starts at 0.51, so the cap advanced on its
cooldown — the degenerate pair `17a83f7` removed, reached from the other side.
The config's calibration caveat had warned the gate might be too *high* and stall
at rung one; seeded, the opposite happened.

The third promotion is the kill, and the damage is hysteretic: the cap demoted
0.13 → 0.10 → 0.08 → 0.06 and the policy did **not** come back with it. At cap
0.06 at the end, graspN read **0.000** against 0.514 at the same cap at the
start, with `physical_release_rate` 0.000 → 0.118 and success-given-pre-grasped
0.99 → 0.65.

Gate re-fitted to what the policy sustains: promote **0.30 → 0.50**, demote
**0.20 → 0.35**, cooldown **15 → 25** (the EMA decay of 0.95 is a ~20-update
window, so 15 let the cap move before the EMA reflected the rung it moved onto —
literally the 562 813 failure).

---

## 5. The gate fix worked, and falsified the diagnosis

`phase4_pick_up_iter2`, resumed from iter1's peak at step 205 806:

| | iter1 | iter2 |
|---|---|---|
| promoted to 0.13? | yes, at 562 k | **never** |
| steps held at cap 0.10 | 179 k | **507 k** |
| graspN above 0.45 until | ~500 k | **~790 k** |

Roughly 400 k more healthy steps, and the fatal rung was never reached.

**But the decay happened anyway, at a fixed cap.** From 385 k to 892 k the cap
sat at 0.10 while graspN fell 0.565 → 0.248 with no promotion at all. The ladder
accelerated the collapse in iter1; it does not cause it.

The metric that moves monotonically in both runs is
`post_grasp_action_z_mean`: **+0.277 → +0.578** here, +0.204 → +0.478 there,
against a plant curve where +0.30 lifts 83 mm and succeeds 14/15. graspN tracks
it inversely the whole way — 0.551 at +0.25, 0.248 at +0.372, 0.032 at +0.578.

This is the campaign's structural failure, not a new one: **one residual serves
both phases**, so an ever-growing upward bias for the lift removes the descent
the approach needs. `split_credit_at_grasp: true` is on and is not sufficient.
What feeds it is that success-given-pre-grasped sits at **0.99 from the first
update** — the seed carried the lift in already — while 48% of groups (35%
pre-grasped + 13% aligned) train nothing else. Half the batch keeps paying for
"command up" on a sub-task that was solved before the run started.

**Both runs converge on ~0.135 validation and then decay.** iter1 reached it
after 200 k steps from the seed; iter2 after 1.0 M. 2.5 M steps of RL between
them did not move the ceiling, which is why pick_up RL was stopped here rather
than iterated again.

---

## 6. Four times the data broke the retention ceiling

Phase 4 §6.1 ruled out epochs and pointed the remaining headroom at slice size.
Cycle 2 tested it: re-harvest every family, balance at the largest quota the
bank supports.

Availability after the new harvest (116 134 decisions pooled):

| instruction | episodes available | decisions available |
|---|---|---|
| move_to_object | 2 973 | ~66 600 |
| put_into_plate | 7 567 | ~83 800 |
| put_into_bowl | 4 152 | ~55 900 |
| **pick_up** | **1 498** | **26 109 — binding** |

Balanced at 26 000 → **~104 000 rows**, against cycle 1's 23 422. pick_up binds
because it is the one family not re-harvested from a fresh RL run; its pooled
source rate of 0.163 blends the 0.260 `step_7505256` rounds with the 0.155
`step_1003315` ones and is a statement about harvest cost, not demo quality —
only successful episodes are banked.

Result: **move_to 48.5% → 67.3% of reference, pick_up 37.1% → 66.8% of source.**
Two families, independent sources, the same ~67%.

### 6.1 The SFT now has a stopping point

Cycle 1's 60 epochs stopped at the cap while still improving. Cycle 2 overfits:
best val_mse **0.023034 at epoch 38**, then monotone rise to 0.024368 by 119
while train_mse kept falling 0.0184 → 0.0141. In optimizer steps that is ~6 840
against cycle 1's ~2 490 — more data supported more training, then turned.

120 epochs was ~3× past the turn and cost nothing: the residual stage runs in
**32 seconds**. The tool saves the best-val checkpoint, so the overfitting did
not reach the artefact. **~40 epochs is the setting for a 100 k-row set.**

### 6.2 The LoRA stage is now pure cost

No epoch beat the 0.02322163 baseline, so no adapter was applied and the
residual-only checkpoint stands — after **1 h 24 m**. The train loss fell
(0.0182 → 0.0173) while validation did not, so this is overfitting rather than a
step-size problem, which is the alternative the tool's own warning names.

Three cycles, three failures to help, and it is the single most expensive stage
in the pipeline. `--lora-epochs 0` from here.

---

## 7. Placement fell, and the reason is the base, not the data

plate 0.5736 → 0.3463 and bowl 0.2721 → 0.1702 look like the bigger dataset
hurting. They are not.

**The two SFTs started from different places.** Cycle 1 ran from
`step_7753190`, a placement policy with placement intact — that half of the mix
was *consolidation*. Cycle 2 ran from `step_1003315`, which is the seed adapter
after 1.2 M steps of pick_up RL, and phase 4 §3 measured what RL does to an
absent instruction: 0.64 → ~0.05. So cycle 2 had to *rebuild* placement from a
deep hole, and a rebuild is partial by construction.

Secondary: the new placement rows are slightly weaker than the ones they joined
— plate 0.613/0.527/0.437 across caps against the old slice's 0.648/0.600/0.490.
Bowl went the other way at the middle rung, 0.304 against 0.270.

So this is the alternating optimisation working as designed, one lap further
round: RL on pick_up eroded placement, the SFT rebuilt it partway, and placement
is now the family owed an RL turn. Exactly the position move_to was in at the
start of phase 4.

---

## 8. State, and what is open

Best single policy for the non-placement families:
`runs/phase4_bank/sft_cycle2/sil_sft_adapter.pt` — pick_up **0.1738** @0.06,
move_to **0.4316** @0.19. Best pick_up RL checkpoint:
`runs/phase4_pick_up_iter2/rl/step_1003315`, validation 0.1328.

Best PLACEMENT checkpoint, and the one to resume from:
`runs/phase5_placement_iter3_20260828_224948/rl/step_2754052` — overall 0.6211,
plate 0.7982, bowl 0.4073 (§11.1). iter4's lineage tops out at 0.3320 and is
superseded.

Open:

1. ~~Placement's RL turn.~~ Done, twice. iter3 took it past phase 4 on both
   families (§11.1); iter4 spent 5.25 M steps measuring what the
   place-not-drop constraint costs (§11.2). Open now: **do the extended
   ladders alone beat 0.6211?** That is the next run, and the criterion is
   exactly that number.
2. **Is place-not-drop worth buying?** The gate works and costs half the
   success rate. Re-arming it needs bowl at ~0.075 rather than 0.065 and
   `placement_wrong_drop_penalty` lowered alongside, run as its own experiment
   from a known peak. Whether the thesis needs the distinction is a separate
   question from whether it is achievable.
2. **Does the ~67% ceiling move again?** Both families landed there from very
   different starting fractions, which is either a coincidence or a second
   ceiling. pick_up is the family to test it on, since it is the one whose bank
   is exhausted at 1 498 episodes — harvesting it from `sft_cycle2` (0.1738,
   above the 0.155 that produced the last slice) would add rows and answer it.
3. **The post-grasp z drift.** §5 identifies it as the cause of both pick_up
   collapses and names the suspect: 48% of groups train only the lift, whose
   success is 0.99 from step 0. The config forbids annealing that fraction and
   cites a measured A/B — but that A/B ran on an *unseeded* policy whose lift
   was genuinely unlearned, so its premise no longer holds. Needs an explicit
   A/B against a fixed-0.5 control from `step_1003315`, not a config edit.
4. **The composed pick-and-place**, still untouched. See phase 4 §10, whose
   claim about the caught-stage gate was corrected in this campaign: there is
   one fraction for both placement families, annealed on the mean of their pass
   rates, so it never moves rather than moving for plate alone.

---

## 9. Tooling built here

* `sil_record --devices cuda:0,cuda:1` — shards a round range across GPUs by
  re-invoking the script per device, so each shard is the ordinary
  single-device path. Record files are named by round index so shards share one
  output directory. Tests cover the split, the argv rebuild and the pooled
  summary.
* `sil_sft --progress {auto,always,never}` — tqdm bars for both stages, gated on
  stdout being a terminal so a redirected run's log is unchanged.
* `train_cdpr_smolvla_pick_up_grpo_mjlab_dual_remote_resume.sh` now sources
  `run_naming.sh`; it was the only trainer entry point with no collision guard.

## 10. Two things watching the dataset videos showed

Both found by playing `sil_dataset_videos` output — the actual frames the SFT
consumed — rather than by reading a metric. Neither is visible in any number the
pipeline currently logs, which is the argument for looking at the data.

### 10.1 The gripper aims its wrist camera at the overview camera

Across instructions the policy rotates yaw so the end-effector camera faces the
overview camera. It is consistent enough to notice within a handful of clips.

**Yaw is an unconstrained axis for every instruction we train.** Grepping the
reward module, `yaw` enters a reward or success term in exactly two places:
`prepositioned_gripper_yaw_penalty`, which belongs to a different instruction
family, and the object-rotation tasks' `total_signed_rotation`. Nothing in
`pick_up`, `put_into_plate` or `put_into_bowl` references gripper yaw at all,
and `lock_non_commanded_axes` is `false`. The policy has a yaw channel at
`action_step_yaw: 0.08` and no reason to prefer any value.

So the behaviour costs nothing, which explains why it survives. What it does not
explain is why *this* orientation. The candidate worth testing: the residual's
state carries a 512-d vision feature pooled from both cameras' connector tokens
under a **fixed random projection** (`residual_vision_pooling: flat_random`).
Pointing the wrist at a fixed, high-contrast landmark makes that half of the
feature far more predictable than pointing it at whatever the workspace happens
to contain — the policy would be stabilising its own observation rather than
solving the task. That is an observation-hacking hypothesis, not a measurement.

Cheap test, if it is worth chasing: log the yaw distribution against success,
and A/B a run with yaw locked (`lock_non_commanded_axes: true`). If success is
unchanged, the behaviour is free and cosmetic. If it *drops*, the policy is
genuinely relying on a stabilised vision feature, which would say something
uncomfortable about how much of the encoder's usable signal is landmark rather
than object.

### 10.2 `put_into_*` succeeds by dropping, not by placing

The videos show the policy carrying the object over the receptacle and opening
the gripper from height. The object falls in, settles, and the episode scores.
Technically correct, and not the behaviour the task name promises.

**The predicate permits it by construction.** For
`CONTAINER_PLACEMENT_INSTRUCTION_TYPES` success requires XY within tolerance
(plate 0.091, bowl 0.057), `abs(object_z - container_z)` within
`put_container_z_tolerance` — **0.12 m** — plus `put_require_release: true` and
grasp history. Nothing anywhere constrains **how high the object was when the
gripper opened**. A release from 10 cm up and a release from 1 cm up produce the
same terminal state once the object settles, so the two are indistinguishable to
the success test and the reward pays them identically.

Tightening `put_container_z_tolerance` does not fix it: a dropped object also
ends up resting low in the container, so it passes the tightened test on a later
step. The discriminator has to be evaluated **at the release moment**, not at
the end — a `put_release_max_height` condition on the object's height above its
resting surface at the step the gripper crosses `put_release_opening_threshold`.

The knob does not exist yet. `put_downward_reward_enabled` is the nearest
existing machinery, but it measures displacement from the episode's initial
object height rather than clearance at release, so a high carry followed by a
high drop satisfies it.

**Time-reversing pick-from-container episodes does not substitute for this**,
which is worth writing down because the idea is appealing and the reason it
fails is not obvious. The bank stores `(frames, actions)` and the SFT trains
action prediction. Reversing a trajectory reverses the *states* kinematically,
but the actions that generate the reverse motion are not the reversed actions:
the plant is dissipative and gravity is not symmetric in time. Lifting a held
object needs a sustained positive z command — the lift-barrier probe measured
0.30 → 83 mm — while lowering the same object needs far less than the mirrored
negative, because gravity does the work and the measured plant gain (0.44-0.54,
asymmetric under load) does not invert. Played backwards the lift's commands are
still positive z during a descent; negated, they command a fall much faster than
the real one. Neither is a valid label. Only the gripper channel reverses
cleanly, and that is one of five.

The sound version of the same intuition is to use extraction episodes as a
**reset distribution** rather than as demonstrations: the poses a successful
lift-out-of-bowl passes through are exactly the poses a successful placement
must pass through, so they define a reverse curriculum of start states -- object
low in the container, gripper closed, task reduced to "release from here" --
which then anneals outward. That uses the trajectories for what they actually
certify (reachable, physically valid states) rather than for labels they cannot
support.

---

## 11. Placement's RL turn: recovered past phase 4, then paid for the gate

Two runs from `sft_cycle2`, whose placement the pick_up RL had left at
0.3463 / 0.1702.

### 11.1 iter3 — placement comes back higher than it started

`phase5_placement_iter3`, 525 updates, step 1.00 M → 5.01 M.

| | phase 4 best | iter3 |
|---|---|---|
| overall peak | 0.633 | **0.6211** @ 2 754 052 |
| put_into_plate | 0.791 | **0.7982** |
| put_into_bowl | 0.442 | **0.4655** @ 3 250 193 |

Both families ended above phase 4's ceiling. **The erode → rebuild → RL cycle
does not merely restore a family, it improves on the previous peak** — a
stronger statement about the loop than cycle 2's move_to result, because
placement had been driven much further down first.

**And 73% of the run was wasted at the ladder ceiling.** Plate reached 0.20 at
step 1 363 484 and bowl 0.19 at 1 529 700; the remaining **3.64 M** and
**3.48 M** steps ran with the cap pinned and pass-rate EMAs of 0.560 and 0.384
against a 0.30 promote gate. That is phase 4 §8 repeating verbatim, now at a
measured cost. Ladders extended to bowl 0.22/0.25 and plate 0.23/0.26.

Unlike pick_up's iter1, the gate behaved: the caps climbed over 500 k steps and
stopped because they ran out of rungs, not because they outran the policy.

### 11.2 iter4 — what the place-not-drop constraint costs

`put_*_release_max_height` armed at 0.080 (plate) / 0.065 (bowl) from the
measurement in §10.2, resumed from iter3's peak, 5.25 M steps to the 8 M
ceiling.

| | iter3 peak | iter4 |
|---|---|---|
| `release_clearance_mean_m` | 0.0867 | **0.0636**, flat from 3.5 M |
| overall validation | 0.6211 | 0.2295 → peak **0.3320** → 0.3027 |
| `release_clearance_worlds` | 821 | **524** |
| `physical_release_rate` | 0.057 | **0.021** |
| bowl cap | 0.19 (top) | 0.22 → **0.07**, never recovered |
| plate cap | 0.20 (top) | 0.23 → 0.20 |

**The gate did what it was built to do and the policy did learn it.** Release
clearance fell 2 cm in 3.5 M steps and then converged. At 0.068 against a plate
resting height of 0.058 (max 0.069) that is a genuine placement rather than a
drop.

**It cost half the success rate and never got it back** over 3.5 M further
steps of oscillation between 0.23 and 0.33.

Two mechanisms, both visible:

* **The bowl threshold was below what the policy could reach.** It converged at
  0.068 — above bowl's 0.065 gate, below plate's 0.080. Bowl releases were
  therefore mostly denied, its cap collapsed to 0.07, and because **one policy
  serves both families** that pulled the plate down with it. A re-arm sets bowl
  to ~0.075.
* **The policy compensated by releasing LESS OFTEN, not lower.** Releasing
  worlds fell 36%. This is the hovering optimum `placement_wrong_drop_penalty`
  was tuned against: at 0.25 releasing beats hovering once success probability
  passes ~11%, and the gate lowers exactly that probability. The penalty has to
  fall with the gate or hovering wins again.

**The run confounded itself, and that was avoidable.** The gate and the extended
ladders were armed together, so "the threshold is too tight" and "two
difficulty increases at once" cannot be separated from this data. Bowl was
promoted straight to 0.22 on resume — its EMA was 0.45, over the gate — into a
task that had simultaneously become harder. The campaign's own history is a
record of single-variable discipline and this run broke it.

Gate disabled; the armed values and this result are kept in the config comment
for whoever re-arms it. The extended ladders are the change with standalone
expected value and go forward on their own.

---

## 12. Predictions made here that came back wrong

* **"The 0.06 rung will hold for a long stretch — that is the ladder working."**
  Said before iter0. The approach half had no gradient to climb with, so waiting
  was never going to help. §2.
* **"Name record files by round index when `--rounds > 1`."** True of a serial
  harvest, false of the shards it was written for: a one-round shard takes
  `--rounds 1`, so two children wrote `record_00.npz` concurrently. One round
  lost, the survivor failed its CRC, and it surfaced hours later at replay.
* **"The promotion to 0.13 caused the collapse."** iter2 never promoted to 0.13
  and decayed anyway, at a fixed cap. The ladder accelerates it; the post-grasp
  z drift causes it. §5.
* **"Expect the first validation to drop."** Said before iter4, about arming
  the release-height gate. It dropped 0.6211 -> 0.2295 and recovered only to
  0.3320 over 5.25 M steps. The direction was right and the magnitude was not
  close, which is the difference between a caveat and a forecast.
* **"Arm the gate and extend the ladders together."** Not a prediction, a
  process error, and the more expensive one: it made the result of a 5.25 M
  step run unattributable. The two changes had independent rationales and
  should have been sequenced. §11.2.
* **"More data will let the SFT train longer."** It let it train longer in
  *steps* (2 490 → 6 840) but *fewer* epochs before overfitting than the epoch
  cap that stopped cycle 1. The two are not the same axis and the report had
  been reading them as one.
