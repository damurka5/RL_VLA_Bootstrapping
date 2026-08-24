# Phase 4 — one policy for every instruction, via a retention bank

Everything measured while unblocking the placement RL stall and building the
retention loop, including the two designs that were tried and abandoned and the
predictions that came back wrong. Written so a fresh session can start from here
without re-deriving any of it.

Supersedes one decision in `configs/examples/cdpr_smolvla_phase4_placement_loop.yaml:20`
("ONE POLICY PER FAMILY"). The deliverable is a single policy that holds every
instruction. That note describes why phase 3 split them, not what phase 4 does.

Tools built: `tools/audit/sil_refresh_priors.py`, and
`--rows-per-instruction` / `--quota-seed` on `sil_record --mode dataset`.
Tests: `tests/test_sil_refresh_priors.py`, plus `RowQuotaTests` and
`test_quota_is_reported_and_reaches_the_arrays` in
`tests/test_sil_record_reporting.py` (173 total, all passing).

---

## 1. Headline results

**Retention works, and it is not paid for out of the new skill.**

Retention SFT on a three-family bank, from
`runs/phase4_placement_iter1_20260823_165011/rl/step_1504301`:

| instruction | cap | before | SFT, 20 epochs | SFT, 60 epochs |
|---|---|---|---|---|
| **move_to_object** | 0.19 | 0.062 / 0.060 / 0.117 → **0.080** | 0.371 / 0.254 / 0.270 → **0.298** | 0.383 / 0.256 / 0.295 → **0.311** |
| put_into_bowl | 0.20 | 0.230 / 0.231 → 0.2305 | 0.250 / 0.238 → 0.244 | 0.319 / 0.234 → **0.2765** |
| put_into_plate | 0.20 | 0.508 / 0.492 → 0.500 | 0.538 / 0.492 → 0.515 | 0.519 / 0.527 → **0.523** |

move_to up **3.9×**, every round positive; placement up on both families and
down on neither. Against the move_to reference of 0.641 the SFT closed **41% of
the gap**, starting from a policy that had lost the skill outright.

Retention was not paid for out of placement — bowl gained 20% and plate 5%
while move_to was being rebuilt. The bank's placement rows are the policy's own
recent output, so that half is consolidation rather than distillation, and it
behaves like it.

**And the RL stall was one config line.** Zeroing `episode_offset_std` (see §2)
took the placement run from 193 updates of frozen caps to both ladders topped
out in 74:

| | stalled run | after the fix |
|---|---|---|
| bowl cap | 0.10 for 4.8 h | 0.10 → 0.13 → 0.16 → **0.19** in 1.25 h |
| plate cap | 0.17 for 4.8 h | 0.17 → **0.20** in 9 min |
| plate `pass_rate_ema` | pinned ~0.28 | 0.295 → **0.455** |
| `log_std_mean`, first ~75 updates | −1.441 → −1.502 | −1.422 → **−1.432** |
| `clip_fraction_mean` | 0.36 → 0.48 | 0.268 → 0.302 |

---

## 2. The gate was reading a metric the offset had depressed

The stalled run sat at `pass_rate_ema` ≈ 0.28 against a 0.30 promote gate while
`log_std_mean` fell −1.44 → −1.93 and validation regressed from a 0.4375 peak
to 0.34 across three consecutive validations.

The gate reads
`instruction_successes_normal_start/<name> / instruction_worlds_normal_start/<name>`
(`smolvla_grpo_mjwarp_cdpr.py:2296`) — the raw training success rate, measured
with `episode_offset_std: [0, 0, 0, 0, 0.3]` injected. `sil_record` contains no
reference to that offset and never applies it. Same checkpoint, same caps:

| | with the offset (trainer) | without it (harvest) |
|---|---|---|
| put_into_bowl @ 0.10 | 0.303 | ~0.44 |
| put_into_plate @ 0.17 | 0.266 | 0.401 |

A ~14-point gap, entirely on the measurement path. The config had already
written the trigger condition for this, before it happened: *"if success falls
as the cap reaches the upper rungs, cut this before anything else."*

The offset is a **discovery tool for the release**, not a training-time
regularizer. The 7.7-sigma argument for it applies to a policy that has never
released; once cap 0.02 scores 0.92/0.98 *without* it, discovery is complete and
the offset is pure cost. Zeroing it also arrested the entropy collapse, which is
the second half of the config's own warning — GRPO cancels a sustained injected
bias by narrowing the policy's own sigma.

`min_log_std: -1.72` was added in the same commit as a floor. It has not bound
(`log_std_saturated_fraction` is driven by the ceiling), so the offset appears
to have been the whole cause.

---

## 3. Forgetting is caused by RL, not by SFT

Measured on the move_to config at cap 0.19, three rounds each:

| checkpoint | move_to |
|---|---|
| move_to reference (`step_11009573`) | 0.643 / 0.619 / 0.662 |
| after 1.0 M steps of placement RL | 0.004 / 0.053 / 0.103 |
| after single-family placement SFT | 0.000 / 0.018 / 0.035 |

RL takes 0.64 → ~0.05; the SFT then removes what is left. So a retention pass is
not *preserving* a skill, it is *rebuilding* one from near zero — which sets the
scale of the mix. A token 5% slice cannot do that job.

---

## 4. The bank stores frames and actions, and nothing else

Three columns, three lifetimes:

* `action` — what the plant executed. True forever, however far the network
  drifts.
* `prior` — the SmolVLA chunk the residual was conditioned on, and the SFT's KL
  anchor. Stale the moment the LoRA moves.
* `state` — proprioception plus a vision feature pooled from *the recording
  adapter's* connector tokens. Same.

So the durable artifact is `runs/phase4_bank/*/replay_*.npz` +
`frames_*.npz`, and everything network-dependent is re-derived at the point of
use by `sil_refresh_priors`. `demonstrations.npz` is derived, not durable: it
gets **rebuilt** when a family joins, because the quota has to rebalance across
the whole pooled set. "Extending" it would leave the old slices at old weights.

### 4.1 Cross-checkpoint replay was the first design, and it does not work

The plan was to obtain fresh priors by replaying the bank's actions under the
current checkpoint — `sil_record --mode replay` pins actions from playback while
`patched_action` records whatever network is loaded, so it looks exactly right.
Measured, move_to bank at cap 0.19:

| replay | survival | max_ee_delta |
|---|---|---|
| own checkpoint, unsmoothed | 333/333 = 1.000 | 9.6 mm |
| own checkpoint, smoothed w5 | 307/333 = 0.922 | 1.32 m |
| **placement checkpoint, smoothed w5** | **30/3072 = 0.010** | — |

Recording, pinning, smoothing and survival all work; loading a different
checkpoint destroys it. The root cause in the reset or horizon was not chased,
because the deeper objection settles it: this should never have been a rollout.
A forward pass over stored pictures is what was wanted, and dragging a
simulator, a reset and a termination predicate through it adds three ways to
diverge and no information. `sil_refresh_priors` does the forward pass. There is
no physics in it, so there is nothing to diverge.

### 4.2 The join, and the guard on it

`resolve_frame_rows` matches `<parent>/<replay stem>/r<round>w<world>` against
`frames_<stem>.npz` through `frame_join_key`. An earlier version of this join
matched 0 of 33 102 rows *after* a whole harvest had been paid for, so
`tests/test_sil_refresh_priors.py` pins it on synthetic files.

`--min-resolved-fraction` (default 0.99) refuses to run on a partly-resolvable
bank: the rows that survive would be the ones whose replay happened to keep
pictures, which is a selection nobody chose. Iteration 1's harvest resolved
11 930 of 50 102 rows at `--frame-worlds 64`; the bank, replayed at
`--frame-worlds 0`, resolves 17 501 of 17 704.

The 203 that did not resolve are **20 episodes losing their whole tail**
(`decision >= entry["decisions"]`), not 200 episodes losing a final decision.
move_to lost nothing. At 1.5% of episodes this cannot move a result of this
size, so the run proceeded at `--min-resolved-fraction 0.98`.

---

## 5. Balance the mix in decisions, not episodes

Episode lengths differ by a factor of four across families, so an episode quota
that reads as balanced is not. `--rows-per-instruction 6000` over the pooled
bank:

| instruction | episodes | decisions | decisions/episode |
|---|---|---|---|
| move_to_object | 281 | 6 009 | 21.4 |
| put_into_bowl | 469 | 5 694 | 12.1 |
| put_into_plate | 622 | 6 001 | 9.7 |

Rows within 5% across families whose episode counts differ by 2.2×. A
300-episode-per-family quota would have handed move_to 2.5× plate's gradient.
Whole episodes only — `_episode_split` holds out whole episodes, and a quota
that cut mid-episode would put consecutive decisions of one trajectory on both
sides of the train/val line.

`put_into_bowl` used all 469 available rather than reaching the quota; at 95% of
budget that was accepted rather than harvested again.

---

## 6. Retention lives in the residual, not in the LoRA

Two SFT runs on the same refreshed bank, differing only in `--epochs`:

| | 20 epochs | 60 epochs |
|---|---|---|
| untrained baseline val_mse | 0.049979 | 0.049979 |
| residual best val_mse | 0.019656 (e19, **still falling**) | 0.017305 (e58, flat from ~e50) |
| LoRA untrained-on-frames | 0.019876 | 0.017591 |
| LoRA best val_mse | 0.019555 (e1) | 0.017575 (e1) |
| LoRA contribution | 1.6% | **0.09%** |
| `headline_over_control` | 1.024 | 1.024 |

The 20-epoch run stopped at its epoch cap, not at convergence. The 60-epoch run
converged and improved val_mse a further 12%.

The LoRA stage contributes essentially nothing and overfits from epoch 2 with
`val_kl` at ~1e-4. This is consistent with the known gradient path: the vision
LoRA reaches the action only through the prior, because the residual's 512-wide
vision feature is pooled under an unconditional `no_grad` behind a fixed random
projection. **Retention is a residual phenomenon.** `--lora-epochs 4` is ample;
8 was twice what it used.

`headline_over_control = 1.024` inside the SFT, against 4.47 reported by the
refresh itself, is the pipeline's own consistency check passing: the refreshed
states agree with the SFT's recompute to within the numerics floor, so the 4.47
was measuring distance to the *recording* network — exactly the distance the
refresh exists to close.

### 6.1 Epochs were not the binding constraint

Scored in simulation, the 12% MSE gain is worth almost nothing on the metric
that matters:

| | 20 epochs | 60 epochs | Δ |
|---|---|---|---|
| residual val_mse | 0.019656 | 0.017305 | −12% |
| move_to @ 0.19 | 0.298 | **0.311** | **+0.013** |
| put_into_bowl @ 0.20 | 0.244 | 0.2765 | +0.033 |
| put_into_plate @ 0.20 | 0.515 | 0.523 | +0.008 |

Every number moved the right way and every one is small; tripling the training
bought 1.3 points of move_to. **The MSE-to-success relationship has saturated**,
which rules out "train longer" as the lever and points the remaining headroom at
the data — the move_to slice is 6 009 rows of 13 363 available — or at the
residual's capacity, or at a ceiling on how far another policy's demonstrations
can carry a skill this network no longer has.

Curiously placement gained more from the extra epochs than move_to did (+0.033
on bowl against +0.013), so the extra fitting went mostly into the
self-imitation half of the mix rather than the distillation half. 60 epochs is
still the better setting; it just is not the way to lift retention.

---

## 7. Predictions made here that came back wrong

In the phase-3 report's spirit, because each cost a step.

* **"Replay the bank under the current checkpoint to refresh its priors."**
  Destroyed the bank, 0.010 survival. §4.1.
* **"The unresolved rows will be the terminal decision of many episodes."**
  They were the whole tail of twenty episodes. Benign either way, but the shape
  was wrong, and the two have different implications — losing 200 success
  moments is a bias, losing 20 tails is not.
* **"`headline_over_control` will drop on the pooled refresh, because the
  placement rows came from this very checkpoint."** It did not: 4.466 against
  4.346. Unexplained at the refresh's own probe; resolved by the SFT's check
  reading 1.024, which is what actually matters.
* **"Bowl may not be bankable at 0.22 validation."** Its harvest measured
  0.449 / 0.297 / 0.231 across caps 0.10 / 0.15 / 0.20 — better than the pre-fix
  checkpoint at every far rung. Validation against a fixed distribution and
  harvest against the training distribution are different questions and the
  first one had been read as if it answered the second.

---

## 8. State, and what is open

Single policy `runs/phase4_bank/sft_retention_e60/sil_sft_adapter.pt`:
move_to 0.311 @ 0.19, put_into_plate 0.523 @ 0.20, put_into_bowl 0.2765 @ 0.20.
Both placement ladders topped out (bowl 0.19, plate 0.20).

Open:

1. ~~Does the 60-epoch checkpoint retain better?~~ Marginally. §6.1. Epochs are
   not the lever.
2. **Does retention hold across a second cycle?** This is the one that decides
   whether the architecture works. The loop is alternating optimization: RL
   erodes, SFT rebuilds. If a second cycle lands near 0.31 again that is a
   stable fixed point and the design is sound; if it lands lower it is a
   sawtooth and the fix moves inside the RL objective — a behaviour-cloning
   anchor on bank data, so the skill never falls to 0.08 in the first place.
   One data point cannot tell these apart.
3. **41% of the gap, or more?** Epochs are excluded. What remains is slice size
   — move_to used 6 009 of 13 363 available rows — which
   `--rows-per-instruction` can test today by raising the quota, at the cost of
   a mix deliberately weighted toward the forgotten skill. A genuine per-family
   weight would need a small extension to that flag.
4. **The composed pick-and-place.** `PlacementCaughtStageCurriculum`
   (`smolvla_grpo_mjwarp_cdpr.py:1016`) already anneals the fraction of
   placement episodes that start with the object caught, and is disabled by
   default. See §9.

---

## 9. Lift-then-place is a curriculum knob, not a new instruction

Placement episodes start with the object already between the fingers
(`curriculum/placement_caught_fraction` logs 1.0). The trainer carries a built,
wired and checkpointed curriculum for annealing that away, currently off:

| key | default | meaning |
|---|---|---|
| `placement_caught_curriculum_enabled` | `false` | the whole stage |
| `placement_caught_object_fraction` | 1.0 | starting fraction caught at reset |
| `placement_caught_fraction_min` | 0.25 | floor — 0.0 demands the grasp every episode |
| `placement_caught_fraction_step` | 0.10 | step per reduction |
| `placement_caught_reduce_success` | 0.60 | success EMA above which the fraction drops |
| `placement_caught_restore_success` | 0.35 | and below which it is restored |
| `placement_caught_cooldown_updates` | 15 | between changes |

It is state-dicted into `extra_state` alongside the approach curriculum, so it
survives a resume the same way the caps do.

The geometry that blocked every earlier grasp attempt is closed:
`pick_grasp_height_offset: 0.0075` (measured pad offset, against the old 0.08)
with the controller floor at 0.18, and the pick_up config records 3/3 successful
grasps under it.

Gate to watch: `reduce_success: 0.60` is measured against placement success,
which at the topped-out caps is 0.244 (bowl) and 0.515 (plate). Bowl would never
clear 0.60 and its fraction would never anneal, so a run started this way trains
the composition on plate alone. Either lower the gate to each family's reach or
start the annealing at a lower cap where success is high — bowl reaches 0.449 at
cap 0.10.

---

## 10. Reference commands

```bash
# Bank a mastered instruction: record from the checkpoint that has it, then
# replay under THAT SAME checkpoint with every frame kept.
python tools/audit/sil_record.py --mode record --rounds 6 --seed-torch 0 \
  --start-distance-cap <cap> --checkpoint <mastering ckpt> --config <its config> \
  --output runs/phase4_bank/<name>_actions
python tools/audit/sil_record.py --mode replay --smooth moving_average \
  --smooth-window 5 --actions runs/phase4_bank/<name>_actions/record_NN.npz \
  --seed-torch 0 --start-distance-cap <cap> --checkpoint <same ckpt> \
  --config <its config> --record-frames --frame-worlds 0 \
  --output runs/phase4_bank/<name>_demos

# Rebuild the balanced pooled bank (every family, every time a family joins).
python tools/audit/sil_record.py --mode dataset \
  --inputs runs/phase4_bank/*_demos/replay_*.npz \
  --rows-per-instruction 6000 --output runs/phase4_bank/dataset

# Re-derive state and prior for the network the SFT will start from.
python tools/audit/sil_refresh_priors.py \
  --dataset runs/phase4_bank/dataset/demonstrations.npz \
  --frames runs/phase4_bank/*_demos/frames_*.npz \
  --checkpoint <current ckpt> --vision-lora \
  --output runs/phase4_bank/refreshed

# Retention SFT. Watch headline_over_control (~1.0) and the LoRA's best epoch.
python tools/audit/sil_sft.py \
  --dataset runs/phase4_bank/refreshed/demonstrations.npz \
  --checkpoint <current ckpt> --frames runs/phase4_bank/*_demos/frames_*.npz \
  --epochs 60 --train-vision-lora --lora-epochs 4 --lora-rows 8192 \
  --output runs/phase4_bank/sft_retention

# Score every family at the caps and seeds its baseline used. A harvest round
# IS a record run of the pre-SFT checkpoint, so it doubles as the baseline arm.
python tools/audit/sil_record.py --mode record --rounds 3 --seed-torch 0 \
  --start-distance-cap <cap> --checkpoint <sft ckpt> --config <family config> \
  --output runs/phase4_bank/eval/after_<family>
```
