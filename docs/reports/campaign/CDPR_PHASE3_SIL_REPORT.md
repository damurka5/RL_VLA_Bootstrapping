# Phase 3 — self-imitation from RL-generated successes

Everything measured while building and running the self-imitation pipeline,
including the results that came back against the plan and the measurement
errors that had to be corrected on the way. Written so a fresh session can
start from here without re-deriving any of it.

Tools built: `tools/audit/sil_record.py` (record / replay / compare /
dataset), `tools/audit/sil_sft.py`, `tools/audit/sil_sweep_table.py`,
`tools/audit/sil_eval_table.py`, `tools/audit/sil_action_stats.py`.
Tests: `tests/test_sil_record_reporting.py`, `tests/test_sil_sft_dataset.py`,
`tests/test_sil_action_stats.py` (71 total, all passing).

---

## 1. Headline results

**Self-imitation lifted the weaker placement task and destroyed anything left
out of the mix.**

| cap | put_into_plate | put_into_bowl | pick_up |
|---|---|---|---|
| 0.01 | +0.017 | **+0.128** | 0.066 → 0.002 |
| 0.03 | −0.001 | **+0.130** | 0.048 → **0.000** |
| 0.06 | +0.011 | +0.020 | 0.033 → **0.000** |
| 0.10 | +0.050 | −0.002 | 0.027 → **0.000** |

* **bowl improved ~+0.13 at the near caps.** Paired t = 2.99 / 3.12 against a
  3.18 bar at df = 3 — under the bar, but all eight round-pairs positive.
  Pooling the two near caps as one contrast gives t = 4.67 (sign test
  p = 0.0078). That grouping was chosen after seeing which caps moved, so it
  is suggestive rather than established. **Confirming it needs ≥12 rounds.**
* **plate is flat.** It was already at 0.86–0.94, near ceiling.
* **pick_up was erased, not degraded.** Zero successes in every round at three
  caps, paired t = −4.9 to −5.2. It was excluded from the SFT mix because its
  source rate (0.043, from the placement checkpoint) was mostly luck. Fine-
  tuning the residual on placement alone removed the skill entirely.
* **The gain vanishes past cap 0.06.**

**The single most important lesson for the next phase: SFT on a subset of
instructions erases the others completely.** This is phase 2's forgetting
result again, now with no rehearsal at all, and total rather than partial.

---

## 2. Validation was never deterministic

LeRobot's `sample_actions` runs `if noise is None: noise = self.sample_noise(...)`
and `sample_noise` is a bare `torch.normal` with **no `generator=`**. The CDPR
wrapper never passes `noise`. So the SmolVLA prior is a fresh global-RNG draw
on every forward, and `deterministic_action_chunks_tensor` is deterministic
only in its *residual* — `action = tanh(prior + residual)` with a stochastic
prior is a stochastic action.

Two identical validation rounds gave pick_up 0.059 vs 0.101.

Decomposition, all measured on the placement checkpoint at cap 0.01, 512 worlds:

| source | magnitude | flips verdicts? |
|---|---|---|
| stochastic prior | dominant | yes — fully closed by `--seed-torch` |
| SmolVLA bf16 forward | **0.000e+00** bitwise at step 0 (512×5) | no |
| MuJoCo Warp, actions pinned | mean 7.6e-06 m, max 7.7e-03 m | **no** — 0 flips |
| closed-loop amplification | 0.218 m EE divergence | yes — **34/512 = 6.6%** |

**Consequences.**

* Always pass `--seed-torch`. Without it the first commanded action differs in
  all 512 worlds by up to 3.6e-02.
* Even seeded, ~6.6% of episodes flip between identical runs, because
  micron-scale physics noise is chaotically amplified through the policy. A
  seeded round is reproducible in its first decision and not in its outcome.
* **Single-round n = 512 cannot resolve a slice whose rate is near or below
  6.6%** — which is the entire pick_up column of the phase-3 briefing
  (0.089 / 0.054 / 0.030 / 0.030 are mutually indistinguishable).
* **With actions pinned the verdict noise floor is zero** (replay agreement
  1.0, 0 flips), so smoothing survival rates are clean.
* bf16 is exonerated; `--deterministic-kernels` was never needed.

---

## 3. Architecture facts that cost time to establish

**The residual actor** (`ResidualChunkActor`, shared with the Octo path):

```
features = cat([state(518), prior.flatten(8*5)])     # 558
residual = tanh(net(features))                        # 558→1024→1024→40
action   = tanh(prior + residual_scale * residual)    # residual_scale = 1.0
```

* The net **sees the prior**, not just the state. A stochastic prior is
  therefore not a moving target for a fixed state — the net can compensate the
  draw exactly. 1.66 M parameters.
* `chunk_size = 8` but `replan_every = 4`, so the actor emits 8 action slots
  and the plant executes 4. Slots 4–7 have no target and are never read
  (`deterministic_action_chunks_tensor` slices `[:, :count]`).
* **The reachable action set is `[tanh(p − s), tanh(p + s)]`** — verified
  empirically that the actor cannot leave it even with weights at σ = 50. A
  target outside it cannot be fitted by any weights and the loss sits at a
  floor that looks exactly like underfitting.
* `state_dim = 6` proprio + `residual_vision_dim = 512` = 518, identical in the
  pick_up and placement configs, so their datasets pool.

**Things the phase-3 briefing said existed and did not.**

* `rl_vla_bootstrapping/lchol/` is **unreachable from the MJWarp path** — zero
  references in `mjwarp_rank_local_collector.py` or `smolvla_grpo_mjwarp_cdpr.py`.
  It lives on the legacy non-batched trainer behind `--lchol-mode`. Its
  `CDPRLCHOLSpec` is a **second, independent implementation** of the success
  predicates, reading a dict the MJWarp collector never builds. Do not reuse
  it; relabel by re-running `evaluate_active_sparse_tasks`.
* `training.sft` builds an **OpenVLA** plan (`build_openvla_sft_plan`). There
  was no SmolVLA SFT path; `tools/audit/sil_sft.py` is new.
* `scripts/render_cdpr_task_reference_episodes.py` has **no action-injection
  path** — it is a closed-loop scripted oracle, not a replay harness.

**LoRA and images.** LoRA sits on the action expert and updating it needs
gradients through the vision tower, which needs the 256×256 frames. The
dataset stores the pooled 512-wide vision feature instead — ~37 MB per round
against ~5.0 GB. `sil_sft.py` therefore trains the residual only and copies the
source checkpoint's `vla_lora` verbatim (dropping it would restart a resume
from a zero adapter).

---

## 4. Recording and replay

The env steps the **whole chunk**, so recording at the chunk producer loses
three of every four executed actions. Recording happens at two points:

* `backend.step(actions, active_mask)` — the executed command and the mask.
* `evaluate_active_sparse_tasks(...)` — patched in the collector's namespace
  and **called through**, giving production's own observations and verdict.

The two report `active` independently and a mismatch raises.

**Replay is a seeded re-run, not a state restore.** The reset is a pure
function of `base_seed + rank·1e6 + update_index·1e7 + round_index·100003`, so
the same cap and round index reproduce the starts by construction; actions are
substituted at `backend.step`.

`terminated = success | wrong_place_settled | timeout`, so an **episode
terminates on success** and the policy keeps emitting into a frozen world
afterwards. Demonstrations must be truncated at `first_success_step`, and the
decision containing the success keeps its dead tail masked rather than dropped
(shortening chunks would misalign the action head). Measured dead-action
fraction: 0.086.

---

## 5. Hindsight relabelling — dead

`pick_up` as a prefix of `put_into_*` was the briefing's strongest relabel.
Measured `would_relabel_fraction = 0.0` over 309 successful placements: peak
held lift is 6.5 mm mean / 15 mm p90 against a 0.05 m threshold. `target_lift`
is measured from `initial_target_positions` captured **at reset**, and
placement starts with the object already up, so the carry moves it sideways
and down and never registers a lift.

`push_forward` / `push_backward` do not exist in `ACTIVE_INSTRUCTION_TYPES`,
and the push predicate is x-axis only (`push_motion = sign * delta[:, 0]`), so
adding them is a new predicate plus reward plus curriculum rung, not a schema
edit.

---

## 6. Smoothing

Moving average **strictly dominates** — higher reduction *and* higher survival
than both alternatives on both families:

| family | method | reduction | survival | ratio |
|---|---|---|---|---|
| pick_up | **moving_average w5** | **0.582** | **0.909** | **6.4** |
| pick_up | ema α=0.3 | 0.547 | 0.804 | 2.79 |
| pick_up | median w5 | 0.519 | 0.708 | 1.78 |
| placement | **moving_average w5** | **0.596** | **0.938** | **9.66** |
| placement | ema α=0.3 | 0.561 | 0.903 | 5.76 |
| placement | median w5 | 0.512 | 0.731 | 1.90 |

**Why:** a centred moving average has zero phase lag. EMA lags by its time
constant and median produces staircase artifacts, and the controller
re-anchors its target to the measured EE pose every step — so a lagged command
fights an anchor that has already moved. The caveat is that MA's zero lag comes
from being non-causal; legitimate for dataset construction, unavailable to a
runtime filter.

**Window sensitivity tracks the success radius, and not the way expected:**

| window | plate (0.091 m) | bowl (0.057 m) |
|---|---|---|
| 5 | **1.000** | 0.864 |
| 9 | 0.981 | 0.836 |
| 13 | 0.932 | 0.829 |

Plate degrades with window; **bowl loses ~14 % at any window and barely more
after**. Bowl's marginal episodes fail under any perturbation — its radius is
tight enough that widening the filter costs little extra.

Per-instruction step-delta reduction at plate=13/bowl=13/pick_up=5:
plate 0.360→0.086 (76 %), bowl 0.349→0.088 (75 %), pick_up 0.374→0.161 (57 %).

**Divergence is baseline, not caused by smoothing.** The `none` control
diverges 18 worlds (pick_up) and 7 (placement); moving average gives 9 and 11.
Both `none` arms survive at exactly 1.0, which is the control passing.

---

## 7. Dataset and SFT

Placement dataset: 4065 episodes / 17 621 decisions / 31.3 MB, state_dim 518.
After dropping the pick_up slice: 15 917 rows, 3 952 episodes.

Per-rung source success rates (bowl / plate):
cap 0.01 0.795/0.922, 0.03 0.733/0.864, 0.06 0.714/0.800, 0.10 0.295/0.562.
**A pooled rate describes no collection that ever ran** — it must be reported
per rung.

SFT (residual only, 300 epochs, lr 1e-4, batch 512, episode-level split):

* `reachable = 0.918` — 8.2 % of targets are unfittable by any weights.
* untrained baseline val MSE **0.1426** → converged **0.0274** (train 0.0135).
* The baseline is large because the observations come from the *smoothed*
  rollout, so it measures the open-loop demonstrator gap, not a bug.
* Training takes seconds. **The loss is not the verdict; the success rate is.**

---

## 8. What the demonstrations actually contain

`aim` = cosine(command, direction-to-goal) minus the same cosine with the
**commands permuted across rows**. Calibrated on synthetic policies:
0.000 = knows nothing, 0.163 = half the rows aiming, 0.327 = clean servo.

| slice | rows | conc | aim |
|---|---|---|---|
| pick_up | 4 048 | 0.058 | 0.122 |
| put_into_bowl | 20 735 | 0.372 | 0.150 |
| put_into_plate | 28 853 | 0.444 | 0.113 |

**The demonstrations contain real but weak goal-conditioning — roughly a third
of a clean servo.** Unambiguously above zero (null spread ±0.002), and
unambiguously far from aiming. Consistent with the known 3–5 cm encoder
localization against tasks needing ~2 cm.

Per object the aim is uneven: pick_up ranges 0.037 (apple) to 0.230 (tomato);
plate ranges 0.022 (tomato) to 0.173 (orange). There is no consistent ordering
across tasks.

**A systematic −x drift.** Mean command `[−0.207, +0.001]` against a spread of
0.412 — the constant component is about a third of typical command magnitude,
and the x histogram is visibly skewed negative.

**Only four of six objects appear.** `robocasa_banana` and `robocasa_mug` are
wider than the gripper's open gap in the seeded pose (both contact at 0.98),
so they contribute attempts and never successes. Any pick_up set harvested on
success is a four-object dataset presenting itself as six.

---

## 9. Measurement errors made and corrected

Recorded because each was a plausible-looking number that meant nothing.

1. **Two nulls certified a policy that knows nothing.** A pure fixed drift
   scored `cosine_gap` 0.383 against a world-shuffle null and `target_cosine`
   0.78 against a rotation null — matching or beating the real recordings.
   Pairing with another world's goal does not hold geometry fixed (the arm
   tracks its own goal, so its own direction is short and variable while
   another's is long); rotating the goal destroys the systematic geometry
   along with the pairing. **Only permuting the commands preserves both
   marginals and breaks only the pairing.**
2. **`direction_concentration` is confounded too.** Where the arm sits
   systematically to one side of the goals, a *genuine servo* scores 0.82. It
   only accuses when `aim` is also near zero.
3. **A constant noise floor cannot judge a small rate.** Adding a ±0.047 band
   derived from the 6.6 % flip rate to a task whose rate is 0.03 labelled the
   pick_up collapse "not resolved". Under a true rate of 0.033 the probability
   of 0/664 is ~2e-10. Replaced with a paired-by-round t test.
4. **Replay must use the recording's own round index.** Defaulting to 0 while
   the harvest walked 0–3 fed one episode set's actions into another's starts;
   the dataset silently became 472 episodes instead of 4230 and plate survival
   read a plausible 0.259. Reset identity is now a precondition that raises.
5. **Episode ids must be unique per source file.** Rung/round/world collides
   across families at one cap (`sil_harvest_0.03` and `sil_pickup_0.03` both
   label `cap_0.03`), silently merging trajectories and splitting one across
   train and validation.
6. **Output names must be unique.** A fixed `replay.npz` overwrote all but the
   last round; naming by stem alone then overwrote all but the last *rung*.
7. **Placement aims at the receptacle.** A placement episode starts holding its
   target, so cosine against the target measures gripper slop and post-release
   retreat — it read −0.40 before the geometry was fixed.

---

## 10. Reference commands

```bash
# Harvest (RL baseline for free: same harness, caps, rounds, seed)
python tools/audit/sil_record.py --mode record --rounds 4 --seed-torch 0 \
  --start-distance-cap 0.03 --checkpoint <adapter> --config <config> \
  --output out/harvest_0.03

# Smooth + re-simulate (round index defaults to the recording's own)
python tools/audit/sil_record.py --mode replay --smooth moving_average \
  --smooth-window 5 --smooth-window-by-instruction put_into_bowl=13 \
  --actions out/harvest_0.03/record_00.npz --seed-torch 0 \
  --start-distance-cap 0.03 --checkpoint <adapter> --config <config> \
  --video-worlds 6 --output out/smooth

# Dataset, SFT, verdict
python tools/audit/sil_record.py --mode dataset --inputs out/smooth/replay_*.npz --output out/ds
python tools/audit/sil_sft.py --dataset out/ds/demonstrations.npz --checkpoint <adapter> --output runs/sft
python tools/audit/sil_eval_table.py --baseline out/harvest_0.03 --candidate out/eval_sft_0.03
python tools/audit/sil_action_stats.py --recordings 'out/smooth/replay_*.npz' --output out/stats --successes-only

# Forensics, no GPU
python tools/audit/sil_record.py --mode compare --actions A.npz --against B.npz --output out/cmp
python tools/audit/sil_sweep_table.py --per-instruction
```
