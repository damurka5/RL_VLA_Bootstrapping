# SmolVLA move-to training: run history and metric reference

CDPR · SmolVLA · GRPO · `move_to_object` · MJWarp backend

- Prepared 2026-07-28
- Config: `configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml`
- Current run: two objects, warm start from `step_7405457`
- Code: `64c1437`

What happened across ~30M training steps, why the 16M run collapsed, and what
every scalar on the TensorBoard page measures.

---

## 1. The finding

The 16M-step run did not fail on the curriculum, and it did not fail on
distractors. It failed because the action distribution slowly inflated for
twelve million steps while every curriculum metric looked healthy.

`log_std_mean` bottomed at −1.227 at 3.6M and rose monotonically for the rest of
the run. The old `max_log_std` ceiling was −0.30, so it never bound — the entire
collapse happened underneath it.

| global step | log_std_mean | validation/success_rate |
|---:|---:|---:|
| 202k | −1.204 | 0.043 |
| 3.56M | **−1.227** (minimum) | 0.030 |
| 4.63M | −1.200 | 0.048 |
| **7.41M** | −1.162 | **0.073** (peak) |
| 7.65M | −1.150 | — |
| 9.81M | −1.131 | 0.063 |
| 11.29M | −1.100 | 0.051 |
| 12.77M | −1.050 | 0.039 |
| 13.73M | −1.000 | 0.033 |
| 15.21M | −0.913 | 0.006 |
| 16.00M | −0.895 | 0.007 |

Supporting evidence over the same span:

- `entropy_mean` 0.83 → 4.15
- `policy_target_cosine_mean` 0.25 → 0.14, touching −0.007 at 12.3M (task-blind)
- `candidate_reward_mean` 0.79 → 0.47
- `group_pass_rate_mean` 0.30 → 0.14

### The distractors are exonerated

Split by phase, the two-object window scored the highest mean reward of the
entire run at unchanged grounding:

| phase | scene objects | reward | cosine | log_std |
|---|---:|---:|---:|---:|
| 0 – 4M | 1 | 0.696 | 0.223 | −1.206 |
| 4 – 8M | 1 | 0.723 | 0.248 | −1.180 |
| **8 – 11M** (post-restart) | 2 | **0.761** | 0.243 | −1.134 |
| 11 – 16M | 2 | 0.581 | 0.137 | −1.004 |

Validation held 5–6% through 11.6M. The curriculum restart at the unlock worked
exactly as designed: cap 0.23 → 0.05, pass rate 0.47, re-climbed to 0.17 within
600k steps.

---

## 2. What happened

Five runs. The first three each hid the next problem, because a broken
measurement upstream made every downstream number look plausible.

### 5M — baseline

Full-workspace starts, one object. Ended around 1–2% validation success. Became
the warm-start source for everything after.

### 2M (2026-07-25 07:52) — the curriculum was never connected

**Discarded. Fixed in `c76cbb1`.**

`BatchedRandomWorkspaceMoveToResetter.reset()` called the base class, which
places the end-effector under the curriculum cap — then overwrote both the pose
and the horizon from its own sampler, which took a *minimum* goal distance and
no maximum. Every curriculum signal was dropped on the floor.

The tell: `group_pass_rate_mean` was 0.044 at a 0.03 m cap and 0.037 at a 0.34 m
cap. A cap that changes nothing produces a pass rate that does not respond to
it. The gate promoted on a distance-independent background rate and marched
0.03 → 0.34 m in 1.6M steps while the realized start distance stayed ~0.15 m
out. `curriculum_horizon_coupling_enabled` was dead for the same reason.

### 350k (2026-07-25 13:06) — the gate thresholds were fitted to the bug

**Discarded. Fixed in `17a83f7`.**

With the plumbing repaired, promote/demote of 0.03/0.01 became degenerate: those
numbers had been tuned while the pass rate was pinned under 0.045 by the
previous bug. The real range is 0.06–0.41, so promote was true on every update
and demote was unreachable. The cap ran away again, for a completely different
reason.

Now 0.30/0.12 with a 15-update cooldown, and the EMA is re-seeded on every cap
change so a promotion is not judged on the easier level's average.

> **Rule this produced:** whenever a measurement bug is fixed, re-derive every
> threshold that was tuned against the broken numbers.

### 16M (2026-07-26 11:35) — the curriculum worked; the policy diffused

**Best checkpoint `step_7405457`.**

Everything curriculum-side behaved. The cap climbed under a real gate, demoted
twice when the policy fell behind, and the distractor unlock at 8M restarted it
cleanly. Validation reached 7.3%.

Underneath that, `log_std_mean` had been rising since 3.6M. Nothing pushes it
down, so a net-positive entropy bonus wins on a long run. Past roughly 12M the
policy was sampling too widely to servo, and validation fell to 0.7%.

Fixed in `7d99c3e`:

- `max_log_std` −0.30 → **−1.10**, just above the [−1.23, −1.15] band the policy
  occupied for its whole productive phase, so diffusion is impossible rather
  than discouraged.
- `entropy_coef` 0.0005 → **0.0**. Lowering it from 0.002 slowed the drift but
  could not stop it.
- Both applied to all three staged configs — they must match across phases or
  the weight hand-off reintroduces the failure.

### Running — diffusion closed off, grounding opened up

Two objects from step 0 (`64c1437`), `ee_workspace_z_bounds` ceiling 0.52 → 0.47.

With one object the instruction is dead text: there is nothing to disambiguate,
so every single-object step buys object-agnostic servoing and no grounding at
all. That is what the tomato/orange and bowl/plate confusions are.

---

## 3. Metric reference

Expected values are from the healthy phase of the 16M run unless noted.

### Curriculum — how hard the task currently is

| metric | what it measures | expect |
|---|---|---|
| `curriculum/start_max_goal_distance_m/move_to_object` | Cap on the 3-D distance from the EE start to the hover point (target XY at z = 0.27). Starts land *on* this sphere, not inside it, because the workspace floor equals the hover height. | 0.03 → 0.34, steps of 0.02 |
| `curriculum/approach_pass_rate_ema/move_to_object` | Decaying average of the group pass rate — the gate's only input. Above 0.30 promotes, below 0.12 demotes, then a 15-update cooldown. Re-seeded on every cap change. | 0.12 – 0.30 |
| `curriculum/horizon_decisions` | Episode length in policy decisions, interpolated from the cap. **Summed across ranks — halve it.** Each decision is 4 environment steps. | 16 → 64 logged (8 → 32 per rank) |
| `curriculum/scene_objects_max` | Objects in the scene. Counts are sampled uniformly per group, so 1-object scenes stay in the mix as rehearsal. | 2 |
| `curriculum/enabled` | **Not the approach curriculum.** This is the Reverse Frontier shell curriculum, which move-to does not use. Reads 0 by design and says nothing about whether the start-distance cap is active. | 0 |

### Policy health — is the policy still a controller

| metric | what it measures | expect |
|---|---|---|
| `log_std_mean` | Mean log standard deviation of the Gaussian action distribution — how widely the policy samples around its mean action. The single most important number in this run. Clamped to [−5.0, −1.10] on the path that produces actions. | −1.23 → −1.15, **flat, not rising** |
| `entropy_mean` | Entropy of the same distribution. Moves with `log_std_mean`; the more legible version of it. | 0.8 – 1.4 |
| `policy_target_cosine_mean` | Cosine between the first commanded action and the true EE→target direction. +1 points at the target, 0 task-blind, negative away. **With two objects this becomes a grounding measurement**, because approaching the wrong object scores negative. | ≥ 0.24 |
| `prior_target_cosine_mean` | The same probe on the frozen SmolVLA alone, without the residual. The control: sits near 0 because the frozen model does not ground this task. | ≈ 0.00 |
| `policy_target_alignment_rate` | Fraction of worlds whose first action has positive cosine. Coarser than the cosine but less noisy. | 0.59 – 0.70 |

### Learning signal — what GRPO has to work with

| metric | what it measures | expect |
|---|---|---|
| `candidate_reward_mean` | Mean terminal return per world, bounded in [0, 1] (see reward inversion below). Falls as the cap widens — expected, not regression. | 0.70 – 0.80 at cap 0.15–0.17 |
| `group_pass_rate_mean` | Fraction of worlds reaching the 2 cm success window. Should *drop* after each cap promotion and recover — that coupling is the proof the cap is real. | 0.20 – 0.30 |
| `candidate_successes`, `candidate_worlds` | Raw counts behind the pass rate, summed across ranks. 1024 worlds per update at the current layout. | sum |
| `advantage_mean`, `advantage_std` | Group-normalized advantage. Pinned at 0 and 1 by construction — a sanity check that normalization ran, nothing more. | 0.0 / 1.0 |
| `informative_groups`, `groups_collected` | Groups with non-zero reward spread, over groups collected. If the first falls well below the second, GRPO is training on an increasingly empty loss mask. | roughly equal |

**Reward inversion.** The dense reward is

```
r = 1 / (1 + (max(d_3d - 0.02, 0) / 0.08)^2)
```

where `d_3d = ||ee_position - hover_target||` and `hover_target = (target_xy, 0.27)`.
It inverts to a distance:

```
d = 0.02 + 0.08 * sqrt(1/r - 1)
```

Because the map is nonlinear, inverting the *mean* reward is biased toward the
close episodes — treat it as an order-of-magnitude read. Worked example: the
2M-run's `candidate_reward_mean` of 0.284 inverts to ~0.147 m, which is how the
broken curriculum was caught while it was logging a 0.03 m cap.

### Optimization — is the update well behaved

| metric | what it measures | expect |
|---|---|---|
| `approx_kl_mean` | KL between the sampling policy and the updated policy. High for PPO conventions, but stable across the whole run. | 0.08 – 0.10 |
| `clip_fraction_mean` | Fraction of samples hitting the PPO clip range (0.20 / 0.28). ~0.40 means the update consistently wants to move further than the trust region allows. | ≈ 0.40 |
| `gradient_norm_mean` | Pre-clip gradient norm of the residual, against `max_grad_norm` 1.0. | 4 – 6 |
| `vla_lora/kl` | How far the LoRA-adapted action expert has moved from the frozen prior. **Currently ~1e−4**: decayed 300× and effectively inert since 3.2M, so all learning is in the residual. The 3.42M LoRA parameters cost ~9% throughput for nothing. Open decision. | ≈ 0.0001 (inert) |
| `vla_lora/grad_norm` | Gradient still flows even though the KL is flat — the adapter is being held, not starved. | 1 – 3 |

### Validation — the only number that generalizes

| metric | what it measures | expect |
|---|---|---|
| `validation/success_rate` | Held-out episodes on *uncapped* full-workspace starts, balanced across all eight objects, every 200k steps. The curriculum does not touch it, which is what makes it the headline number. Previous peak 7.3%. | new baseline |
| `validation/final_xy_distance_mean_m` | Mean terminal XY distance. Context: a uniform random point in this workspace averages ~0.25 m from a uniform target, so 0.27 m means the average episode is not closing distance — the successes are a minority mode. | down from 0.272 |
| `validation/reward_mean` | Same shaped reward on validation episodes. Rises earlier than the success rate because it is continuous — the better early-progress signal. | up from 0.18 |
| `validation/by_object/…` | Per-object breakdown, 128 episodes each. At these rates one episode is ±0.8%, so read the ordering, not the individual values. Banana has been persistently last. | noisy |

### Throughput — is the box working

| metric | what it measures | expect |
|---|---|---|
| `selected_actions_per_second_global` | Environment actions kept for the gradient, per second, across both ranks. Lower at short horizons because per-update overhead amortizes over fewer actions. | 105 – 141 |
| `global_step_increment` | Steps added per update. Scales with the coupled horizon: ~3k at cap 0.03, ~14k at cap 0.21. A falling value after a cap restart is expected, not a stall. | 3k – 16k |
| `trajectory_work_amplification` | Sampled actions over selected actions — the chunk size, 8. Constant unless the action codec changes. | 8.0 |

---

## 4. Before you read a number, check how it was reduced

At the update boundary every metric is all-reduced with `SUM` across both ranks
(`_synchronize_update_metrics_once`). Only then are some keys divided back down
by world size — those ending in `_mean`, `_max`, `_std`, `_rate` or `_time_s`,
those starting with `loss_`, and a short explicit list.

**Everything else is a two-rank sum.** That is why:

- `curriculum/horizon_decisions` reads 64 when the horizon is 32
- `updates` reads 2 for one update per rank
- `groups_collected` reads 128 for 64 groups each

Divide by `distributed_world_size` before interpreting any of them.

The curriculum cap and pass-rate EMA are the exception among `curriculum/*`
keys: the trainer writes them after the collective, so they are already per-run
values and need no correction.

---

## 5. Reading the current run

Roughly the first million steps are re-climb — the warm start resets the cap to
0.03 by design (`load_weights_only` discards curriculum state), and last time a
trained policy took about 600k steps to get back to 0.17.

Object separation is at least 0.16 m, so **while the cap is below ~0.15 the
named target is simply the nearest one** and selection is not yet exercised. A
flat cosine at cap 0.09 is not failure.

| watch | good | means something else |
|---|---|---|
| `log_std_mean` | Flat near −1.15 to −1.20 → diffusion fixed, run to 16M | Pins at −1.10 and sits there → the clamp is holding but upward pressure survives `entropy_coef = 0`, so it comes from the policy gradient. Survivable, but worth investigating. |
| `policy_target_cosine_mean` | ≥ 0.24 and rising once the cap passes 0.15 | Flat at 0.24 with the cap above 0.16 → the residual's frozen vision feature cannot separate target from distractor. That becomes the next problem to attack. |
| `group_pass_rate_mean` | Drops after each promotion, recovers | Lower than the single-object numbers once past cap 0.15 → chance target selection. Expected, not regression. |
| `validation/success_rate` | Rising from its new baseline | Not comparable to the 7.3% peak — the start-height ceiling changed 0.52 → 0.47, which makes uncapped starts easier. |

### Attribution caveat

This run changes the entropy ceiling, the entropy coefficient, the object count
and the start-height bound at once. The evidence that two objects are safe is
strong — it was the best phase of the 16M run — so the trade is worth it against
burning steps on a phase that provably teaches no grounding. But if the run
disappoints, that ambiguity is the price, and the first move is a single-object
re-run to separate them.

---

## 6. Settled questions

Recorded so they are not re-litigated.

**No image crop exists.** Both preprocessing sites (`smolvla_cdpr.py:168`,
`:307`) use `F.interpolate` to 256×256 with no cropping. The 320×240 render is
squashed (1.33× horizontal compression), not cut.

**Nothing is out of frame.** Projecting the object envelope through the overview
frustum (camera at `(0, -0.541, 0.5125)`, view direction `(0, 0.866, -0.5)`,
fovy 45° → 57.8° horizontal at 4:3): the worst case, a near corner at
`(0.205, -0.205)`, sits at 83% of the half-width and 67% of the half-height. The
camera covers |x| ≤ 0.254 m at that depth; objects reach 0.205 m.

**Rendering 256×256 would crop the workspace.** `fovy` is vertical; horizontal
is derived from the aspect ratio. At 256×256 the horizontal FOV drops 57.8° →
45°, covering only |x| ≤ 0.190 m at the near desk edge — objects at 0.205 m
would fall outside. Preserving coverage requires raising `fovy` to ~57.8°, which
also widens the vertical view. Not done.

**Moving objects closer to the overview camera makes framing worse.** The camera
looks toward +y and 30° down, so closer means shorter forward distance and a
narrower frustum at that depth:

| grid position | forward distance | % of half-width | % of half-height |
|---|---:|---:|---:|
| far corner (0.205, +0.205) | 0.817 m | 45% | 24% |
| near corner today (0.205, −0.205) | 0.460 m | 81% | 67% |
| shifted 5 cm toward camera | 0.417 m | 89% | 88% |

The direction that centres objects is *away* from the camera, or a smaller x
spread — though the latter collides with the 0.16 m separation invariant that
keeps a plate and bowl from overlapping.

**"Moves upward when it loses the object" is the frozen prior's +0.45 mean Z
bias**, documented next to `residual_scale`. When the residual fails to
localize, the output is the prior, and the prior goes up. It is the signature of
failed grounding, not a separate bug.

**The wrist camera already sees the target for every cap ≤ 0.11.** Footprint
half-width at the desk is `(ee_z − 0.13) × 0.797` — 11.2 cm at hover height,
27 cm at the 0.47 ceiling. Since the cap bounds 3-D distance and
`curriculum_cap_includes_z` puts starts on the cap sphere, wrist visibility is
guaranteed by geometry through the whole early curriculum. Making it a hard
constraint at wider caps would widen the train/validation gap, since validation
is uncapped and guarantees no such thing.

**The strict 2 cm threshold is not what hides successes.** From the 24-episode
qualitative audit at `step_7804192`: 2 cm → 8.3%, 5 cm → 12.5%, 10 cm → 20.8%,
15 cm → 41.7%. If episodes ended just outside the window, 5 cm would jump. 79%
never get within 10 cm, and per-object best-over-episode distances are
0.15–0.33 m.
