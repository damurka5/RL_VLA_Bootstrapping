# move_to validation — phase 4, iteration 0, step 11009573

Everything measured on the checkpoint
`runs/phase4_move_to_iter0_resume_20260818_080928/rl/step_11009573/smolvla_grpo_adapter.pt`
in one six-leg run on 2026-08-21, plus the geometry preflights that fix the
regimes those legs test. Written so a paper can be drafted from this file
without re-deriving anything, and so a fresh session can reproduce every number
with one command.

Run: `runs/move_to_validation/20260821_085119_phase4_move_to_iter0_resume_20260818_080928_step_11009573/`
Harness: `tools/audit/move_to_validation_videos.py`, driven by
`scripts/validate_cdpr_smolvla_move_to_remote.sh`.
Totals: **6656 episodes, 832 distinct scenes, 30 recorded MP4s, 1 diverged
world.**

---

## 1. Headline

**On the distribution it was trained on, the policy reaches the named object in
63.0% of episodes (645/1024).**

| | |
|---|---|
| Success | **0.6299** (645 / 1024) |
| 95% CI, clustered by reset | **± 0.0731** |
| Distinct scenes | 128 (8 GRPO replicas each) |
| Median start distance | 0.145 m (XY), 0.166 m (3-D to the hover point) |
| Median best XY distance reached | 0.014 m |
| Success criterion | XY ≤ 0.02 m of the named object **and** gripper height in [0.26, 0.28] m, at any env step |
| Episode length | 32 policy decisions = 128 env steps |

Three facts make this number readable as a validation metric rather than a
demo:

* it is produced by the trainer's own `collector.validate_round` — the same
  reset, reward, horizon, termination and success predicate the training loop
  uses, with the deterministic residual mean and nothing reimplemented;
* the start distribution is the one the checkpoint earned, not a proxy: the
  approach-curriculum cap restored from `extra_state` is **0.19 m, the top of
  the configured ladder**, and 0/1024 episodes started outside it;
* the two preconditions are measured per episode rather than assumed —
  1024/1024 episodes had the named object inside the overview frame and inside
  the reachable workspace.

An independent replication of the same configuration on a different scene seed
(leg F, seed 1000505 instead of 1000000) scores **0.5947 ± 0.0731** — the same
number within noise, on 128 different scenes.

---

## 2. All six legs

Each leg is the same harness with one knob moved. Only leg A is the metric;
B–F carry `counts_toward_validation_metric: false` in their manifests and must
not be pooled with it.

| leg | what differs | episodes | success | 95% CI |
|---|---|---:|---:|---:|
| **A `train_config`** | nothing — the training configuration | 1024 | **0.6299** | ±0.0731 |
| F `one_object_many_places` | A on a different scene seed (clips pinned to one object) | 1024 | 0.5947 | ±0.0731 |
| B `multi_object` | 2–3 objects instead of 1–2 | 1024 | 0.4600 | ±0.0778 |
| E `scattered_objects` | 2–3 objects + placement lattice loosened | 1024 | 0.3516 | ±0.0724 |
| C `multi_object_wrist_blind` | 2–3 objects + start cap 0.33 m (past the trained 0.19) | 1536 | 0.1751 | ±0.0471 |
| D `uncapped_workspace` | cap disabled — start anywhere ≥ 0.12 m away | 1024 | 0.1719 | ±0.0579 |

CIs are clustered by reset: the eight candidates of a GRPO group share their
scene *and* their start pose, so the independent unit is the group (128, 128,
128, 128, 192, 128 respectively), not the episode. Treating episodes as
independent would understate the interval by ≈ √8.

Tolerance sweep on the best XY distance reached (leg A / leg B / leg D):

| threshold | A train_config | B multi_object | D uncapped |
|---|---:|---:|---:|
| ≤ 2 cm | 0.658 | 0.497 | 0.205 |
| ≤ 3 cm | 0.738 | 0.553 | 0.253 |
| ≤ 5 cm | 0.788 | 0.608 | 0.302 |
| ≤ 10 cm | 0.876 | 0.726 | 0.348 |
| ≤ 15 cm | 0.938 | 0.889 | 0.445 |

The 0.658 → 0.630 gap in leg A is the height window: the XY condition alone is
met more often than XY and the 0.26–0.28 m band simultaneously.

---

## 3. What the number depends on

### 3.1 Start distance

The dominant variable. Leg A, by start-distance bin:

| start XY | episodes | success |
|---|---:|---:|
| < 0.12 m | 280 | 0.746 ± 0.116 |
| 0.12–0.15 m | 288 | 0.715 ± 0.134 |
| 0.15–0.18 m | 320 | 0.506 ± 0.139 |
| 0.18–0.21 m | 136 | 0.500 ± 0.208 |

Leg D, which removes the cap and therefore extends the axis, shows where the
policy stops:

| start XY | episodes | success |
|---|---:|---:|
| 0.12–0.15 m | 72 | 0.389 |
| 0.15–0.18 m | 128 | 0.422 |
| 0.18–0.21 m | 128 | 0.359 |
| 0.21–0.25 m | 128 | 0.227 |
| 0.25–0.30 m | 144 | 0.132 |
| 0.30–0.40 m | 280 | **0.000** |
| > 0.40 m | 144 | **0.000** |

Beyond 0.30 m the policy scores nothing at all: 0/424.

**This is not the episode budget running out.** 32 decisions × 4 actions = 128
commanded steps at a measured effective XY step of ~0.0075 m (the plant
realises ~0.5 of the nominal 0.015), i.e. ~0.9 m of reachable travel — four
times the largest start distance in the leg. What the far episodes actually do,
by bin (median closed distance = start − best):

| start XY | median start | median closed | median best | closed > 5 cm |
|---|---:|---:|---:|---:|
| 0.21–0.25 m | 0.228 m | 0.148 m | 0.086 m | 73.4% |
| 0.25–0.30 m | 0.269 m | 0.049 m | 0.220 m | 47.9% |
| 0.30–0.40 m | 0.352 m | **0.043 m** | 0.300 m | 45.7% |
| > 0.40 m | 0.447 m | 0.074 m | 0.357 m | 56.9% |

(leg A closes 0.116 m of a 0.145 m start for comparison). At 0.21–0.25 m the
policy still travels most of the way and misses on the last 8 cm; past 0.25 m
it stops closing at all, with ~0.9 m of budget left unused. The far-start
collapse is therefore a localisation failure, not a horizon limitation.

**Distance-matched comparison.** Restricting every leg to the same start band
[0.12, 0.19) m removes most, but not all, of the gap:

| leg | matched episodes | success |
|---|---:|---:|
| A train_config | 744 | 0.586 ± 0.090 |
| F same config, other seed | 744 | 0.561 ± 0.086 |
| B multi_object | 776 | 0.409 ± 0.088 |
| C wrist_blind (cap 0.33) | 464 | 0.410 ± 0.109 |
| D uncapped | 224 | 0.402 ± 0.167 |
| E scattered | 800 | 0.326 ± 0.081 |

The residual gap between A and C/D at matched XY is explained by the third
dimension: with the cap active the start height is confined so the **3-D**
distance to the hover point also respects the cap (p95 = 0.190 m), while with
the cap off the same XY band reaches p95 = 0.217 m. Matching on the 3-D
distance instead moves A to 0.609 and D to 0.447 — the ordering survives, so
distance is not the whole story either.

### 3.2 Distractors

Within leg A, which mixes one- and two-object scenes by construction:

| scene | episodes | success |
|---|---:|---:|
| 1 object | 552 | 0.745 ± 0.089 |
| 2 objects | 472 | 0.496 ± 0.111 |

Legs B and E extend the axis:

| scene | B multi_object | E scattered |
|---|---:|---:|
| 2 objects | 0.487 ± 0.109 | 0.427 ± 0.099 |
| 3 objects | 0.433 ± 0.112 | 0.255 ± 0.102 |

Adding the *first* distractor costs ~25 points; the second costs ~5 more. That
shape matters: it says the cost is not "clutter" but "ambiguity about which
object is meant", and the next section measures exactly that.

### 3.3 Language grounding — the named object versus the nearest one

Object separation on the desk is ≥ 0.15 m and the trained cap is 0.19 m, so in
most multi-object episodes the named object is *already* the closest thing to
the gripper and the instruction is not load-bearing. Splitting on that:

| leg | subset | episodes | success | ended closer to the **named** object |
|---|---|---:|---:|---:|
| B | named already nearest | 792 | 0.515 ± 0.088 | 0.832 ± 0.065 |
| B | **named NOT nearest** | 232 | **0.272 ± 0.147** | **0.388 ± 0.166** |
| E | named already nearest | 752 | 0.435 ± 0.087 | 0.822 ± 0.071 |
| E | **named NOT nearest** | 272 | **0.121 ± 0.091** | 0.294 ± 0.134 |
| C | named already nearest | 656 | 0.354 ± 0.089 | 0.834 ± 0.075 |
| C | **named NOT nearest** | 880 | **0.042 ± 0.032** | 0.185 ± 0.066 |

Where the episodes actually end, on multi-object scenes:

| leg | subset | reached the named object | reached a **wrong** object (≤ 2 cm) | reached neither |
|---|---|---:|---:|---:|
| A | all multi-object | 0.496 | 0.083 | 0.422 |
| A | named not nearest | 0.125 | **0.323** | 0.552 |
| B | all multi-object | 0.460 | 0.133 | 0.407 |
| B | named not nearest | 0.272 | **0.409** | 0.319 |
| E | named not nearest | 0.121 | 0.368 | 0.511 |

Read together these say something specific and, for the paper, more interesting
than a single success rate: **grounding is real but partial.**

It is not zero. 27.2% of the episodes that genuinely require reading the
instruction still succeed, and 38.8% of them end closer to the named object
than to any other. The floor for that second number does not have to be
assumed: leg C, where the same split is applied to episodes the policy cannot
solve at all (far starts, target often out of the wrist frame), scores 0.185 —
that is what "ends closer to the named object" looks like when nothing is
driving the choice. Leg B's 0.388 sits well above it.

It is also not solved. In 40.9% of those episodes the gripper arrives within
2 cm of the *wrong* object: it executes the task competently against the wrong
referent. The failure mode is misidentification, not incompetence.

### 3.4 The wrist camera is load-bearing

Leg C raises the cap to 0.33 m specifically so that some episodes begin with
the named object outside the wrist camera's field of view. Whether an object is
in that view is not read from pixels but bounded geometrically: the wrist
camera hangs off the end-effector with a 15° tilt and fovy 60°, and the ball
joint makes its realised orientation only partly a function of the commanded
pose, so an object more than 58.9° from nadir is **certainly** out of frame
regardless of how the wrist hangs (`tools/audit/start_distance_probe.py`).

| condition | episodes | success |
|---|---:|---:|
| target certainly outside the wrist frame (leg C) | 200 | **0.0000** |
| target possibly inside it (same leg) | 1336 | 0.2013 ± 0.053 |
| every object in the scene outside the wrist frame (leg C) | 64 | **0.0000** |
| target certainly outside the wrist frame (leg D) | 392 | 0.0051 |

Controlled by start distance, because blind episodes also start farther away
(median 0.293 m vs 0.216 m):

| start XY bin | blind | in-frame |
|---|---:|---:|
| 0.20–0.25 m | 0.000 (n=40) | 0.095 ± 0.069 (n=432) |
| 0.25–0.28 m | 0.000 (n=32) | 0.087 ± 0.087 (n=208) |
| 0.28–0.31 m | 0.000 (n=56) | 0.055 ± 0.074 (n=128) |
| 0.31–0.35 m | 0.000 (n=72) | 0.000 (n=40) |

Leg D reproduces it independently (0/32 blind at 0.25–0.28 m against 0.194
in-frame).

Pooling the two legs, which is the honest way to state it: **2 successes in 592
wrist-blind episodes, 0.34%** (exact 95% upper bound 1.1%); leg C alone is
0/200 (upper bound 1.5%). The two successes are in leg D, whose blind episodes
are the least far of the pooled set. A further 11 blind episodes (4 in C, 7 in
D) reached the 2 cm XY tolerance at some step without ever satisfying the
height window simultaneously — so the collapse is not quite absolute, and the
residue lands exactly where the predicate is strictest.

The overview camera saw the target in **1.0000** of all 6656 episodes, in every
leg. So this is not "the scene was unobservable": the object was on screen the
whole time in a camera the policy also receives. The claim the data supports is
narrower and stronger — *the localisation the policy acts on comes through the
wrist view; the overview alone is not sufficient for it.*

Inside the trained regime the same effect appears as a gradient rather than a
cliff (leg A, by wrist angle from nadir):

| wrist angle | episodes | success |
|---|---:|---:|
| 20–30° | 280 | 0.693 ± 0.132 |
| 30–40° | 352 | 0.645 ± 0.124 |
| 40–50° | 344 | 0.555 ± 0.134 |

### 3.5 Object placement

Leg E widens the placement lattice: the common scene shift goes ±0.015 → ±0.03 m
and the per-object jitter ±0.01 → ±0.015 m, with the guard that the closest two
objects can come (grid step − jitter = 0.15 m) stays above the widest realistic
pair (plate 0.091 + bowl 0.057 = 0.148 m). Measured: offset from the lattice
cell rises from 10.5 mm std / 24.7 mm max to 19.3 mm / 44.4 mm, and every
object stays inside the overview frame and the reachable workspace
(max |coord| 0.225 m).

Cost: 0.460 → 0.352 at equal object counts, and 0.433 → 0.255 on three-object
scenes. So the policy is measurably tuned to the placement statistics it
trained on — worth stating plainly in the paper, because the 3×3 lattice is a
property of the environment, not of the method.

### 3.6 Per object

Leg A, 128 episodes each, catalogs balanced by construction:

| object | success | mean best XY | distinct lattice cells used |
|---|---:|---:|---:|
| robocasa_bowl | 0.828 ± 0.167 | 0.0223 m | 8/9 |
| robocasa_tomato | 0.820 ± 0.163 | 0.0240 m | 8/9 |
| robocasa_plate | 0.758 ± 0.162 | 0.0126 m | 8/9 |
| robocasa_orange | 0.727 ± 0.197 | 0.0271 m | 7/9 |
| robocasa_potato | 0.609 ± 0.233 | 0.0370 m | 8/9 |
| robocasa_mug | 0.500 ± 0.223 | 0.0494 m | 9/9 |
| robocasa_apple | 0.492 ± 0.216 | 0.0520 m | 9/9 |
| robocasa_banana | 0.305 ± 0.173 | 0.0540 m | 7/9 |

The spread (0.31 → 0.83) is wider than the interval on any single object and
orders roughly by apparent size and compactness in the image: the large flat
receptacles are easiest, the thin elongated banana is hardest. This is a
perception-side result, not a control-side one — the controller is identical
across objects.

### 3.7 Aiming, not stumbling

Cosine between the first commanded XY action and the true direction to the
named object, at decision 0, before any feedback:

| leg | all episodes | successes | failures |
|---|---:|---:|---:|
| A train_config | +0.042 | **+0.185** | −0.202 |
| B multi_object | +0.052 | +0.304 | −0.162 |
| D uncapped | +0.028 | +0.173 | −0.002 |

The very first action already separates the outcomes. A policy that succeeded
by random walking into the target would show no such split.

### 3.8 Terminal geometry

Leg A: median final gripper height 0.2784 m against the 0.27 m hover target
(successes 0.2783, failures 0.2818) — the descent is learned and the height
window is not what most failures fail on. Median final XY distance to the named
object 0.0193 m overall, 0.0146 m on successes, 0.0617 m on failures. Of the
failures, 42.7% still came within 5 cm at some point: the modal failure is a
near miss under a 2 cm criterion, not a departure.

---

## 4. Methodological guarantees

Each of these was a measurement error in this project's history, so each is now
checked per episode and recorded in `episodes.csv`.

| guarantee | leg A | all six legs |
|---|---|---|
| named object inside the overview frame | 1024/1024 | 6656/6656 |
| named object inside the reachable workspace | 1024/1024 | 6656/6656 |
| start within the restored curriculum cap (3-D) | 1024/1024 | 6656/6656 (trivially true in leg D, whose cap is disabled by design) |
| worlds diverged into the non-finite reset | 0 | 1 (of 6656) |
| episodes outside the training preconditions | 0 | 0 |

Further:

* **The distribution is the trained one.** The cap reaches the simulator only
  when `random_workspace_gripper_start` is true; the tool refuses to run
  otherwise, because a resetter that silently drops the cap is what made 52M
  steps of this project's logged validation a number about a different task.
* **Scenes are not degenerate.** Target positions cover all 9 lattice cells
  with a χ² of 5.3 on 8 dof against uniform, and P(cell | object) is
  indistinguishable from a uniform null for all eight catalogs (measured over
  4096 independent scenes with `tools/audit/start_distance_probe.py`). Within
  leg A the apple alone occupies 9/9 cells with 0.154 m position std, so
  "the instruction names a fixed coordinate" is excluded as an explanation of
  the success rate.
* **The verdicts carry irreducible noise.** The frozen SmolVLA prior draws
  fresh flow-matching noise on every forward, so per-episode verdicts are not
  reproducible even at a fixed seed (~6.6% measured previously). Round-to-round
  spread here: A 0.656 / 0.604, B 0.395 / 0.525, E 0.438 / 0.266, C 0.176 /
  0.158 / 0.191, D 0.113 / 0.230, F 0.562 / 0.627. Read rates at the sample
  size, never at the third decimal.
* **Video selection is separated from the metric.** Which episodes get filmed
  is drawn from its own generator (`video_seed`, recorded in each manifest);
  the reset stream stays deterministic, so the rate is reproducible while the
  clips differ between runs.

---

## 5. Artifacts

30 MP4s. Each frame is the overview and wrist images the policy actually
received, side by side, one frame per policy decision; the filename carries
leg, round, world, outcome, object, scene size, start distance and wrist angle.

**Leg A — training configuration, five successes** (`train_config/videos/`):

| clip | object | target (x, y) | objects | start | best |
|---|---|---|---:|---:|---:|
| `..._r00_w040_success_robocasa_mug_obj1_d170mm_wrist48deg` | mug | (+0.175, −0.173) | 1 | 170 mm | 5 mm |
| `..._r00_w086_success_robocasa_tomato_obj1_d137mm_wrist37deg` | tomato | (+0.189, +0.193) | 1 | 137 mm | 13 mm |
| `..._r00_w155_success_robocasa_orange_obj1_d177mm_wrist41deg` | orange | (+0.001, +0.000) | 1 | 177 mm | 11 mm |
| `..._r00_w243_success_robocasa_plate_obj2_d142mm_wrist29deg` | plate | (+0.007, +0.188) | 2 | 142 mm | 21 mm |
| `..._r00_w379_success_robocasa_bowl_obj1_d139mm_wrist37deg` | bowl | (+0.008, +0.186) | 1 | 139 mm | 16 mm |

**Leg F — one object, many places** (`one_object_many_places/videos/`): nine
clips of the *same* apple, in **7 distinct lattice cells** spanning the desk
— (+0.185, −0.002), (+0.181, +0.181), (−0.189, −0.173), (−0.192, −0.008),
(−0.005, +0.182), (+0.165, +0.173), (+0.180, −0.159), (−0.194, +0.182),
(−0.181, +0.197). This is the artifact that answers "is the named object
pinned to one place" by eye rather than by table.

**Legs B and E — two and three objects** (4 clips each), targets spread across
the desk, scene sizes 2 and 3.

**Leg C — wrist-blind** (`multi_object_wrist_blind/videos/near_miss/`): three
near-misses at 293/312/328 mm starts. There are no successes to show, which is
the finding of §3.4, not a gap in the recording.

**Leg D — uncapped** (3 successes + 2 near-misses), including a success from a
261 mm start and one near-miss worth watching in full: a potato at a **394 mm**
start with the target certainly outside the wrist frame, which the policy still
closed to **9 mm in XY** — and then failed the predicate at a terminal height of
0.316 m, outside the 0.26–0.28 m window. One clip carries three of this
report's points at once: the budget is not the binding constraint (0.38 m
travelled of ~0.9 m available), the height window is a real part of the
criterion, and a wrist-blind episode can still get the XY right occasionally.

---

## 6. Claims this run supports, with the numbers behind them

Stated at the strength the data carries, for direct reuse in a paper.

1. **RL alone, with no demonstrations, learns instruction-conditioned reaching
   on a cable-driven parallel robot to 63.0% ± 7.3% (n = 1024, 128 scenes) at a
   2 cm tolerance**, from a frozen SmolVLA prior plus a trainable residual and
   an action-expert LoRA, trained for 11.0M environment steps.
2. **The learned behaviour is aiming, not search**: the cosine between the very
   first commanded action and the direction to the target is +0.185 on
   successes and −0.202 on failures.
3. **Language grounding is present but partial.** On episodes where the named
   object is not the nearest one, success is 27.2% (against 51.5% when it is),
   and the gripper arrives at the wrong object in 40.9% of them.
4. **Localisation flows through the wrist camera.** In 592 episodes where the
   named object was certainly outside the wrist frame — while remaining inside
   the overview frame, which the policy also receives — success was 0.34%
   (2/592, exact 95% upper bound 1.1%; 0/200 in the leg designed for it),
   against 5.5–9.5% for distance-matched in-frame controls in the same legs.
5. **Performance degrades smoothly with start distance and collapses past the
   trained range**: 0.746 under 0.12 m, 0.506 at 0.15–0.18 m, 0.227 at
   0.21–0.25 m, 0.000 beyond 0.30 m.
6. **The policy is tuned to the placement statistics of its environment**:
   widening the object lattice by 2× drops success from 0.460 to 0.352 at equal
   object counts.
7. **Validation must be measured on the training start distribution.** The same
   checkpoint scores 0.630 with the earned curriculum cap applied and 0.172
   with the cap omitted — a 3.7× difference produced by the reset alone, with
   the policy, the seed, the predicate and the horizon held fixed. The uncapped
   figure is what a validation resetter that never receives the curriculum
   state produces, which is the configuration this project logged as
   `validation/success_rate` for 52M steps before the fix at
   `smolvla_grpo_mjwarp_cdpr.py:2041`.

Claim 7 is the methodological contribution and is worth a paragraph of its own:
in a curriculum-driven RL setup, a held-out evaluator that does not receive the
curriculum state measures a different task, and the discrepancy is large enough
to invert conclusions about whether the method works at all.

---

## 7. Limitations

* **One checkpoint, one task, simulation only.** Nothing here transfers a claim
  to `pick_up` or `put_into_*`, and nothing was run on hardware.
* **128 independent scenes per leg** bound the metric leg's interval at ±7.3
  points. Narrowing it is rounds, not analysis: episodes within a GRPO group
  are eight replicas of one reset.
* **The wrist result is measured in a shifted regime.** Wrist-blind episodes
  only exist at caps above the trained 0.19 m, so §3.4 controls for distance by
  binning rather than by design. A cleaner test is an observation-level
  ablation (zeroing the wrist channel at a fixed start distribution); the hook
  for it already exists (`RLVLA_EVAL_ZERO_WRIST`).
* **"Named vs nearest" is a proxy for grounding**, not a direct measure of
  language use. It cannot distinguish a policy that reads the instruction from
  one that has learned a per-object visual prior correlated with it.
* **Leg E changes two things at once** (placement spread and object count vs
  leg A), so its 0.352 should be compared against leg B's 0.460 rather than
  against the metric.
* **The far-distance collapse is measured, not explained.** The budget
  hypothesis is ruled out (§3.1: ~0.9 m reachable, 0.043 m closed), but the
  data cannot separate "cannot localise a target this far off-axis" from
  "never trained above 0.19 m and does not extrapolate". Separating them needs
  either a longer curriculum or an oracle-position arm at matched starts
  (`tools/audit/xy_approach_probe.py --legs oracle`).

---

## 8. Reproduction

```bash
cd /root/repo/RL_VLA_Bootstrapping
bash scripts/validate_cdpr_smolvla_move_to_remote.sh
```

Six legs, 13 rounds, ~35 min on one A40. Every leg writes `episodes.csv` (one
row per episode, 38 columns), `validation_summary.csv` (the splits of §3),
`manifest.json` (provenance: checkpoint, cap, seeds, camera geometry, counts)
and `videos/`. The run root additionally carries
`all_legs_validation_summary.csv` and `validation.log`.

Knobs used by the legs, all of them measured rather than chosen: the
wrist-blind cap (0.33 m → 14.5–17.2% of episodes certainly out of frame,
against 0.000 at the trained 0.19 m), the scatter parameters (§3.5), and the
scene seeds (leg A on the checkpoint's own validation seed for comparability,
the diagnostics on their own so their scenes differ).

Supporting preflights, all CPU-only and runnable without a GPU:
`tools/audit/start_distance_probe.py --backend fake` for the start
distribution, the cap ladder and the camera framing.

---

## 9. Appendix — where each number lives

A copy of every numeric artifact — the six `episodes.csv`, the six
`validation_summary.csv`, the six `manifest.json`, the combined table and the
console log — is tracked in `docs/artifacts/move_to_validation_20260821/`. The
MP4s stay outside the repository; they are the run directory named above.

| section | file |
|---|---|
| headline, per-object, per-split rates | `<leg>/validation_summary.csv` |
| every episode, 38 columns | `<leg>/episodes.csv` |
| provenance, seeds, camera geometry, counts | `<leg>/manifest.json` |
| all legs in one table | `all_legs_validation_summary.csv` |
| per-round console trace | `validation.log` |
| clips | `<leg>/videos/`, near-misses in `<leg>/videos/near_miss/` |

Columns worth knowing in `episodes.csv`: `success`,
`start_xy_distance_m`, `start_3d_distance_to_hover_m`,
`start_within_curriculum_cap`, `best_xy_distance_m`,
`final_xy_distance_to_named_m`, `final_xy_distance_to_nearest_other_m`,
`named_is_nearest_at_start`, `ended_closer_to_named`, `target_in_overview`,
`target_wrist_angle_deg`, `target_certainly_out_of_wrist`,
`all_scene_objects_out_of_wrist`, `target_reachable`, `cosine_decision0`,
`scene_object_count`, `target_x`, `target_y`, `video`.
