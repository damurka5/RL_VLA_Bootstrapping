# Phase 2 preflight — is `put_into_plate` / `put_into_bowl` reachable?

Answering the one thing asked for first: **F3 reproduces at HEAD, and success is
currently unreachable for any policy, oracle included.** The oracle scores
**0/6**. It is not one bug, it is three, and two of them were introduced or made
worse by the phase-1 geometry correction that was believed to have fixed F3.

Do not launch phase 2. Do not run the oracle arm on a checkpoint yet either —
until the gating below is fixed the oracle arm can only return zero, and a zero
would say nothing about localization.

## How this was measured

`scripts/render_cdpr_task_reference_episodes.py` against the phase-2 config
(`cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml`), scripted oracle,
production resetter + production reward + the trainer's own
`_update_physical_grasp` called unbound. Physics is **MuJoCo CPU**, not MJWarp —
same MJCF, different solver. Every finding below is either config arithmetic or
a contact/threshold ordering, so the *direction* is solver-independent; the
exact opening at which a pad loads is not, and is flagged where it matters.

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MUJOCO_GL=disable PYTHONPATH="<tf-stub>:.:robots/cdpr" python scripts/render_cdpr_task_reference_episodes.py --config configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml --instructions put_into_plate put_into_bowl --episodes-per-instruction 3 --physics mujoco_cpu --no-video --output <out>
```

## The headline

| arm | plate | bowl | how episodes end |
|---|---|---|---|
| as configured | 0/3 | 0/3 | plate at **env step 1** of 64; bowl at 14–15 |
| + `--reseat-held-object` | 0/3 | 0/3 | correct placements scored as wrong drops |
| + release threshold 0.55 | 0/3 | **2/3** | plate still dies at step 1 |
| + release 0.55 + `put_plate_release_height=0.10` | **2/3** | **2/3** | reward 3.48–3.49, the ladder's own top |

The reward ladder is sound. Everything between the ladder and the policy is not.

## F3 confirmed — and re-attributed

The consistency report blamed F3 on the hard-coded `0.08` in the caught-object
reset. That hard-code **is** fixed (`mjwarp_rank_local_collector.py:1749` now
reads `pick_grasp_height_offset`), and the object is now spawned at the pad
centre. F3 survived the fix, because the offset was never the binding
constraint. Measured at env step 1 of an as-configured placement start:

| catalog | seeded opening (`fitted − 0.033`) | left/right pad force | `released` at t=0 |
|---|---|---|---|
| robocasa_tomato | 0.651 | 0.00 / 0.29 N | **true** |
| robocasa_orange | 0.735 | 0.00 / 0.33 N | **true** |

The pads are not bilateral and carry ~0.3 N on one side. The object is in free
fall from step 1. "Starts with the object already caught" is false as
configured — the object starts *adjacent to* an open gripper.

Worse: because `released = gripper_opening >= max(0.55, fitted + 0.04)` uses a
**0.55 floor** (`BatchedTaskThresholds.release_opening`), and the seeded opening
is ≥ 0.55 for five of the six pool catalogs, **the reward believes the object
has already been released before the policy acts.** Placement is not a task the
policy performs; it is a coin flip on where a dropped object lands.

### Root cause: `fitted_gripper_opening` is not the contact opening

Measured by closing the fingers on a pad-centred object until the physics reports
bilateral contact (0.02 increments, so ±0.02; the harness then adds a 0.04
squeeze, and the "contact" column below adds it back):

| catalog | table `fitted` | seated opening | contact ≈ | left/right force at seat | `released` at seeded opening |
|---|---|---|---|---|---|
| robocasa_apple | 0.785 | 0.552 | 0.59 | 3.7 / 15.7 N | true |
| robocasa_banana | 0.368 | 0.275 | 0.32 | 57.4 / 60.6 N | false |
| robocasa_mug | 0.800 | 0.707 | 0.75 | 8.2 / 25.2 N | true |
| robocasa_orange | 0.768 | 0.375 | 0.41 | 2.7 / 11.4 N | true |
| robocasa_potato | 0.602 | 0.369 | 0.41 | 3.6 / 15.7 N | true |
| robocasa_tomato | 0.685 | 0.372 | 0.41 | 3.0 / 10.0 N | true |

Every tabulated value overstates the real contact opening. This is F4, confirmed
with an independent measurement. Note it does **not** match F4's own table
(F4 had apple 0.45, orange 0.30, potato 0.25) — F4 measured a desk-resting
object at grasp height, this measures a pad-centred carried one, which is the
pose the placement reset actually creates. Treat neither as canonical until it is
re-measured on MJWarp; treat "the table is too open by 0.2–0.4" as established.

## The second bug: a correct placement is scored as a wrong drop

This is F6, and it is **not** plate-specific as the consistency report believed.
The bowl loses it too. `put_into_bowl`, orange, reseated so the grip is real:

```
step  phase          opening  obj_z   settled  released  container_xy_err  reward
  34  descend         0.386   0.2738     0        0          0.0008        1.469
  35  open_gripper    0.407   0.2722     0        0          0.0008        1.486   <- object leaves the pads
  ...
  41  open_gripper    0.675   0.2283     0        0          0.0011        1.484
  42  open_gripper    0.725   0.2143     1        0          0.0012        1.233   <- WRONG DROP
```

The object is **1.2 mm from the bowl centre** against a 57 mm radius. It is
penalised. The mechanism:

* the object physically leaves the pads at opening ≈ **0.41**;
* `released` needs `max(0.55, fitted + 0.04)` = **0.808** for the orange;
* the gripper opens at 0.05 per env step, so crossing 0.41 → 0.808 costs **8
  env steps**;
* the object falls and settles in **7**.

`released` can never become true before `target_has_settled`, so `container_ok`
is never true, so `wrong_place_settled` fires on every correct placement.
The race is lost by arithmetic, for every catalog whose `fitted + 0.04` exceeds
its contact opening by more than ~0.30 of opening — which is all six.

Dropping the threshold to the 0.55 floor alone takes the bowl from 0/3 to
**2/3 at reward 3.485**. That is the single highest-value line in this report.

## The third bug: the plate's hover point is below its own settle threshold

`target_has_settled` is `target_z <= support_surface_z + target_rest_height +
0.045`. For an orange: `0.15 + 0.0278 + 0.045 = 0.2228`.
The plate's carried hover point is `plate_z + put_plate_release_height` =
`0.16 + 0.045 = 0.205`. **The carried object is 18 mm below the height at which
the reward declares it has come to rest on the desk.**

So for the plate, `target_has_settled` is true at t=0 while the object is still
gripped. Combined with bug 4 below, that terminates the episode on **env step
1 of 64**, before a single policy decision can matter. Raising
`put_plate_release_height` to 0.10 (the bowl's value) takes the plate from 0/3
to 2/3.

The 0.045 is not arbitrary and cannot simply be lowered: an object resting
*correctly* on the plate sits at ~0.205 too. The plate's release height and its
settled height are the same number. The carried object must hover materially
above where it will come to rest, or the two states are indistinguishable.

Note the settle margin was raised 0.025 → 0.045 to stop bowl placements timing
out silently. That change is correct for the bowl and is what pushed the plate
over the line. Both configs were individually defensible; the pair is not.

## The fourth bug: `bilateral_contact_steps` is not seeded

`mjwarp_rank_local_collector.py:2061` seeds `bilateral_contact_steps` to zeros
while seeding `physical_grasp=True` on the same reset. `physical_grasp` needs
`bilateral_contact_steps >= _GRASP_PERSISTENCE_STEPS` (2), so a caught start
that is *genuinely* holding the object reads `grasped = False` for its first two
env steps. On a plate episode, where `target_has_settled` is already true, that
is exactly enough for `wrong_place_settled` to fire on step 1. Observed with 46 N
and 22 N on the pads:

```
step  ee_z   opening  obj_z   settled  Lf     Rf     physical_grasp  terminated
  1   0.2113  0.326   0.2049     1     46.2   21.6        0              1
```

A held object, a settled flag, a detector that has not warmed up, and the episode
is over.

## Also observed, lower confidence

* **The banana never latches.** Bilateral contact for 3 steps at 1.2–1.4 N with
  1.6–1.8 mm of position slip, and `physical_grasp` stays 0 — where the orange
  latches on step 2 at *lower* force. Position slip is well inside the 8 mm
  bound, so the rejecting gate is `relative_orientation_slip` (the banana rotates
  in the pads). It is not exported to telemetry, so this is inference from
  elimination, not a direct reading. Banana is 0/4 in every arm.
* **The mug cannot be served by a 0.55 threshold.** Its contact opening is 0.75,
  above the floor, so `released` is true while it is still gripped, and its
  episodes time out (never terminate) rather than fail. A single global release
  threshold cannot cover a pool spanning contact openings 0.32–0.75.
* Both are CPU-physics readings and should be re-confirmed on MJWarp.

## What has to be true before the oracle arm means anything

The release threshold must sit **above** each catalog's contact opening (so
`released` is false while held) and **below** contact opening + ~0.30 (so the
gripper crosses it before the object settles). Per catalog, from the table above:

| catalog | contact ≈ | admissible threshold window | current `max(0.55, fitted+0.04)` |
|---|---|---|---|
| apple | 0.59 | 0.59 – 0.89 | 0.825 ✅ (marginal) |
| banana | 0.32 | 0.32 – 0.62 | 0.55 ✅ |
| mug | 0.75 | 0.75 – 1.00 | 0.84 ✅ (marginal) |
| orange | 0.41 | 0.41 – 0.71 | 0.808 ❌ |
| potato | 0.41 | 0.41 – 0.71 | 0.642 ✅ |
| tomato | 0.41 | 0.41 – 0.71 | 0.725 ❌ |

The window exists for every catalog. The threshold just has to be derived from
the contact opening instead of from a mesh-bounds number. That is one constant
per catalog, measured once.

Proposed, in dependency order — **none of these are applied yet**, because the
numbers are CPU-physics and the run is MJWarp:

1. Re-measure the contact opening for all catalogs on MJWarp and write it into
   `cdpr_object_catalog.py` as the value `fitted_gripper_opening` was always
   meant to be. Everything else falls out of this.
2. Derive the release threshold as `contact_opening + margin` with the margin a
   config knob, and drop the hard `0.55` floor in
   `BatchedTaskThresholds.release_opening` — the floor is what breaks the orange
   and tomato, and what makes the seeded start read as already-released.
3. `put_plate_release_height: 0.045 → 0.10`.
4. Seed `bilateral_contact_steps` to `_GRASP_PERSISTENCE_STEPS` on caught starts,
   with a regression test.
5. Re-confirm the banana's orientation-slip rejection; export
   `relative_orientation_slip_rad` to telemetry so it can be read rather than
   inferred.

## The measurement plan, pre-registered

Once 1–4 land, run these **in this order**. Each states what it shows and what
result falsifies the claim it is testing.

### M0 — the harness re-run (free, local, CPU)

Same six-episode oracle sweep with the fixes in the config rather than on the
command line.

* **Shows**: whether the fixes hold together with no override flags.
* **Expect**: ≥ 4/6, reward ≈ 3.48. Banana and mug are the permitted failures.
* **Falsified if**: still 0/6, or the plate still terminates before step 10. Then
  a fifth mechanism exists and the oracle arm is still premature.

### M1 — the placement oracle arm (GPU, one probe run, no training)

`tools/audit/xy_approach_probe.py --legs oracle`, with the placement sources
added (below). `oracle_place` hands the policy the true receptacle position and
servos the gripper to the hover point, then opens; the policy keeps nothing.

* **Shows**: the ceiling. Whether the plant, the horizon (26 decisions × 4 env
  steps at a realized XY gain of ~0.0075 m/step) and the release timing permit
  the task at all when localization is free.
* **Expect**: success ≥ 0.7 for the bowl if the gating is fixed.
* **Falsified if**: `oracle_place` is near zero while M0 passes. That would mean
  the *training* horizon, not the reward, is the binding constraint — the oracle
  in M0 gets 64 env steps and used 39–46 of them, while training gives 26 × 4 =
  104, so this should have margin, but the curriculum's coupled horizon starts at
  **16 decisions = 64 env steps** and the successful oracle episodes used 39–46.
  That is thin, and M1 is the measurement that prices it.
* **This is the measurement that must not be skipped**, because a failure here is
  cheap to fix and would otherwise be misread as a localization result.

### M2 — the localization ladder (same run, extra arms)

`oracle_place_err_{0.01,0.02,0.03,0.05,0.08}m`, error drawn once per episode and
held — the existing `_make_oracle_xy_source` convention, and the right one: a
bad feature is wrong in a consistent direction, and per-decision resampling would
let the servo average it away and price a bad feature as usable.

* **Shows**: the accuracy the receptacle position must be known to. This is the
  number the phase exists to produce.
* **Expect**: if the campaign's framing is right, success should hold up to
  ~0.05 m for the plate (91 mm radius) and degrade sharply past ~0.03 m for the
  bowl (57 mm radius).
* **Falsified if**: success collapses at 0.01–0.02 m. Then the radius is not the
  operative tolerance and something else — release timing, hover height — is
  doing the gating, and the "9 cm target radius" framing in the phase brief is
  wrong.

### M3 — the policy arm against the ladder (same run)

The deterministic phase-1 checkpoint, no substitution, on the placement
instructions it has never trained on.

* **Shows**: where the frozen prior + residual sits on the M2 ladder. That
  placement is the ladder's own units is the whole point: it converts "the
  encoder localizes to 3–5 cm" into "and that is / is not enough here".
* **Expect**: nothing in particular. A warm-started policy that has never seen
  a placement instruction is a floor, not a prediction.
* **Falsified if**: — nothing. This arm cannot be falsified because it is
  descriptive, and that is exactly why it must not be the arm any conclusion
  rests on.

### The trap this plan is built to avoid

The phase brief's framing is that M2 "prices the localization requirement for a
9 cm radius the same way it did for grasping". **It does not, quite, and the
difference matters.** For `pick_up`, the oracle handed over the position of the
thing the gripper had to reach. For placement, the *held object's* position is
known from proprioception — it rides the gripper — and only the receptacle needs
localizing. So a passing M1 does not show that 3–5 cm suffices; it shows the
plant and the horizon are not the ceiling. Only M2's degradation curve, read
against the 3–5 cm figure from §4.12–§4.16, answers the phase's question. And
only if M1 passes first, since a floored M1 makes every M2 arm read zero for a
reason that has nothing to do with localization.

Second trap, from the campaign's own list (#6): do not read a
success-vs-failure gap in M3 as evidence of aiming. Selecting on success selects
on alignment for a blind policy too.

## Commands

**Re-measure the contact openings on MJWarp** (fix 1). No new tool: the harness
already reports `held_object_reseated.seated_gripper_opening` and the pad forces
per episode, which is the measurement. One pass per catalog, read the manifest:

```bash
for cat in robocasa_apple robocasa_banana robocasa_tomato robocasa_orange robocasa_potato robocasa_mug; do RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab python scripts/render_cdpr_task_reference_episodes.py --config configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml --instructions put_into_bowl --episodes-per-instruction 1 --physics mjlab_mjwarp --no-video --reseat-held-object --target-catalogs $cat --output runs/placement_grip/$cat; done
```

The seating loop closes in 0.02 increments and then adds a 0.04 squeeze, so the
contact opening is `seated_gripper_opening + 0.04 ± 0.02`.

**M0**, once fixes 1–4 are in the config:

```bash
RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab python scripts/render_cdpr_task_reference_episodes.py --config configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml --instructions put_into_plate put_into_bowl --episodes-per-instruction 3 --physics mjlab_mjwarp --no-video --output runs/placement_m0
```

**M1 + M2 + M3** are one probe run. `--legs placement` is new in
`tools/audit/xy_approach_probe.py`; it is the `oracle` leg's placement
counterpart, aims at `reference_slots` rather than `target_slots`, and commits
to the release once aligned rather than opening one increment per decision:

```bash
RLVLA_HF_OFFLINE=1 MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab python tools/audit/xy_approach_probe.py --checkpoint runs/<phase1-run>/smolvla_grpo_adapter.pt --config configs/examples/cdpr_smolvla_catch_release_dense_grpo_mjlab_resume.yaml --legs policy,placement --output runs/placement_probe
```

`policy` gives M3, `oracle_place` gives M1, the `oracle_place_err_*m` arms give
M2 at 0.01/0.02/0.03/0.05/0.08 m of receptacle error — a finer, shorter ladder
than the grasp leg's, because the question is where between 1 and 8 cm the curve
breaks against a 91 mm and a 57 mm radius. Error is drawn once per episode and
held, and corrupts Z as well as XY.

Read `oracle_place` **first**. Every rung of the ladder is that arm plus noise,
so if it is floored the ladder measures nothing and the answer is a horizon or a
gating problem, not a localization one.

Note the probe reports `command_cosine` against the receptacle for placement
arms and against the object for grasp arms, and it detects which from the trace
rather than from the arm name. It also drops the `~holding` gate for placement —
holding is the normal state there, and keeping the gate would discard every step.

## What is NOT applied

Nothing in this report has been committed to the training path. The probe leg is
new; `configs/`, `cdpr_object_catalog.py`, `cdpr_batched_tasks.py` and
`mjwarp_rank_local_collector.py` are untouched.

That is deliberate. Fixes 1–3 depend on contact openings measured on **CPU**
physics, and the run is MJWarp. Landing one of the four (say the
`bilateral_contact_steps` seeding, which needs no measurement) while the other
three wait would leave M0 measuring a state neither this report nor the config
describes. Re-measure on MJWarp first; the fixes then follow from the numbers in
one commit. Say the word and I will write them.
