# Phase 2 preflight — is `put_into_plate` / `put_into_bowl` reachable?

Answering the one thing asked for first: **F3 reproduces at HEAD, and success is
currently unreachable for any policy, oracle included.** The oracle scores
**0/6**. It is not one bug, it is three, and two of them were introduced or made
worse by the phase-1 geometry correction that was believed to have fixed F3.

Do not launch phase 2. Do not run the oracle arm on a checkpoint yet either —
until the gating below is fixed the oracle arm can only return zero, and a zero
would say nothing about localization.

## How this was measured

**Confirmed on MJWarp.** M0 was re-run on the training engine and reproduces the
CPU result: `put_into_plate` terminates on env step **1** of 64 in all three
episodes, `put_into_bowl` at 7/14/12. The gating failure is not a CPU-physics
artefact. (The report's prediction was that the *direction* is solver-independent
and the exact contact openings are not; the direction held.)

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

**Fixed and measured.** The reset now seeds the counter to
`_GRASP_PERSISTENCE_STEPS` on caught starts, with a regression test that pins the
counter and `physical_grasp` to agree rather than to a literal. Re-running the
same six episodes:

| arm | before | after |
|---|---|---|
| as configured | plate 1/1/1, bowl 14/14/15 env steps | **identical** |
| `--reseat-held-object` | plate **1**/1/32, bowl 14/44/42 | plate **35**/1/32, bowl 14/44/42 |

The as-configured column does not move at all, and that is the fix behaving
correctly rather than failing: with the pads unloaded `persistent_candidate` is
false, so `_update_physical_grasp` zeroes the counter on step 1 regardless of
what the reset seeded. The counter is a floor for a grasp the physics already
supports, never a head start for a world that has to earn one.

Where the grip **is** real, a plate episode goes from dying on env step 1 to
surviving 35 — through the traverse, the descent and into a genuine release
attempt. So bug 4 is real and now closed, but it is entirely masked by bug 1
until the seating opening is fixed. Do not expect this commit alone to change a
training run. (The remaining step-1 plate failure is the banana, which never
latches — see below.)

## The fifth bug: the placement curriculum starts inside its own success radius

Found from a training launch, not from the harness, and it is the one that would
have done the most damage.

`random_workspace_start_distance_initial: 0.03` is a **single scalar shared by
all three instructions**. The caps that follow are per-instruction
(`_start_cap_table`), but they all start here. Realized start distance to the
release hover point, read from the reset rather than from the logged cap —
`manifest.json` → `episodes[].start.ee_to_placement_hover_m`:

| instruction | realized start distance | success radius |
|---|---|---|
| `put_into_plate` | 0.0292, 0.0300, 0.0300 | **0.091** |
| `put_into_bowl` | 0.0300, 0.0300, 0.0245 | **0.057** |

The caught object starts **already inside the receptacle's success radius**, held
above it. For `put_into_bowl` the task at the initial cap is: open the gripper.
Nothing else. For `pick_up`, whose tolerance is ~2 cm, a 3 cm start is a genuine
approach — the same number means opposite things for the two instructions.

This is the shape of trap #1 and of the shaping-window-vs-success-radius bug: a
knob calibrated on one instruction, silently inherited by another whose scale is
different. The placement caps must **start above their own success radius** —
~0.12 for the bowl, ~0.15 for the plate — or better, be expressed as a multiple
of each instruction's own tolerance. Until then the curriculum's first rung has
no task in it.

### What this cost, and what it nearly cost

A 15M-step run was launched and stopped at update 1 (4676 steps), reporting
`success=144/1024`. That is not learning — there has been one gradient step. It
is the reset: release, and the object falls the 2–3 cm into a receptacle it is
already over.

The composition is consistent with that reading, though the log is not broken
down by instruction so this is arithmetic rather than proof. Of ~1024 episodes
under `uniform_cycle` over three instructions, ~341 are `put_into_bowl`; the
plate contributes ~0 because it terminates on env step 1; `pick_up` contributes
at roughly its phase-1 rate. 144 is ~42% of the bowl third.

Had this run continued, it would have reported a healthy and rising placement
success rate, the per-instruction gate would have promoted the placement caps
briskly past 0.057, and success would then have **fallen** as the starts finally
moved outside the radius — after millions of steps, and reading as a regression
rather than as the first honest measurement. Trap #6 in the phase brief warns
that a success-vs-failure gap is not evidence of skill; this is the same warning
one level up. **A success rate is not evidence of skill either, if the reset
already satisfies the predicate.**

Verify before trusting any placement number: `ee_to_placement_hover_m` from the
reset, against the instruction's own radius. Never the logged cap.

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
4. ~~Seed `bilateral_contact_steps` to `_GRASP_PERSISTENCE_STEPS` on caught
   starts, with a regression test.~~ **Done** — needs no measurement, and it is
   inert until 1 lands. See "the fourth bug" above for the before/after.
5. Re-confirm the banana's orientation-slip rejection; export
   `relative_orientation_slip_rad` to telemetry so it can be read rather than
   inferred.
6. Give the placement instructions their own
   `random_workspace_start_distance_initial` above their own success radius
   (~0.12 bowl, ~0.15 plate), or express the cap as a multiple of each
   instruction's tolerance. Without this the first curriculum rung is vacuous
   and every placement success number is the reset's, not the policy's.

## Fixes 1, 3 and 6 applied — and what M0 says afterwards

`fitted_gripper_opening` re-measured on **MJWarp** and written into the catalog;
`put_plate_release_height` 0.045 → 0.10; per-instruction first rung
(`random_workspace_start_distance_initial_by_instruction`: bowl 0.12, plate
0.15), honoured by both the trainer and the reference harness.

**Fix 2 turned out to be unnecessary and was not made.** I proposed decoupling
the release threshold and dropping the hard 0.55 floor. Deriving it from the
corrected constant is enough on its own: the threshold is
`max(0.55, fitted + 0.04)` and the seeded opening is `fitted − 0.033`, so both
move together. Against the measured contact openings the gap between the
threshold and the opening at which the object leaves the pads is now
0.040–0.235 — all inside the ~0.30 the gripper travels before the object
settles — and `released` is false at t=0 for every catalog. One constant per
catalog, no reward change. Smaller diff than advertised.

The MJWarp sweep also **corrects a claim in this report.** I wrote that the
wrong-drop race is "lost by arithmetic ... which is all six". That was measured
on CPU and over-generalised. On MJWarp, with only the seating repaired and the
production thresholds untouched, apple, potato and tomato already **succeed**
(reward 3.478–3.479) and only the orange loses the race — its gap was 0.373
against a ~0.30 budget, while tomato's 0.293 squeaks in. The mechanism was
right; "all six" was not. Contact openings agreed CPU-to-MJWarp within 0.04,
which is what flipped tomato across the line.

### M0, and the two things still in the way

| arm | old 0.03 cap | honest per-instruction cap |
|---|---|---|
| as configured | **4/8** (was 0/6) | **0/8** |
| `--reseat-held-object` | — | 1/8, and the failure mode changes |

Raising the first rung to a real start distance costs everything. That is not a
regression, it is the first honest measurement — at 0.03 the traverse was ~3 cm
and the task was mostly the reset. Two separate causes, separated by the reseat
arm:

**(a) The seeded grip does not survive a real carry.** At the honest cap all
eight episodes take the wrong-drop penalty, and most drop *mid-traverse* with
the object still 0.08–0.14 m from the receptacle. Reseating — 0.007 more
squeeze (`contact − 0.04` rather than `contact − 0.033`) plus a settle step —
stops the dropping outright: tomato and orange then carry the object to the end
of the budget. So this is a constant, `_GRASP_SQUEEZE = 0.001/0.03`, not a
design problem. It was invisible at a 3 cm traverse and only appears over 20–40
steps of transport. **Not changed here**: the same constant seeds pick_up's
pre-grasped stage, so it is a judgement call rather than a measurement.

**(b) Once the grip holds, the horizon binds.** The reseat arm's plate episodes
run to 88/88 env steps without finishing; the one success (bowl, tomato) needs
**66**. At a realized XY gain of ~0.0075 m/step a 0.15 m traverse is ~20 env
steps before the raise, descend, open and settle are paid. The coupled horizon
gives 84–88 here. The margin is thin, and the scripted oracle is deliberately
slow (damped servo, settles at every phase), so a policy could be faster — but
this is exactly what M1 exists to price, and it should be read before assuming
so.

**Do not launch on this.** The oracle is 0/8 at the start distribution training
would use.

### The squeeze constant, applied — and why the local M0 can no longer judge it

`0.001/0.03 = 0.0333` was inlined in five places. It is now
`CAUGHT_START_GRIP_SQUEEZE = 0.04` in `cdpr_object_catalog.py`, next to the
`fitted_gripper_opening` it is measured against, and imported everywhere.

Measured effect on CPU — last env step at which both pads are still loaded:

| episode | before | after | drop-point XY error |
|---|---|---|---|
| plate / tomato | 28 | **47** | 0.042 → 0.021 |
| plate / orange | 22 | 22 | 0.103 → 0.091 |
| bowl / tomato | 34 | **47** | 0.0072 → 0.0051 |
| bowl / orange | 23 | **28** | 0.093 → 0.015 |
| banana ×4 | 1–3 | 1–3 | unchanged |

The grip holds ~40–70% longer and the object is dropped much closer to the
receptacle. Banana is untouched, as expected — its failure is the orientation
gate, not squeeze.

But M0 stays 0/8 locally, and **that number is no longer trustworthy on this
machine.** The catalog now carries **MJWarp** contact openings, and the CPU's are
0.02–0.04 *lower* (tomato 0.412 vs 0.432, potato 0.409 vs 0.449, apple 0.592 vs
0.632). Production closes to `fitted − 0.04`, so against CPU physics it now
under-grips by exactly that gap, while the `--reseat-held-object` arm re-measures
per episode and does not. That is why reseat reaches the receptacle and the
production reset does not, on this box.

So the CPU harness has served its purpose and is now the wrong instrument: it
was the right tool while the constants were engine-independent arithmetic, and
it stopped being one the moment the constants became engine-specific
measurements. **M0 must be re-run on MJWarp before the squeeze change is judged
either way.** Do not read the local 0/8 as evidence that it failed, and do not
read a local pass as evidence that it worked.

## M0 on MJWarp: the gating is closed, for objects that fit the gripper

Re-run on the training engine at the honest per-instruction cap, with every fix
above applied and no override flags. **4/8.**

| instruction | catalog | held until | ended | success | XY error | radius | reward |
|---|---|---|---|---|---|---|---|
| bowl | potato | 56 | 62 | **yes** | 0.0019 | 0.057 | 3.310 |
| bowl | potato | 35 | 43 | **yes** | 0.0013 | 0.057 | 3.474 |
| plate | potato | 59 | 65 | **yes** | 0.0011 | 0.091 | 3.293 |
| plate | potato | 30 | 37 | **yes** | 0.0019 | 0.091 | 3.467 |
| bowl | banana | 0 | 1 | no | 0.1009 | 0.057 | 0.184 |
| bowl | banana | 0 | 3 | no | 0.1512 | 0.057 | 0.346 |
| plate | banana | 0 | 1 | no | 0.1208 | 0.091 | 0.148 |
| plate | banana | 0 | 3 | no | 0.0667 | 0.091 | 0.127 |

**Every non-banana episode succeeds, landing 1.1–1.9 mm from the receptacle
centre against radii of 57 and 91 mm, and carrying the object 30–59 env steps to
get there.** The five gates are closed. Placement is reachable, on the training
engine, at a start distance that is a real task.

Every banana episode fails with `held_until = 0` — the pads never both load, at
any step.

### That banana failure is a bug in this report's own measurement

`seat_held_object` began its closing sweep at **the opening the reset had
already seeded** rather than at fully open. When contact is already present
there, it breaks on the first decrement and reports that value — a lower bound
presented as a measurement. The MJWarp sweep that produced the catalog constants
reported `closing_increments = 1` for **banana and mug**, against 6–15 for every
other catalog. Both constants were bracketed from the wrong side. Fixed; the
sweep now starts fully open, and the docstring says to re-measure any catalog
reporting `closing_increments = 1`.

Re-bracketed properly, banana and mug report contact at **0.98** — i.e. the pads
touch them even with the gripper fully open, in the orientation the caught reset
seeds. Tomato, as a control, takes 30 increments down to 0.40 and is unaffected.

So the finding is not that these two were mismeasured by a little. **The banana
and the mug are wider than the gripper's open gap in the seeded pose.** The
0.315 constant is why the banana seats at 175/183 N against 3–17 N for every
other object, is ejected at reset, and never loads a pad afterwards.

Do **not** write 0.98 into the catalog for them: the release threshold is
`max(0.55, fitted + 0.04)`, which clamps at 1.0, so `released` could never fire
and every episode would fail differently. They need either a seeded orientation
that presents their narrow axis to the pads, or removal from the placement pool.

`gripper-geometry-mismatch` already records that banana grasps only when the
approach yaw is perpendicular to its long axis, and explicitly says not to drop
it pre-emptively — so this is a scope decision, not a measurement, and it is
left open here.

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

## What is and is not applied

Applied: fix 4, the `bilateral_contact_steps` seeding in
`mjwarp_rank_local_collector.py`, plus its regression test and the new
`--legs placement` arm in `tools/audit/xy_approach_probe.py`.

Not applied: fixes 1–3. `configs/`, `cdpr_object_catalog.py` and
`cdpr_batched_tasks.py` are untouched. They depend on contact openings measured
on **CPU** physics while the run is MJWarp, so re-measure first; the three then
follow from the numbers in one commit.

Fix 4 was safe to land ahead of them precisely because it is inert without them —
measured, not assumed: the as-configured arm is byte-identical before and after.
It cannot move a training run on its own, and it cannot mask what M0 measures.
