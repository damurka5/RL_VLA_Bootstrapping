# WidowX-200 sustained-grip repair: GPU preflight passed

The user-supplied **2026-11-04 post-repair CUDA preflight passes**, with
`passed: true` and an empty failure list. All 24 logged reset batches are
reachable and finite, every tested placement/joint caught-start hold reaches
100%, and scripted grasp/lift/five-second hold rises from 0% to **100%**.
The original [JSON](../../artifacts/widowx200_grip_repair_20261104/gpu_preflight/preflight.json)
and [camera grid](../../artifacts/widowx200_grip_repair_20261104/gpu_preflight/cameras.png)
are preserved byte-for-byte with [source hashes](../../artifacts/widowx200_grip_repair_20261104/gpu_preflight/source.json)
and a [derived review summary](../../artifacts/widowx200_grip_repair_20261104/gpu_preflight/review_summary.json).

The earlier CUDA preflight under the original contact model failed despite
reachable, finite resets: placement caught-start proximity after five seconds
ranged from 0% to 45.31%, and both scripted grasp/lift/hold rounds scored 0%.
The new result validates the repair's sustained-grip checks on production
MJWarp as well as the local CPU checks below. It does not measure the
learned policy's complete placement success.

The pickup reset's 100% `caught_held_fraction` is a proximity test on
pre-grasped starts at desk height. An object resting on the desk beneath its
gripper can pass it without supporting its weight. It does not contradict the
failed lift/hold experiment. Likewise, 17 mm object-centre separation is
compatible with a fruit resting inside the bowl; centre separation alone
cannot establish interpenetration.

## Physical changes

`robots/widowx200/widowx200_mujoco/wx200.xml` now selects **elliptic friction
cones and `impratio=10`**. The original model inherited pyramidal cones and
`impratio=1`. In a stationary apple trace, each pad supplies approximately
0.345 N normal force and 0.196 N vertical friction, well inside its configured
friction cone, while the object creeps down at approximately 10 mm/s until it
leaves the pads. This is consistent with MuJoCo's soft-contact regularization.
The [official slip guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)
recommends elliptic cones and higher friction-to-normal impedance to reduce
this behavior. [MJWarp's supported-feature documentation](https://github.com/google-deepmind/mujoco_warp#feature-parity)
does not list these options as unsupported; this candidate uses no NoSlip
post-solver.

A six-setting CPU sweep distinguishes the changes: pyramidal/1 fails all five
20-second holds; pyramidal/10 and elliptic/1 each retain only one; elliptic/10,
/20 and /50 retain all five. We selected the lowest tested setting retaining
all five. This changes the soft-contact model, rather than adding adhesion or
increasing the friction coefficients, mass, servo gains or object dimensions.
This initial sweep used the original travel-stop settings; the final candidate
and its regression tests also include the mechanical-stop correction below.

The finger travel stops also use `solreflimit="0.004 1"` and
`solimplimit="0.99 0.999 0.001"`. Testing the new contact model exposed an
existing mechanical-limit weakness: a 69 mm sphere forced raw finger travel
to 44.5/47.4 mm despite a declared 37 mm upper limit. Stronger limit impedance
keeps that test near 34.65/37.01 mm. A sphere can still be pinched at its top
cap while its centre hangs more than 4 cm below the end effector. The size
regression now checks actual finger travel and centred-grasp proximity as
well as lift, instead of incorrectly assuming that every top-cap lift proves
the full sphere fits between the pads.

First-contact openings were remeasured under the repaired solver and written
to the WidowX-only table: apple 0.809, pepper 0.822, tomato 0.807, orange 0.811,
potato 0.797 and mug 0.627. The caught-start squeeze remains 0.20. The shared
CDPR values and its robot XML are unchanged. These measurements used CPU
MuJoCo 3.3.7; the server should remeasure with its installed version before
the GPU check, as requested in the supplied command block. The returned
folder contains the resulting preflight and cameras, but not the server's
updated opening table.

## Verification and remaining limits

| Local CPU check | Result |
|---|---:|
| Existing and new relevant tests, including rendered cameras | 59 passed |
| Actual descend/close/lift followed by a 20-second stationary hold | 5/5 objects |
| Sampled caught-placement reset poses, five-second hold | 384/384 |
| Scripted carry retained until deliberate opening | 60/60 |
| Central drops supported and visibly inside after five seconds | 60/60 |
| Scripted placements passing the strict final settling window | 59/60 |

The caught-pose check uses the validation sampler with balanced catalogs,
seed 12345, rounds 0/1 and caps 0.01/0.10/0.18. It spans scene shifts and
object/EE yaws, but is not the exact production GPU preflight sampler.

The remaining potato-on-plate case stays supported, remains within 0.72 mm
of the plate centre and retains its grip until release. It moves 2.234 mm
over the final 0.5-second window, above the audit's 2 mm limit. **The full CPU
scale audit therefore still exits with failure**; its strict settling
criterion was not loosened. Its five-second observation period is explicit
(`--settle-seconds 5`); at the old 2.5-second observation period, another
potato/plate case also needed longer to settle.

## Returned CUDA verification

| Reported check | Result |
|---|---:|
| Reset batches / worlds per batch | 24 / 512 |
| Reachable fraction in every batch | 100% |
| Logged reset states finite | All |
| Five-second placement caught-start proximity checks | 100% across 3,072 world checks |
| Five-second joint caught-start proximity checks | 100% across 816 world checks |
| Scripted grasp/lift/hold rate | 100% |
| Minimum logged object-centre separation | 45.2 mm |
| Final reported contact / constraint overflow | 0 / 0 |

The hold lasts 100 actions at 0.05 s/action. The scripted success check
requires at least 4 cm remaining lift and object/EE distance below 3 cm after
the hold. The returned mean is 1.0 across the requested two rounds of 512
worlds. These are scripted physical checks, not learned-policy results.
GRPO copies and repeated caps are correlated; the world counts are diagnostic
exposures, not that many independent scenes or a statistical transfer test.

Reported versions match the repository lock: MuJoCo 3.10.0, MJWarp 3.10.0.1,
Warp 1.14.0, MJLab 1.5.0 and Torch 2.7.1+cu128. The final capacity snapshot is
1,615 contacts out of capacity 131,072 and at most 28 constraints/world out of
1,024. That snapshot is not a measurement of peak usage over the whole run.

The camera grid shows the lifted potato between the fingers, with both
overview and wrist views nonblank. Its scale looks plausible for this small
arm. The potato mesh is visibly faceted. The four rows are the first four
worlds from one GRPO scene, not four independent visual examples; they do not
verify the appearance of every fruit or a completed bowl/plate placement.

The returned JSON lacks explicit solver settings and model/config/catalog/
asset hashes. The run follows the supplied model-installation workflow, but
these two returned files alone cannot reconstruct the exact simulator revision
or the server's recalibrated table. Preserve those files alongside future
checkpoint comparisons.

The **local** preflight now measures actual signed object/object contact penetration
(default allowed depth 3 mm) at reset and after holding. It retains centre
distance as a diagnostic, so a legitimate fruit/bowl stack is not rejected
solely for having nearby centres. New unit tests cover valid contacts,
actual overlap, inactive contacts and finger/object contacts. The report also
records solver options and simulator XML/asset hashes. This added GPU contact
diagnostic was not installed by the model-only server installer. The returned
server run uses the original centre-distance check, which also passes now.

This repair does not resolve the visual/collider shape mismatches documented
in the [scale audit](WIDOWX200_OBJECT_SCALE_AUDIT_20261103.md). In particular,
the orange's visible XY width remains about 50 mm around a 42 mm collision
sphere. Whole-object bowl containment is also stricter than the training
predicate's centre-distance criterion. Stable gripping is one part of the
requested physical/visual verification, not a full policy or visual-fidelity
certificate.

## Reproducible evidence

- [Contact-option sweep](../../artifacts/widowx200_grip_repair_20261104/contact_sweep.json)
- [Caught-placement pose probe](../../artifacts/widowx200_grip_repair_20261104/caught_start_probe.json)
- [Remeasured openings](../../artifacts/widowx200_grip_repair_20261104/remeasured_openings.json)
- [Final physical audit and source hashes](../../artifacts/widowx200_grip_repair_20261104/scale_audit.json)
- [Pasteable server model installer](../../artifacts/widowx200_grip_repair_20261104/install_server_model.py)
- Regression tests: `tests/test_widowx200_sustained_grip.py` and
  `tests/test_widowx200_preflight_contacts.py`.

The installer was tested against HEAD's original XML, tested a second time
for idempotency, and compiled with MuJoCo to confirm it installs the same
contact and mechanical-stop settings as the local candidate. It backs up
the old model. It updates the model only; run the existing first-contact
measurement tool with `--write` and then the existing five-second CUDA
preflight, as shown in the chat. The original remote verification used the
pasteable installer before publication. The complete repair is now included
in the repository changes; the [fresh-run handoff](WIDOWX200_FRESH_RUN_20261104.md)
uses a normal repository update and recalibration instead of reapplying it.

The complete local candidate also includes the improved preflight contact
diagnostic, tests and CPU audit option. The supplied server model installer
does not replace the server's preflight source; its existing five-second
hold check remains sufficient to test whether this contact repair fixes the
reported grip failure.

This is a simulator revision. Preserve the original failed report and old
checkpoints. The returned CUDA preflight clears the sustained-grip gate.
Matched policy evaluations, complete carry/release behavior on GPU, the
strict local potato/plate settling edge case and the known visual/collider
mismatches remain separate measurements for the corrected fresh experiment.
The user executed the GPU preflight; no policy training was launched from this chat.
