# WidowX-200 object scale and placement audit

**2026-11-04 update:** the server's five-second CUDA preflight also fails
sustained gripping. A [contact-model repair candidate](WIDOWX200_GRIP_REPAIR_20261104.md)
passes local long-hold/carry checks, and the subsequent returned CUDA
preflight **passes**, including 100% scripted grasp/lift/five-second hold.
The linked repair report includes the preserved GPU artifacts and remaining
verification limits. The original measurements below are retained as
evidence of the earlier model.

Reviewed 2026-11-03 against local HEAD `3c6486d`, with the current 42 mm
graspable collider revision. The user confirms the server has the latest
commit. The sizes fit the gripper and receptacles, and the rendered proportions
are plausible for small tabletop props. **The setup is not fully verified:
sustained gripping fails in local CPU MuJoCo.** Confirm that failure on the
server's MJWarp backend before attributing it to the training environment.

## Evidence

- [Measured results and source hashes](../../artifacts/widowx200_object_scale_20261103/scale_audit.json)
- [Fresh placement preview](../../artifacts/widowx200_object_scale_20261103/fit_preview.png)
- [Fresh wrist and inspection views immediately after lift](../../artifacts/widowx200_object_scale_20261103/grips.png)
- [All selected drop and scripted-placement frames](../../artifacts/widowx200_object_scale_20261103/placements.png)
- [Reach summary](../../artifacts/widowx200_object_scale_20261103/reach_summary.json)
- Reproducible diagnostic: `tools/audit/audit_widowx200_object_scale.py`.

The physics diagnostic used CPU MuJoCo **3.3.7**, a 2 ms timestep and the
repository's mounted robot/controller, asset meshes, collision primitives,
mass and inertia. Mixed object/container scenes were freshly compiled so their
collision BVHs matched the selected variants. This is not a trained-policy
evaluation, CUDA parity check, or a measure of the completed remote run.

## What passes

| Check | Result |
|---|---:|
| Existing robot, kinematics, controller, camera and pipeline tests | 53 passed |
| Sampled reset/required-waypoint reach checks | 1,536/1,536 |
| First-contact measurements and short lift at the configured squeeze | 5/5 |
| Central drops supported by their receptacle after settling | 60/60 |
| Central drops with the complete visible XY extent conservatively inside | 60/60 |

The reach audit sampled 384 scenes for each of move-to, pickup, caught
placement and composed placement, over rounds 0, 1, 2, 3, 100 and 101.
Its carry waypoints follow an arc around the waist keep-out. These checks do
not establish collision-free motion along every possible trajectory.

The drop panel tests every combination of five graspables, plate/bowl and six
lattice destination cells. It isolates container fit from grasp-controller
errors by releasing each object centrally from the configured 70 mm
body-origin height above the receptacle. Support is measured across the final
0.5 simulated seconds, with container contact in at least half the samples and
less than 2 mm object motion. This avoids judging contact chatter from one
instantaneous velocity sample. All visible XY bounds remain inside the
conservative container footprint.

Scaling consistently changes visual meshes, primitive sizes/offsets, rest
heights, mass and inertia. The measured first-contact openings reproduce the
configured values to 0.001. The mug remains a reach target/distractor rather
than a grasp target; banana and carrot stay outside the WidowX pools.

## Visual scale

Measured visible mesh bounds, in the settled upright orientation:

| Object | X × Y × Z, mm | Mass, g |
|---|---:|---:|
| Apple | 37.3 × 37.0 × 42.7 | 40.0 |
| Tomato | 43.2 × 43.2 × 40.5 | 40.0 |
| Orange | 50.4 × 50.4 × 40.1 | 48.5 |
| Potato | 42.6 × 82.8 × 45.0 | 56.0 |
| Bell pepper | 47.9 × 62.7 × 42.3 | 40.0 |
| Mug, including handle | 36.2 × 52.9 × 41.6 | 40.0 |
| Plate | 148.0 × 148.0 × 16.0 | 112.6 |
| Bowl | 102.0 × 102.0 × 44.7 | 122.8 |

The plate/bowl proportions look reasonable for these small props in the fresh
renders. This is an appearance judgment, not validation against a physical
object set. The settled visual bottoms differ from the desk by approximately
−0.9 to +1.6 mm, rather than showing a large global scale/rest-height error.

The collider width is not the visible width. In particular, the orange's
collision sphere is **41.76 mm** across while its visible XY width is
**50.4 mm**. A pad contacting that sphere can intersect the visible orange by
roughly 4.3 mm on each side, before contact compliance. The apple has the
opposite mismatch: a 42.09 mm collision sphere around an approximately 37 mm
visible width can leave a visible gap to the pads. The pepper also has a
coarse multi-sphere approximation. Uniform rescaling preserves these
pre-existing shape mismatches. Smaller scales alone will not repair them.

The bowl's collision walls provide an 85 mm square inner opening. The
approximately 83 mm long visible potato fits centrally, with little margin
along its long axis. The configured 48 mm **object-centre** success tolerance
is much looser than whole-object containment: a conservative square-opening
check leaves only about 1 mm of longitudinal centre error for the upright
potato's visible bounds. The training predicate is not a visual containment
certificate. Changing its meaning would require a separately evaluated task
revision.

## What fails locally

At the calibrated first-contact opening minus the configured 0.20 squeeze:

- All five objects pass the short lift measurement.
- **0/5** remain held after five stationary simulated seconds following an
  actual descend/close/lift sequence.
- **0/60** of the scripted arc carries retain the object until deliberate
  opening; **0/60** finish supported by the intended container in that panel.

For example, the apple initially lifts 79.6 mm and sits 4.7 mm from the
tracked end effector. After one second it is 14.8 mm away; by two seconds it
has returned to the desk and is 91.2 mm away. The raw report contains the
one-second traces for all five objects.

An exploratory fully closed carry probe also lost its grip, and a fully
closed stationary apple probe dropped by three seconds. These observations
motivate a sustained-hold check; they do not prove that every controller,
gripper action sequence or grasp pose must fail. The scripted motions are not
the learned policy, and their duration is not constrained to the RL horizon.
CPU MuJoCo and the server's MJWarp contact behavior may differ.

The previous GPU preflight observed caught starts for only four actions
(0.20 s at this control period), and held after scripted lift for six actions
(0.30 s). Those windows can miss the delayed slip seen here. The local
preflight now checks five simulated seconds in both places, checks post-hold
finiteness, and requires both remaining lift and proximity to the gripper.
`--hold-seconds` records and controls the duration. Its production pass-rate
thresholds remain unchanged.

## Server verification

Sync the changed preflight and new audit tool first. The supplied
`widowx200_scale_checks.patch` contains only this audit's code/documentation
changes, excluding the other in-progress pipeline/report edits in the local
checkout. Applying the patch does not change object dimensions, contact
parameters, reward constants or policy weights.

Run the stronger check directly, without launching training:

```bash
cd /root/repo/RL_VLA_Bootstrapping
conda run --no-capture-output -n cdpr-mjlab python3 scripts/preflight_widowx200_pipeline.py \
  --device cuda:0 --worlds 512 --rounds 2 --hold-seconds 5 \
  --output runs/widowx200_geometry_preflight_20261103
```

For the same CPU diagnostic on the server:

```bash
conda run --no-capture-output -n cdpr-mjlab python3 tools/audit/audit_widowx200_object_scale.py \
  --output runs/widowx200_scale_audit_20261103
```

Add `MUJOCO_GL=egl` before `conda run` and `--render` to the audit command
for headless Linux RGB previews if EGL is configured. Rendering is optional;
the physical checks run without it. The audit exits nonzero if either fit or
motion checks fail, and writes the measured failure details to JSON.

The launcher also picks up the stronger preflight on its next invocation;
the tracked code diff changes its validation fingerprint, preventing reuse of
the old pass stamp. No remote command or training was executed in this review.
Changing files on disk does not change the catalog in an already-running
Python process: scales are selected once at import. The server's current HEAD
alone cannot establish which sizes an earlier checkpoint learned with.

Do not shrink the props further on the strength of these results. First
check sustained holding on MJWarp; if it reproduces, investigate grip/contact
stability before another full pipeline run. Review visual/collider alignment
and bowl placement margins independently. The present evidence supports
container fit and plausible overall proportions, not a complete natural,
stable pick-and-place sequence.

Local verification: the 53 existing relevant tests passed, including three
camera tests with macOS graphics access. The new CPU audit ran through all
60 placement pairs and deliberately returned failure for its motion checks;
Python compilation, CLI help and whitespace checks passed. The strengthened
GPU preflight has not been executed locally because this machine has no CUDA
device.
