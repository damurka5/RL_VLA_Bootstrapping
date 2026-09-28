# CDPR manipulation task consistency review (pick_up, put_into_plate, put_into_bowl)

Pre-flight review requested before launching
`scripts/train_cdpr_smolvla_pick_up_grpo_mjlab_dual_remote.sh`.

**Verdict: do not launch yet.** The grasp phases cannot succeed as configured.
The dense reward's optimum is 7.25 cm above where the fingers actually are, and
the controller floor forbids the end-effector from ever reaching grasp height.
Both are single-number config errors traceable to one wrong assumption about the
gripper's geometry.

## How this was checked

`scripts/render_cdpr_task_reference_episodes.py` runs a scripted oracle -- not a
learned policy -- through the production task code:

| component | source |
|---|---|
| episode start | `BatchedReverseFrontierResetter.reset` (the trainer's own resetter) |
| reward + success | `evaluate_active_sparse_tasks` + `BatchedCatchReleaseDenseReward.from_metadata` |
| grasp evidence | `RankLocalMJWarpGRPOCollector._update_physical_grasp`, called unbound |
| scene | `robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml`, same catalogs/cameras |
| physics | **MuJoCo CPU** (`MujocoReferenceBatchedBackend`) -- the one substitution |

MJ-Lab/MJWarp is CUDA-only and cannot run on macOS, so physics is MuJoCo's CPU
pipeline on the identical MJCF. Every finding below except **F7** is model
arithmetic or config arithmetic and is independent of the solver.

The overlay's reward breakdown is asserted every step to sum to the reward the
production function returned (tolerance 2e-4), so the numbers on screen cannot
drift from the real reward.

## Findings

### F1 — `pick_grasp_height_offset: 0.08` is ~10x the real gripper offset

Measured from the MJCF, relative to `ee_base` (which is what
`low_dim.ee_position` reports, in both backends -- `mjlab_mjwarp_backend.py:287`):

| feature | offset below `ee_base` |
|---|---|
| finger pad **centre** | **0.0075 m** |
| pad vertical span | −0.010 … +0.025 m |
| deepest gripper point (finger tip edge) | 0.0395 m |

The config asserts 0.08 m ("the gripper hangs 0.08 m below it"). With 0.08 the
reward's maximum is at a pose where the pads sit **7.25 cm above the object
centre** — a hover the fingers cannot close on.

The as-configured `pick_up` episodes show exactly this: the oracle reaches the
reward's own optimum (`distance_coarse` = 1.000, `distance_fine` ≈ 0.48, total
≈ 1.48) and sits there for the whole 64-step budget with **0.000 N on both
pads**. It never earns `contact_bonus`, `grasp_bonus`, `lift` or `success_bonus`.

### F2 — the controller floor blocks grasp height independently

`CDPRBackendConfig.workspace_z = (0.25, 0.60)` is used unmodified by the trainer
(`smolvla_grpo_mjwarp_cdpr.py:984` passes no override), and clamps both
`set_end_effector_poses` and every `step`. So the pads can never be commanded
below 0.2425 m, while desk objects' centres sit at 0.160–0.195 m.

Sweeping every graspable catalog at the 0.25 floor: only `robocasa_mug` produces
any pad force at all. Even a corrected `pick_grasp_height_offset` would be
clamped away, so **F1 and F2 must both be fixed**.

The comment justifying 0.25 ("keeps the grasp point 2 cm clear") was derived
from the same wrong 0.08 offset.

### F3 — placement never actually holds the object

The reset places the carried object at `ee_z − 0.08`
(`mjwarp_rank_local_collector.py:1507,1515`, hard-coded, *not* read from
metadata) and sets `ever_grasped = grasped = physical_grasp = True`. At the real
pad offset that leaves the object 7.25 cm below the fingers, in free space.

Observed, as configured: pad forces 0.000/0.000 N from step 1, the object
free-falls, and `wrong_place_settled` fires — **the episode terminates on env
step 1 of 64 for `put_into_plate`, step 6–7 for `put_into_bowl`**, taking
`placement_wrong_drop_penalty`. The policy never gets a decision that matters.

### F4 — `fitted_gripper_opening` is too open for 6 of 8 graspable objects

Opening at which bilateral pad contact first appears, measured at the corrected
grasp height:

| catalog | table | measured | best lift |
|---|---|---|---|
| robocasa_apple | 0.785 | **0.45** | 118 mm |
| robocasa_bell_pepper | 0.868 | **0.55** | 119 mm |
| robocasa_tomato | 0.685 | **0.30** | 120 mm |
| robocasa_orange | 0.768 | **0.30** | 119 mm |
| robocasa_potato | 0.602 | **0.25** | 116 mm |
| robocasa_mug | 0.800 | 0.80 | 115 mm |
| robocasa_banana | 0.368 | never | — |
| robocasa_carrot | 0.200 | never | — |

The reset seats a "caught" object at `fitted − 0.033`, so for these six the
fingers are left wide open around an object they are supposed to be holding.

### F5 — banana and carrot cannot be gripped at all

Pad separation ranges 0.0969 m (open) to **0.0403 m (fully closed)**. Banana and
carrot are 0.028 m across, below the closed gap. They are ungraspable by this
gripper at any opening — and `robocasa_banana` is in the phase-1
`target_object_pool`, so ~1/6 of `pick_up` episodes would be unsolvable even
after F1–F4 are fixed.

### F6 — `put_into_plate` loses a correct placement to a wrong-drop race

`released` requires `gripper_opening >= max(0.55, fitted + 0.04)` (0.825 for the
apple), but the object leaves the pads at ~0.45 and the gripper opens at 0.05 per
env step. `target_has_settled` is a pure height test with no contact or velocity
condition, and at `put_plate_release_height: 0.045` the carried object is
*already* below that height while still gripped. So the instant the pads let go,
`wrong_place_settled` fires with `released` still false: a placement landing
1.9 mm from the plate centre is scored as a wrong drop and penalised.

`put_into_bowl` escapes this only because its 0.10 release height keeps the
object above the settle threshold while carried.

Related: `caught_object_start_release_opening_margin: 0.04` in the config is dead
— the resetter hard-codes `max(0.55, fitted + 0.04)`.

### F7 — wrist camera goes pure white at grasp height *(reference renderer only)*

Below `ee_z ≈ 0.24` the `ee_camera` image saturates to 255 on every pixel; at
grasp height it is uniformly blank. Cause is the scene headlight
(`cdpr.xml:32`, `diffuse=".62 .62 .62"`); at 0.30 the saturation disappears
(mean 202, no clipped pixels) with the view still well lit.

If this reproduces under MJWarp's renderer, SmolVLA's wrist input carries **zero
information exactly when the grasp decision is made**. This is the one finding
that must be re-confirmed on the GPU renderer before acting on it.

### F8 — `ee_workspace_z_bounds` will need re-deriving

`[0.27, 0.38]` currently brackets the *wrong* grasp point. Against a corrected
one (~0.19 m) the start is always ≥ 8 cm away, so the 3 cm approach-curriculum
cap silently cannot bind — observed: cap 0.03, realised start distance 0.083–0.089 m.

## Evidence that the reward is right once the geometry is

With `--grasp-height-offset 0.0075 --controller-z-floor 0.18` and nothing else
changed, the same oracle and the same reward function give:

| instruction | result | terminal reward |
|---|---|---|
| `pick_up` | **3/3 success** | 5.70 – 5.72 |
| `put_into_bowl` | **2/2 success** | 3.48 |
| `put_into_plate` (+`put_plate_release_height=0.10`, release threshold 0.55) | **3/3 success** | 3.48 |

The `pick_up` reward ladder matches the config's own predicted terminal returns
almost exactly (config comment: 1.5 hovering → 1.75 contact → 2.75 grasp →
3.75 lift → 5.75 success; measured 1.49 → 1.74 → 2.85 → 3.68 → 5.72).

**So the reward shaping itself is sound and does describe the instruction.** It
is measuring the right quantities against the wrong geometric constant.

## Suggested order of fixes

1. `pick_grasp_height_offset: 0.08 → ~0.0075` (both configs).
2. `CDPRBackendConfig.workspace_z` low bound `0.25 → ~0.18`, and re-derive the
   "fingers through the desk" guard from `ee_z − 0.0395`.
3. Un-hard-code the `0.08` in the resetter's caught-object start
   (`mjwarp_rank_local_collector.py:1507,1515`) so it follows the metadata.
4. Re-measure `fitted_gripper_opening` for all catalogs from contact, not from
   mesh bounds; drop banana and carrot from the graspable pools.
5. Gate `wrong_place_settled` on `released`, or raise
   `put_plate_release_height`, or both.
6. Re-derive `ee_workspace_z_bounds` around the corrected grasp point.
7. Check F7 on MJWarp's renderer.

## Reproducing

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".:robots/cdpr" python scripts/render_cdpr_task_reference_episodes.py --instructions pick_up --episodes-per-instruction 2 --output runs/as_configured
```

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".:robots/cdpr" python scripts/render_cdpr_task_reference_episodes.py --instructions pick_up --grasp-height-offset 0.0075 --controller-z-floor 0.18 --target-catalogs robocasa_apple robocasa_tomato --output runs/corrected
```

Videos (overview, wrist, side-by-side composite; telemetry and reward burned in),
per-step CSV and a manifest land under `runs/cdpr_task_reference_episodes/`.
The manifest records `exact_production_backend: false` on every episode.
