# Fresh WidowX-200 pipeline after implementation and contact repairs

Prepared **2026-11-04**, Europe/Moscow. The next run starts from the pretrained
SmolVLA base and a fresh residual at stage 1, with empty pipeline state and
new SIL banks. The completed October 24--November 1 run remains evidence.
Its checkpoints and datasets are not reused by this launch.

Startup update: the fresh server attempt stopped on a real bowl-rim
intersection in a caught placement reset. The
[reset-clearance fix](WIDOWX200_RESET_CLEARANCE_FIX_20261104.md) adds an
object-centre height floor for caught placement only and provides a command
to continue the existing directory into full production training.

## What changed

The training schedule and configured hyperparameters remain the existing
dense skills -> binary skills with SIL -> final-prompt composition SFT ->
joint staged RL pipeline. This rerun evaluates repaired implementation and
simulation behavior, rather than a newly designed training algorithm.

Repairs already present in branch history, applied consistently from the
beginning in the fresh run:

- Fixed-distance validation and per-skill controller bounds are used for
  checkpoint selection and retention. Binary pickup retains terminal credit
  through the approach rather than cutting credit at first grasp.
- Recorder divergence masks match the collector's scored episodes; partial
  sharded recordings and interrupted replay/frame writes cannot be accepted
  as complete artifacts. Empty harvests preserve an existing usable bank.
- Source identity distinguishes same-named recordings from different banks.
  Dataset filtering removes unfilmed rows before stacking arrays, and prior
  refresh processes frame archives individually.
- Frames are selected across scenes before taking additional GRPO copies.
  Scene IDs are retained, and SFT keeps copies of one recorded scene on the
  same side of its train/validation split.
- Bank harvesting uses rounds **100 onward**, distinct from retention,
  candidate and final-evaluation rounds starting at zero. Overlap is refused.
- Final-prompt composition SFT refreshes priors/features under the student
  prompts and preserves original-label supervision in both fitting passes.
  These skill samples are not represented as continuous full-task chains.
- Manual checkpoint finishing preserves harvest/retention, and interrupted
  old joint attempts cannot bypass the composition-SFT handoff.

New repair and reporting changes included with this publication:

- WidowX contacts use **elliptic cones and impratio 10**; stronger finger
  limit constraints prevent a wide object from forcing the gripper beyond
  its mechanical travel. First-contact openings were remeasured. Object
  dimensions, friction coefficients and the shared CDPR settings are retained.
- Preflight now checks five simulated seconds of holding after caught resets
  and scripted lifting, with post-hold finiteness. Actual signed object/object
  contact penetration replaces rejection by centre distance alone. Solver
  options and simulator XML/asset hashes are recorded.
- The supplied post-repair CUDA preflight passes all 24 reset batches and both
  scripted lift/hold rounds. This is evidence of physical feasibility, not a
  newly trained policy result. [Repair and evidence](WIDOWX200_GRIP_REPAIR_20261104.md).
- Final reporting separates best RL, selected RL and SIL handoff policies,
  accepted/attempted cycles, reset protocols, exact episode counts and final
  reference-based retention status.
- The launcher keeps the required pipeline path in its production argument
  array, avoiding Bash 3.2's empty-array/nounset failure when no optional
  stop/restart flags are set. The no-memory dry run covers this fresh path.

The acceptance rule still uses mean gain over degraded instructions and a
collateral-loss limit. It does not require each degraded family to recover.
Joint RL still has `sil: false`. Those choices are unchanged; no guaranteed
per-family recovery or new success-rate result is claimed for this rerun.

## Remote command: new run, no memory stress pass

The server previously received manual model/calibration edits. Save those
tracked edits before pulling the published sources; do not reapply the old
patches on top of the repaired source. The stash is a recoverable backup.
Generated `runs/` data are left in place.

```bash
set +e
(
set -euo pipefail
cd /root/repo/RL_VLA_Bootstrapping
git stash push -m "widowx-before-fresh-$(date +%Y%m%d_%H%M%S)" -- \
  robots/widowx200/widowx200_mujoco/wx200.xml \
  rl_vla_bootstrapping/simulation/cdpr_object_catalog.py \
  scripts/preflight_widowx200_pipeline.py
git pull --ff-only origin main

RLVLA_EMBODIMENT=widowx200 conda run --no-capture-output -n cdpr-mjlab python3 \
  tools/audit/measure_widowx200_fitted_openings.py --write

mkdir -p runs
RUN_DIR="$(mktemp -d "$PWD/runs/widowx200_pipeline_fresh_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
env -u STOP_BEFORE_STAGE -u FINISH_STAGE_AT_CHECKPOINT \
  -u RESTART_JOINT_FROM_SFT -u SMOKE_DIR -u MEMORY_DIR \
  MODE=full DRY_RUN=0 SMOKE_FIRST=1 MEMORY_CHECK_FIRST=0 \
  CUDA_VISIBLE_DEVICES=0,1 PIPELINE_DIR="$RUN_DIR" \
  bash scripts/train_widowx200_pipeline_remote.sh
)
WIDOWX_LAUNCH_STATUS=$?
printf 'WidowX launcher exited with status %s\n' "$WIDOWX_LAUNCH_STATUS"
```

Calibration uses the server's installed MuJoCo version. The launcher then
runs the repaired GPU geometry/grip/camera preflight, a small smoke run, and
all seven production stages. **The production-size memory stress pass is
skipped.** A unique directory name prevents automatic reuse of any prior
stage checkpoint, completion marker or bank.

The strict error settings are confined to a child shell. A failing calibration
or preflight therefore returns to the interactive tmux shell instead of
closing its last pane.

Keep the printed `pipeline dir` path. To resume an interruption, use that same
path with `MEMORY_CHECK_FIRST=0 PIPELINE_DIR=...`; omit the `mktemp` step.
Inspect the new run's final report and export its evidence before comparing
it with the earlier experiment. Contact behavior and data handling changed,
so this is a complete corrected rerun, not an isolated causal ablation.

## Pre-publication verification

**208 CPU regression checks passed; three camera checks were skipped** because
the local process could not create a CoreGraphics context. This includes the
new contact/preflight tests, all five objects' 20-second physical holds,
pipeline state and checkpoint handling, composition-SFT and SIL recording
tests. Memory/stress cases were excluded from this verification run.

A launcher dry run on the fresh-state path executed the smoke and production
drivers with `--dry-run`, confirmed **zero `--memory-check` invocations**, and
started no training. The independent theory appendix's ten numerical sanity
checks also pass; they do not validate policy performance. The command syntax,
documentation links and whitespace checks pass. The
[verification record](../../artifacts/widowx200_fresh_run_20261104/verification.json)
records the source hashes and local versions. No new GPU training or memory
stress run was performed for publication.
