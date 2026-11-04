# WidowX caught-placement reset clearance

The user's fresh server run
`widowx200_pipeline_fresh_20261104_162138_CBEU9r` stopped before training.
Its reported failure list contains one entry: placement at caps plate 0.10 m /
bowl 0.08 m has **0.0064 m object/object penetration**, exceeding the unchanged
0.003 m preflight limit. The reported final capacity snapshot has zero contact
and constraint overflow. The supplied log shows both scripted grasp/lift rounds
holding all objects for five seconds. This is a reset collision, not evidence
that the repaired gripper cannot hold an object.

## Cause and correction

The caught-placement sampler moves the target fruit to the sampled gripper
position. Low carry starts allowed its centre down to 0.20 m in the placement
skill, and down to 0.19 m in joint starts. At some XY offsets the bottom of that
fruit intersects the bowl wall before the first action.

A CPU reproduction of the actual reset sampler at the middle placement rung,
two rounds of 64 independent GRPO groups, produced four collision contacts
deeper than 3 mm across three scene groups. The worst was an orange/bowl-wall
intersection of 4.859 mm. CPU and CUDA use different random streams; this
reproduces the failure mechanism rather than the exact server scene.

Both arm placement configurations now specify
`placement_caught_object_min_height: 0.215` metres. The sampler applies that
object-centre floor only to caught container starts, adding the configured
grasp offset to obtain the corresponding end-effector floor. The stationary
bowl collision rim is at approximately 0.186 m; carried colliders extend at
most 0.0211 m below their centres, leaving approximately 8 mm initial clearance.
The old low carry band remains available above that physical floor.

The change draws no extra random numbers and preserves catalogs, XY positions,
yaws, grip openings, other objects, horizons, pickup starts and uncaught
placement starts. Raising the affected starts moves their height toward the
release goal. CDPR configurations do not enable this new floor. Rewards,
optimizers, stage budgets and the 3 mm penetration limit are unchanged.

Preflight also repeats its failure list at the end so a short log tail shows
the reason for stopping.

## Verification

The new `tests/test_widowx200_reset_clearance.py` exercises the actual sampler
with a capture backend across all four skills, all three preflight ladder
points and both rounds. It checks preservation of unaffected starts and the
distance to the curriculum goal, then compiles actual MuJoCo collision models
for the legacy and corrected middle-rung placement poses. Contacts deeper than
3 mm fall from **4 to 0** in those 128 scene groups.

All **12 targeted checks pass**, including the sampler/collision regression,
the signed-contact checks, pipeline config parsing and arm reach projection.

This is a CPU geometry/reset regression, not a new CUDA preflight or policy
evaluation. The launcher must rerun the CUDA preflight after pulling the fix;
its source/config fingerprint invalidates any earlier completion marker.
No memory stress test or training was run locally for this fix.

## Continue the existing run into full production

Inside an interactive tmux shell, use the existing directory. Preflight failed
before either smoke or production training began, so this still starts the
fresh stage-1 policy. The command skips the optional toy smoke run and memory
stress pass; after preflight passes it runs all seven production stages.
It does not create or forge a preflight `PASSED` marker.

```bash
set +e
tmux set-option -w remain-on-exit on
(
set -euo pipefail
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only origin main
RUN_DIR="$PWD/runs/widowx200_pipeline_fresh_20261104_162138_CBEU9r"
if [[ -d "$RUN_DIR/preflight" && ! -e "$RUN_DIR/preflight_before_clearance_fix" ]]; then
  cp -a "$RUN_DIR/preflight" "$RUN_DIR/preflight_before_clearance_fix"
fi

env -u STOP_BEFORE_STAGE -u FINISH_STAGE_AT_CHECKPOINT \
  -u RESTART_JOINT_FROM_SFT -u SMOKE_DIR -u MEMORY_DIR \
  MODE=full DRY_RUN=0 SMOKE_FIRST=0 MEMORY_CHECK_FIRST=0 \
  CUDA_VISIBLE_DEVICES=0,1 \
  PIPELINE_DIR="$RUN_DIR" \
  bash scripts/train_widowx200_pipeline_remote.sh
)
WIDOWX_LAUNCH_STATUS=$?
printf 'WidowX launcher exited with status %s\n' "$WIDOWX_LAUNCH_STATUS"
```

The strict shell settings stay in the child shell, so a failed command does
not exit the interactive tmux shell. The existing `launcher.log` records the
new attempt, and preflight writes its current JSON report in the same run.
