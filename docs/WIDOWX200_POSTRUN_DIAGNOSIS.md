# Inspecting the completed WidowX run on the server

For the subsequent grasp/reset correction and commands to reevaluate or
warm-start stage 7, see [the contact-v2 repair](WIDOWX200_CONTACT_GRASP_FIX.md).
Keep the original diagnostics: their opening-v1 scores describe the old run.

The reported composed rates are 35/520 bowl (6.7%) and 26/504 plate (5.2%),
61/1024 overall (6.0%). `put_into` starts holding the object;
`put_into_composed` starts empty-handed, already aligned over the object. It
does not test an approach from a distant pose. The stage-6 validation minimum
of 74.0% therefore does not mean stage 6 could complete 74% of whole tasks.

SIL candidates were accepted after stages 2, 3, 4 and 6. Stage 5 rejected its
only candidate. Stage 6 accepted its first candidate and rejected its second;
the first is the handoff. Stage 1 had no previous skill to retain. Stage 7
has SIL disabled in this recipe. An accepted candidate need only improve the
**mean** gain on degraded instructions by 2 percentage points while keeping
loss on other instructions within 5 points. It need not fully restore every
instruction. Read actual before/after rates below.

Run the checks in this order. These commands create diagnostic outputs and
do not resume training, change pipeline state or replace checkpoints. Keep
the server's simulator/code unchanged for the first audit; record its revision
before updating anything. The new `policy_outcome_videos.py` must be copied to
`tools/audit/` on the server if it is not there yet.

## 1. Export numerical evidence and print the SIL recoveries (no GPU)

Run from the repository root on the server. Keep this shell open for the later
commands. `py` uses the same conda environment as the training launcher.

```bash
cd /root/repo/RL_VLA_Bootstrapping
export PIPELINE_DIR="$PWD/runs/widowx200_pipeline_fresh_20261104_162138_CBEU9r"
export AUDIT_DIR="$(mktemp -d "$PWD/runs/widowx_diagnosis_XXXXXX")"
export RLVLA_EMBODIMENT=widowx200 MUJOCO_GL=egl PYTHONUNBUFFERED=1
py() { conda run --no-capture-output -n cdpr-mjlab python "$@"; }
git rev-parse HEAD > "$AUDIT_DIR/server_commit.txt"
git diff HEAD > "$AUDIT_DIR/server_changes.patch"
py tools/audit/export_widowx200_pipeline_results.py \
  --pipeline-dir "$PIPELINE_DIR" --output "$AUDIT_DIR/snapshot"
```

Print the exact per-instruction effects of retention SIL:

```bash
py - <<'PY' | tee "$AUDIT_DIR/sil_recovery.txt"
import json, os
from pathlib import Path
p = Path(os.environ['PIPELINE_DIR'])
s = json.loads((p / 'pipeline_state.json').read_text())
for name, stage in s['stages'].items():
    cycles = stage.get('sil_cycles') or []
    print('\n' + name, 'SIL', sum(bool(c.get('accepted')) for c in cycles), '/', len(cycles))
    print('  selected RL:', stage.get('selected_checkpoint') or stage.get('best_checkpoint'))
    print('  handoff:    ', stage.get('policy'))
    before = stage.get('retention_rates') or {}
    after = stage.get('retention_rates_after_sil') or before
    for instruction, b in before.items():
        a = after.get(instruction)
        if a is None:
            continue
        print(f"  {instruction}: {b['successes']}/{b['episodes']} ({b['rate']:.1%})"
              f" -> {a['successes']}/{a['episodes']} ({a['rate']:.1%});"
              f" delta {100*(a['rate']-b['rate']):+.1f} percentage points")
    if not before:
        print('  No retention evaluation recorded for this stage.')
    for i, c in enumerate(cycles, 1):
        print('  cycle', i, 'ACCEPT' if c.get('accepted') else 'REJECT', c.get('reason'))
        print('    candidate gains:', c.get('gains', {}))
print('\nComposition transition:', json.dumps(s.get('composition_sft', {}), indent=2))
print('\nJoint initialization:', json.dumps({k: v for k, v in
      s['stages'].get('s7_joint_staged', {}).items()
      if k in ('init', 'composition_sft_id', 'composition_sft_skipped')}, indent=2))
final = json.loads((p / 'final_report.json').read_text())
print('\nFinal retention:', json.dumps(final.get('retention', {}), indent=2))
PY
```

`snapshot/report.md` and `retention.csv` show the same before/after comparisons.
`stages.csv` separates selected RL from post-SIL handoff; `banks.csv` identifies
the actual retained teacher for each skill. `validation_slices.csv` shows
instruction/object curves. Compare final skills against the matching retention
protocol, rather than subtracting a best validation maximum from a final test.
The snapshot includes numerical reports/configs, not models or camera arrays.

## 2. Decompose the already recorded composed failures (no new rollout)

```bash
export COMPOSED_CONFIG="$PWD/configs/widowx200_pipeline/$(basename "$PIPELINE_DIR")__eval_put_into_composed.yaml"
py tools/audit/placement_failure_decomposition.py \
  --config "$COMPOSED_CONFIG" \
  --recordings "$PIPELINE_DIR/eval/final/put_into_composed/record_*.npz" \
  --output "$AUDIT_DIR/final_composed_decomposition"
```

This reports grasp/release/settle/geometry terms, lost grip, timeouts, release
clearance, bounce and spawn-distance checks. Read
`failure_decomposition.json` and `container_episodes.csv`. The tool must agree
with the recorded success verdicts. If its predicate check fails, investigate
config/code drift; do not raise the disagreement allowance to obtain a table.
Numbers in that script's historical CDPR docstring are not a WidowX ceiling.
Even a measured scripted-oracle score is a reference, not a mathematical upper
bound on what another controller could do.

Interpret `spawn outside [0.06, 0.1]` with care. This compares object-to-receptacle
XY separation against the distance-sampled reset's range, not distance from the
robot base. With `placement_uncaught_object_at_cell: true`, composed resets leave
the object on its lattice cell and do not use that 6–10 cm sampling range.
Consequently, `520 ABOVE` does not establish an invalid reset or unreachable
object. Use the controller/IK audit below to test robot reach. A long actual
object-to-receptacle carry can still consume the available horizon.

The supplied final bowl decomposition gives 341/520 ever grasped, 107/341
released after grasp, and 35/104 correct XY after release/settling. Of 485
failures, 234 first fail release (48.2%), 179 fail grasp (36.9%), 69 miss XY
(14.2%), and 3 fail settling (0.6%). Within the 234 without release, 131 end
holding the object and 103 end without it. Watch both groups: increasing the
budget may help unfinished carries, but does not directly fix grip loss. The
reported zero inside-radius release failures gives no evidence that bounce or
the final Z predicate is the main loss in these evaluated bowl episodes.

## 3. Check sampled poses, controller reach and camera visibility (CPU)

Build the coverage input from the **saved final evaluation configs and caps**,
so a later edit to the example YAML does not silently change the task being
audited. The helper resolves per-device summaries as well as unsharded ones.

```bash
py - <<'PY'
import os, yaml
from pathlib import Path
from tools.audit.policy_outcome_videos import pipeline_selection
from tools.audit.widowx200_scene_coverage import pipeline_view, eval_config
p, out = Path(os.environ['PIPELINE_DIR']), Path(os.environ['AUDIT_DIR'])
skills = {}
for skill in ('move_to', 'pick_up', 'put_into'):
    selection = pipeline_selection(p, skill, 'final', 'handoff')
    path = selection['config']
    config = yaml.safe_load(path.read_text())
    config['simulator']['fixed_scene_xml'] = str((path.parent / config['simulator']['fixed_scene_xml']).resolve())
    skills[skill] = {'eval_cap': selection['start_distance_cap'], 'overrides': config}
coverage = out / 'coverage_pipeline.yaml'
coverage.write_text(yaml.safe_dump({'widowx200_pipeline': {'skills': skills}}, sort_keys=False))
actual = yaml.safe_load(pipeline_selection(p, 'put_into_composed', 'final', 'handoff')['config'].read_text())
rebuilt = eval_config(pipeline_view(coverage), 'put_into', True)
if rebuilt['task'] != actual['task']:
    raise SystemExit('Composed-task reconstruction differs from the saved task; inspect code/config drift before coverage.')
print(coverage)
PY

# Reach/IK/controller clamps only; no camera rendering.
py tools/audit/widowx200_scene_coverage.py \
  --config "$AUDIT_DIR/coverage_pipeline.yaml" \
  --rounds 0 1 100 101 102 103 --no-render \
  --output "$AUDIT_DIR/reach"

# Add both policy cameras, segmentation/occlusion and a contact sheet.
py tools/audit/widowx200_scene_coverage.py \
  --config "$AUDIT_DIR/coverage_pipeline.yaml" \
  --rounds 0 1 100 101 102 103 \
  --output "$AUDIT_DIR/reach_and_visibility"
```

Inspect `scenes.csv`, `summary.json` and `contact_sheet.png`:

- `reach_grasp`, `reach_lift`, `reach_release`, `reach_carry`, `ik_*` and
  `grasp_z_below_floor`: distinguish workspace clamps, IK failure and height.
- `grasp_yaw_coverage`: orientation-dependent feasibility.
- `overview_target_pixels_visible`, wrist visibility, and occlusion columns:
  distinguish a reachable object from one the policy cannot see well.

This is a kinematic/visibility audit, not proof of collision-free dynamic
graspability. It uses the current scene assets/controller implementation; the
saved YAML alone cannot recover older geometry. Rounds 0/1 are the final test
resets, while rounds 100–103 check the usual harvest resets at each skill cap.
CPU and CUDA random streams differ. These CPU samples characterize the sampler;
they do not reproduce the exact GPU scenes, even with identical seeds. Use the
CUDA preflight and the video recording's true reset poses for that check.

## 4. Watch live policy successes AND failures (GPU)

```bash
py tools/audit/policy_outcome_videos.py \
  --pipeline-dir "$PIPELINE_DIR" --evaluation put_into_composed \
  --rounds 4 --track-scenes 16 --per-outcome 8 \
  --output "$AUDIT_DIR/final_composed_videos"
```

This runs the original 512-world, group-size-8 evaluation setup, with the task's
controller bounds and explicit recorded cap. It keeps one randomly sampled
world from each of 16 distinct reset groups per round for video. It saves
telemetry for all 512 worlds, including failures. Video capture is from the
same live rollout, with no action smoothing or replay. The views are overview
and wrist side by side, one frame per policy decision, plus an extra final-state
render. Filenames identify instruction, object and success/failure; JSON
sidecars give reset positions, grasp/lift measurements and divergence flags.

`summary.json` has success and ever-grasped rates by instruction/object/spawn
position, using **all worlds**, not just the filmed ones. The spawn grouping
rounds XY to 1 mm; raw coordinates remain in `episodes_*.csv`. Copies within a
GRPO reset are not independent scenes. A small video sample may contain no
success for an instruction; increase `--rounds` or `--track-scenes` to collect
more, and use `videos.json` to see what was actually filmed. Camera RAM scales
with tracked scenes; 16 scenes is roughly 300 MB at 40 decisions and two
320×240 RGB views, excluding other rollout/model memory.

To check other final skills, rerun with `--evaluation pick_up`, `move_to`, or
`put_into`, each with a new `--output`. Use rounds 0/1 for matching the original
test. Use a disjoint round range such as `--round-index 200 --rounds 8` for a
fresh diagnostic test; confirm the range was not used by training/harvesting
before treating it as held out. Seeded runs may still differ from stored runs;
the builder disables model compilation, and GPU arithmetic can differ.

## 5. Compare checkpoints on the SAME task, resets and budget

Stage 6 before SIL versus its accepted handoff is the most useful first pair.
Evaluate both empty-handed as well as with a held object. The helper keeps
final-eval scene settings fixed while loading each set of policy weights.

```bash
for kind in rl handoff; do
  py tools/audit/policy_outcome_videos.py \
    --pipeline-dir "$PIPELINE_DIR" --stage s6_put_into_binary \
    --policy-kind "$kind" --evaluation put_into_composed \
    --rounds 2 --output "$AUDIT_DIR/s6_${kind}_composed"
done
```

Repeat that pair for `--evaluation put_into`, using new output directories.
Compare with the final stage-7 policy from step 4. If the stage-6 SIL handoff
loses composed performance while restoring original-label skills, that is a
tradeoff the retention acceptance rule did not test. Repeat at stage 5 for
pickup; its rejected SFT candidate is **not** its handoff. To test a rejected
candidate explicitly, pass its `candidate` path from `pipeline_state.json` as
`--checkpoint` while keeping `--pipeline-dir` and `--evaluation` unchanged.

For a budget check, repeat the final composed command with
`--horizon-decisions 80` and a new output directory. Compare per-object
success, grasp and release times with the original budget. Report this as a
changed-budget diagnostic. `--oracle` runs the scripted controller through the
same reset/predicate stack; inspect its failures too before interpreting it as
evidence that a task is physically infeasible.

The decomposition command from step 2 also accepts the new video's
`record_*.npz` files with the same saved evaluation config.

## 6. Watch the actual stored SIL demonstrations (CPU + ffmpeg)

These are different from new policy rollouts. They show exactly the stored
camera frames that supervised SIL. Failed episodes are excluded from this
successful-experience bank; inspect policy failures through step 4.

```bash
py - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
p, out = Path(os.environ['PIPELINE_DIR']), Path(os.environ['AUDIT_DIR'])
s = json.loads((p / 'pipeline_state.json').read_text())
cycle = next(c for c in s['stages']['s6_put_into_binary']['sil_cycles'] if c.get('accepted'))
label = cycle.get('label') or Path(cycle['candidate']).parents[1].name
base = p / 'sil' / label
frames = sorted({str(path) for path in (p / 'banks').rglob('frames_*.npz')})
if not frames:
    raise SystemExit('No stored frames found; inspect banks.*.frames paths in pipeline_state.json.')
subprocess.run([sys.executable, 'tools/audit/sil_dataset_videos.py',
    '--dataset', str(base / 'refreshed' / 'demonstrations.npz'),
    '--frames', *frames, '--per-instruction', '4', '--camera', 'both',
    '--output', str(out / 's6_actual_sil_demonstrations')], check=True)
PY
```

The video tool resolves dataset row IDs against frame-file identities; inspect
its coverage report for missing files. A moved bank may require the original
frame paths from the state. `replay_survival.csv` in the snapshot shows whether
smoothing/replay preserved source successes; survival is not task success.

## Choosing the next training change

| Measured result | Next controlled experiment |
|---|---|
| Failures cluster at clamped/unreachable poses or poor camera visibility | Correct reset support, controller bounds or camera placement; rerun physical audits and establish a new evaluation baseline after the task changes. |
| Pickup works under `pick_up`, but the same objects fail to grasp under final placement prompts | Train final-prompt grasp/lift behavior on matching receptacle-present scenes, ideally continuous successful grasp-carry-release trajectories; verify composed rollouts before joint RL. |
| Objects are grasped/lifted but lost during carry | Inspect grip persistence, yaw, clearance, object-specific collision fit and grip-release actions; compare the stage-6 specialist with stage 7. |
| Objects arrive but remain held, or timeout with useful motion | Test the longer horizon; inspect release actions and reward milestones before increasing training time. |
| Released objects bounce out, or visually correct placement fails scoring | Inspect release height, settling time and predicate/geometry agreement; keep the original predicate as the baseline. |
| Accepted SIL helps old prompts but harms composed behavior | Add matched composed evaluation to candidate selection and collect relevant final-prompt data. Original-label retention acceptance alone does not certify composition. |
| Stage 7 regresses relative to its exact initializer | Check joint sampling mix, caught-start fraction, per-instruction caps, non-informative groups and validation curves; compare an earlier joint checkpoint on the same resets. |

Do not infer a need for more RL steps solely from `early stop plateau`. Check
the actual stage-7 config/curves: a 0.02 absolute improvement threshold is large
relative to a 0.04 composed minimum. Lower-rate/longer training and finer stop
thresholds are experiments only after the failure mechanism is measured.
Relabelled independent skill-stage demonstrations do not by themselves certify
a continuous, final-prompt full-task behavior.

For review, copy back `snapshot.zip`, `sil_recovery.txt`, the decomposition
JSON/CSV, reach/visibility outputs, and a few success/failure MP4+JSON pairs.
Full NPZ recordings and model checkpoints are unnecessary for an initial review.
