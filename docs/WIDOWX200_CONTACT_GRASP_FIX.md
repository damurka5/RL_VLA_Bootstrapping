# WidowX grasp scoring and open-start repair

For a direct test of whether live physical grasps are counted, use the
[grasp-gate recording and annotated videos](WIDOWX200_GRASP_DETECTOR_CHECK.md).

The old WidowX grasp detector rejected a stable bilateral grasp whenever the
measured opening exceeded the per-object release threshold. The task predicate
then independently rejected `opening > 0.94`. Both can reject a wide object
that physically holds the fingers open. This removes pickup/lift credit and
can prevent stage-7 progress. It does not establish that every lifted object
was grasped, or that all stage-7 learning was invalid.

WidowX now uses `contact_v2`: both pads must contact the target, each carry at
least 0.05 N, and the relative pose must remain stable (at most 8 mm and 0.15
rad per environment step) for two consecutive steps. Opening does not veto
that grasp. Release still requires the opening threshold, plus no physical
grasp and neither pad carrying at least 0.05 N. Thus a wide grip over a bowl
cannot earn placement while still holding the object. Pickup, staged rewards,
stage-7 milestone credit, standalone evaluation and recording analysis share this
contract. Existing CDPR scoring retains its previous behavior.

Open aligned/uncaught WidowX resets now hover 4 cm above the grasp centre
instead of 1 cm. CPU MuJoCo reproduces greater-than-1-mm initial overlap for
bell peppers and potatoes at the old height; the 4-cm height clears both in
the 36-angle canonical yaw regression. This is not a CUDA-scene reproduction.
The production reset additionally checks the actual device contact buffer,
raises a blocked GRPO group in 1-cm increments, and fails if six corrections
cannot clear target/robot penetration deeper than 1 mm. Object poses and RNG
draws are preserved. Intentionally caught starts are excluded from this guard.

Both collision pads are now visible in the policy cameras at their actual
positions and sizes. The pad physics and object collision proxies retain their
existing shapes. This corrects the invisible pad surface; it does not make
every object collider identical to its visual mesh.

New NPZ files store the grasp contract, either-pad load mask, and initial
object/tool quaternions. Old NPZ files retain opening-v1 reconstruction. They
cannot recover missed grasps without the original contact forces, so do not
relabel them by object height alone. Replay rejects a bank with a different
grasp contract; re-record demonstration banks under the fixed environment.

## Server checks before training

Run from the repository root. Use fresh output directories to retain the old
results. The new evaluation changes both grasp scoring and open-start geometry;
compare it as a repaired protocol, not as a pure detector-only counterfactual.

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only origin main
export RLVLA_EMBODIMENT=widowx200 MUJOCO_GL=egl PYTHONUNBUFFERED=1
export PIPELINE_DIR="$PWD/runs/widowx200_pipeline_fresh_20261104_162138_CBEU9r"
export CONTACT_RUN="widowx200_s7_contact_v2_$(date +%Y%m%d_%H%M%S)"
export CONTACT_AUDIT="$PWD/runs/${CONTACT_RUN}_checks"
py() { conda run --no-capture-output -n cdpr-mjlab python "$@"; }

py scripts/preflight_widowx200_pipeline.py \
  --config configs/examples/widowx200_smolvla_pipeline.yaml \
  --output "$CONTACT_AUDIT/preflight"

py tools/audit/policy_outcome_videos.py \
  --pipeline-dir "$PIPELINE_DIR" --evaluation put_into_composed \
  --rounds 4 --track-scenes 16 --output "$CONTACT_AUDIT/final_composed"

py tools/audit/placement_failure_decomposition.py \
  --config "$CONTACT_AUDIT/final_composed/source_eval_config.yaml" \
  --recordings "$CONTACT_AUDIT/final_composed/record_*.npz" \
  --output "$CONTACT_AUDIT/decomposition"
```

Preflight must pass its existing held-grip, scripted-lift and reach checks, plus
`max_open_grasp_start_penetration_m <= 0.001`. Its entries and the video
manifest should say `grasp_contract: contact_v2`. Inspect successful and failed
bell-pepper/potato episodes, then compare the grasp/release/XY funnel and
per-object rates. The GPU preflight and a new policy evaluation have not been
run on the local Mac; CPU regressions do not establish the new success rate.

## Warm-start a fresh stage 7 after the checks pass

The one-command launcher prepares the saved stage-7 config with the phased
milestone reward, runs CUDA preflight on that exact config, and starts joint RL:

```bash
conda run --no-capture-output -n cdpr-mjlab python tools/train/continue_widowx200_stage7.py \
  --pipeline-dir runs/widowx200_pipeline_fresh_20261104_162138_CBEU9r --execute
```

Run it inside tmux from the repository root after pulling the fix. The default
initializer is stage 7's selected/handoff policy. `--checkpoint /path/to/policy.pt`
can choose a different saved policy, and `--steps 10000000` sets a new 10M-action
budget. Otherwise the saved stage-7 budget is retained (30M actions for this
run). Placement gets 0.2 per reached milestone: approach, grasp, held lift,
carry over the container, successful placement. Move-to and pickup retain their
own shorter milestone chains. This is the existing staged reward throughout
the run, rather than a schedule that switches reward formulas at step counts.

The launcher prints the new run directory. It saves `continuation.json`,
`preflight.log`, and `training.log` there, and selects checkpoints using
`min_by_instruction_composed`. It runs only stage-7 RL; earlier stages,
composition SFT, and SIL are not rerun. Source configs and pipeline state are
preserved. A failed preflight stops before training. Omitting `--execute` only
prepares the run and prints commands.

### Retry an expired-token failure during model loading

The initial stage-7 launcher omitted the main shell launcher's public-model
authentication setup. An expired cached OAuth token was consequently sent to
the public SmolVLM2 processor endpoint, producing HTTP 401 before RL updates.
The corrected launcher removes inherited tokens, disables implicit credentials,
and redirects the child process's token-file lookup to the null device. It
does not delete credentials or change `HF_HOME`/`HF_HUB_CACHE`.

After pulling the correction, retry this already prepared run:

```bash
RLVLA_HF_PUBLIC_MODELS_ONLY=1 conda run --no-capture-output -n cdpr-mjlab python tools/train/continue_widowx200_stage7.py \
  --retry-run runs/widowx200_s7_contact_v2_20261113_135223 --execute
```

This preserves its config, selected initializer, budget, phased rewards and
output directory; only HF environment settings are repaired. A passed
preflight is reused for this startup retry. If preflight did not pass it runs
again. Startup retry refuses runs with saved training checkpoints so progress
cannot be silently restarted. Use a new continuation when changing the task
or simulator, rather than treating this authentication retry as a new preflight.

`RLVLA_HF_OFFLINE=1` now enables both Hub and Transformers cache-only loading
for this launcher, if all model and processor files are already cached. Public
mode is the default; `RLVLA_HF_PUBLIC_MODELS_ONLY=0` retains authentication for
an intentionally private model. These modes follow the repository's existing
shell helper and Hugging Face's [environment-variable documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

### Plateau stop at 5,006,757 actions

The supplied `early_stop.json` confirms a normal plateau stop at step 5,006,757.
The monitored metric is `min_by_instruction_composed`: the worst instruction,
using the uncaught/composed validation leg for plate and bowl. Its earlier best
was 0.0535714 (5.36%); the best within the latest 16 validations was 0.0446429
(4.46%). The rule required a gain greater than 0.02 (two percentage points) and
was allowed to stop after 5M actions. The 30M budget was an upper bound.

The progress bar's `1053/2048` counts training candidates across mixed
instructions and assisted starts. It is not a composed placement rate. Inspect
the per-instruction curves and the actual best checkpoint before choosing an
extension:

```bash
conda run --no-capture-output -n cdpr-mjlab python tools/audit/inspect_widowx200_stage7.py \
  --run-dir runs/widowx200_s7_contact_v2_20261113_135223 \
  --output runs/widowx200_s7_contact_v2_20261113_135223/stage7_diagnosis.json
```

The report uses the trainer's own metric reader, matches the configured
validation caps, identifies the bottleneck instruction and best saved policy,
and reports non-finite force/loss metrics. It loads no checkpoints and starts
no GPU work. No stop file means the reason is unconfirmed, rather than an
assumed plateau.

The force warning is emitted when TensorBoard skips non-finite scalars; that
warning does not itself terminate training. A separate backend masking bug can
produce those scalars: `NaN * 0` and `Inf * 0` from an unused or unrelated
contact contaminate the target's pad load. Contact-force reduction now selects
matched pad contacts with `torch.where` before summing. Real matched non-finite
pad forces remain visible; this does not establish that every warning came
from unused contacts or that every physical rollout was finite.

An extension of this trained contact-v2 run should resume its full checkpoint,
including optimizer and curriculum. `--retry-run` is only for failures before
the first saved checkpoint. A full resume also restores stopping history, so
the same plateau rule remains active unless it is deliberately changed. The
diagnosis report supplies the per-instruction evidence for that decision.

Reuse learned weights, with a fresh optimizer, curriculum and validation-best
history. The completed pipeline skips finished stages, and a full checkpoint
resume preserves the old histories and may fail the changed XML fingerprint.
Use a new run instead. This starts from the old final selected policy; keep the
stage-6 handoff available for a separate initializer comparison if needed.

The following copies the saved stage-7 configuration into the same directory
so relative asset paths stay correct. It preserves the original training
recipe and budget, explicitly records the new hover, and refuses overwrites.
It does not reuse historical SIL banks or run the whole pipeline's post-stage
retention machinery.

```bash
export CONTACT_CONFIG="$PWD/configs/widowx200_pipeline/$CONTACT_RUN.yaml"
py - <<'PY'
import json, os
from pathlib import Path
import yaml

p = Path(os.environ['PIPELINE_DIR'])
state = json.loads((p / 'pipeline_state.json').read_text())
source = Path(state['stages']['s7_joint_staged']['config']).resolve()
destination = Path(os.environ['CONTACT_CONFIG']).resolve()
if source.parent != destination.parent:
    raise SystemExit('Keep the new YAML beside the original stage-7 YAML.')
run = os.environ['CONTACT_RUN']
if (Path.cwd() / 'runs' / run).exists():
    raise SystemExit('Choose a fresh CONTACT_RUN; run directory exists.')
config = yaml.safe_load(source.read_text())
config.setdefault('project', {})['name'] = run
config['task'].setdefault('metadata', {})['open_grasp_hover_height'] = 0.04
config['training']['rl']['args'].pop('resume_checkpoint', None)
checkpoint = Path(json.loads((p / 'final_report.json').read_text())['policy'])
if not checkpoint.is_file():
    raise SystemExit(f'Missing initializer: {checkpoint}')
with destination.open('x') as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
out = Path(os.environ['CONTACT_AUDIT'])
out.mkdir(parents=True, exist_ok=True)
(out / 'warmstart.txt').write_text(str(checkpoint) + '\n')
(out / 'training_source.json').write_text(json.dumps({
    'source_config': str(source), 'config': str(destination),
    'warmstart': str(checkpoint), 'grasp_contract': 'contact_v2',
    'source_pipeline': str(p),
}, indent=2) + '\n')
print(destination)
print('Weights-only initializer:', checkpoint)
PY

unset RLVLA_SMOLVLA_RESUME_CHECKPOINT
export RLVLA_SMOLVLA_WARMSTART_CHECKPOINT="$(cat "$CONTACT_AUDIT/warmstart.txt")"
export RLVLA_SMOLVLA_RESET_LOG_STD=1
py -m rl_vla_bootstrapping.cli.train --config "$CONTACT_CONFIG" \
  --stage rl --run-name "$CONTACT_RUN" --execute
```

Training/validation logs expose `grasp_contract_version = 2`. Improved grasp
credit should remove this learning obstacle; the carry/release policy still
needs to learn. A higher composed success rate must be measured after the fix.
