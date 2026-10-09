# Zero-init correction and latent likelihood: experiment note

Implements `docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_IMPLEMENTATION.md`.
This note keeps four things separate: implementation status, local test
evidence, GPU preflight, and training/evaluation results. GPU preflight and
one-update smoke evidence are recorded in §5. As of 2026-10-09, both
10-update diagnostics and 2M training pilots are complete. Pause further
training for the repeated matched evaluation; candidate simulator-failure
rates increased during the pilot (see §9).

## 1. Implementation status

| Piece | Where |
|---|---|
| Actor `frozen_reference_logit_correction_v1`, conversion, version tags, latent log-density | `rl_vla_bootstrapping/policy/latent_correction_policy.py` |
| Legacy actor exposes its direct logit (`ResidualChunkActor.reference_terms`); `forward` unchanged | `rl_vla_bootstrapping/policy/octo_finetune_cdpr.py` |
| Trainer: `--policy-architecture`, `--action-likelihood`, `--legacy-init-checkpoint`; optimizer over trainable params only; latent sampler; latent PPO branch; conversion/resume/save; integrity guards | `rl_vla_bootstrapping/policy/smolvla_grpo_finetune_cdpr.py` |
| Collector: `executed_action` / `policy_sample` / `behavior_mean_offset` / `likelihood_version` records; offset gate helper; component, clipping, offset and target-y-quartile telemetry | `rl_vla_bootstrapping/policy/mjwarp_rank_local_collector.py` |
| Runtime: conversion vs resume, frozen LoRA, explicit LR check, zero-update checkpoint, pilot counters, protocol JSON | `rl_vla_bootstrapping/policy/smolvla_grpo_mjwarp_cdpr.py` |
| Env knobs `RLVLA_SMOLVLA_LEGACY_INIT_CHECKPOINT`, `RLVLA_SMOLVLA_POLICY_ARCHITECTURE` | `rl_vla_bootstrapping/policy/smolvla.py` |
| Dedicated config (differs only in the two tags) | `configs/examples/cdpr_smolvla_three_stage_put_into_latent_correction.yaml` |
| Launcher (ARM=candidate/control) | `scripts/train_cdpr_latent_correction_pilot_remote.sh` |
| GPU preflight | `tools/audit/latent_correction_preflight.py`, `scripts/preflight_cdpr_latent_correction_remote.sh` |
| Evaluator provenance and direct-component traces | `tools/audit/evaluate_cdpr_full_put_into.py`, `episode_kinematic_trace.py`, `summarize_kinematic_traces.py` |
| Pilot promotion rule and target-y quartiles in the repeat comparison | `tools/audit/compare_put_into_repeats.py`, `scripts/compare_cdpr_three_stage_repeats_remote.sh` (`LABEL=PATH`, `RETENTION_MARGIN`, `SCENE_MANIFEST`) |

Version-aware loading:

- Training, in-run validation, the evaluator (`_build_world`), the video
  evaluator and `validate_cdpr_smolvla_policy.py` all build the actor from the
  checkpoint's architecture.
- A correction checkpoint is never evaluated as its reference branch alone.
- Legacy-only readers fail with an explicit message instead of loading
  partial weights: `sil_sft`, the staged-demonstration teachers,
  `ReferenceAnchor` and `reference_anchor_drift`.
- An untagged checkpoint is read as legacy. An untagged one that holds
  correction weights is refused.

Unsupported in the latent mode, failing before training:

- the CPU-backend GRPO loop;
- `sample_action_group`, `sample_action_chunks_batch` and `update`;
- LoRA updates and VLA capture;
- the reference anchor;
- the demonstration bank;
- weights-only warm start;
- passing `offset_std` to the sampler.

Mixed records fail at update time: legacy `action`/`offset_std` rows, missing
latent fields, or a wrong `likelihood_version`.

## 2. The estimator, including persistent offsets

For every executed slot of a decision:

```text
b_t   = realized episode offset * gate_t     (gate read once per decision)
u_t   = mu_t + b_t + sigma * eps_t           -> stored as policy_sample (detached)
a_t   = clamp(u_t, -1, 1)                    -> stored as executed_action, sent to the controller
logp  = sum_dims log N(u_t; mu_t + b_t, sigma^2)
ratio = exp(logp_new(u_t | b_t) - logp_old)
```

Clipping is a deterministic map applied after the policy's sample, so the
latent policy-gradient estimator is exact.

The offset `b` is drawn once per episode from a density with no policy
parameters. In the trajectory likelihood that density is a common factor of
the old and new policies, so it cancels in the ratio. The score is then
`sum_t (u_t - mu_t - b_t) / sigma^2 * d mu_t`, conditioned on the realized
`b`.

The offset still shapes learning. It changes the returns and later states
(and so later means), and those enter the estimator like any other exogenous
noise.

The old claim that "conditioning makes the offset invisible" was based on a toy
whose advantage was the offset itself. The policy mean cannot influence that
advantage, so ~0 was the correct answer there. That claim is now marked
superseded in `_marginal_log_std`, in the collector comment, and in the old
test's docstring.

The legacy path is kept bit-for-bit for historical runs. In the measurements
below it is biased in two ways:

- it scores the clipped action;
- it scores each step against an independent widened marginal, which counts
  one persistent offset as T independent ones.

The surrogate keeps the existing per-slot factorization, executed-slot and
live-episode masks, stage-mass normalization, clip 0.20/0.28, entropy
coefficient, `log_std` projection and gradient clipping. The entropy bonus is
the latent Gaussian's (`latent/gaussian_entropy_mean`), not the entropy of the
clipped executed action. `approx_kl_mean` (also `latent/sampled_kl_estimate_mean`)
is a sampled latent-KL estimate, not an exact KL over executed actions.

## 3. Local test evidence (2026-10-03, macOS, CPU, torch 2.13)

Command:

```text
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=.:<tensorflow stub> \
  python -m unittest discover -s tests -p 'test_latent_correction_*.py'
Ran 39 tests in 2.271s
OK
```

Printed measurements:

```text
[A1] max |mean_new - mean_old| over 320 inputs x 8 slots x 5 dims = 0.000e+00; inner-tanh saturated share 0.76-0.79
[C2] resumed vs uninterrupted continuation, max |param diff| = 0.000e+00 (tolerance 0)
[B5 single-step] true 0.3543766  latent-conditional 0.3543764  historical clipped/marginal 0.3604168
[B5 multi-step] FD [1.25259, 0.20078]  latent [1.25182, 0.20114] z=[-0.13, 0.22];
                clipped-marginal z=[52.5, -151.5]; independent-marginal z=[112.5, 81.9]
```

Coverage by spec item:

- **A1–A5.**
  - A1: means bitwise equal at zero init, with a synthetic source that has a
    nonzero prior, reference scale 0.7 and saturated residuals.
  - A2: correction exactly zero; disjoint storage; reference, `log_std` and
    LoRA exact; train/eval never unfreezes the reference.
  - A3: the first optimizer step reaches only the output layer; later steps
    reach the hidden layers; the reference never changes.
  - A4: authority below the legacy bound on y and z.
  - A5: unexecuted slots get zero gradient.
  - Also: construction consumes the same global RNG as the legacy trainer, so
    both arms start rollouts from the same streams.
- **B1–B7.**
  - B1: identical executed commands under the same noise and offsets.
  - B2: latents beyond both bounds are stored and scored.
  - B3: the density matches `torch.distributions.Normal` to 1e-12.
  - B4: ratio is exactly 1 at unchanged parameters; after a change it matches
    an independent float64 NumPy ratio.
  - B5: the gradient checks above. The single-step expectation is integrated
    on a deterministic grid. The multi-step check uses CRN finite differences
    with a fixed seed and a 5-SE tolerance, and the persistent offset moves
    later states.
  - B6: the gate records the offset of the decision that sampled the chunk;
    new episodes draw fresh offsets and start ungated; fields survive
    `concatenate_collector_rounds`, padding and slot indexing.
  - B7: missing or mismatched fields fail.
- **C1–C5.**
  - C1: legacy and untagged checkpoints give the historical output.
  - C2: save, resume and continuation are exact.
  - C3: the evaluator-style loader uses a nonzero saved correction; legacy-only
    readers refuse the checkpoint.
  - C4: the optimizer holds exactly the correction and `log_std`, with no
    moments.
  - C5: the config differs only in the two declared fields;
    staged-reward/divergence/filter suites still pass.

**Two-rank DDP, CPU/gloo only** (`tests/_latent_correction_ddp_smoke.py`):

```text
{"world_size": 2, "ranks_identical": true, "reference_unchanged": true,
 "max_param_change": 0.0010, "max_error_vs_full_batch": 7.45e-09, "optimizer_steps": 1.0}
```

On macOS `--standalone` rendezvous hangs. Use
`--master-addr 127.0.0.1 --master-port <p> --nnodes 1` with `GLOO_SOCKET_IFNAME=lo0`.
This checks the DDP reducer and the partition. It is **not** an MJWarp/GPU
check.

**Full suite:** 1604 tests. The only failures are the three pre-existing ones
(`test_inference_step_preserves_persistent_controller_buffers`,
`test_predict_normalized_action_chunk_uses_action_head_path`,
`test_grpo_bootstraps_from_td3_actor_checkpoint_directory`). I confirmed two of
them on a clean stash; the third is the documented pre-existing failure in code
this change does not touch.

**Not run locally**: anything on the real `step_56072006` checkpoint, the
SmolVLA runtime or MJWarp. The checkpoint is only on the training host; its
subsequently supplied SHA-256 and remote preflight results are recorded in §5.

## 4. Remote commands

Run on the training host after `git pull`:

```bash
cd /root/repo/RL_VLA_Bootstrapping && git pull
SRC=runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006/smolvla_grpo_adapter.pt
sha256sum "$SRC"        # record it; pass it as EXPECTED_SOURCE_SHA256 below
```

**Unit tests and CPU DDP smoke on the host:**

```bash
conda run --no-capture-output -n cdpr-mjlab python3 -m unittest discover -s tests -p 'test_latent_correction_*.py'
PYTHONPATH=.:tests conda run --no-capture-output -n cdpr-mjlab torchrun --standalone --nproc_per_node 2 tests/_latent_correction_ddp_smoke.py
```

**GPU preflight, before any optimization** (checks 9.1–9.4):

```bash
EXPECTED_SOURCE_SHA256=<sha> bash scripts/preflight_cdpr_latent_correction_remote.sh
```

It writes `latent_correction_preflight.json`, which holds:

- source/config/manifest hashes, Git commit and runtime versions;
- both arms' resolved optimizers;
- mean-equivalence errors on captured real inputs, grouped by strict/failed ×
  early/middle/late decisions;
- the sampled-command check and the save/reload check.

Exit code 1 on any failure.

**One-update two-rank smoke** (check 9.5):

```bash
ARM=candidate LEGACY_INIT_CHECKPOINT="$SRC" EXPECTED_SOURCE_SHA256=<sha> \
  MAX_TRAIN_STEPS=1000000 MAX_UPDATES=1 WORLDS_PER_RANK=64 RUN_LABEL=latent_smoke_candidate \
  bash scripts/train_cdpr_latent_correction_pilot_remote.sh
```

Confirm in `train.log` and `metrics.jsonl`:

- `rl/latent_pilot_protocol.json` and `rl/step_0000000/` exist;
- `correction/grad_norm_final_mean > 0`; the hidden gradient is zero only on
  the first **optimizer step**, not necessarily on the first rollout update
  (which contains hundreds of optimizer steps);
- `correction/reference_max_abs_change == 0` and `policy/lora_max_abs_change == 0`;
- the losses are finite.

Then confirm that evaluation uses the learned correction:

```bash
CANDIDATE_CHECKPOINT=runs/latent_smoke_candidate_<ts>/rl/latest.pt EXPECTED_SOURCE_SHA256=<sha> \
  bash scripts/preflight_cdpr_latent_correction_remote.sh
```

Repeat the smoke with `ARM=control`.

**10-update diagnostic, 1M selected-action cap, both arms:**

```bash
ARM=candidate LEGACY_INIT_CHECKPOINT="$SRC" EXPECTED_SOURCE_SHA256=<sha> MAX_TRAIN_STEPS=1000000 MAX_UPDATES=10 bash scripts/train_cdpr_latent_correction_pilot_remote.sh
ARM=control   LEGACY_INIT_CHECKPOINT="$SRC" EXPECTED_SOURCE_SHA256=<sha> MAX_TRAIN_STEPS=1000000 MAX_UPDATES=10 bash scripts/train_cdpr_latent_correction_pilot_remote.sh
```

**Equal-budget 2M pilot, only if the diagnostic passes.** The update cap is
explicitly off. Checkpoints are written at 0, every 250k (`save_every_steps`),
and at the end. Limits and save intervals are checked at update boundaries:
actual saved steps can overshoot the requested budget/interval. Report each
arm's actual selected and sampled action counts, not an exact 2M claim:

```bash
ARM=candidate LEGACY_INIT_CHECKPOINT="$SRC" EXPECTED_SOURCE_SHA256=<sha> MAX_TRAIN_STEPS=2000000 MAX_UPDATES=0 bash scripts/train_cdpr_latent_correction_pilot_remote.sh
ARM=control   LEGACY_INIT_CHECKPOINT="$SRC" EXPECTED_SOURCE_SHA256=<sha> MAX_TRAIN_STEPS=2000000 MAX_UPDATES=0 bash scripts/train_cdpr_latent_correction_pilot_remote.sh
```

**Resume after an interruption.** Same arm, same total budget; this writes a
new run directory:

```bash
ARM=candidate RESUME_CHECKPOINT=runs/latent_pilot_candidate_<ts>/rl/latest.pt MAX_TRAIN_STEPS=2000000 MAX_UPDATES=0 bash scripts/train_cdpr_latent_correction_pilot_remote.sh
```

Both arms use the same pilot settings:

- `PPO_EPOCHS=1` and `LR_OVERRIDE=1e-5` (launcher defaults);
- the same seed (config default 0), split, sampler and selected-action budget.

Recorded per run:

- `launch_provenance.json`: deviations from the YAML;
- `latent_pilot_protocol.json`: resolved optimizer, trainable/frozen names,
  offset std and gate.

Logged per update: `pilot/*` counters (selected and sampled simulator actions,
episodes, optimizer steps, wall seconds), plus `pilot/source_global_step`.
`global_step` in a pilot is pilot selected actions only; the source lineage
stays in `source_global_step = 56072006`.

**Evaluation.** Development data: 512 `student_validation` scenes, excluding
the in-run panel. Four repeats each, all under the historical config. The
primary comparison is the fixed 2M final checkpoint of each arm:

```bash
CHECKPOINTS="reference=$SRC candidate=runs/latent_pilot_candidate_<ts>/rl/step_<final> control=runs/latent_pilot_control_<ts>/rl/step_<final>" \
  REPEATS=4 WORLDS=64 ROUNDS=8 RETENTION_MARGIN=0.03 SCENE_MANIFEST=runs/three_stage/scenes_8192.json \
  bash scripts/compare_cdpr_three_stage_repeats_remote.sh
```

Architecture attribution: candidate vs. the matched control, reusing the
same evaluation directories:

```bash
conda run --no-capture-output -n cdpr-mjlab python3 tools/audit/compare_put_into_repeats.py \
  --checkpoint control=$OUT/control/rep1,$OUT/control/rep2,$OUT/control/rep3,$OUT/control/rep4 \
  --checkpoint candidate=$OUT/candidate/rep1,$OUT/candidate/rep2,$OUT/candidate/rep3,$OUT/candidate/rep4 \
  --retention-margin 0.03 --scene-manifest runs/three_stage/scenes_8192.json --output $OUT/candidate_vs_control.json
```

What the comparison reports:

- pooled strict SR;
- plate/bowl and target-y Q1–Q4;
- grasp, lift and carry slip;
- the pilot rule: promote only if the strict paired 95% CI is above 0 and the
  grasp and lift CI lower bounds are above −3 pp.

`final_test` stays untouched.

## 5. GPU preflight results

**2026-10-03, checks 9.1–9.4: PASSED** (`scripts/preflight_cdpr_latent_correction_remote.sh`, 64 worlds).

- Source `step_56072006/smolvla_grpo_adapter.pt`, SHA-256
  `af8e31f654e7cafed47356260c15d197f4b15a4deb70fc08e88fda373378dbcb`.
- Captured real `(state, prior)` rows from the legacy rollout, shared by every
  actor:

  | Outcome | Early | Middle | Late |
  |---|---|---|---|
  | Strict | 152 | 97 | 12 |
  | Failed | 360 | 305 | 309 |

- Converted candidate and control: max |mean − legacy mean| = 0, and
  max |correction| = 0.
- Sampled controller commands under fixed noise and offsets are identical.
- After save and reload through the resume path, all of the above is
  unchanged.
- Integrity: the reference equals the source, the LoRA equals the source, and
  both are frozen.

**2026-10-04, check 9.5: PASSED.** One update, two ranks, 512 worlds per rank.

| | candidate (`latent_smoke_candidate_20261003_224201`) | control (`latent_smoke_control_20261004_154929`) |
|---|---|---|
| optimizer steps / LR | 554 / 1e-5 | 564 / 1e-5 |
| sampled-latent KL estimate | 0.00076 | 0.0053 |
| PPO clip fraction | 0.0045 | 0.0196 |
| grad norm mean (pre-clip; cap 1.0) | 10.39 | 5.79 |
| correction grad norm final / hidden / log_std | 10.39 / 0.023 / 0.187 | n/a |
| reference / LoRA max abs change | 0 / 0 | n/a / 0 |
| latent clip fraction x/y/z/yaw/gripper | .156/.192/.204/.221/.040 | .154/.188/.200/.218/.039 |
| offset gate occupancy; mean abs gripper offset | 0.089; 0.127 | 0.094; 0.130 |
| non-finite live episode rate | 0.0146 | 0.0117 |
| rollout pickup / placement milestone rate | 0.619 / 0.256 | 0.632 / 0.253 |
| selected / sampled actions, wall time | 87,009 / 689,634, 890 s | 85,406 / 688,838, 888 s |

Readings:

- **Hidden-layer gradient.** The hidden norm is a mean over 554 optimizer
  steps. Only the first step's is exactly zero (unit-tested); it becomes
  nonzero once the output layer moves.
- **Matched LR is not matched step size.** At the same LR the candidate moved
  about 7× less KL than the control, plausibly because only its output layer
  and `log_std` train at first. A flat candidate curve may therefore reflect
  a smaller effective step, not the architecture. Report KL alongside any
  result.
- **Clipping is common.** 15–22% of latent samples on x/y/z/yaw lie outside
  [-1, 1]. That is the share of samples the legacy likelihood mis-scored.
  This is a measurement of the data, not a task result.
- **Evaluator path.** The preflight with `CANDIDATE_CHECKPOINT=…/step_0087009`
  passed: the checkpoint loaded as the correction architecture, with a nonzero
  correction, through the evaluator's loader.

## 6. Inspecting runs before extending (2026-10-06)

The newly supplied log excerpt identifies
`latent_smoke_control_20261004_154929/rl/step_0085406`: it is the already
recorded one-update control smoke, not evidence of a completed diagnostic.
Its initial in-run strict validation is 0.3828 and final is 0.3809 (about
−0.19 percentage points); this single comparison cannot establish a gain or
regression. The separately supplied preflight passed frozen-state integrity
on 1,286 captured inputs (336/328/302 failed early/late/middle and 176/24/120
strict early/late/middle). The excerpt does not identify a trained checkpoint
or establish ten completed updates.

Use the standard-library reporter on the remote host after pulling:

```bash
python3 tools/audit/summarize_latent_pilot.py --list
python3 tools/audit/summarize_latent_pilot.py runs/latent_smoke_control_20261004_154929 --expect-updates 10
# If the diagnostic directories exist, inspect all matches explicitly:
python3 tools/audit/summarize_latent_pilot.py runs/latent_diag_candidate_* runs/latent_diag_control_* --expect-updates 10
```

The smoke command intentionally flags fewer than ten updates. The reporter
reads `launch_provenance.json`, `rl/latent_pilot_protocol*.json`,
`rl/metrics.jsonl`, and `rl/validation.jsonl`. It prints arm, source hash,
Git commit, configured limits, per-update diagnostics, cumulative interaction
counts, and validation rows. Missing/corrupt metrics and frozen-weight changes
are flagged, rather than hidden behind `-` columns. It does not select a
checkpoint or approve a longer run automatically.

New launcher runs also write `launch_result.json` with training/logging exit
codes and print the report plus exact run directory on exit. Existing runs
without an exit record are labeled by the limit evidenced in saved metrics;
this is not proof of the process's current state. A killed launcher can also
leave no exit record. A metrics row appears only after that update's validation
and checkpoint, so an empty file while an update is running is not a failure
by itself. A preflight directory has no training metrics.

`MAX_UPDATES=1` can end normally at 9% of a 1M-action progress bar.
`MAX_UPDATES=10 MAX_TRAIN_STEPS=1000000` stops at whichever limit comes first;
ten updates are not guaranteed if selected-action throughput changes. The
report distinguishes invocation-local updates from cumulative resumed counts.
The EPA warning alone does not explain an early exit; inspect exit codes,
saved counters and the log for an actual failure.

If no diagnostic exists, run both fresh arms sequentially from the original
source (do not resume the smoke and call it an identical fresh diagnostic):

```bash
SRC=runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006
SHA=af8e31f654e7cafed47356260c15d197f4b15a4deb70fc08e88fda373378dbcb
for arm in candidate control; do
  ARM="$arm" LEGACY_INIT_CHECKPOINT="$SRC" EXPECTED_SOURCE_SHA256="$SHA" \
    MAX_TRAIN_STEPS=1000000 MAX_UPDATES=10 RUN_LABEL="latent_diag_$arm" \
    bash scripts/train_cdpr_latent_correction_pilot_remote.sh || break
done
```

Before the matched 2M pilot in §4, inspect both reports for finite optimization
metrics, exactly unchanged frozen reference/LoRA, active correction gradients,
KL/clip and correction trends, non-finite live-episode rates, and pickup,
placement and strict-validation behavior. Non-finite **simulation episode
rates** were already around 1–1.5% in the smoke; they are not the same as NaN
losses, and their trend needs review rather than a made-up zero threshold.
Keep the matched LR and objective unchanged for this screen. Missing diagnostic
evidence is a reason to collect it, not evidence that the actor needs repair.
After the 2M pilot, perform the repeated matched development evaluation and
retention checks in §4 before choosing a longer training direction.

## 7. Training and evaluation results

**2026-10-08: control diagnostic completed**, ten updates, 852,587 selected
actions and 6,888,452 sampled actions; training/log exits both zero. Initial
strict validation 394/1024 (38.48%), final 415/1024 (40.53%). Loss/gradient/std
metrics are finite and LoRA remains unchanged. KL spikes on updates 3/4 and
NaN pad-force averages on updates 1/8 remain caveats; the latter now produce
reporter warnings rather than a false impression of a failed training process.
Simulator behavior is unchanged. Full evidence hashes, subgroup tradeoffs and
limitations are in the [control diagnostic analysis](CDPR_LATENT_CONTROL_DIAGNOSTIC_20261008.md).
**2026-10-08: candidate diagnostic completed**, from the user's pasted reporter
output (raw candidate JSONL/protocol files were not attached):
`latent_diag_candidate_20261006_191536`, Git
`c4811a77414bb65bd58b6c871bacbf9e81bb3689`, legacy conversion from source SHA-256
`af8e31f654e7cafed47356260c15d197f4b15a4deb70fc08e88fda373378dbcb`.
Resolved architecture `frozen_reference_logit_correction_v1`, LR 1e-5.
Ten updates end at 827,913 selected actions, approximately 6.63673M sampled
actions, 20,480 episodes, 5,434 optimizer steps and 9,413.89 pilot wall seconds.
The report says `completed: update cap` and has no CHECK lines.

| Diagnostic | Candidate | Control |
|---|---:|---:|
| Initial strict validation | 37.89% | 38.48% |
| Final strict validation | 37.99% | 40.53% |
| Median sampled latent KL | 0.000863 | 0.003975 |
| PPO clip fraction range | 0.263–0.471% | 1.70–2.69% |
| Non-finite live-episode rate, mean over updates | ~2.178% | 1.587% |
| Rows with NaN pad-force averages | 3, 4, 6, 10 | 1, 8 |

Candidate initial strict validation is **37.8906% (388/1024)**; its final is
**37.9883% (389/1024)**, a one-episode difference, **+0.098 pp**. The supplied
intermediate rows are 39.8438% at step 504,088 and 38.9648% at 750,484. Treat
the final candidate curve as flat, not as evidence of a gain. The pasted
report contains four validation rows; no raw candidate file was available
to investigate the absence of a row near its first 250k crossing.

Reference and LoRA max change are exactly zero throughout. Hidden-layer
gradient means rise from 0.0243 to 0.1126, and absolute correction magnitude
becomes nonzero (last reported z 0.07399, gripper 0.02409). The zeros in the
first correction row are expected: component statistics describe the rollout
collected before that update's optimization. These are not a measurement of
the final saved checkpoint's correction on a fixed input bank.

The candidate's typical sampled KL is about 4.6 times smaller than control's;
equal LR did not produce equal effective policy movement. Do not infer that
the branch is disconnected, or automatically raise LR to match the control.
Candidate non-finite episode rates climb through much of updates 1–9 (peak
2.686%) and fall to 1.758% on update 10. This is a simulator-health caveat,
not evidence of a NaN loss; the last lower value does not prove the issue is
resolved. No raw force samples or subgroup validation were supplied for it.

## 8. Decision after both diagnostics (2026-10-08)

Proceed to the **bounded, matched 2M selected-action pilot per arm** already
specified in §4, retaining the conservative LR and objective. Both diagnostics
establish functioning optimization and frozen-weight integrity in the reported
metrics; neither establishes performance superiority or fully healthy contact
telemetry. Retain simulator failures and force warnings in the comparison.
This is an experiment under known caveats, not checkpoint promotion or an
automatic 10M extension. Do not silently repair contact values by zeroing them.

Start both arms fresh from the same verified source so this follows the
predeclared independent pilot protocol and does not splice differing
diagnostic histories into the primary comparison:

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only
unset RUN_NAME RESUME_CHECKPOINT WARMSTART_CHECKPOINT
SRC=runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006
SHA=af8e31f654e7cafed47356260c15d197f4b15a4deb70fc08e88fda373378dbcb
for arm in candidate control; do
  ARM="$arm" LEGACY_INIT_CHECKPOINT="$SRC" EXPECTED_SOURCE_SHA256="$SHA" \
    MAX_TRAIN_STEPS=2000000 MAX_UPDATES=0 PPO_EPOCHS=1 LR_OVERRIDE=1e-5 \
    WORLDS_PER_RANK=512 SMOLVLA_MICROBATCH_SIZE=256 PRIOR_NOISE_SCALE=1 \
    CONFIG=configs/examples/cdpr_smolvla_three_stage_put_into_latent_correction.yaml \
    SCENES=runs/three_stage/scenes_8192.json CUDA_VISIBLE_DEVICES=0,1 \
    RUN_PREFLIGHT=1 DRY_RUN=0 RUN_LABEL="latent_2m_$arm" \
    bash scripts/train_cdpr_latent_correction_pilot_remote.sh || break
done
```

`MAX_UPDATES=0` disables the update cap; the 2M-action cap remains active.
Both may overshoot at update boundaries; report actual action counts.
Use the final budget checkpoint for each primary comparison, followed by the
four-repeat matched development evaluation against the source and against
each other in §4. Keep `final_test` untouched. Inspect reports during the run
for any growing numerical failure trend; stop and investigate an actual
non-finite optimizer metric or frozen-weight violation before continuing.

This update records supplied evidence and commands only; no model, optimizer,
reward, simulation or training code was changed, and no remote run was started.

## 9. Both 2M pilots completed (2026-10-09)

The four supplied metrics/validation JSONL files establish completed budgets,
finite recorded optimization metrics and unchanged frozen weights. Candidate
ends at step **2,007,079** after 25 updates; control at **2,052,558** after 24.
Initial → final strict validation is **39.36% → 38.18%** for candidate and
**38.77% → 40.92%** for control. This favors control descriptively, but the
in-run panel does not establish a reliable gain over the source.

Candidate non-finite live training episodes increase from 1.96% in its first
five updates to 3.27% in its last five (2.90% overall, versus control 1.57%).
Final validation non-finite rates are 3.61% and 1.17%, respectively. Contact
health is therefore more concerning than at the diagnostic stage. Do not
interpret finite losses or small sampled KL as proof of healthy simulation.

**Next: repeated matched evaluation, no further training yet.** Use the
predeclared final-budget checkpoints, not retrospectively selected peaks.
[Full review, evidence hashes and exact remote commands](CDPR_LATENT_2M_REVIEW_20261009.md)
cover all three checkpoints and candidate-versus-control attribution. No new
checkpoint is promoted and no training/simulator code was changed.
