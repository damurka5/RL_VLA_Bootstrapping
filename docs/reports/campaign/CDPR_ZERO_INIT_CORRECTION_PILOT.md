# Zero-init correction and latent likelihood: experiment note

Implements `docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_IMPLEMENTATION.md`.
This note keeps four things separate: implementation status, local test
evidence, GPU preflight, and training/evaluation results. Only the first two
exist as of 2026-10-03.

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
SmolVLA runtime or MJWarp. Its SHA-256 is **not yet recorded**: the checkpoint
is only on the training host.

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

- `latent_pilot_protocol.json` and `rl/step_0000000/` exist;
- `correction/grad_norm_final_mean > 0` and `correction/grad_norm_hidden_mean == 0` on update 1;
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
and at the end:

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

Not run yet.

## 6. Training and evaluation results

None. No claim about success rates follows from the tests above.
