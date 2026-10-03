# Implementation instruction: zero-initialized correction over step_56072006

Prepared 2026-10-03 against repository commit `63bdac8`.

## Objective and scope

Implement an opt-in GRPO policy that starts with exactly the action means of
`step_56072006`, freezes that controller, and learns an additional correction
in the final action's pre-tanh coordinates. Repair the action likelihood by
retaining unclipped Gaussian samples and conditioning on the actual episode
exploration offset. Keep the existing staged sparse reward, task, observations,
scene distribution, controller, and evaluation protocol unchanged.

This is an implementation and bounded-pilot handoff. Implement and run local
checks, provide working remote commands, and report any unavailable GPU checks.
Do not automatically launch a long training run. Do not claim a success-rate
gain from unit tests, action equivalence, or a single noisy evaluation.

Use the repository's existing conventions and applicable instructions. Preserve
unrelated working-tree changes. The repository root is the directory containing
`CDPR_CONSOLIDATED_PROGRESS_REPORT.md`, not its enclosing workspace directory.
Paths below are repository-relative.

Reference checkpoint on the training host:

```text
runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006/smolvla_grpo_adapter.pt
```

Resolve this path explicitly and record its SHA-256. Do not substitute the latest
checkpoint or either 66M continuation. If the checkpoint is unavailable locally,
implement synthetic-checkpoint tests and mark real-checkpoint validation pending.

## 1. Motivation and the experiment's hypothesis

The existing actor computes:

```text
x         = concat(state, flatten(prior))
r_ref     = tanh(reference_net(x))
logit_ref = prior + reference_scale * reshape(r_ref)
mean_ref  = tanh(logit_ref)
```

The bounded residual spends most of its z range cancelling a nearly constant
positive prior. At representative measured priors, deterministic mean z is
approximately restricted to [-0.013, 0.963], and y to [-0.204, 0.946]. These are
illustrative, observation-dependent mean bounds, not bounds on noisy actions.

Hypothesis: an additional trainable path without the inner tanh restriction can
learn useful corrections while preserving the current policy at initialization.
This experiment does not assume that action authority is the only bottleneck.

## 2. Policy architecture

Introduce a separately identifiable architecture, provisionally
`frozen_reference_logit_correction_v1`. Keep legacy checkpoints loadable through
their legacy architecture. Do not silently reinterpret legacy state dictionaries.

The new actor must compute:

```text
x          = concat(state, flatten(prior))
logit_ref  = prior + reference_scale * tanh(reference_net(x))
correction = correction_net(x)             # linear output; NO output tanh
mean       = tanh(logit_ref + correction)
```

Reshape outputs to `[batch, chunk_size, action_dim]` as appropriate. Derive all
dimensions and the reference scale from checkpoint metadata, checking them
against runtime observations. The expected current dimensions are 518 state
values, 40 prior values, two 1024-wide hidden layers, and 40 output values.
Do not hard-code them when metadata is available.

Implementation requirements:

1. Copy the reference residual network exactly from the source checkpoint.
   Freeze all its parameters and keep its inference behavior fixed.
2. Construct `correction_net` by deep-copying the reference MLP's architecture
   and hidden-layer weights. Zero only its final linear layer's weight and bias.
   Make all correction-network parameters trainable. Do not zero hidden layers.
3. Do not share parameter storage between the frozen and trainable networks.
4. Use the reference's directly computed pre-tanh value. Do not reconstruct it
   with `atanh(mean_ref)`, which creates avoidable precision/boundary problems.
5. Keep correction scale at 1.0 for this pilot. Do not bound the correction,
   subtract a hand-measured prior bias, change the inherited reference scale,
   or add a correction penalty.
6. Preserve proprioception, vision pooling/projection, camera ordering, prior
   conditioning, and chunk layout. Do not introduce task embeddings, privileged
   state, a new encoder, or additional cameras in this experiment.
7. Restore the checkpoint's VLA and action-expert LoRA exactly and keep them
   frozen. `train_vla_lora` may still be necessary to attach/load the adapter;
   it must not enable LoRA optimizer updates. Vision LoRA stays disabled.
8. Copy `log_std` and its existing bounds; keep it trainable with the existing
   projection-after-update behavior. Include it in the fresh optimizer.

At initialization, correction must be exactly zero for every finite input.
Thus the old and new mean actions are equal. Copying hidden weights also avoids
unnecessary initialization RNG consumption. Test that construction/conversion
does not perturb the rollout RNG streams, or explicitly preserve those streams.

Only the final correction layer normally receives an actor gradient on the
first update: its zero weights initially block gradients to its hidden layers.
That is expected. Hidden-layer gradients should become available after the
output weights move. The reference must never receive gradients.

This freezes the learned reference parameters, not its output. It must still
recompute reference actions from each current observation and current prior.

## 3. Valid action likelihood while preserving exploration behavior

Choose **unclipped latent Gaussian actions** for this experiment. Do not also
switch to a squashed Gaussian; that would change the sampling behavior and add
another experimental variable.

Define, for every actually sampled action slot:

```text
mu_t       = new_actor(state_t, prior_t)
b_t        = actual_episode_offset * existing_gate_t
sigma      = exp(clamped_log_std)
u_t        = mu_t + b_t + sigma * epsilon_t
a_t        = clamp(u_t, -1, 1)
old_logp_t = sum_dims Normal(mu_t + b_t, sigma).log_prob(u_t)
```

The controller receives only `a_t`. The policy likelihood uses `u_t`, including
when `u_t` lies outside [-1, 1]. Clipping is a deterministic environment action
mapping; this is a valid latent-action policy-gradient formulation.

Preserve the existing episode-offset sampling, persistence, and gate exactly.
Currently the configured offset is gripper-only, standard deviation 0.15, and
the three-stage collector activates it after the held-lift milestone. Resolve
the actual configuration and assert its value in provenance rather than relying
only on this document.

### Required records and update behavior

Store separately:

- `executed_action`: clipped action sent to the controller;
- `policy_sample`: unclipped `u_t`, detached before storing;
- `behavior_mean_offset`: actual effective `b_t`, not only its standard deviation;
- `old_log_prob`: conditional latent Gaussian likelihood;
- the existing observation, prior, action-slot index, stage, mask, and grouping
  fields required to reconstruct the sampled decision.

At PPO update time:

```text
new_mu   = actor(recorded_state, recorded_prior)[recorded_slot]
new_logp = sum_dims Normal(new_mu + recorded_offset, new_sigma)
                       .log_prob(recorded_policy_sample)
ratio    = exp(new_logp - recorded_old_logp)
```

Never regenerate the offset, infer the latent from the clipped action, or score
`executed_action` as a Gaussian draw. Keep stored latent samples detached: the
PPO score-function update must not backpropagate through the old sampling step.

The persistent offset is an exogenous episode-level random variable. Condition
on its realized value in both old and new action likelihoods. Its parameter-free
density cancels in the policy ratio. Do not use `_marginal_log_std` for this new
mode: treating each step as an independent widened Gaussian does not generally
represent the process conditioned on an offset-dependent state/history.

This explicitly supersedes the existing comment claiming that conditional
likelihood necessarily makes offset exploration invisible to learning. A toy
test whose advantage is only the exogenous offset is not a valid proof of the
trajectory policy-gradient estimator. Test against an actual return-dependent
stochastic process instead.

Keep the current factorization and action-slot PPO reduction; do not silently
change to whole-trajectory or whole-chunk likelihood ratios. Record the policy
and offset associated with the decision that sampled the chunk, not a later
observation or gate. Apply the existing executed-slot and live-episode masks.

Preserve the existing latent Gaussian entropy bonus and label it as such. It is
not entropy of the clipped executed-action distribution. Preserve the current
`log_std` limits, entropy coefficient, and gradient clipping.

Add a versioned mode, provisionally `latent_gaussian_conditional_offset_v1`.
Every rollout and update in that mode must require the new record fields and
fail clearly if they are missing. Legacy stored clipped-action records cannot
be migrated into exact latent samples. Do not reuse them as on-policy data.

## 4. Keep the task and sparse objective unchanged

Start from `configs/examples/cdpr_smolvla_three_stage_put_into.yaml`. Create a
dedicated opt-in config and launcher; preserve the historical config.

The existing objective is **staged sparse credit**, not a newly introduced
terminal-only reward. Preserve:

- the three ordered predicate-defined stages and their credit assignment;
- stage loss weights `[1.0, 1.0, 1.0]`;
- full-task bonus `1.0`;
- achieved-milestone negative bonus scale `0.0`;
- strict success via the shared `FullTaskOutcome`;
- reward geometry, termination/retry behavior, and divergence exclusion;
- group size 8, existing group filters, dynamic sampling, and refill limits;
- PPO clip bounds 0.20/0.28 and existing loss normalization;
- existing controller scales, workspace limits, yaw contract, camera settings,
  prior noise scale 1, eight-slot output, and four executed slots;
- 128-decision full-task horizon, training `collection` split, and current
  full-task validation reset/instruction protocol.

Do not add dense shaping, action penalties, a release interlock, a ceiling
intervention, a curriculum, recovery resets, far-side oversampling, SFT, or an
off-policy critic. Do not enable the previous reference-anchor loss: the frozen
branch is architectural initialization, not a behavior-preservation penalty.

## 5. Checkpoints, initialization, and resume

Implement an explicit conversion/initialization mode from the named legacy
checkpoint. It is a weights-only initialization with a fresh optimizer over
the correction parameters and `log_std`. Do not load the old actor's Adam
moments into the new branch. Freeze the reference before constructing optimizer
parameter groups or DDP wrapping.

Store architecture and likelihood versions, the full reference residual state,
correction state, `log_std`, required LoRA state, optimizer state, all applicable
runtime metadata, and source checkpoint SHA-256 in new checkpoints. Do not rely
on the original adapter remaining at its old filesystem path for inference.
Normal external dependencies such as the base SmolVLA model remain explicit.

Track pilot steps from zero and source lineage separately:

```text
source_global_step = 56072006
pilot_global_step  = newly collected selected-action count
```

Also report all sampled simulator actions, episodes, optimizer steps, and wall
time. Do not present selected-action counts as total simulator interaction cost.

Resume of a new checkpoint must restore both branches, the frozen/trainable
partition, `log_std`, pilot counters, optimizer state, and protocol metadata.
Do not re-zero a learned branch on resume. Distinguish conversion from resume
in the CLI and reject ambiguous combinations.

Route training, standalone evaluation, validation, trace generation, and relevant
checkpoint readers through a version-aware actor factory. A new checkpoint must
never silently evaluate only its reference branch. Update consumers needed for
this experiment; unsupported legacy-only utilities must fail with an explicit
message rather than load partial weights. Preserve legacy loading behavior.

## 6. Repository integration map

Inspect these files and their callers before editing:

| File | Required work |
|---|---|
| `rl_vla_bootstrapping/policy/octo_finetune_cdpr.py` | Existing `ResidualChunkActor`; preserve legacy semantics and expose/share direct reference-logit computation if appropriate. |
| `rl_vla_bootstrapping/policy/smolvla_grpo_finetune_cdpr.py` | Policy selection, new actor, sampling, conditional latent likelihood, optimizer partition, conversion/resume/save. Audit every sampling and log-probability call site. |
| `rl_vla_bootstrapping/policy/mjwarp_rank_local_collector.py` | Separate latent/executed actions, store effective offset, propagate fields and masks into updates; keep controller commands unchanged. |
| `rl_vla_bootstrapping/policy/smolvla_grpo_mjwarp_cdpr.py` | Runtime construction, flags, metadata, DDP synchronization, initialization and resume. |
| `rl_vla_bootstrapping/cli/validate_cdpr_smolvla_policy.py` | Version-aware checkpoint inference, including the learned correction. |
| `tools/audit/evaluate_cdpr_full_put_into.py` | Follow its actual loader path; preserve strict evaluation and add architecture provenance. |
| `tools/audit/episode_kinematic_trace.py` and `summarize_kinematic_traces.py` | Distinguish reference logits, correction logits, mean actions, and executed noisy actions. |
| `scripts/train_cdpr_three_stage_sparse_grpo_remote.sh` | Reuse validated launch/preflight logic through an opt-in launcher/config with explicit conversion vs resume. |
| `tests/test_grpo_episode_offset_exploration.py` | Audit obsolete estimator assumptions; add behavior and gradient-validity coverage. |
| `tests/test_grpo_three_stage_sparse_credit.py` | Retain the unchanged task/credit contract and regression coverage. |

This is a map, not a requirement to change every file. Prefer a small shared
actor/distribution implementation to divergent copies. Also inspect record
packing, concatenation, distributed padding, VLA capture, and secondary sampler
paths. Paths not supported in the new mode must fail before training starts.

Existing trace reconstruction `atanh(final) - prior` must not be described as
the old bounded residual after adding the correction. Store direct components
and label them accurately; preserve legacy trace interpretation.

## 7. Required tests and acceptance criteria

### A. Policy initialization and learnability

1. Convert a synthetic legacy checkpoint with nonzero prior, nonunit reference
   scale, and saturated residuals. New and old means must match on varied batches
   and all slots. Use strict dtype-appropriate tolerances and report max error.
2. Verify zero correction, disjoint parameter storage, and exact reference and
   LoRA preservation. Verify train/eval mode changes do not unfreeze the reference.
3. After optimizer steps, reference state is unchanged; correction changes.
   Check the expected first-step gradient behavior described above.
4. Show a controlled negative correction can produce mean z and y below the
   legacy actor's attainable bounds for the same input. This is a mathematical
   authority test, not a claim of learned task improvement.
5. Check unexecuted chunk slots receive no direct action-loss supervision.

### B. Likelihood correctness

1. With identical externally supplied Gaussian noise and offsets, old and new
   sampling produce identical clipped controller actions when means/std match.
2. Include latent samples beyond both action boundaries; verify stored latents
   differ from executed actions and likelihoods use the latents.
3. Verify log probability against `torch.distributions.Normal` at known values,
   with zero and nonzero offsets and per-slot/per-dimension standard deviations.
4. Unchanged parameters yield ratios of one, including clipped samples. Also
   test changed parameters against independent analytical likelihood ratios;
   ratio-one alone is insufficient.
5. Use a small stochastic control problem with actual action-dependent returns
   to compare estimated score gradients against finite differences or an
   analytically integrated expected return. Include clipping and a persistent
   offset; add a short multi-step case where the offset affects later state.
   Use deterministic integration or sufficient seeded samples with justified
   statistical tolerances, avoiding flaky thresholds.
6. Gate transitions preserve the actual recorded effective offset. Resets do
   not leak a previous episode's offset. Latents/offsets survive record packing,
   action-slot indexing, masks, and distributed padding unchanged.
7. Missing new-mode fields and mismatched likelihood versions fail clearly.

### C. Checkpoint and end-to-end regression

1. Legacy checkpoints still load with their historical outputs.
2. New checkpoint save/load and resume preserve learned corrections and action
   outputs; deterministic fixed-record optimizer continuation matches an
   uninterrupted continuation within the declared numerical tolerance.
3. Evaluator inference includes a deliberately nonzero saved correction.
4. New optimizer contains exactly the intended trainable parameters.
5. Existing staged reward, divergence handling, group filtering, and action
   masking tests pass. Add a config-difference assertion allowing only declared
   architecture/likelihood and pilot-protocol fields.
6. Run an appropriate two-rank smoke test when the required environment is
   available. Report tests not run and why; do not substitute CPU results for
   MJWarp/GPU validation.

## 8. Instrumentation

Log the following with clear units and definitions:

- per-axis reference logit, correction logit, combined logit, and mean action
  summaries (mean and relevant quantiles);
- `abs(correction)` and combined output-tanh slope `1 - mean**2`;
- inherited inner-tanh saturation, labeled as frozen-reference diagnostics;
- latent-to-executed clipping fraction by action dimension;
- realized offset, gate occupancy, `log_std`, and latent Gaussian entropy;
- sampled PPO KL estimate, clip fraction, gradient norms, optimizer LR, and
  trainable parameter count; do not label the sample KL an exact executed KL;
- strict success, grasp, held lift, commanded premature release/carry slip,
  and existing failure categories;
- informative groups and selected stage records by target-y quartile and
  destination, without changing the sampler or using those diagnostics as input.

If gradient contribution by stage/quartile can be measured cheaply, report it
separately from record counts; counts alone are not gradient mass. Avoid
per-minibatch full-network diagnostic backward passes that dominate runtime.

## 9. GPU preflight and bounded experiment

Deliver executable commands using the actual implemented flags; do not present
the provisional names in this document as already existing CLI options.

### Preflight, before optimization

1. Load the exact checkpoint and config; record source hash, Git revision,
   config/manifest hashes, resolved optimizer settings, and runtime versions.
2. On saved real inputs spanning successful and failed phases, verify reference
   vs zero-correction mean equivalence and frozen-state integrity. Share the
   same already-computed prior tensors; separate stochastic prior calls do not
   test actor equivalence.
3. With fixed latent noise and offsets, verify sampled command equivalence.
4. Save/reload the initialized candidate and repeat equivalence checks.
5. Run a small full-task rollout and one update, confirming new record fields,
   finite losses, nonzero correction gradients, frozen reference/LoRA, and
   active correction use in evaluation. Check two-rank execution before scaling.

Do not demand identical full-episode success verdicts from independent MJWarp
runs as proof of equivalence: the report documents substantial repeat noise.
Use fixed-input equality for exactness and repeated evaluation for performance.

### Experimental arms

Retain the original checkpoint as the evaluation reference. Prepare two matched
training arms:

- **Control:** legacy actor architecture, train its original residual, but use
  the repaired latent conditional likelihood and a fresh optimizer.
- **Candidate:** frozen-reference correction architecture, train only the new
  branch and `log_std`, using that same repaired likelihood and fresh optimizer.

These arms separate the architecture's contribution from the likelihood repair.
Do not call the historical continuation an equivalent likelihood-only control.

Use the same starting action distribution, sparse objective, seeds, training
split, sampler, optimizer settings, and interaction budget in both arms. For
the bounded pilot, explicitly set `PPO_EPOCHS=1` and optimizer LR `1e-5` in both
arms. These are conservative pilot choices, not claimed optima. Record them as
deviations from the historical YAML, and verify the resolved LR after loading.
Use existing Adam epsilon, entropy coefficient, std bounds, and gradient clipping.

First run a 10-update diagnostic with a 1M selected-action cap. This is a
correctness/stability screen, not sufficient evidence for a small SR gain.
If it passes, provide commands for an equal-budget 2M selected-action pilot
per arm, with the update cap explicitly disabled. Do not let the launcher's
default 10-update cap silently truncate that pilot. Report actual simulator
steps and wall time as well as selected actions.

Save the zero-update candidate, fixed-budget checkpoints, and final checkpoint.
Use matched schedules for both arms. Avoid an open-ended learning-rate sweep or
automatic 10M extension. Failed numerical/integrity gates require repair before
continuation; a flat short learning curve is inconclusive, not proof of benefit.

### Evaluation and promotion

Preserve unassisted empty-gripper starts, final instruction from step zero,
strict geometry, 128 decisions, and the usual deterministic-mean deployment
mode with prior noise scale 1. No correction-free evaluation of the new actor.

Use the existing repeated matched protocol: 512 development scenes excluding
the fixed in-run 128-scene panel, four repeats per candidate/reference, with
matched run conditions. Reuse baseline artifacts only after checking exact
protocol compatibility. The analyzed 512-scene set is development data; keep
`final_test` untouched until a candidate is selected.

Report paired scene-level mean differences and scene-clustered uncertainty,
following the existing repeat-comparison tool. Episodes from one scene are not
independent units. Report pooled strict SR plus Q1-Q4, plate/bowl, grasp/lift,
and premature-release results. Correction magnitude alone is not improvement.

Predeclare one primary checkpoint comparison per arm (e.g. the fixed 2M final
checkpoint) or account for multiple selection/comparison when screening more.
For promotion, require a positive strict-success effect against the original
checkpoint with the paired 95% interval above zero. Use an explicit preservation
margin of 3 percentage points for grasp and held-lift rates: require the lower
confidence bound on each difference to exceed -3 pp; otherwise label retention
inconclusive/failed and do not automatically promote. This margin is a pilot
decision rule, not a measured property of the task. Report all subgroup tradeoffs.

To attribute a gain to the architecture, also compare candidate against the
matched likelihood-repair control. A gain only against the original checkpoint
supports the combined intervention, not the correction branch alone.

## 10. Deliverables and completion report

Deliver:

1. Opt-in actor/likelihood implementation with checkpoint migration and resume.
2. Updated record plumbing, evaluator support, and truthful trace telemetry.
3. Dedicated config, launcher/preflight, and exact commands for conversion,
   smoke testing, both pilot arms, resume, evaluation, and matched comparison.
4. Tests with actual pass/fail/skip output and a concise explanation of the
   likelihood estimator, including handling of persistent offsets.
5. An experiment note containing the resolved protocol, hashes, initialization
   equivalence errors, resource costs, and results if actually run.

Keep implementation status, unit-test evidence, GPU preflight, training results,
and promotion verdict separate. Append to the consolidated report only what
has actually been verified. Never overwrite the reference checkpoint.

If this experiment remains flat after a valid bounded comparison, report that
result. Do not silently add dense rewards, recovery resets, new visual features,
or off-policy learning; those are separate future experiments.
