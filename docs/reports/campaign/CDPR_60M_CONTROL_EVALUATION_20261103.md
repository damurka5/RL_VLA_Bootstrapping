# Evaluate the 60M CDPR control against promoted 40M

Prepared **2026-11-03**, Europe/Moscow; **completed results received November 4**.
The endpoint passes promotion at **1,416/2,048 = 69.14% strict**, versus
40M **1,245/2,048 = 60.79%**. See the
[matched result, selected checkpoint and next command](CDPR_60M_CONTROL_REVIEW_20261104.md).
The instructions below retain the evaluation protocol for reproduction.

The [TensorBoard review](CDPR_60M_CONTROL_TENSORBOARD_REVIEW_20261028.md)
identifies endpoint **`step_60025885`**, in-run strict 723/1024 = 70.61%, and
panel peak **`step_59268127`**, 728/1024 = 71.09%. Use the exact integer
event step, not the float32 scalar's rounded `60025884`. The new run directory,
originally unknown, is now identified as `latent_control_to60m_20261019_224018`.
The existing launcher resolves a bare `step_<N>` under
`runs/*/rl/` only if exactly one matching directory exists, then checks its
adapter file. If resolution fails, pass the actual checkpoint path; do not
substitute a nearby step or assume the file exists.

Baseline: `runs/latent_control_to40m_20261012_230538/rl/step_40056193`.
Promoted adapter SHA-256:
`63aff785cc0202ecea2f9c1e713f58648f14a0695a593975fa3022b4f2505e91`.
Its previous 61.43% separate score is context, not the baseline rate to
subtract from the new evaluation; rerun it contemporaneously.

## Copy/paste on the remote server

This uses the existing comparison script, without depending on the new wrapper
being synced to the server. It evaluates two checkpoints, **4,096 episodes**
total: 512 distinct off-panel `student_validation` scenes × four repeats
per checkpoint. The 40M/60M policies get matching scenes and repeat-major job
scheduling across the two GPUs. Residual actions are deterministic; the prior
retains noise scale 1, matching the previous comparison. Scene exclusion
removes the 128-scene training panel; `final_test` stays unused.

```bash
cd /root/repo/RL_VLA_Bootstrapping
BASE=runs/latent_control_to40m_20261012_230538/rl/step_40056193
OUT="runs/latent_control_60m_comparison_$(date +%Y%m%d_%H%M%S)"

env -u SCENE_LIST -u SCENE_CLASS -u CONTROLLER_Z_MAX -u STOCHASTIC_SEED \
  RLVLA_EMBODIMENT=cdpr \
  CHECKPOINTS="control40m=$BASE final60m=step_60025885" \
  OUT_ROOT="$OUT" REPEATS=4 GPUS="0 1" WORLDS=64 ROUNDS=8 \
  CONFIG=configs/examples/cdpr_smolvla_three_stage_put_into.yaml \
  SCENES=runs/three_stage/scenes_8192.json SPLIT=student_validation \
  SCENE_MANIFEST=runs/three_stage/scenes_8192.json RETENTION_MARGIN=0.03 \
  DECISIONS=128 SETTLE_DECISIONS=0 PRIOR_NOISE_SCALE=1 EXCLUDE_VALIDATION_PANEL=1 \
  PANEL_VALIDATION_SEED=2000000 PANEL_GROUPS_PER_RANK=64 PANEL_RANKS=2 \
  bash scripts/compare_cdpr_three_stage_repeats_remote.sh
```

Keep the printed `out=...` path. Resume interruptions with the **same** `OUT`
and the same settings/checkpoint bytes; finished `evaluation.json` files are
reused. Use a fresh output directory after any protocol or checkpoint change.
Return `$OUT/comparison.json`; per-repeat `evaluation.json`, `checkpoint.sha256`
and `git_commit.txt` remain alongside it for provenance.

After syncing the new local wrapper, the same protocol is one command with
an additional check that the baseline bytes match the promoted adapter:

```bash
bash scripts/evaluate_cdpr_60m_control_remote.sh
```

Set `ENDPOINT_CHECKPOINT=/actual/run/rl/step_60025885` if discovery is ambiguous.
Set `OUT_ROOT` to the prior output directory to resume.

## Optional panel-peak comparison

If `step_59268127/smolvla_grpo_adapter.pt` exists, include it as a secondary
candidate. With the new wrapper:

```bash
PEAK_CHECKPOINT=step_59268127 bash scripts/evaluate_cdpr_60m_control_remote.sh
```

This runs **6,144 episodes** across three checkpoints and writes
`comparison.json` plus `peak_vs_final.json`. The latter reuses the same
episodes for a direct peak-versus-endpoint comparison. Both candidates'
strict comparisons against 40M receive Holm correction in the main report.
The peak was selected from 81 panel readings and exceeds the endpoint by only
five panel successes; that is no basis to prefer it without matched evidence.
If adding it after the two-checkpoint run, use a new output directory so the
repeat/device schedule remains balanced for the three-checkpoint design.

## Decision after evaluation

Use the scene-clustered paired strict interval and Holm p-value, not the panel
rate difference. Require a demonstrated strict gain and the declared 3 pp
grasp/lift retention criterion. Review non-finite rates and their paired
intervals; the automated retention rule does not gate simulator health. The
comparison also reports plate/bowl and target-y breakdowns. These are
development model-selection results, not independent final-test evidence.

The returned comparison now supports replacing 40M with the 60M endpoint.
The peak was not evaluated. For a future comparison in which both 60M
and the peak pass and the peak has no demonstrated advantage over the
endpoint, prefer the endpoint. Do not launch another long continuation solely
because the in-run panel reached 70.61%.

The new wrapper passed shell syntax checks and mocked launch checks for the
two-checkpoint protocol, optional peak comparison, clearing diagnostic
interventions, explicit CDPR geometry and rejection of the wrong baseline hash.
These checks do not validate remote checkpoint existence or simulation results.
