# Matched latent-likelihood 2M pilots: training review

Reviewed 2026-10-09 from four user-attached JSONL files and the control's
console report. These artifacts suffice for a training diagnosis. They do
not contain the independent repeated scene-level evaluation needed to select
a checkpoint for longer training.

## Evidence

| File | SHA-256 |
|---|---|
| `latent_2m_candidate_metrics.jsonl` | `603ff3b248ffa89cde37eddc41444c83f5f967fc818e4b1b569eb6e6502d6d81` |
| `latent_2m_candidate_validation.jsonl` | `3b7827bff34b83796e394e1eb5fdd3fc2263b3197f1c1becdd357359eeb5beaf` |
| `latent_2m_control_metrics.jsonl` | `a85b8340d647158b87255d941ff65296680542348d5a610f2e856605e042fb05` |
| `latent_2m_control_validation.jsonl` | `5819f887fd97f0de94bbdcf08674e7677dcd29564272fab6a1267ef7ffa17319` |

Control run: `latent_2m_control_20261009_033534`, Git
`b257db96f031286778940e897fc95c5d3c1af5db`, source SHA-256
`af8e31f654e7cafed47356260c15d197f4b15a4deb70fc08e88fda373378dbcb`,
legacy conversion, architecture `bounded_residual_v0`, completed action cap.
Candidate run-directory name and full launch/protocol files were not supplied.
Both metrics streams identify source global step 56,072,006, LR 1e-5 and
512 worlds per rank; this alone does not establish exact full-protocol identity.

## Completion, cost and integrity

| Metric | Candidate | Control |
|---|---:|---:|
| Final selected actions | 2,007,079 | 2,052,558 |
| Updates | 25 | 24 |
| Sampled actions | 16,091,547 | 16,320,292 |
| Episodes | 51,200 | 49,152 |
| Optimizer steps | 13,264 | 13,349 |
| Training elapsed hours | 7.47 | 7.27 |
| Pilot counter hours | 6.43 | 6.23 |
| Validation rows | 9 | 9 |

The budget is checked at update boundaries: these are approximately matched
2M budgets, not identical interaction counts. Neither records a zero-signal
stop. Both have consecutive update indices, consistent global/selected
counters and final validation at their final training step. The expected
final checkpoint directories are `step_2007079` and `step_2052558`; their
existence/content must be verified on the remote host by the evaluation runner.

All recorded numeric training metrics are finite except the named left/right
pad-force averages. Candidate frozen reference and LoRA change remain exactly
zero; control LoRA change also remains zero. Candidate hidden gradient means
increase from 0.0240 to 0.2154, while its last rollout's absolute correction
magnitudes reach z 0.13289 and gripper 0.04532. This is an active correction
branch; these rollout statistics are not final-checkpoint fixed-input probes.

Candidate sampled KL median/range: 0.000849 / 0.000716–0.001115.
Control: 0.003367 / 0.002796–0.014098. PPO clip fractions are 0.244–0.526%
and 1.560–1.942%, respectively. These show different policy movement at equal
LR, not a reason by themselves to raise the candidate LR. No NaN optimization
metric or sustained explosion is present in the supplied aggregates.

## Validation results

Each validation has 1,024 episodes on the in-run panel. These aggregate
measurements do not provide paired scene-level uncertainty. Zero-update
scores already differ by six episodes between arms, consistent with the need
for the project's repeated evaluation protocol rather than exact rollout
equality across independent runs.

| Metric | Candidate initial → final | Control initial → final |
|---|---:|---:|
| Strict successes | 403 → 391 / 1024 | 397 → 419 / 1024 |
| Strict rate | 39.36% → 38.18% (−1.17 pp) | 38.77% → 40.92% (+2.15 pp) |
| Physical grasp | 74.41% → 73.63% | 75.88% → 75.68% |
| Held lift | 66.31% → 66.60% | 67.19% → 68.65% |
| Plate strict | 40.63% → 39.45% | 40.63% → 42.97% |
| Bowl strict | 38.09% → 36.91% | 36.91% → 38.87% |
| Carry slip (lower is better) | 26.17% → 27.54% | 27.44% → 26.86% |

The candidate fluctuates without a sustained strict gain. Control looks
stronger in these measurements, ending 2.73 pp above candidate, but neither
arm is promoted from the in-run panel. Do not replace the predeclared final
budget checkpoint with the candidate's earlier 41.31% peak after seeing it.

## Simulator health is a substantive caveat

| Metric | Candidate | Control |
|---|---:|---:|
| Non-finite live training episodes | 1,483 / 51,200 (2.896%) | 770 / 49,152 (1.567%) |
| First five updates, pooled rate | 1.963% | 1.572% |
| Last five updates, pooled rate | 3.271% | 1.650% |
| Worst update rate | 4.102% | 2.100% |
| Excluded records | 285,290 | 158,317 |
| Final validation non-finite episodes | 37/1024 (3.613%) | 12/1024 (1.172%) |
| Updates with NaN force averages | 11/25 | 6/24 |

Candidate force-warning updates: 1, 6, 7, 13, 17, 18, 20, 21, 22, 23, 24.
Control: 7, 13, 16, 17, 20, 21. Finite losses do not make these simulator
failures harmless. Candidate training and validation both show increased
non-finite incidence. The aggregates cannot establish whether the growing
correction causes it or locate the failing state/contact. Do not simply
extend training or raise LR to compensate for flat validation.

## Decision and exact next evaluation

Pause further optimization. Run the existing repeated matched evaluation on
the final budget checkpoints and original source: 512 development scenes,
excluding the in-run panel, four repeats per checkpoint (6,144 episodes total).
Use deterministic residual means, prior noise scale 1, 128 decisions, no
controller intervention, and preserve non-finite failures in the outcomes.
The comparator already reports scene-clustered strict/grasp/lift and
non-finite differences. Keep `final_test` untouched.

The runner supports `LABEL=step_N` and refuses ambiguous/missing matches, so
the unknown candidate timestamp does not need to be guessed:

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only
unset SCENE_LIST SCENE_CLASS CONTROLLER_Z_MAX STOCHASTIC_SEED
SRC=runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006
OUT="runs/latent_2m_comparison_$(date +%Y%m%d_%H%M%S)"
CHECKPOINTS="reference=$SRC candidate=step_2007079 control=runs/latent_2m_control_20261009_033534/rl/step_2052558" \
  OUT_ROOT="$OUT" REPEATS=4 GPUS="0 1" WORLDS=64 ROUNDS=8 \
  CONFIG=configs/examples/cdpr_smolvla_three_stage_put_into.yaml \
  SCENES=runs/three_stage/scenes_8192.json SPLIT=student_validation \
  SCENE_MANIFEST=runs/three_stage/scenes_8192.json RETENTION_MARGIN=0.03 \
  DECISIONS=128 SETTLE_DECISIONS=0 PRIOR_NOISE_SCALE=1 EXCLUDE_VALIDATION_PANEL=1 \
  bash scripts/compare_cdpr_three_stage_repeats_remote.sh &&
conda run --no-capture-output -n cdpr-mjlab python3 tools/audit/compare_put_into_repeats.py \
  --checkpoint "control=$OUT/control/rep1,$OUT/control/rep2,$OUT/control/rep3,$OUT/control/rep4" \
  --checkpoint "candidate=$OUT/candidate/rep1,$OUT/candidate/rep2,$OUT/candidate/rep3,$OUT/candidate/rep4" \
  --retention-margin 0.03 --scene-manifest runs/three_stage/scenes_8192.json \
  --output "$OUT/candidate_vs_control.json"
```

Keep the printed output directory. If interrupted, reuse that **same** `OUT`
value when rerunning the evaluation command; do not regenerate the timestamp
if the intent is to resume completed evaluations. Return `comparison.json`
and `candidate_vs_control.json` from that directory. No further raw training
logs are needed to perform this next comparison. If repeated evaluation
confirms candidate's higher simulator-failure rate, investigate that before
any candidate extension; a favorable strict result alone does not resolve it.

No algorithm change or remote evaluation was performed locally. The documented
command block and both evaluator launchers passed shell syntax checks; file
hashes, metrics, update counters and all validation rows were inspected locally.
