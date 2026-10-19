# 40M control continuation: training result and checkpoint comparison

Reviewed 2026-10-19. The user identifies the completed run as
`latent_control_to40m_20261012_230538`, with endpoint
`rl/step_40056193/smolvla_grpo_adapter.pt`. Unlike the earlier console paste,
this event file contains the 20M → 40M continuation.

## Evidence and training outcome

[Source hash and summary](../../artifacts/latent_control_40m_20261019/summary.json),
[81 validation readings](../../artifacts/latent_control_40m_20261019/validation.csv),
and [299 update readings](../../artifacts/latent_control_40m_20261019/training.csv)
are retained. Scalar sampling was disabled when reading the event file.
Steps are exact integer Event.step metadata; the final selected-action scalar
rounds to 40,056,192, so it must not be used as the checkpoint name.

Training advances **20,068,312 → 40,056,193**, adding **19,987,881 selected
actions**, 299 updates (560 cumulative), 609,792 episodes and 141,277 optimizer
steps. Recorded training elapsed time is **89.14 hours**. Cumulative pilot
work timers have a different scope and are not used for that elapsed figure.
No launch provenance or checkpoint bytes were supplied; the run directory
comes from the user and checkpoint hashes remain to be captured by evaluation.

| In-run validation, 1024 episodes | Resume 20,068,312 | Peak 39,804,332 | Final 40,056,193 |
|---|---:|---:|---:|
| Strict successes | 555 | 664 | 627 |
| Strict rate | 54.20% | 64.84% | 61.23% |
| Physical grasp | 83.40% | 91.41% | 90.14% |
| Held lift | 78.32% | 87.30% | 85.35% |
| Plate strict | 60.35% | 72.66% | 70.90% |
| Bowl strict | 48.05% | 57.03% | 51.56% |
| Carry slip | 23.54% | 21.78% | 23.63% |
| Wrong place | 21.58% | 18.46% | 19.04% |
| Non-finite episodes | 0.68% | 0.68% | 0.59% |

Endpoint strict gains **7.03 pp** versus this invocation's initial validation.
The first 20 validation readings average 55.05%, the final 20 average 61.91%.
These correlated panel averages describe the trend, not independent statistical
replication. The initial 54.20% differs from the prior run's 53.03% endpoint
reading of the same 20M checkpoint; stochastic closed-loop evaluations vary.
Do not compare 61.23% directly against the earlier off-panel 53.47% as a
matched gain. No 40M off-panel result has yet been supplied.

![40M validation and simulator health](../../artifacts/latent_control_40m_20261019/training_review.png)

## Optimization and health

All 299 recorded policy-loss, entropy, KL, PPO clipping, mean-gradient, log-std,
LR and LoRA-change readings are finite with full update coverage. LR remains
1e-5 and LoRA change zero. KL median 0.002942, maximum 0.021169; PPO clipping
ranges 1.247–1.960%; gradient means 4.093–5.659. No recorded optimization
collapse is evident. This does not prove every intermediate tensor was finite.

Mean reported non-finite episode rate is **1.619%**, close to the prior leg's
1.622%. First/last 20 update means are 1.523% / 1.665%, with no large late
escalation visible. The aggregate episode count gives
**9,881/609,792 = 1.620%**. Five updates contain 1,536 episodes rather than
2,048; their steps are retained in summary.json. The logged rate is a mean
across ranks, whereas episode counts are summed, so multiplying every rate
by a fixed 2,048 would give an incorrect failure count. The global count and
episode total are used for the aggregate health rate.

Both pad-force streams contain only 228 of 299 readings. The writer skips
non-finite scalars, so the 71 missing readings are unavailable contact-force
diagnostics, not evidence of finite forces. The raw JSONL can clarify them.

## Decision and checkpoints

Continuing the existing parameters was productive on the training validation
panel. There is no demonstrated reason to switch architecture, rewards or LR
now. **Next perform a checkpoint comparison, not another training pilot.**
Use the promoted 20M control as baseline, final40m as the primary endpoint,
and peak39m as a secondary panel-selected candidate:

| Role | Directory relative to the server repository |
|---|---|
| Promoted baseline | `runs/latent_control_long_20261009_160735/rl/step_20068312` |
| Primary endpoint | `runs/latent_control_to40m_20261012_230538/rl/step_40056193` |
| Panel-selected alternative | `runs/latent_control_to40m_20261012_230538/rl/step_39804332` |

Each must contain `smolvla_grpo_adapter.pt`. The final checkpoint path was
supplied by the user; peak existence is expected from validation checkpointing
but has not been inspected on the server. The launcher checks file existence.
The peak has 37 more panel successes than final (3.61 pp), but it was selected
from 81 readings. That does not establish an off-panel advantage or a collapse
at the final checkpoint. Both deserve comparison; retain both.

The same 512 off-panel development scenes × four repeats × three checkpoints
requires 6,144 episodes. The second command compares peak versus final on CPU
from those saved episodes, with no additional simulation. Keep final_test
untouched. Continue to use the 3 pp grasp/lift retention criterion and inspect
non-finite outcomes. Until these results arrive, 20M remains the promoted
checkpoint; 40M is a completed training result awaiting comparison.

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only
unset SCENE_LIST SCENE_CLASS CONTROLLER_Z_MAX STOCHASTIC_SEED
BASE=runs/latent_control_long_20261009_160735/rl/step_20068312
LONG=runs/latent_control_to40m_20261012_230538/rl
OUT="runs/latent_control_40m_comparison_$(date +%Y%m%d_%H%M%S)"

CHECKPOINTS="control20m=$BASE final40m=$LONG/step_40056193 peak39m=$LONG/step_39804332" \
  OUT_ROOT="$OUT" REPEATS=4 GPUS="0 1" WORLDS=64 ROUNDS=8 \
  CONFIG=configs/examples/cdpr_smolvla_three_stage_put_into.yaml \
  SCENES=runs/three_stage/scenes_8192.json SPLIT=student_validation \
  SCENE_MANIFEST=runs/three_stage/scenes_8192.json RETENTION_MARGIN=0.03 \
  DECISIONS=128 SETTLE_DECISIONS=0 PRIOR_NOISE_SCALE=1 EXCLUDE_VALIDATION_PANEL=1 \
  bash scripts/compare_cdpr_three_stage_repeats_remote.sh &&
conda run --no-capture-output -n cdpr-mjlab python3 tools/audit/compare_put_into_repeats.py \
  --checkpoint "final40m=$OUT/final40m/rep1,$OUT/final40m/rep2,$OUT/final40m/rep3,$OUT/final40m/rep4" \
  --checkpoint "peak39m=$OUT/peak39m/rep1,$OUT/peak39m/rep2,$OUT/peak39m/rep3,$OUT/peak39m/rep4" \
  --retention-margin 0.03 --scene-manifest runs/three_stage/scenes_8192.json \
  --output "$OUT/peak_vs_final.json"
```

Return `comparison.json` and `peak_vs_final.json` from `$OUT`; no model upload
is needed. Reuse the same OUT when resuming an interrupted evaluation. If both
beat 20M and peak has no demonstrated advantage, prefer the primary final
endpoint. A clear peak advantage must also retain grasp/lift and acceptable
simulator health. If promotion passes, another substantial continuation with
the same parameters is reasonable; choose its source and budget after this
comparison. No remote training or evaluation was run locally.
