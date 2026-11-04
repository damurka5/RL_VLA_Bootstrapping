# Matched 60M CDPR evaluation: promote the endpoint

Reviewed **2026-11-04**, Europe/Moscow. The remote comparison directory is
`latent_control_60m_comparison_20261103_183216`. The 60M endpoint passes the
declared development promotion gate and replaces 40M as the selected
controller. The [supplied comparison](../../artifacts/latent_control_60m_comparison_20261104/comparison.json)
is retained verbatim, with [source hash](../../artifacts/latent_control_60m_comparison_20261104/source.json)
and [checked counts](../../artifacts/latent_control_60m_comparison_20261104/review_summary.json).

## Main result

| Checkpoint | Strict successes | Strict | Grasp | Held lift | Non-finite |
|---|---:|---:|---:|---:|---:|
| control40m | 1,245/2,048 | 60.79% | 89.45% | 86.13% | 20/2,048 = 0.98% |
| final60m | **1,416/2,048** | **69.14%** | **92.14%** | **89.45%** | 24/2,048 = 1.17% |

The candidate adds **171 strict successes**, **+8.35 percentage points**,
scene-clustered paired 95% CI **[+6.05, +10.69] pp**, Holm p=**0.00005**.
Every repeat improves: **318→353, 306→354, 313→344, 308→365** successes
out of 512, gains of 35, 48, 31 and 57. These repeats estimate rollout
variation for two fixed checkpoints; they are not independent training seeds.

The report's 3 pp retention criterion passes:

- Grasp **+2.69 pp**, CI **[+1.17, +4.20]**.
- Held lift **+3.32 pp**, CI **[+1.61, +5.08]**.
- Carry slip **24.51%→19.43%**, −5.08 pp, CI **[−7.23, −2.98]**.
- Wrong placement **20.56%→14.89%**, −5.66 pp, CI **[−7.62, −3.71]**.
- Native placement **66.02%→74.95%**, +8.94 pp, CI **[+6.74, +11.18]**.
- Strict given held lift **70.58%→77.29%**, a descriptive ratio.

These grasp/lift measurements are within the full `put_into` task; they do
not establish retention on separate reaching or pickup instructions.

Non-finite outcomes change by **+0.20 pp**, CI **[−0.44, +0.83]**,
p=0.63292. No increase is detected, but this interval does not establish
equivalence or an absence of simulator failures. The promotion rule checks
grasp/lift retention; simulator health is reviewed separately. Event flags can
overlap and non-finite attempts remain in the denominator.

## Protocol and evidence limits

Both policies use **512 distinct off-panel student_validation scenes × four
repeats**, 2,048 episodes each, 4,096 total. The scene manifest hash is
`30dc9d53027c7e7ba871379e1ec17f0f148a08517f03fc4fa531549a15ef9208`.
The 128-scene training validation panel is excluded: split size 1,230→1,102,
panel seed 2,000,000, 64 groups per rank and two ranks. Evaluation uses
64 worlds × eight distinct-scene rounds, 128 decisions, no extra settling,
unassisted starts, deterministic residual actions and prior noise scale 1.
There is no controller-ceiling override. `final_test` is not used.

The baseline adapter hash matches the previously promoted 40M controller.
Counts inferred from four-decimal rates are unambiguous at these denominators;
strict counts agree with the repeat sums and destination totals. Per-episode
evaluation files were not supplied, so paired intervals, p-values and scene
matching are reported from the comparison and were not independently rebuilt.
Checkpoint bytes were not inspected locally; their hashes are recorded by the
remote report.

Use the contemporaneous **60.79%** 40M baseline for this gain. Its earlier
61.43% off-panel score came from different rollouts. The 60M panel's **70.61%**
and peak **71.09%** are also different measurements. This new result confirms
transfer beyond the training panel, with development model-selection scope;
it is not an untouched final-test result or proof that a single method change
caused the improvement.

## Destination and difficult-scene results

| Destination | 40M strict | 60M strict | Difference, paired CI |
|---|---:|---:|---|
| Bowl, 255 scenes × 4 | 554/1,020 = 54.31% | **642/1,020 = 62.94%** | +8.63 pp [+5.39, +11.96] |
| Plate, 257 scenes × 4 | 691/1,028 = 67.22% | **774/1,028 = 75.29%** | +8.07 pp [+4.67, +11.48] |

The difficult target-y Q4 improves **26.94%→43.02%**, +16.09 pp,
CI [+11.43, +20.74]. Q3 improves 9.77 pp and Q2 5.80 pp. Q1's +1.73 pp
interval includes zero. Potato remains weakest at **51.84%**; its +3.28 pp
interval [−1.84, +8.40] includes zero. Apple, orange and tomato gains have
positive supplied intervals. Subgroup and secondary-metric intervals are
exploratory and not jointly multiplicity-adjusted; Holm correction refers to
the primary checkpoint comparison. Do not claim a demonstrated gain in every
object or spatial subgroup.

The panel-selected `step_59268127` was not included. There is no separate
peak-versus-endpoint conclusion and no need to replace the endpoint on the
five-success panel difference alone.

## Promoted checkpoint and next continuation

```text
runs/latent_control_to60m_20261019_224018/rl/step_60025885/smolvla_grpo_adapter.pt
SHA-256: f69360d4a7afeac36b147c6f3a335a7199c9a385f871c568587c85d9d3d86711
```

The gain supports another unchanged control continuation to **80M total
selected actions**, adding about **19,974,115** before rollout overshoot.
This is a proposed next budget, not a prediction of further gain. Keep the
bounded residual, latent conditional likelihood, frozen LoRA, LR 1e-5, one PPO
epoch, task, rewards and simulator settings. A further method change is not
indicated by this comparison. Preserve 40M and 60M, and compare the new
endpoint against 60M before promotion. Keep `final_test` reserved.

Copy/paste on the server when ready to run the continuation:

```bash
cd /root/repo/RL_VLA_Bootstrapping
env -u LEGACY_INIT_CHECKPOINT -u WARMSTART_CHECKPOINT \
  RLVLA_EMBODIMENT=cdpr ARM=control \
  RESUME_CHECKPOINT=runs/latent_control_to60m_20261019_224018/rl/step_60025885 \
  EXPECTED_SOURCE_SHA256=f69360d4a7afeac36b147c6f3a335a7199c9a385f871c568587c85d9d3d86711 \
  MAX_TRAIN_STEPS=80000000 MAX_UPDATES=0 \
  PPO_EPOCHS=1 LR_OVERRIDE=1e-5 PRIOR_NOISE_SCALE=1 \
  CONFIG=configs/examples/cdpr_smolvla_three_stage_put_into_latent_correction.yaml \
  SCENES=runs/three_stage/scenes_8192.json \
  CUDA_VISIBLE_DEVICES=0,1 WORLDS_PER_RANK=512 SMOLVLA_MICROBATCH_SIZE=256 \
  RUN_PREFLIGHT=1 DRY_RUN=0 RUN_LABEL=latent_control_to80m \
  bash scripts/train_cdpr_latent_correction_pilot_remote.sh
```

This requests full-state resume, not weights-only initialization. The launcher
checks the source hash and resume metadata before training. No new remote job
was launched during this review.
