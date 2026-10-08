# Latent-likelihood control: completed 10-update diagnostic

Reviewed 2026-10-08 from the user's three attached files and terminal excerpt.
Run: `latent_diag_control_20261006_223344`. No remote process was run locally.

## Evidence and completion

| Supplied file | SHA-256 |
|---|---|
| `prflight_control_223344_metrics.jsonl` | `c560188df8a51209144245a6bc344ea2ce5975dbe3ebf58079ba88c55d9c7c38` |
| `prflight_control_223344_validation.jsonl` | `3a9eae5f20e7bbb947bd3f8207fe65162325d067ad72009aba791098e2831019` |
| `prflight_control_223344_preflight.json` | `3b54f9eaf42032ae1c9627fdb554a656b32aaeff1a08bb79004461b57b5e11d1` |

The metrics contain ten updates and validation contains five rows. Final
counters are 852,587 selected actions, 6,888,452 sampled actions, 20,480
episodes and 5,594 optimizer steps. Source global step is 56,072,006 and
optimizer LR is 1e-5 throughout. The terminal excerpt identifies the resolved
architecture as `bounded_residual_v0` and reports `train_exit=0 log_exit=0`.
The 10-update limit was reached before the 1M-action budget. No zero-signal
stop was recorded. Neither launch provenance nor the resolved protocol file
was attached; exact initializer hash and complete configuration matching
cannot be reverified from these three files alone.

`pilot/wall_time_s` is 9,532.74 s (2.65 h); end-to-end
`training/elapsed_time_s` is 11,462.71 s (3.18 h). The latter includes in-loop
validation/checkpoint work, but starts after initial validation and preflight.

## Numerical behavior

- Loss, entropy, gradient statistics, log standard deviation and all other
  recorded numeric metrics are finite except the two force averages described
  below. LoRA max absolute change remains exactly zero.
- PPO clip fraction ranges from 1.70% to 2.69%. Pre-clip mean gradient norm
  ranges from 5.36 to 6.35. A norm above the configured clipping threshold
  is expected for a statistic measured before gradient clipping.
- Sampled latent KL has median 0.003975, with spikes to **0.07544 on update 3**
  and **0.03682 on update 4**. Corresponding pre-clip maximum gradient norms
  are 185.81 and 115.04. KL later returns to approximately 0.003–0.0054.
  This is a weighted sample mean of old minus current log probability during
  minibatch optimization, not exact final-policy KL. The aggregate files do
  not identify the samples responsible for the spikes or establish their
  cause. Retain these as stability caveats for the next comparison.
- Non-finite live simulation episodes total **325/20,480 = 1.587%**, ranging
  from 1.22% to 2.05% per update, without sustained growth over these ten
  updates. The collector reports 64,916 excluded records in total. These
  simulation failures must not be mistaken for NaN optimization losses.

## Why the summarizer printed a conda error

On updates 1 and 8, `left_pad_normal_force_mean_n` and
`right_pad_normal_force_mean_n` are NaN. The summarizer previously returned
exit code 1 for every non-finite numeric metric, causing `conda run` to print
an error **after the training process had already exited successfully**.

The collector already applies `torch.where` to exclude inactive or ineligible
worlds from force sums, and its observation denominator is positive. Thus
this is not simply an empty-mean or masked `NaN * 0` explanation. Non-finite
included force readings or corrupted/overflowed accumulation remain possible;
the aggregate JSONL cannot distinguish them. Raw contact forces also feed the
physical-grasp predicate, so this is **not proof of harmless simulator behavior**.

The reporter now labels these two named observational averages as explicit
warnings and preserves their raw values. They no longer alone cause a nonzero
report exit; NaN losses, gradients, reward metrics, other numeric metrics,
missing data and integrity failures still do. No force samples are replaced
with zero, and no simulator, reward, optimizer or collector behavior changes.
This fixes failure classification, not the underlying contact-force issue.

## Validation and interpretation

Strict success on the in-run panel:

| Selected actions | Strict successes / 1024 | Rate |
|---:|---:|---:|
| 0 | 394 | 38.48% |
| 257,736 | 409 | 39.94% |
| 514,087 | 382 | 37.30% |
| 769,030 | 423 | 41.31% |
| 852,587 | 415 | 40.53% |

| Metric | Initial | Final | Difference |
|---|---:|---:|---:|
| Strict success | 38.48% | 40.53% | +2.05 pp |
| Physical grasp | 74.51% | 77.05% | +2.54 pp |
| Held lift | 65.92% | 70.70% | +4.79 pp |
| Plate strict | 38.67% | 44.14% | +5.47 pp |
| Bowl strict | 38.28% | 36.91% | −1.37 pp |
| Carry slip | 26.17% | 29.49% | +3.32 pp (worse) |

The control completed without an observed optimizer collapse, and its final
aggregate strict score is higher. This is a diagnostic result, not promotion:
the gain is uneven by destination, carry slip worsens, and repeated in-run
panel measurements do not provide the planned paired scene-level uncertainty.
Do not choose the update-9 peak as a new primary checkpoint after observing it.

The generic GPU preflight has `ok=true`, every required check true and no
errors. Its static-only `compatible=false` reflects checks that static parsing
does not execute; the completed dynamic compatibility report is true. This
preflight is not the actor-equivalence or learned-correction preflight.

## Next step

Keep this completed control diagnostic; do not rerun it merely because the
report command returned 1. Obtain the **candidate** diagnostic metrics and
validation next (plus each arm's launch provenance and resolved protocol for
exact matching). Inspect its correction gradients, frozen reference/LoRA,
KL and simulator-failure trends before committing to the matched 2M pilot.
The original loop ran candidate before control, so inspect existing candidate
artifacts first. No candidate results were supplied in this turn.

Local verification of the reporting change: 17 focused tests pass, including
contact-only warning vs NaN optimizer/reward failures. Applied directly to the
attached rows, classification finds exactly the two contact-warning rows and
no other non-finite numeric metrics; all five validation rows are finite.
