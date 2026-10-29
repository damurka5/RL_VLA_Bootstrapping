# Sparse and Dense Reward Training Comparison for SimpleVLA-RL

This report compares the available SimpleVLA-RL training logs for reward shaping experiments against the default terminal reward baseline.

Analyzed logs:

| Run label | Source log | Notes |
|---|---|---|
| `terminal_reward` | `terminal_reward_training.txt` | Default terminal success reward baseline. |
| `dense_reward` | `dense_reward_training.txt` | Dense reward enabled, but logged subgoal phase metrics are mostly inactive. |
| `dense_0.2_phase_0.3_gradclip_0.5` | `dense-0.2-phase-0.3-gradclip-0.5-training.txt` | Partial log starting at trainer step 27; stronger phase/subgoal shaping and grad clipping. |
| `sparse_phase_transfer_0.5` | `sparse_phase_transfer_05_training.txt` | Sparse phase-transfer reward. The filename was listed twice in the request, so it was analyzed once. |

## Executive Summary

The best validation score is tied across all reward variants: every run reaches `val/test_score/all = 0.983` on LIBERO spatial evaluation with 60 rollouts. This means the present logs do not prove a higher peak success rate from dense or sparse rewards.

The strongest paper-worthy difference is not peak validation, but training signal density. Terminal reward produces nonzero actor gradient norm in only `16.2%` of parsed training steps, while shaped reward runs produce nonzero actor gradients in `71.6%` to `96.1%` of steps. This supports the claim that dense or phase-based reward shaping gives more frequent credit assignment than pure terminal reward.

Among shaped-reward runs, `dense_0.2_phase_0.3_gradclip_0.5` is the best candidate for a main dense-reward ablation: it reaches the same best validation score as terminal reward, has much richer reward/gradient signal, and avoids the extreme reward spikes seen in `sparse_phase_transfer_0.5`. The sparse phase-transfer run starts strong, but its final validation score falls to `0.917`, suggesting instability without checkpoint selection or tuning.

Important caveat: these are single logs, not repeated seeds. Treat claims as preliminary until replicated.

## Validation Performance

| run | log span | train steps | val evals | best val (step) | final val (step) | mean val | val std | first >=0.967 | first >=0.983 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `terminal_reward` | 0-98 | 99 | 9 | 0.983 (79) | 0.983 (89) | 0.944 | 0.026 | 69 | 79 |
| `dense_reward` | 0-161 | 162 | 16 | 0.983 (19) | 0.933 (159) | 0.957 | 0.028 | 19 | 19 |
| `dense_0.2_phase_0.3_gradclip_0.5` | 27-148 | 122 | 12 | 0.983 (79) | 0.933 (139) | 0.953 | 0.021 | 69 | 79 |
| `sparse_phase_transfer_0.5` | 0-101 | 102 | 10 | 0.983 (59) | 0.917 (99) | 0.957 | 0.019 | 9 | 59 |

Interpretation:

- `terminal_reward` has the strongest final checkpoint among these logs: `0.983`.
- `dense_reward` reaches `0.983` earliest, at step 19, but later regresses to `0.933`.
- `sparse_phase_transfer_0.5` reaches `0.967` immediately at step 9 and peaks at `0.983`, but later drops to `0.917`.
- `dense_0.2_phase_0.3_gradclip_0.5` matches terminal peak performance, but its log is partial and final validation regresses.

## Common Validation Steps

These are the validation steps shared by all four logs.

| step | terminal | dense | dense 0.2 phase 0.3 gradclip 0.5 | sparse phase transfer 0.5 |
|---:|---:|---:|---:|---:|
| 29 | 0.917 | 0.900 | 0.933 | 0.967 |
| 39 | 0.933 | 0.967 | 0.933 | 0.967 |
| 49 | 0.950 | 0.917 | 0.933 | 0.950 |
| 59 | 0.933 | 0.950 | 0.933 | 0.983 |
| 69 | 0.967 | 0.967 | 0.967 | 0.933 |
| 79 | 0.983 | 0.983 | 0.983 | 0.967 |
| 89 | 0.983 | 0.983 | 0.983 | 0.967 |

Mean difference against terminal reward on these shared validation steps:

| run | mean diff vs terminal | wins | ties | losses | reading |
|---|---:|---:|---:|---:|---|
| `dense_reward` | 0.000 | 2 | 3 | 2 | Roughly tied with terminal over shared checkpoints. |
| `dense_0.2_phase_0.3_gradclip_0.5` | 0.000 | 1 | 5 | 1 | Nearly identical validation curve over shared checkpoints. |
| `sparse_phase_transfer_0.5` | +0.010 | 3 | 1 | 3 | Better early, worse late. |

## Training Signal Metrics

| run | train verifier mean | reward_all mean | reward_all last10 | subgoal_dense mean | grad nonzero % | grad_norm mean | reward spread mean | reward spread max | pg_clipfrac mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `terminal_reward` | 0.946 | 4.729 | 4.750 | n/a | 16.2 | 0.260 | 0.960 | 5.000 | 0.004 |
| `dense_reward` | 0.936 | 4.812 | 5.225 | 0.133 | 71.6 | 1.756 | 1.963 | 6.000 | 0.016 |
| `dense_0.2_phase_0.3_gradclip_0.5` | 0.940 | 6.317 | 6.775 | 1.619 | 93.4 | 2.392 | 3.708 | 19.600 | 0.020 |
| `sparse_phase_transfer_0.5` | 0.930 | 5.933 | 7.669 | 1.282 | 96.1 | 2.436 | 5.392 | 198.500 | 0.021 |

Interpretation:

- Terminal reward often gives uniform rewards inside the batch, causing zero advantages and zero actor gradients in most steps.
- Dense and phase-transfer rewards create richer intra-batch reward variation, visible in higher reward spread, nonzero gradient frequency, and higher `pg_clipfrac`.
- `sparse_phase_transfer_0.5` has very large reward spread spikes. Its maximum observed reward spread is `198.5`, much larger than `19.6` for the weighted dense run. This is likely an instability signal.
- `dense_0.2_phase_0.3_gradclip_0.5` gives strong nonzero training signal without the extreme sparse-transfer spike.

## Subgoal and Phase Metrics

| run | phase_completed mean | reward_total mean | positive_delta mean | subgoal_progress mean | phase_completed last10 |
|---|---:|---:|---:|---:|---:|
| `terminal_reward` | n/a | n/a | n/a | n/a | n/a |
| `dense_reward` | 0.000 | 0.015 | 0.000 | 0.015 | 0.000 |
| `dense_0.2_phase_0.3_gradclip_0.5` | 4.336 | 0.034 | 0.042 | 0.107 | 4.588 |
| `sparse_phase_transfer_0.5` | 4.737 | 0.026 | 0.039 | 0.111 | 8.938 |

Interpretation:

- The older `dense_reward` log appears to have a weak or incomplete subgoal signal: `phase_completed` and `positive_delta` are always zero.
- The weighted dense and sparse phase-transfer logs expose richer phase metrics and should be used for paper analysis of subgoal credit assignment.
- The high `phase_completed last10` for sparse phase transfer, together with the extreme reward spread, suggests the reward may be over-amplifying phase events late in training.

## Per-Task Validation

Best checkpoint per task:

| task | terminal best | dense best | dense 0.2 phase 0.3 gradclip 0.5 best | sparse phase transfer 0.5 best |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | 0.750 | 0.750 | 0.750 | 0.750 |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 9 | 1.000 | 1.000 | 1.000 | 1.000 |

Final checkpoint per task:

| task | terminal final | dense final | dense 0.2 phase 0.3 gradclip 0.5 final | sparse phase transfer 0.5 final |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1.000 | 0.750 | 0.750 | 0.750 |
| 5 | 0.750 | 0.750 | 0.750 | 0.750 |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 0.750 | 0.750 | 0.500 |
| 9 | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation:

- The task-level ceiling is almost saturated: every run reaches perfect best-checkpoint performance on 9 of 10 tasks, with task 5 capped at `0.750` in these logs.
- Final-checkpoint degradation mostly appears on tasks 4 and 8 for shaped rewards.
- The sparse phase-transfer final checkpoint is especially weak on task 8, dropping to `0.500`.

## Runtime Metrics

| run | gen mean s | actor update mean s | gen+update mean s | testing mean s |
|---|---:|---:|---:|---:|
| `terminal_reward` | 276.7 | 656.3 | 933.0 | 1986.3 |
| `dense_reward` | 278.8 | 658.0 | 936.9 | 1986.3 |
| `dense_0.2_phase_0.3_gradclip_0.5` | 275.3 | 648.6 | 923.9 | 1963.1 |
| `sparse_phase_transfer_0.5` | 280.8 | 663.0 | 943.8 | 1996.6 |

Interpretation:

- Reward shaping does not materially change runtime in these logs.
- Average rollout generation and actor update times are very similar across runs.
- Validation is expensive, around 33 minutes per evaluation.

## Paper-Relevant Claims to Test Further

Useful claims supported by these logs:

1. Dense and phase-based rewards increase the frequency of nonzero policy-gradient updates compared with terminal reward.
2. Dense reward can reach terminal-reward peak validation performance while exposing more informative subgoal/phase metrics.
3. Sparse phase-transfer reward may improve early validation performance, but can regress late without careful scaling, clipping, or checkpoint selection.
4. Best-checkpoint reporting is important: shaped rewards often match peak terminal performance, while final checkpoints can be worse.

Claims not yet supported strongly enough:

1. Dense or sparse reward improves peak final success rate over terminal reward. All variants peak at `0.983`.
2. Any reward method is statistically superior. These are single-run logs and should be repeated across seeds.
3. Sparse phase transfer is better overall. It has good early scores, but the final checkpoint is the weakest.

## Recommendation for the Paper

Use `terminal_reward` as the baseline and `dense_0.2_phase_0.3_gradclip_0.5` as the primary dense-reward comparison. It is the cleanest shaped-reward run because it produces much denser training signal while matching the terminal baseline's best validation score.

Use `sparse_phase_transfer_0.5` as an ablation showing that phase-transfer rewards can accelerate early validation, but may become unstable or over-scaled. In a future run, test lower transfer weights, reward normalization, stricter clipping, or early stopping based on validation.

For the paper tables, report both:

- best checkpoint validation score, because shaped rewards can peak early;
- final checkpoint validation score, because it reveals late-training stability.

Also include training-signal metrics such as nonzero gradient percentage, reward spread, subgoal dense reward, phase completion, and policy clip fraction. These metrics tell the core story better than validation score alone.
