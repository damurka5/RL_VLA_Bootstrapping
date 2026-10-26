# Proposed results and experiment priorities

All new experiments below are proposals for author review. No training, evaluation, hardware run, or final-test access was performed for this outline.

## Main quantitative table

Use matched endpoint evaluation for the paper's central claims. Rows: mapped frozen SmolVLA; bootstrapped initializer; terminal-only residual RL; staged residual RL with legacy likelihood; staged residual RL with latent likelihood; latest selected checkpoint. For every row record the actual observation model, initialization, training budget, and evaluation contract. An unavailable row stays pending rather than borrowing a number from a different evaluator.

Columns: strict successes/trials, strict rate and uncertainty, plate/bowl results, native placement, and interaction/compute budget. Use the same scene list, decision horizon, assistance settings, stochasticity, and failure handling. Show separate final-test and development panels if both are included.

## Retained comparison available now

| Arm | Strict successes | Strict rate | Difference vs source (paired 95% CI) | Non-finite episodes |
|---|---:|---:|---:|---:|
| Historical source | 756/2048 | 36.91% | Reference | 0.98% |
| Latent-likelihood bounded-residual continuation | 843/2048 | 41.16% | +4.25 pp [+2.10, +6.40] | 1.12% |
| Added correction-network continuation | 769/2048 | 37.55% | +0.63 pp [-1.76, +3.03] | 3.56% |
| Latest run | Pending | >50%, author-reported | Pending | Pending |
| Physical robot | Pending | Pending | Pending | Not applicable; report hardware aborts separately |

The first three rows share 512 development scenes with four repeats each, excluding the fixed in-run panel. They support a development improvement of the bounded-residual continuation over its source. They do not isolate the likelihood effect from additional training. The two continuation arms were each compared with source; a direct paired control-versus-correction interval is not in this JSON. Do not compute it from marginal intervals.

## Priority 1: establish the headline

- Record latest checkpoint path/hash, parent checkpoint, Git commit, architecture and likelihood tags, resolved configuration, exact successes/trials, date, and strict/native metric.
- Determine whether >50% is a single in-run reading, an average, or a separately evaluated endpoint. Report it at that scope.
- Freeze the checkpoint-selection rule; repeat matched development evaluation and prepare an untouched final-test evaluation after development decisions are fixed.
- Preserve per-scene outcomes, repeats, stochastic settings, non-finite flags, destination/object IDs, and videos. Cluster inference by scene; multiple evaluation repeats are not independent training seeds.

## Priority 2: explain why the method works

| Question | Comparison | Necessary control |
|---|---|---|
| Does staged credit help full-task learning? | Terminal-only vs three milestones | Same initialization, actor, budget and evaluation |
| Does upstream completion credit help? | Full-task bonus off/on | Same stage normalization and data selection |
| Does the likelihood repair help? | Legacy vs latent likelihood continuation | Same source, budget, optimizer schedule, offsets and task |
| What does the VLA contribute? | Normal vs fixed/removed action prior | Matched visual/proprioceptive inputs and training budget; specify how actor initialization is handled |
| Is the added correction network helpful? | Bounded residual vs correction network | Both use latent likelihood, matched source and budget; report simulator failures |

First prioritize staged credit, likelihood, and the VLA-prior contribution. Use independent training seeds where feasible. If these cannot be run, narrow the claims rather than interpreting campaign history as a controlled ablation. Retention/SFT experiments belong in the main paper only if consolidation remains a central contribution; otherwise summarize initialization and lineage.

## Priority 3: stage and robustness analyses

- Stage funnel: physical grasp, held lift, strict success, completion conditional on lift, with denominators.
- Destination and object breakdowns; distinguish in-distribution scene diversity from genuinely held-out objects/tasks.
- Fixed target-position quartiles and a workspace success map, especially the historically weak far side.
- Failure taxonomy with mutually exclusive categories for plots; retain overlapping raw event labels separately.
- Training curves against all sampled/executed actions as well as selected actions; report inherited compute and training-seed coverage.

## Physical experiment in the main Results section

- Setup and protocol are specified in `sections/05_experimental_setup.tex`; result slots are in `sections/06c_real_world_results.tex`.
- Draft design: 4 objects x 2 destinations x 10 trials = 80 trials per policy, subject to hardware feasibility and desired statistical precision. Predeclare trial list, resets, success rubric, baseline and ordering.
- Record calibration and any adaptation on hardware. Claim zero-shot/direct transfer only if the actual procedure warrants it.
- Report successes/trials, uncertainty, intervention/abort counts, completion time, object/destination breakdowns, representative videos, and observed failure mechanisms. Use paired or clustered analysis if layouts/sessions repeat.
- Include unsuccessful outcomes. Do not fill pending entries with simulated values or predicted percentages.

## Claim boundaries to preserve

No claim of >50% strict final-test performance before its protocol is verified; no real-world success claim before measurement; no isolated likelihood improvement from source-versus-continuation alone; no novel residual-RL/SFT claim by itself; no demonstration-free lineage claim when scripted data are present; no claim that a VLA action prior is useful without the relevant comparison.


## Mathematical mechanism: required additions

The paper should connect each mathematical statement to an observation; see `theory/MATHEMATICAL_FOUNDATION.md` for proofs and scope.

| Result | Available now | Still needed on the robot policy |
|---|---|---|
| Conditional latent-score identity | Derivation; single-step clipping and persistent-offset finite-model checks | Captured policy-record audit and matched likelihood ablation |
| Binary mixed-group probability | Exact formula; exhaustive group checks | Per-scene milestone probabilities, observed mixed groups and losses from validity/filter masks |
| Protected-credit sequential-gate result | Proof; 336 exhaustively enumerated groups | Protected vs symmetric credit, gradient interference and grasp/lift retention |
| Local neural-policy improvement condition | Smoothness inequality and error-bound derivation | Independent directional alignment, actual optimizer displacement and separate strict evaluation |
| Training/deployment connection | Explicit difference in exploration law | Evaluate stochastic training policy and deterministic-mean deployment separately |

The current code uses achieved-milestone negative scale 0 in the inspected configuration. Verify the latest run's resolved setting; do not reconstruct it from defaults. A proposed direction must be frozen before collecting its independent alignment audit. L and estimator-error bounds are currently unknown, so no numerical improvement certificate is asserted.

The finite-model verification is reproducible using `python3 theory/verify_mathematics.py`. These values belong to analytic validation, not the robot success table. Standard score identities and existing binary-GRPO theory must receive attribution. Do not present a proof of a modified algorithm as a proof of the implemented stage-filtered optimizer.
