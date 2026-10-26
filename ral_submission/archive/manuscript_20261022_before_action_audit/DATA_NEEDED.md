# What is needed to finish the paper

Updated 22 October 2026. Writing has started: no additional files are required to continue prose. The items below are needed to replace provisional statements, make the results reproducible, and support the final claims. Raw CSV/JSON/event logs are preferable to screenshots. Existing measurements are reused; you do not need to recreate the plots already generated.

## Evidence for the revised central idea

The main story is now dense-reward skill acquisition -> binary reward and self-imitation -> full-task staged RL. Prioritize the evidence below alongside final task and hardware outcomes:

1. **Action-interface initialization:** identify the original model, action dimensions, exact projections/expert/controller weights reset or replaced, retained VLA weights, initialization seed/checkpoint, and which modules each phase trained. The retained final implementation uses a residual MLP; this alone does not prove the original action expert was reinitialized.
2. **Per-skill learning timeline:** skill name/instruction, reset distribution, dense reward, binary predicate, dense-to-binary switch point/rule, RL and self-imitation ordering, success before/after each phase, scene list and evaluation counts. Distinguish approach-to-object, pickup, and held-object placement from full-task empty-start execution.
3. **Self-imitation/retention table:** per-skill outcomes before/after consolidation on matched resets; bank sizes and source policy IDs, scripted/oracle versus policy-generated rows, mixing weights, regression loss, trainable modules, and selected epochs. The historical skill-retention results and unsuccessful late full-task SFT attempts answer different questions.
4. **Composition comparison:** binary terminal-only versus staged RL from the same consolidated initializer, with the same full instruction, empty-start scenes, budget, and evaluator. A chain of externally switched specialists is a separate baseline, not the final single-policy result.
5. **Central ablations:** dense versus binary from a fresh target action interface; transition to binary versus continued dense skill training; consolidation with/without self-imitation and retained skills; staged versus terminal-only full-task RL. Prioritize these above likelihood/correction-network ablations when supporting the revised main contribution.

The proposed schedule must be tied to the actual checkpoint lineage: do not invent a uniform per-skill transition or claim every component was controlled in the current evidence. The LIBERO component study tests phase-aware shaping on a trained manipulator, not adaptation after action-head reinitialization.

## Highest priority: minimum evidence package

| Priority | Send / recover | Why it matters | Where it goes |
|---|---|---|---|
| 1 | Complete physical trial log, selected hardware checkpoint/hash, actual protocol, calibration/adaptation history, baseline trials, unedited video references | Confirms the success-assumption prose and gives honest denominators, failures, uncertainty, and transfer condition | Hardware setup, results table, abstract, conclusion |
| 1 | Real setup photo and timestamped rollout frames, with trial IDs/instruction/object/destination | Replaces prepared photo slots with actual evidence | Physical figure |
| 1 | LIBERO per-episode evaluations for all five reward rows, initialization, and low-success study; run configs/seeds/checkpoints | The PDF reports percentages but no episode counts, independent seeds, or confidence intervals | LIBERO tables and reproducibility text |
| 1 | Selected CDPR run's resolved config, launch command/commit and complete checkpoint lineage; actual control timing and actor input schema | Avoids treating example-YAML defaults as the run's actual settings; establishes training and deployment contract | Method/setup |
| 1 | Locked final-test evaluation after checkpoint/method selection is frozen, with baseline on the same scene list | Current 61.43% result is valid development evidence, not untouched final-test performance | Main endpoint table |
| 2 | Matched CDPR staged-vs-terminal and latent-vs-legacy likelihood runs, plus action-prior control | Supports causal method claims beyond improvement with additional training | Ablation table |
| 2 | LIBERO wall-time, rollout/acceptance, environment-action and update logs | Quantifies the PDF's faster accepted-batch claim without confusing more updates with better learning per update | Training/throughput figure |

## Tables to complete

### Main CDPR comparison

Already populated: 20M (1056/2048), 40M final (1258/2048), 39.8M peak (1256/2048); native/NF rates and paired strict differences. We have checkpoint hashes and scene-manifest hash.

Still needed: mapped frozen VLA, exact bootstrapped initializer, terminal-only residual RL, and selected full method on a common protocol. Include checkpoint, training seed, strict successes/trials, grasp/lift/native counts, numerical failures, plate/bowl breakdown, total sampled/executed/selected actions, inherited interactions, GPU-hours, and evaluation split. If a baseline cannot be run, narrow the associated claim; do not borrow an unmatched historical value.

### Credit and likelihood ablations

Prioritize three comparisons: (1) terminal-only versus three-stage credit; (2) legacy versus conditional-latent likelihood from the same source and budget; (3) normal versus fixed/removed VLA action prior with the same visual/proprioceptive inputs. Next add protected versus symmetric terminal bonus and bonus on/off. Preserve initialization, optimizer settings, task lists, horizon, offsets, noise, and budget. Provide independent training seeds where feasible; otherwise explicitly report a single-seed study. The failed correction-network pilot is already available but is not a replacement for these controls.

### LIBERO reward table

All five endpoint percentages are already present, plus low-success initial/intermediate/final values. Supply per-task success counts/attempts, initial-state IDs, trial seeds, training seeds, checkpoint steps, exact final-selection rule, horizon, actual validation dates, and task names. Preserve the common 61.0% initialization and clarify whether aggregate success is a task macro-average or a pooled episode rate. Recover the exact phase functions and filtering configuration for every row. Do not infer trial counts from decimal percentages.

### Physical table

One row per evaluated policy, with successes/attempts, interval, intervention count, abort count, transfer/adaptation condition, and completion-time summary. Keep object/destination and session/layout breakdowns in the raw data, with sample counts. Specify whether time summaries include only successful trials and how timeouts are handled. Report every attempt using a declared success rubric. The earlier 4 objects x 2 destinations x 10 repetitions = 80 proposal is optional and is not an observed sample size.

## Figures and source data

| Figure | Status now | Data still needed |
|---|---|---|
| Shared two-branch learning schematic | Included as an editable LaTeX diagram | Exact initialization lineage/assistance boundaries for a richer version |
| CDPR validation vs selected actions | Generated from 81 retained readings, 20M--40M; unsmoothed | Earlier legs, all sampled/executed-action counters, wall times, independent seeds for a complete training-cost curve |
| CDPR milestone and target-y breakdown | Generated from matched 20M/40M summaries | Episode-level records for confidence bands, continuous workspace map and exclusive failure categories |
| LIBERO reward endpoints | Included as a table to avoid duplicating the PDF bar chart | Counts/uncertainty and per-task results |
| LIBERO success and accepted batches vs time | Not drawn; PDF supplies no numerical time series | Wall-time/checkpoint evaluations, attempted/accepted groups, rejection reasons, collection seconds, updates, environment actions, GPU model/count |
| Hardware setup and rollout | Real image slots ready | Setup photo showing cables/gripper/cameras/workspace; 5 timestamped frames for approach/grasp/lift/carry/release; actual task text and trial ID; representative failure/video |

## Reproducibility details that affect the claims

- CDPR: full config and commit; base VLA and inherited LoRA checkpoints; residual dimensions/scale; chunk length, replanning interval and real Hz; observation tensors (especially hardware availability); action mapping/units/workspace limits; exact approach/pickup/placement events and termination; masks, group size, advantage normalization, stage masses, bonus/protection values, Gaussian/offset schedules, PPO/Adam/entropy settings.
- Bootstrapping: skill order, teacher policies, source checkpoint for each stage, scripted/oracle assistance, successful trajectory counts, SFT objective/epochs, and total inherited interaction/compute. Claims about no *new target-robot teleoperation* must not become claims of no demonstrations or human engineering anywhere in the lineage.
- LIBERO: source TeX/figures if available; precise OpenVLA-OFT/token policy implementation; phase predicates and equations; simulator/benchmark versions; group size (aggregate batch 32 is not automatically GRPO group size); reward aggregation and discount; raw evaluation and filter logs.
- Hardware: dimensions, cable actuation/control/state estimation, camera models/poses/calibration, object/receptacle specification, reset distribution, policy checkpoint, frequency/latency, any adaptation, trial ordering/stopping, intervention policy, success rubric, and videos.
- Statistics: retain scene/layout/session IDs. Repeated rollouts are not independent training seeds. Pair comparisons by scene and cluster repeated layouts/sessions as appropriate. Existing scene-paired intervals are quoted from source analyses; raw episodes are needed to recompute them. Do not attach binomial intervals treating the 2,048 CDPR attempts as independent scenes.

## Mechanism measurements if the mathematical claims remain central

Capture `(policy_sample, executed_action, behavior_mean_offset, old_log_prob, mean, log_std, stage, masks)` from actual training records to audit the likelihood. Supply scene-level milestone probabilities and attempted/mixed/retained group counts. For the protection claim, collect protected/symmetric prefix retention and stage-gradient interference. A neural improvement certificate would additionally require independent directional alignment and valid smoothness/error bounds; neither is currently available. The draft keeps these as limitations, not proven robot guarantees.

## Intake files

Header-only schemas in `data_templates/`:

- `cdpr_eval_episodes.csv`: one row per attempt/checkpoint.
- `libero_eval_episodes.csv`: one row per benchmark evaluation episode.
- `training_timeseries.csv`: time, actions, update and acceptance counters with explicit scope.
- `physical_trials.csv`: every physical attempt, including failures and aborts.
- `run_manifest.json`: placeholders for configuration, selection, lineage and deployment provenance.

Use blank for unknown values, not zero. Set booleans to 0/1. Use seconds/metres/radians where indicated, stable IDs across paired evaluations, and timezone-qualified timestamps. Store raw observations and videos separately and reference their paths. No remote training, final-test access, or hardware execution has been launched by this writing task.
