# Reports

The canonical record of the project is
[`CDPR_CONSOLIDATED_PROGRESS_REPORT.md`](../../CDPR_CONSOLIDATED_PROGRESS_REPORT.md)
in the repository root. It holds the headline results, the method, what was
deliberately excluded, and a dated ledger of every experiment. Start there.

The reports below are its evidence chain and history, grouped by role.

## `campaign/` — the SmolVLA + MJWarp campaign, in order

These are the source reports the consolidated report cites (its §2.1). They
are ordered by phase.

| Phase | Report | Date | What it establishes |
|---|---|---|---|
| 0 | [`SMOLVLA_MOVE_TO_TRAINING_REPORT.md`](campaign/SMOLVLA_MOVE_TO_TRAINING_REPORT.md) | 2026-07-28 | `move_to` run history; the 16M-step collapse was `log_std` inflation. Also defines every TensorBoard metric |
| 0–1 | [`CDPR_SMOLVLA_CAMPAIGN_REPORT.md`](campaign/CDPR_SMOLVLA_CAMPAIGN_REPORT.md) | 2026-08-05 | `move_to` 28M steps; `pick_up` is limited by the frozen encoder's 3–5 cm localization |
| 1 | [`CDPR_MANIPULATION_TASK_CONSISTENCY_REPORT.md`](campaign/CDPR_MANIPULATION_TASK_CONSISTENCY_REPORT.md) | 2026-07-28 | Pre-launch audit: gripper geometry made grasping impossible as configured |
| 2 | [`CDPR_PLACEMENT_PHASE2_PREFLIGHT_REPORT.md`](campaign/CDPR_PLACEMENT_PHASE2_PREFLIGHT_REPORT.md) | 2026-08-13 | RL alone learns placement from zero: plate 0.71, bowl 0.32 at 15M steps |
| 3 | [`CDPR_PHASE3_SIL_REPORT.md`](campaign/CDPR_PHASE3_SIL_REPORT.md) | 2026-08-17 | Self-imitation toolchain (record → smooth → replay → SFT); leaving a task out of the mix erases it |
| 4 | [`CDPR_PHASE4_LOOP_DESIGN.md`](campaign/CDPR_PHASE4_LOOP_DESIGN.md) | 2026-08-19 | Design of the alternating RL ⇄ SFT loop (in Russian) |
| 4 | [`CDPR_MOVE_TO_VALIDATION_REPORT.md`](campaign/CDPR_MOVE_TO_VALIDATION_REPORT.md) | 2026-08-21 | Six-leg controlled validation: `move_to` 63.0% (645/1024) at 2 cm |
| 4 | [`CDPR_PHASE4_RETENTION_REPORT.md`](campaign/CDPR_PHASE4_RETENTION_REPORT.md) | 2026-08-26 | Retention bank of frames and actions; one policy for three instructions |
| 5–6 | [`CDPR_PHASE5_REPORT.md`](campaign/CDPR_PHASE5_REPORT.md) | 2026-08-31 | `pick_up` joins the single policy; a 4× larger slice breaks the retention ceiling; Cycle 3; composed-task seed |
| 8 | [`CDPR_THREE_STAGE_PUT_INTO_SFT_DESIGN.md`](campaign/CDPR_THREE_STAGE_PUT_INTO_SFT_DESIGN.md) | 2026-09-12 | Continuous `move_to → pick_up → put_into` chains, the SFT bank, and the full-task evaluator |

Phases 7 and later (sparse joint RL, staged-sparse GRPO, the 56M-step best
checkpoint and the September review) are recorded in §7–§8 and the §14 ledger of the
consolidated report.

## `plans/` — superseded plans from 2026-09-08 … 09-10

Kept because configs and the consolidated report (§7.15) refer to them.
Their outcome is recorded in the consolidated report.

| Plan | Status |
|---|---|
| [`CDPR_MANIPULATION_UPGRADE_PLAN.md`](plans/CDPR_MANIPULATION_UPGRADE_PLAN.md) | Asymmetric actor–critic proposal, **rejected** (the user chose to keep GRPO). The comparison utility remains |
| [`CDPR_NEXT_CAMPAIGN_PLAN.md`](plans/CDPR_NEXT_CAMPAIGN_PLAN.md) | "Four instructions above 70%" campaign contract and experiment order |
| [`CDPR_GRPO_DEMONSTRATION_PLAN.md`](plans/CDPR_GRPO_DEMONSTRATION_PLAN.md) | Demonstration-start GRPO. The 1M-action pilot ran and showed no gain, which led to the three-stage design |

## `side_tracks/`

| Report | What it is |
|---|---|
| [`SMOLVLA_LORA_FINETUNE_PLAN.md`](side_tracks/SMOLVLA_LORA_FINETUNE_PLAN.md) | Design of the action-expert LoRA path. It is implemented and used by the SFT stage |
| [`WIDOWX200_MIGRATION_REPORT.md`](side_tracks/WIDOWX200_MIGRATION_REPORT.md) | Port to an Interbotix WidowX-200 arm. The action contract transfers; batched training is not wired |

## `history/` — before the current method

| File | What it merges |
|---|---|
| [`OPENVLA_OCTO_ERA_SUMMARY.md`](history/OPENVLA_OCTO_ERA_SUMMARY.md) | 11 reports plus the old README (March – July 2026): OpenVLA-OFT PPO → GRPO, LC-HOL, reverse frontier, Octo-Small |
| [`SIMULATOR_CHOICE_SUMMARY.md`](history/SIMULATOR_CHOICE_SUMMARY.md) | The June 2026 MuJoCo audit and migration decision |

`history/legacy/` and `history/session_briefs/` hold the original texts those
summaries replace, plus three prompts written to open work sessions. They are
queued for deletion; the full texts stay in git history (commit `9339dd9`).

## Elsewhere in `docs/`

- [`research/CDPR_RESEARCH_THEORY.md`](../research/CDPR_RESEARCH_THEORY.md) — the mathematical draft of the dissertation framing
- [`cdpr_mjlab_architecture.md`](../cdpr_mjlab_architecture.md), [`cdpr_mjlab_remote.md`](../cdpr_mjlab_remote.md), [`cdpr_mjwarp_compatibility.md`](../cdpr_mjwarp_compatibility.md) — simulator backend and remote setup
- [`artifacts/`](../artifacts/) — small evaluation artifacts cited by the consolidated report
