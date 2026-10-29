# Data-lineage audit of the reported CDPR controller — 29 October 2026

This audit answers checklist items 1 and 2 of `../REVISION_CHECKLIST_20261029.md`: what the 70.61% controller is made of, and what target-embodiment data its training ancestry used. It is a reading of local reports, launchers, configurations, and code. No model bytes were downloaded and no training or evaluation was run.

Primary source: `../../CDPR_CONSOLIDATED_PROGRESS_REPORT.md` (cited as **CPR** with line numbers at the time of reading), plus the launchers and configurations named in each row.

## Verdict

| Question | Answer | Confidence |
|---|---|---|
| Human teleoperation demonstrations on the CDPR | **None found** in any ancestor. The report, launchers, and configurations do not mention teleoperation, and there is no human-collected CDPR dataset in the repository. The author must still confirm the count of zero before the paper says "zero human demonstrations". | High for the documented ancestry; author confirmation pending |
| Only the learner's own RL trajectories | **No.** Two ancestor stages trained by supervised regression on automated teacher data: (a) **scripted-oracle** composed pick-and-place demonstrations (Phase 6/7 seeds); (b) a **policy-plus-scripted-controller** three-stage bank (Arm C), recorded by chaining earlier learned checkpoints with a yaw tail controller, a gripper-open hold, and a descent controller, then relabelled to the final prompt. | High |
| "No demonstration dataset of any kind" | **Not supported.** The automated oracle and controller banks are demonstration datasets, even though no human produced them. | High |
| Training-only assistance | Yes: object-pre-grasped ("caught") placement starts, aligned-gripper pick-up starts, distance curricula, and assisted handoffs during teacher-bank collection. None of these is active in the reported full-task evaluation. | High |
| Reported evaluation | Empty-gripper start, one full-task instruction, no teacher, no servo, no stage switching, in-run development validation panel. | High |

Wording supported by this lineage: *"without collecting human demonstrations on the target robot; supervised stages in the lineage use only automatically generated data (successful policy rollouts, a scripted simulator oracle, and policy rollouts with scripted controller assistance)."*

Not supported: "zero-demonstration", "demonstration-free", "only the learner's own RL trajectories", "no demonstration data of any kind".

## Lineage table

Read top to bottom, from the pretrained model to the reported endpoint. "Supervised targets" are trajectories whose actions the policy regressed onto. "Start states only" means the data or assistance shaped where episodes began but did not provide action targets. Interaction counts are selected environment actions where reported; "n/r" means not reported in the local evidence.

| # | Checkpoint (run) | Parent | Objective | Data origin | Assistance / special starts | Interaction budget | Source |
|---|---|---|---|---|---|---|---|
| 0 | `lerobot/smolvla_base` + new residual MLP + new `log_std` | — | Pretrained VLA (frozen); residual created by `nn.Linear` default init | SmolVLA pretraining: 481 community LeRobot datasets, 22.9k episodes, documented as SO100 data | — | — | `policy/smolvla_cdpr.py:674–680`; `policy/octo_finetune_cdpr.py:765–817`; SmolVLA paper |
| 1 | move-to scratch `…move_to_scratch_mjwarp_w512_20260719_081705/rl/step_5000081` | 0 | GRPO, dense distance reward | Learner's own rollouts | Distance curriculum | 5.0M | `ACTION_MODULE_AUDIT_20261022.md`; campaign report lines 134–153, 231–232 |
| 2 | Phase 1 pick-up source | 1 | RL (dense) | Learner's own rollouts | **Aligned-gripper starts** above the object; cap 0.06 m | n/r | CPR §4.2 |
| 3 | Phase 2 placement (15M) | earlier phase | RL only, from a frozen VLA prior | Learner's own rollouts | **Caught starts** (object already held) | 15M | CPR table l.317 |
| 4 | Phase 4/5 retention cycles 1–3 (`sft_cycle1..3`) alternating with family RL | 3 | Residual SFT on a balanced bank of **successful policy rollouts** (move-to, pick-up, placement), then family GRPO | Learned-policy trajectories, including the separate Phase 1 pick-up branch | Bank episodes from caught/aligned starts | n/r per cycle | CPR §5 |
| 5 | `runs/phase4_bank/sft_phase6` | `sft_cycle3` | Residual SFT, 45 epochs, 5,867 episodes | Cycle bank **plus scripted-oracle composed demonstrations** (`sil_record --mode oracle`): oracle supplies the five executed action values; states/priors from the checkpoint | Oracle drives the plant | SFT only | CPR §8.2–8.3, l.4626–4643 |
| 6 | `runs/phase4_bank/sft_phase7` | `sft_phase6` | Residual SFT, re-harvested pool | Pooled bank with a **re-harvested oracle composed slice** (~16% composed share, knob off) | Oracle drives the plant | SFT only | `scripts/run_cdpr_phase7_reharvest.sh`; `scripts/run_cdpr_phase7_composed_fraction_sweep.sh:12–15` |
| 7 | `phase7_sparse_joint_20260904_212930/rl/step_2017690` | 6 (weights only) | GRPO, one sparse binary reward, four instructions | Learner's own rollouts | Mixed caught/composed container starts; curricula | ~2.0M | `scripts/train_cdpr_phase7_sparse_joint_remote.sh:47`; CPR l.4522–4540 |
| 8 | `release_recovery_pilot_20260907_193019/rl/step_0527307` | 7 (weights only) | GRPO, sparse binary | Learner's own rollouts | **Aligned gripper starts** for containers; fixed caps | 0.53M | CPR l.4320–4340 |
| 9 | `release_recovery_continue_3m_20260908_102004/rl/step_2117145` | 8 (full resume) | GRPO, sparse binary | Learner's own rollouts | As 8 | to 2.12M of 3.54M cumulative | CPR l.4252–4300, l.3213–3240 |
| 10 | `three_stage_expanded_bowl_init_bc_e1/sft_armC/sil_sft_adapter.pt` (expanded Arm C) | 9 | Residual SFT (1 epoch), then **action-expert LoRA SFT** (112 modules, 3,416,064 params, 8 epochs) | Three-stage bank, 31,892 rows / 842 scenes: **chained learned teacher checkpoints with scripted controller assistance** (yaw tail controller, single-channel gripper-open hold, descent controller); instructions **relabelled** to the final prompt; priors refreshed under 9 | Stage handoffs between teachers; `action_source` marks substituted actions | SFT only; collection 3,968 chains + 325-scene expansion | CPR §7.16, l.3103–3135, l.3249–3300 |
| 11 | `three_stage_sparse_grpo_*` → `step_28309431` → `step_52791642` | 10 | Staged-sparse GRPO, three ordered binary milestones, LoRA frozen, residual only | Learner's own rollouts | **None**: empty start, one final prompt from step 0 | ~57M cumulative (10.07M + 18.2M + ~27M) | CPR l.2961–2990, l.2861–2875, l.2727–2745 |
| 12 | `three_stage_sparse_grpo_20260925_105132/rl/step_56072006` | 11 | Same, grasp-preservation continuation | Learner's own rollouts | None | ~3.3M | CPR l.2595–2605 |
| 13 | `latent_2m_control_20261009_033534/rl/step_2052558` | 12 | Bounded-residual GRPO with offset-conditioned pre-clipping likelihood; fresh optimizer; VLA/LoRA frozen | Learner's own rollouts | None | 2.05M (new counter) | CPR l.4799–4830 |
| 14 | `latent_control_long_20261009_160735` → final20m → `latent_control_to40m_20261012_230538/rl/step_40056193` | 13 | Same | Learner's own rollouts | None | to 40.06M | CPR l.4830–4968 |
| 15 | **Reported endpoint**, step `60025885` (70.61%) | 14 | Same, LR 1e-5 | Learner's own rollouts | None | +19.97M → 60.03M since conversion | CPR l.4969–4994; `latent_control_60m_tensorboard_20261028.json` |

Rejected branches that are **not** ancestors: the step_52791642 strict-success SFT arms (with and without retention), the zero-initialized correction candidate, the Phase 7 SFT sweep arms, the demonstration-start GRPO pilot, and the z-offset pilot.

## Data categories, reported separately

| Category | Present in the lineage? | Role | Stages |
|---|---|---|---|
| Human teleoperation on the CDPR | Not found | — | — |
| Human interventions during rollouts | Not found | — | — |
| Scripted-oracle trajectories | **Yes** | Supervised action targets | 5, 6 |
| Learned-policy trajectories with scripted controller assistance | **Yes** | Supervised action targets (relabelled) | 10 |
| Learned-policy successes (self-imitation) | Yes | Supervised action targets | 4 (and the source policies of 10) |
| Learner's own RL rollouts | Yes | RL return / advantage | 1–3, 7–9, 11–15 |
| Assisted starts (caught object, aligned gripper, curricula) | Yes | Start states only | 2, 3, 4, 7–9 |
| Pretrained robot data (SO100 community datasets) | Yes, via SmolVLA | Pretraining only; not target-embodiment data | 0 |

## Module table (for checklist item 1)

| Module | Pretrained source | Initialization | Trained in | Frozen in | Parameters |
|---|---|---|---|---|---|
| SmolVLA VLM backbone | `lerobot/smolvla_base` | Pretrained | Never | All stages | ≈350M (450M total minus expert, as documented by the SmolVLA paper) |
| SmolVLA action expert | `lerobot/smolvla_base` | Pretrained; never reset | Never directly | All stages | ≈100M (documented) |
| Action-expert LoRA (rank 16, α 32, 112 modules) | — | A Kaiming, B zero (no initial change) | Supervised stage 10 (and possibly earlier LoRA RL; per-stage use not reconstructed) | Stages 11–15 (`vla_lora_updates_enabled: false`; logged LoRA change 0) | 3,416,064 |
| Visual feature projection | — | **Fixed random Gaussian** (seed 20260723), 2 cameras × 16 tokens × 960 → 512 | Never | All stages | 15.7M fixed, non-trainable |
| Residual MLP 558→1024→1024→40 | — | **New**, PyTorch default `nn.Linear` init at stage 1 | All RL and SFT stages | — | 1,663,016 |
| Exploration `log_std` (8×5) + episode offset | — | Constant −1.2 | RL stages | — | 40 (offset std is a configuration constant) |

Residual count: 558·1024+1024 + 1024·1024+1024 + 1024·40+40 = 1,663,016.

## Action interface and low-level controller (for checklist item 5)

`robots/cdpr/cdpr_mujoco/policy_control.py:81–130`: each executed action is a normalized 5-vector in [−1, 1] → Cartesian end-effector increment (×0.015 m per axis), yaw increment (×0.08 rad), and gripper-target increment (×0.05). The end-effector target goes to `set_target_position`, whose controller computes cable lengths by **inverse kinematics** for four spatial tendons driven by position-controlled sliders (`headless_cdpr_egl.py:797–803`; `cdpr.xml:291–340`; 2 ms MJWarp timestep). The policy therefore learns the mapping from images and language to Cartesian increments. Cable-length control is a fixed, hand-built kinematic controller, not learned. The policy emits 8-slot chunks and executes 4 slots per replan.

The control frequency is `1 / (dt · (hold_steps + 1))` in `policy_control.py:38–45`. With the YAML's `hold_steps: 6` and a 2 ms step, this would be ≈71 Hz. The MJWarp collector's effective setting must be read from the resolved run configuration before the paper states it.

## Records still missing

- Resolved configurations and code revisions for every row. The report gives commits for some (e.g. `3409d4c`, `bb37ca3`, `e3a2740`, `2b156603…`), but the launch manifests are on the training host.
- Residual initialization seed at stage 1. Not recorded locally.
- Adapter hashes for rows 1–12. Only rows 12–15 have reported SHA-256.
- Interaction counts for stages 2–6 and for oracle and teacher-bank collection. Without them, total adaptation interaction cannot be summed.
- Number of distinct scenes behind the 1,024-episode validation panel, and whether panel scenes overlap with selection.
- Author confirmation that no human teleoperation or manual intervention entered any bank.
