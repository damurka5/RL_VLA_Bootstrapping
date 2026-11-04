# WidowX-200 pipeline evidence snapshot

This is an interim snapshot of one continuing training lineage. Best RL validation scores are checkpoint-selection results, not independent final-test estimates.
Placement skill tests can start with an already-held object. Replay survival is conditional on original success; it is not task success.

| Stage | Status | Best RL selection metric | Value | Step | Stop | SIL accepted/attempted |
|---|---|---|---:|---:|---|---:|
| s1_move_to_dense | done | validation/success_rate | 48.5% | 7757735 | plateau | 0/0 |
| s2_pick_up_dense | done | validation/success_rate | 12.1% | 501004 | plateau | 0/0 |
| s3_put_into_dense | done | min_by_instruction | 71.9% | 7505414 | plateau | 0/1 |
| s4_move_to_binary | done | validation/success_rate | 56.9% | 4510257 | plateau | 0/1 |
| s5_pick_up_binary | done | validation/success_rate | 0.0% | 2258951 | plateau | 2/2 |
| s6_put_into_binary | done | min_by_instruction | 65.8% | 1016667 | user_checkpoint | 0/1 |
| s7_joint_staged | done | min_by_instruction_composed | 6.7% | 7504473 | plateau | 0/0 |

## Retained banks

| Skill | Source stage | Instruction | Source harvest successes/episodes | Replays | Frame files |
|---|---|---|---:|---:|---:|
| move_to | s4_move_to_binary | move_to_object | 111/2048 | 4 | 4 |
| pick_up | s2_pick_up_dense | pick_up | 200/2048 | 4 | 4 |
| put_into | s3_put_into_dense | put_into_bowl | 369/992 | 4 | 4 |
| put_into | s3_put_into_dense | put_into_plate | 486/1056 | 4 | 4 |

## Skill evaluations before/after retention SFT

| Stage | Evaluation | Candidate accepted | Instruction | Successes/episodes | Rate |
|---|---|---|---|---:|---:|
| s2_pick_up_dense | before_sil | - | move_to_object | 405/1024 | 39.6% |
| s2_pick_up_dense | before_sil | - | pick_up | 140/1024 | 13.7% |
| s3_put_into_dense | before_sil | - | move_to_object | 55/1024 | 5.4% |
| s3_put_into_dense | before_sil | - | pick_up | 0/1024 | 0.0% |
| s3_put_into_dense | before_sil | - | put_into_bowl | 316/520 | 60.8% |
| s3_put_into_dense | before_sil | - | put_into_plate | 372/504 | 73.8% |
| s3_put_into_dense | after_sil | - | move_to_object | 55/1024 | 5.4% |
| s3_put_into_dense | after_sil | - | pick_up | 0/1024 | 0.0% |
| s3_put_into_dense | after_sil | - | put_into_bowl | 316/520 | 60.8% |
| s3_put_into_dense | after_sil | - | put_into_plate | 372/504 | 73.8% |
| s3_put_into_dense | sil_1_candidate | False | move_to_object | 139/1024 | 13.6% |
| s3_put_into_dense | sil_1_candidate | False | pick_up | 2/1024 | 0.2% |
| s3_put_into_dense | sil_1_candidate | False | put_into_bowl | 180/520 | 34.6% |
| s3_put_into_dense | sil_1_candidate | False | put_into_plate | 281/504 | 55.8% |
| s4_move_to_binary | before_sil | - | move_to_object | 551/1024 | 53.8% |
| s4_move_to_binary | before_sil | - | pick_up | 0/1024 | 0.0% |
| s4_move_to_binary | before_sil | - | put_into_bowl | 103/520 | 19.8% |
| s4_move_to_binary | before_sil | - | put_into_plate | 217/504 | 43.1% |
| s4_move_to_binary | after_sil | - | move_to_object | 551/1024 | 53.8% |
| s4_move_to_binary | after_sil | - | pick_up | 0/1024 | 0.0% |
| s4_move_to_binary | after_sil | - | put_into_bowl | 103/520 | 19.8% |
| s4_move_to_binary | after_sil | - | put_into_plate | 217/504 | 43.1% |
| s4_move_to_binary | sil_1_candidate | False | move_to_object | 373/1024 | 36.4% |
| s4_move_to_binary | sil_1_candidate | False | pick_up | 1/1024 | 0.1% |
| s4_move_to_binary | sil_1_candidate | False | put_into_bowl | 209/520 | 40.2% |
| s4_move_to_binary | sil_1_candidate | False | put_into_plate | 291/504 | 57.7% |
| s5_pick_up_binary | before_sil | - | move_to_object | 0/1024 | 0.0% |
| s5_pick_up_binary | before_sil | - | pick_up | 0/1024 | 0.0% |
| s5_pick_up_binary | before_sil | - | put_into_bowl | 62/520 | 11.9% |
| s5_pick_up_binary | before_sil | - | put_into_plate | 122/504 | 24.2% |
| s5_pick_up_binary | after_sil | - | move_to_object | 372/1024 | 36.3% |
| s5_pick_up_binary | after_sil | - | pick_up | 1/1024 | 0.1% |
| s5_pick_up_binary | after_sil | - | put_into_bowl | 204/520 | 39.2% |
| s5_pick_up_binary | after_sil | - | put_into_plate | 329/504 | 65.3% |
| s5_pick_up_binary | sil_1_candidate | True | move_to_object | 311/1024 | 30.4% |
| s5_pick_up_binary | sil_1_candidate | True | pick_up | 0/1024 | 0.0% |
| s5_pick_up_binary | sil_1_candidate | True | put_into_bowl | 198/520 | 38.1% |
| s5_pick_up_binary | sil_1_candidate | True | put_into_plate | 294/504 | 58.3% |
| s5_pick_up_binary | sil_2_candidate | True | move_to_object | 372/1024 | 36.3% |
| s5_pick_up_binary | sil_2_candidate | True | pick_up | 1/1024 | 0.1% |
| s5_pick_up_binary | sil_2_candidate | True | put_into_bowl | 204/520 | 39.2% |
| s5_pick_up_binary | sil_2_candidate | True | put_into_plate | 329/504 | 65.3% |
| s6_put_into_binary | before_sil | - | move_to_object | 9/1024 | 0.9% |
| s6_put_into_binary | before_sil | - | pick_up | 0/1024 | 0.0% |
| s6_put_into_binary | before_sil | - | put_into_bowl | 165/520 | 31.7% |
| s6_put_into_binary | before_sil | - | put_into_plate | 267/504 | 53.0% |
| s6_put_into_binary | after_sil | - | move_to_object | 9/1024 | 0.9% |
| s6_put_into_binary | after_sil | - | pick_up | 0/1024 | 0.0% |
| s6_put_into_binary | after_sil | - | put_into_bowl | 165/520 | 31.7% |
| s6_put_into_binary | after_sil | - | put_into_plate | 267/504 | 53.0% |
| s6_put_into_binary | sil_1_candidate | False | move_to_object | 12/1024 | 1.2% |
| s6_put_into_binary | sil_1_candidate | False | pick_up | 0/1024 | 0.0% |
| s6_put_into_binary | sil_1_candidate | False | put_into_bowl | 165/520 | 31.7% |
| s6_put_into_binary | sil_1_candidate | False | put_into_plate | 251/504 | 49.8% |

## Interpretation and files

- `stages.csv` separates the best RL checkpoint, the selected RL checkpoint (including a user override), and the policy handed to the next stage after any accepted SIL. `checkpoint_selection` identifies user choices; `best_rl_*` remain validation maxima.
- `validation.csv` and `validation_slices.csv` contain learning curves, instruction/object slices, episode denominators, and composed legs when available.
- `training.csv` holds the logged rollout/curriculum/throughput diagnostics; training success is not fixed-distance validation success.
- `record_rounds.csv` contains per-round harvest and retention evaluations. Pooled sharded wrappers are not counted again. A false completeness flag identifies a partial request; blank means no sharded-completeness claim.
- `replay_survival.csv` reports survivors among source successes. `banks.csv` inventories the banks retained by the controller. Marker sums cover all instructions in a bank and are not instruction-specific demo counts.
- `raw/` preserves source JSON/JSONL/CSV and configs. `manifest.json` records their paths, SHA-256 digests, capture time and warnings. No checkpoints, images or trajectory arrays are included.
- Successive validations reuse evaluation seeds. Do not pool them as independent tests or treat stages as independent training seeds. Perform a separate fixed-checkpoint evaluation for final reported performance.
- Composition SFT is reported separately below when recorded; its relabelled skill samples are stage supervision, not certified continuous full-task trajectories.

Controller final report recorded: /root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201/final_report.json

## Pre-joint composition SFT

Recorded status: done; identifier: 5af896659773bb52
Initializer: /root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s6_put_into_binary/rl/step_1764679/smolvla_grpo_adapter.pt
SFT policy: /root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201/composition_sft/5af896659773bb52/sft/sil_sft_adapter.pt

| Evaluation | Protocol | Instruction | Successes/episodes | Rate |
|---|---|---|---:|---:|
| before | skill | move_to_object | 6/1024 | 0.6% |
| before | skill | pick_up | 0/1024 | 0.0% |
| before | skill | put_into_bowl | 172/520 | 33.1% |
| before | skill | put_into_plate | 264/504 | 52.4% |
| before | composed | put_into_bowl | 0/520 | 0.0% |
| before | composed | put_into_plate | 11/504 | 2.2% |
| after | skill | move_to_object | 26/1024 | 2.5% |
| after | skill | pick_up | 0/1024 | 0.0% |
| after | skill | put_into_bowl | 191/520 | 36.7% |
| after | skill | put_into_plate | 258/504 | 51.2% |
| after | composed | put_into_bowl | 0/520 | 0.0% |
| after | composed | put_into_plate | 8/504 | 1.6% |
