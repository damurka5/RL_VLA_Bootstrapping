# WidowX-200 pipeline widowx200_pipeline_20261024_131201

Final policy: `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s7_joint_staged/rl/step_7504473/smolvla_grpo_adapter.pt`

| evaluation | instruction | success | episodes |
|---|---|---:|---:|
| move_to | move_to_object | 0.333 | 1024 |
| pick_up | pick_up | 0.240 | 1024 |
| put_into | put_into_bowl | 0.138 | 520 |
| put_into | put_into_plate | 0.581 | 504 |
| put_into_composed | put_into_bowl | 0.062 | 520 |
| put_into_composed | put_into_plate | 0.081 | 504 |

## Stages

- **s1_move_to_dense**: best validation/success_rate=0.48535 at step 7757735; early stop plateau; SIL cycles 0; policy `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s1_move_to_dense/rl/step_7757735/smolvla_grpo_adapter.pt`
- **s2_pick_up_dense**: best validation/success_rate=0.12109 at step 501004; early stop plateau; SIL cycles 0; policy `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s2_pick_up_dense/rl/step_0501004/smolvla_grpo_adapter.pt`
- **s3_put_into_dense**: best min_by_instruction=0.71875 at step 7505414; early stop plateau; SIL cycles 1; policy `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s3_put_into_dense/rl/step_7505414/smolvla_grpo_adapter.pt`
- **s4_move_to_binary**: best validation/success_rate=0.56934 at step 4510257; early stop plateau; SIL cycles 1; policy `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s4_move_to_binary/rl/step_4510257/smolvla_grpo_adapter.pt`
- **s5_pick_up_binary**: best validation/success_rate=0.0 at step 2258951; early stop plateau; SIL cycles 2; policy `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201/sil/s5_pick_up_binary_sil2/sft/sil_sft_adapter.pt`
- **s6_put_into_binary**: best min_by_instruction=0.6582 at step 1016667; early stop user_checkpoint; SIL cycles 1; policy `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s6_put_into_binary/rl/step_1764679/smolvla_grpo_adapter.pt`
- **s7_joint_staged**: best min_by_instruction_composed=0.06696 at step 7504473; early stop plateau; SIL cycles 0; policy `/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201__s7_joint_staged/rl/step_7504473/smolvla_grpo_adapter.pt`
