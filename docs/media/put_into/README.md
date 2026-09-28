# Full-task `put_into` episodes

Eight **strict** successes, one per object × receptacle, from the standalone
evaluation of `runs/three_stage_sparse_grpo_20260914_195542/rl/step_28309431`
on 2026-09-17 (evaluator commit `d0e802d`; 57/256 strict overall). The
evaluation is the consolidated report's 2026-09-17 §14 entry.

Protocol: an empty gripper and one instruction (`put <object> into <plate|bowl>`)
from the first action. There is no teacher, servo or stage switch. The scenes
come from the held-out `student_validation` split. Each frame shows the
overview camera on the left and the wrist camera on the right, one frame per
executed action at 20 fps.

| File | Instruction | Original video | Env steps |
|---|---|---|---:|
| `apple_into_plate` | put apple into plate | `strict_plate_robocasa_apple_r06_w001` | 181 |
| `orange_into_plate` | put orange into plate | `strict_plate_robocasa_orange_r02_w017` | 174 |
| `potato_into_plate` | put potato into plate | `strict_plate_robocasa_potato_r03_w010` | 218 |
| `tomato_into_plate` | put tomato into plate | `strict_plate_robocasa_tomato_r00_w009` | 197 |
| `apple_into_bowl` | put apple into bowl | `strict_bowl_robocasa_apple_r05_w008` | 163 |
| `orange_into_bowl` | put orange into bowl | `strict_bowl_robocasa_orange_r00_w025` | 179 |
| `potato_into_bowl` | put potato into bowl | `strict_bowl_robocasa_potato_r02_w019` | 235 |
| `tomato_into_bowl` | put tomato into bowl | `strict_bowl_robocasa_tomato_r05_w014` | 132 |

`*.mp4` is the original evaluator output, `*.json` its per-episode record (event
frames for approach, grasp, lift and release), `*.gif` a 400 px / 8 fps preview
for the README, and `*.jpg` the final frame, used as a poster.
