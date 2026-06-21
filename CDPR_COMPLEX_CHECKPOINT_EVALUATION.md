# Complex CDPR checkpoint evaluation

Run the comprehensive evaluation on the training server:

```bash
cd /root/repo/RL_VLA_Bootstrapping
bash scripts/evaluate_cdpr_complex_checkpoint_remote.sh
```

The launcher defaults to:

```text
/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_grpo_complex_tasks_20260618_221123/rl/step_0048000
```

It evaluates every instruction configured in
`configs/examples/cdpr_openvla_grpo_complex_tasks.yaml`. Reverse-frontier tasks are evaluated on
every shell. Canonical prompts are evaluated on every shell, two synonymous prompts are evaluated
on each task's final normal-reset shell, and ten additional free-form prompts are recorded as
qualitative checks.

Every validation scene uses three or four randomized objects. `move_to_object` remains stratified by
target object while retaining distractors, so its score tests whether the policy follows the named
object rather than moving toward the only visible item.

The evaluator first uses metric episodes to compute success rates. If a task is missing a success or
failure example, it can run extra canonical attempts exclusively for video coverage; these attempts
are marked `metric_episode=false` and never change the reported rates.

The output directory contains:

- `validation_report.md`
- `validation_manifest.json`
- `instruction_success_rates.csv`
- `normal_scene_canonical_success_rates.csv`
- `instruction_shell_success_rates.csv`
- `instruction_prompt_success_rates.csv`
- `evaluation_case_success_rates.csv`
- `target_object_success_rates.csv`
- `instruction_text_success_rates.csv`
- `move_to_object_threshold_sweep.csv`
- `episode_results.csv`
- `video_coverage.csv`
- `video_validation.csv` and `video_validation.json`
- `videos/`, including an `*_actions.csv` trace beside every recorded MP4
- `evaluation.log` with the exact shell-escaped command

Each recorded frame identifies whether the action is a newly generated OpenVLA output or a cached
action from the current chunk. It also shows the exact normalized five-dimensional action, its
scaled physical command, the policy-call count, chunk index, end-effector pose, yaw, and gripper
state. `episode_results.csv` and the video summary JSON files record exact OpenVLA call counts.

MP4 files are checked with `ffprobe` when available and with an `imageio` decode fallback otherwise.
By default the launcher exits non-zero after saving all reports if a video is invalid or if any
instruction still lacks either a successful or failed rollout video. Invalid reset states are
randomized and retried up to ten times by default. If every attempt fails, the raised error lists
the reason for each attempt; this is an environment/reset failure, not a policy failure.

Useful overrides:

```bash
EPISODES_PER_CASE=30 \
VIDEO_SEARCH_EXTRA_EPISODES=80 \
REQUIRE_COMPLETE_VIDEO_COVERAGE=false \
bash scripts/evaluate_cdpr_complex_checkpoint_remote.sh
```

Additional validator arguments can be appended to the launcher command.
