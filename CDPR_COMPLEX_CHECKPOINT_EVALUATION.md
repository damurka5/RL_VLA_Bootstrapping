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
on each task's final normal-reset shell, and eight additional free-form prompts are recorded as
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
- `instruction_shell_success_rates.csv`
- `instruction_prompt_success_rates.csv`
- `evaluation_case_success_rates.csv`
- `target_object_success_rates.csv`
- `instruction_text_success_rates.csv`
- `episode_results.csv`
- `video_coverage.csv`
- `video_validation.csv` and `video_validation.json`
- `videos/`
- `evaluation.log` with the exact shell-escaped command

All MP4 files are checked with `ffprobe`. By default the launcher exits non-zero after saving all
reports if a video is invalid or if any instruction still lacks either a successful or failed
rollout video.

Useful overrides:

```bash
EPISODES_PER_CASE=30 \
VIDEO_SEARCH_EXTRA_EPISODES=80 \
REQUIRE_COMPLETE_VIDEO_COVERAGE=false \
bash scripts/evaluate_cdpr_complex_checkpoint_remote.sh
```

Additional validator arguments can be appended to the launcher command.
