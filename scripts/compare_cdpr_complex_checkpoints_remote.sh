#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-openvla-oft}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/cdpr_openvla_grpo_complex_tasks_20260622_142556}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/rl}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-step_0192000=$CHECKPOINT_ROOT/step_0192000 step_0336000=$CHECKPOINT_ROOT/step_0336000}"
COMPARISON_NAME="${COMPARISON_NAME:-cdpr_checkpoint_comparison_$(date +%Y%m%d_%H%M%S)}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/evaluation_comparisons/$COMPARISON_NAME}"
EVALUATION_SCOPE="${EVALUATION_SCOPE:-simple}"
DENSE_GATE_THRESHOLD="${DENSE_GATE_THRESHOLD:-0.70}"
CONTINUE_ON_EVAL_ERROR="${CONTINUE_ON_EVAL_ERROR:-true}"

# Success/failure video coverage is useful but can block low-success checkpoints after all metrics
# have already been written. For comparisons, default to metric completion.
export REQUIRE_COMPLETE_VIDEO_COVERAGE="${REQUIRE_COMPLETE_VIDEO_COVERAGE:-false}"
export STRICT_VIDEO_VALIDATION="${STRICT_VIDEO_VALIDATION:-true}"

case "$EVALUATION_SCOPE" in
  simple|dense|first_stage)
    scope_args=(
      --instruction-types
      move_left
      move_right
      move_top
      move_bottom
      move_up
      move_down
      move_to_object
      open_gripper
      close_gripper
      rotate_gripper_clockwise
      rotate_gripper_counterclockwise
      --no-evaluate-reverse-shells
      --no-include-synonyms
      --arbitrary-instructions-count 0
    )
    ;;
  complex|sparse|second_stage)
    scope_args=(
      --instruction-types
      move_to_object
      grab_object
      pick_up
      push_left
      push_right
      push_forward
      push_backward
      put_into_plate
      move_left_of_object
      move_right_of_object
      move_in_front_of_object
      move_behind_object
      put_in_front_of_object
      put_behind_object
      move_between_objects
    )
    ;;
  all|comprehensive)
    scope_args=()
    ;;
  *)
    echo "Unknown EVALUATION_SCOPE: $EVALUATION_SCOPE" >&2
    echo "Supported scopes: simple, complex, all" >&2
    exit 2
    ;;
esac

mkdir -p "$COMPARISON_ROOT/evals"
status_file="$COMPARISON_ROOT/evaluation_status.tsv"
printf 'label\tcheckpoint_dir\trun_dir\texit_code\n' > "$status_file"

read -r -a checkpoint_specs <<< "$CHECKPOINT_SPECS"
if [[ ${#checkpoint_specs[@]} -eq 0 ]]; then
  echo "CHECKPOINT_SPECS is empty." >&2
  exit 2
fi

cd "$REPO_ROOT"

run_specs=()
failures=()
for spec in "${checkpoint_specs[@]}"; do
  if [[ "$spec" == *=* ]]; then
    label="${spec%%=*}"
    checkpoint_dir="${spec#*=}"
  else
    checkpoint_dir="$spec"
    label="$(basename "$checkpoint_dir")"
  fi
  if [[ -z "$label" || -z "$checkpoint_dir" ]]; then
    echo "Invalid checkpoint spec: $spec" >&2
    exit 2
  fi

  run_dir="$COMPARISON_ROOT/evals/$label"
  echo "Evaluating $label: $checkpoint_dir"
  set +e
  CHECKPOINT_DIR="$checkpoint_dir" \
  RUN_DIR="$run_dir" \
  EVAL_OUTPUT_ROOT="$COMPARISON_ROOT/evals" \
    bash scripts/evaluate_cdpr_complex_checkpoint_remote.sh "${scope_args[@]}" "$@"
  status=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$label" "$checkpoint_dir" "$run_dir" "$status" >> "$status_file"

  if [[ -f "$run_dir/validation_manifest.json" ]]; then
    run_specs+=("$label=$run_dir")
  else
    failures+=("$label:$status:missing_manifest")
  fi
  if [[ "$status" -ne 0 ]]; then
    failures+=("$label:$status")
    if [[ "$CONTINUE_ON_EVAL_ERROR" != "true" ]]; then
      echo "Stopping after failed evaluation for $label (exit $status)." >&2
      exit "$status"
    fi
  fi
done

if [[ ${#run_specs[@]} -eq 0 ]]; then
  echo "No completed validation manifests found; cannot write comparison report." >&2
  exit 5
fi

conda run --no-capture-output -n "$ENV_NAME" \
  python3 -m rl_vla_bootstrapping.cli.compare_cdpr_checkpoint_evaluations \
  --output-dir "$COMPARISON_ROOT" \
  --dense-gate-threshold "$DENSE_GATE_THRESHOLD" \
  "${run_specs[@]}"

echo "Evaluation status: $status_file"
echo "Comparison root: $COMPARISON_ROOT"

if [[ ${#failures[@]} -gt 0 ]]; then
  printf 'Evaluation failures: %s\n' "${failures[*]}" >&2
  exit 1
fi
