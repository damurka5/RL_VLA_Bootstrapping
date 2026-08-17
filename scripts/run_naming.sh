#!/usr/bin/env bash
# Run naming and run-directory safety, shared by the training launchers.
#
# Every launcher already builds a timestamped default run name. The hole is
# that passing RUN_NAME explicitly -- the obvious thing to do when you want a
# readable name -- silently discards the timestamp, and `mkdir -p` then writes
# the new run straight into the old run's directory. What collides there is not
# cosmetic: `latest.pt` is overwritten, `step_XXXXXXX/` directories from two
# different runs interleave under one tree, and `tee` truncates train.log. A
# later resume can then pick a checkpoint written by a different config.
#
# So: give the label, get the timestamp for free, and refuse to start on top of
# an existing run unless that is explicitly what you meant.
#
#   RUN_LABEL=phase4_move_to_iter0   -> phase4_move_to_iter0_20260817_141530
#   RUN_NAME=exactly_this            -> exactly_this   (verbatim, still guarded)
#   neither                          -> <launcher default>_<timestamp>
#
# ALLOW_EXISTING_RUN_DIR=1 permits writing into a populated directory.

# Compose the run name. $1 is the launcher's own default prefix.
cdpr_compose_run_name() {
  local default_prefix="$1"
  local timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
  if [[ -n "${RUN_NAME:-}" ]]; then
    # An exact name was asked for. Honour it -- resuming into a known directory
    # is a real need -- but say so, because the timestamp is being dropped.
    if [[ -z "${RUN_LABEL:-}" ]]; then
      echo "[run-naming] RUN_NAME set explicitly: '$RUN_NAME' (no timestamp appended; use RUN_LABEL to get one)" >&2
    else
      echo "[run-naming] RUN_NAME and RUN_LABEL are both set; RUN_NAME wins and RUN_LABEL='$RUN_LABEL' is ignored." >&2
    fi
    printf '%s' "$RUN_NAME"
    return 0
  fi
  if [[ -n "${RUN_LABEL:-}" ]]; then
    printf '%s_%s' "$RUN_LABEL" "$timestamp"
    return 0
  fi
  printf '%s_%s' "$default_prefix" "$timestamp"
}

# Abort if $1 already holds a run, unless ALLOW_EXISTING_RUN_DIR=1.
cdpr_guard_run_dir() {
  local run_dir="$1"
  [[ -d "$run_dir" ]] || return 0
  # Checkpoints and the training log are the artifacts a second run destroys.
  # A bare directory (someone ran with DRY_RUN, or mkdir'd it by hand) is not a
  # collision and must not block a launch.
  local existing
  existing="$(find "$run_dir" -maxdepth 3 \
    \( -name 'latest.pt' -o -name '*_adapter.pt' -o -name 'train.log' \) \
    -print -quit 2>/dev/null || true)"
  [[ -n "$existing" ]] || return 0
  if [[ "${ALLOW_EXISTING_RUN_DIR:-0}" == "1" ]]; then
    echo "[run-naming] WARNING: reusing populated run dir $run_dir (ALLOW_EXISTING_RUN_DIR=1); existing checkpoints and train.log may be overwritten." >&2
    return 0
  fi
  cat >&2 <<EOF
[run-naming] Refusing to start into a run directory that already holds a run:
  $run_dir
  found: $existing

A second run here overwrites latest.pt and train.log and interleaves
step_*/ directories from two different runs, after which a resume can load a
checkpoint written by a different config.

Use RUN_LABEL=<name> to get <name>_<timestamp>, or set
ALLOW_EXISTING_RUN_DIR=1 if overwriting is what you meant.
EOF
  return 2
}
