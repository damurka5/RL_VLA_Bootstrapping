# CDPR MJLab remote setup, validation, benchmark, and operations

Commands below assume the checkout is
`/root/repo/RL_VLA_Bootstrapping`. Override `REPO_ROOT` if needed.

## Install

The setup script creates a Python 3.12 Conda environment, installs the exact
CUDA 12.8 lock, installs this repository editable without dependency drift,
runs `pip check`, and executes the strict preflight:

```bash
cd /root/repo/RL_VLA_Bootstrapping
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
  bash scripts/setup_cdpr_mjlab_remote.sh
```

To install without system packages or defer preflight:

```bash
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
  INSTALL_APT_DEPS=0 RUN_PREFLIGHT=0 \
  bash scripts/setup_cdpr_mjlab_remote.sh
```

The host must expose two NVIDIA A40s through `nvidia-smi`, use driver major
570 or newer, and have enough host RAM/VRAM for two full SmolVLA replicas.
Configure Hugging Face credentials/cache before the first model download when
the checkpoint is private or not already cached.

## Preflight and backend smoke

```bash
cd /root/repo/RL_VLA_Bootstrapping

conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/preflight_cdpr_mjlab.py \
  --config configs/examples/cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml \
  --require-gpus 2 \
  --worlds 16 \
  --output runs/mjlab_preflight.json

conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/smoke_cdpr_mjlab_backend.py \
  --config configs/examples/cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml \
  --worlds 16 \
  --device cuda:0 \
  --steps 8 \
  --output runs/mjlab_backend_smoke.json
```

The preflight is deliberately strict. A static XML pass is insufficient:
`put_model`, world allocation, forward dynamics, renderer creation, BVH refit,
both camera renders, and GPU tensor checks must execute.

Run the CPU regression and reward/group fixtures in the pinned environment:

```bash
conda run --no-capture-output -n cdpr-mjlab python3 -m pytest -q tests
```

## CPU/MJWarp parity

Run the identical-state scripted-action fixture on one GPU:

```bash
cd /root/repo/RL_VLA_Bootstrapping
MUJOCO_GL=egl conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/validate_cdpr_mjwarp_parity.py \
  --xml robots/cdpr/cdpr_mujoco/cdpr_mjwarp_smoke.xml \
  --device cuda:0 \
  --steps 24 \
  --output-dir runs/cdpr_mjwarp_parity
```

Review both `runs/cdpr_mjwarp_parity/parity.json` and `parity.md`. The default
limits are intentionally explicit and may be tightened only after the first
measured fixture; do not loosen them merely to pass. Camera MAE has an explicit
acceptance limit to catch orientation/channel mistakes while allowing renderer
differences; shape, order, dtype, RGB convention, range, and third-slot alias
are also hard checks.

## Two-GPU update/checkpoint/resume smoke

```bash
cd /root/repo/RL_VLA_Bootstrapping
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
  CUDA_VISIBLE_DEVICES=0,1 \
  WORLDS_PER_RANK=8 \
  BASE_CHECKPOINT=lerobot/smolvla_base \
  bash scripts/smoke_cdpr_mjlab_two_gpu.sh
```

This runs one process per GPU with `torchrun`, performs one update, checks
`latest.pt`, resumes it for another update, and asserts MJWarp checkpoint
metadata. Passing it demonstrates matching DDP backward schedules for the
observed smoke update; distributed schedule unit tests cover imbalanced and
zero-informative rank cases.

## End-to-end benchmark

```bash
cd /root/repo/RL_VLA_Bootstrapping
CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/benchmark_cdpr_mjlab_grpo.py \
  --repo-root "$PWD" \
  --worlds 8 16 32 64 128 \
  --updates 3 \
  --microbatch 16 \
  --base-checkpoint lerobot/smolvla_base \
  --cuda-visible-devices 0,1 \
  --output-dir runs/cdpr_mjlab_benchmark
```

Repeat with compiled inference after the eager sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/benchmark_cdpr_mjlab_grpo.py \
  --repo-root "$PWD" \
  --worlds 8 16 32 64 128 \
  --updates 3 \
  --microbatch 16 \
  --compile-model \
  --output-dir runs/cdpr_mjlab_benchmark_compile
```

Each world count gets an isolated two-rank process. The first update is
warmup/compile; the final update is reported. Failures such as OOM are retained
in the machine-readable sweep rather than aborting later sizes.
The benchmark enables synchronized CUDA component timers. Normal training
leaves those timing barriers disabled so instrumentation does not serialize the
GPU hot path.
`benchmark.json` records commands, logs, sampled and selected actions/s,
SmolVLA batch, physics/render/SmolVLA/reward/update/sync time, GPU utilization,
power, VRAM, CPU utilization, RAM, and comparison with the supplied CPU
end-to-end baseline. `benchmark.md` is the concise report. Physics-only FPS is
never presented as training speedup.

## Training

Start conservatively at 16 worlds/rank:

```bash
cd /root/repo/RL_VLA_Bootstrapping
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
  CUDA_VISIBLE_DEVICES=0,1 \
  WORLDS_PER_RANK=16 \
  SMOLVLA_MICROBATCH_SIZE=16 \
  MAX_TRAIN_STEPS=10000000 \
  bash scripts/train_cdpr_smolvla_complex_grpo_mjlab_dual_remote.sh
```

After benchmark selection, 64 worlds/rank means eight complete local groups on
each A40 and 128 server-wide candidates:

```bash
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
  CUDA_VISIBLE_DEVICES=0,1 \
  WORLDS_PER_RANK=64 \
  SMOLVLA_MICROBATCH_SIZE=16 \
  RUN_NAME=cdpr_mjwarp_w64_production \
  bash scripts/train_cdpr_smolvla_complex_grpo_mjlab_dual_remote.sh
```

Resume only a metadata-compatible checkpoint:

```bash
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
  CUDA_VISIBLE_DEVICES=0,1 \
  WORLDS_PER_RANK=64 \
  SMOLVLA_MICROBATCH_SIZE=16 \
  CHECKPOINT="$PWD/runs/cdpr_mjwarp_w64_production/rl/latest.pt" \
  RUN_NAME=cdpr_mjwarp_w64_resume \
  bash scripts/train_cdpr_smolvla_complex_grpo_mjlab_dual_remote.sh
```

To inspect the generated command without executing:

```bash
DRY_RUN=1 REPO_ROOT="$PWD" WORLDS_PER_RANK=16 \
  bash scripts/train_cdpr_smolvla_complex_grpo_mjlab_dual_remote.sh
```

## Troubleshooting

### Contact or constraint capacity overflow

The backend checks global contacts and maximum constraints/world once per
update. It aborts both ranks instead of training on truncated physics. Increase
`simulator.nconmax`/`--mjwarp-nconmax` for contacts per world or
`simulator.njmax`/`--mjwarp-njmax` for constraints per world, then repeat the
smoke and parity suites. Capacity grows with worlds and contact-rich object
states; do not extrapolate a safe value from empty scenes.

### Contact/grasp mismatch

Use the parity fixture and inspect contact agreement, target trajectories, and
gripper opening together. Confirm the primitive catalog, friction, equality,
seven substeps, fitted opening, and support height. Do not mask a mismatch with
a larger success tolerance. The active latch accepts a finger-pad contact in
the broad alignment window or the configured contactless centered-close
fallback, exactly as the CPU path does.

The checked 16-world capacity is `nconmax=256` contacts/world and
`njmax=1024` constraints/world. The latter includes headroom above the 789
constraints/world observed during the first A40 partial-reset smoke. Do not
lower it to the former value of 512.

### Camera memory or OOM

Rendering memory scales with worlds, pixels, and camera count. Reduce
`WORLDS_PER_RANK` first. SmolVLA-only pressure can be reduced with
`SMOLVLA_MICROBATCH_SIZE`; this does not change complete group ownership.
Keep the raw 320×240 and model 256×256 resolutions during parity. Use
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` and
check measured VRAM/SM utilization rather than assuming free VRAM means free
compute.

### DDP hang

Confirm exactly two visible devices and one rank per device. Keep
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`; use `NCCL_DEBUG=INFO` for diagnosis.
Never filter records into a variable number of local backward calls. The
trainer's synchronized padded schedule must remain the only MJWarp PPO update
path. A simulator exception on one rank can look like an NCCL timeout on the
other; inspect both rank logs for the first failure.

### Version or import failure

Do not mix latest packages into the pinned environment. Re-run the setup script
and `pip check`, then preflight. Explicit MJWarp selection has no silent CPU
fallback. The existing CPU script remains available independently.

### Renderer/color/orientation mismatch

Run parity with `MUJOCO_GL=egl`, compare the reported order and shapes, and
inspect selected exported frames outside the training loop. MJWarp output must
be BCHW float32 RGB in `[0,1]`; CPU EGL is HWC uint8 RGB. A vertical flip,
BGR swap, or camera-id reversal is a blocker, even if image means look
reasonable.
