# CDPR MJLab remote setup, validation, benchmark, and operations

Commands below assume the checkout is
`/root/repo/RL_VLA_Bootstrapping`. Override `REPO_ROOT` if needed.

## Install

The setup script creates a Python 3.12 Conda environment, installs the exact
CUDA 12.8 lock, installs this repository editable without dependency drift,
runs `pip check`, stages the curated RoboCasa subset at
`assets/externals/robocasa`, and executes the strict preflight:

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
per-world real-mesh selection, both camera renders, GPU contact-force output,
GPU tensor checks, deterministic XYZ target tracking, gripper hold stability,
and the complete `training_put_into_bowl` / `training_put_on_plate` trajectories
must execute.

The following optional command runs unit/regression fixtures. It is not part of
the production simulator hot path and does not enable CPU contact handling:

```bash
conda run --no-capture-output -n cdpr-mjlab python3 -m pytest -q tests
```

## Stage the RoboCasa object pack

The MJ-Lab scene no longer depends on YCB or LIBERO visuals. Stage and verify
the curated RoboCasa subset before preflight:

```bash
cd /root/repo/RL_VLA_Bootstrapping
conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/stage_cdpr_robocasa_assets.py
conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/stage_cdpr_robocasa_assets.py --verify-only
```

Only the ten selected model/visual packs are transferred. RoboCasa collision
meshes are omitted because the production backend uses fixed native primitive
colliders.

## GPU real-object grasp videos

Run the strict pick/lift/transport/release probe. It uses eight simultaneous
MJWarp worlds on CUDA, shows every active real-object catalog, and overlays
bilateral contact, solved pad forces, relative-pose slip, lift, grasp, and
release evidence:

```bash
cd /root/repo/RL_VLA_Bootstrapping
conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/render_cdpr_mjwarp_physical_grasp_videos.py \
  --config configs/examples/cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml \
  --device cuda:0 \
  --output-dir runs/cdpr_mjwarp_physical_grasp_videos
```

The script has no CPU simulator fallback. Camera frames and compact scalar
diagnostics leave the GPU only for MP4/CSV encoding. Its default strict mode
exits nonzero unless all eight grippable RoboCasa types demonstrate physical
grasp and release. Inspect `manifest.json`, the grid MP4, the overview/wrist
MP4, and `physical_grasp_trace.csv`.

## Training-camera episode videos

Render the same overview/wrist camera contract and scripted manipulation
episodes used to inspect training resets. On the CUDA host, force the
production backend so the command cannot silently fall back to CPU MuJoCo:

```bash
conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/render_cdpr_mjlab_camera_videos.py \
  --backend mjlab-mjwarp \
  --device cuda:0 \
  --scenarios training_put_into_bowl training_put_on_plate \
  --output-dir runs/cdpr_robocasa_training_videos \
  --no-timestamped
```

Each scenario writes overview, wrist, and side-by-side MP4s with VLA-like
normalized actions, executed deltas, controller targets/errors, gripper state,
object/receptacle telemetry, and cable lengths. The expanded action CSV carries
the same telemetry, and the shared manifest/contact sheet records whether the
production MJLab/MJWarp backend or the local MuJoCo reference backend produced
the files. Scenario verification is enabled by default: the command exits
nonzero if a phase does not reach its target, the caught object slips out, the
gripper jitters, or the final placement misses the 3 cm success tolerance.

## Optional CPU/MJWarp parity

This offline fixture compares identical scripted states against CPU MuJoCo. It
is useful for migration diagnosis, but production training does not import or
call it. Skip it if GPU-only acceptance is your policy:

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

### Step-zero SmolVLA move-to-object run

The dedicated scratch configuration is
`configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml`.
It samples only `move_to_object`, targets the selected RoboCasa apple, banana,
tomato, orange, potato, mug, plate, and bowl variants, and converts their
catalog names to prompts such as `move to apple` and `move to plate`. Variant
suffixes such as `_20` and `_12` never enter the language prompt.

The run starts a new 1024-hidden-unit GRPO residual/readout head at global step
zero on top of the frozen `lerobot/smolvla_base` prior. The scratch launcher
rejects both checkpoint environment variables so a stale remote-shell setting
cannot silently turn the run into a resume.

Before a long run, sweep complete rank-local group counts on the actual two A40
cards:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output -n cdpr-mjlab python3 \
  scripts/benchmark_cdpr_mjlab_grpo.py \
  --repo-root "$PWD" \
  --worlds 16 32 64 128 \
  --updates 3 \
  --microbatch 16 32 64 \
  --compile-model \
  --output-dir runs/cdpr_mjlab_move_to_a40_sweep
```

`benchmark.json` records the fastest successful
`recommended_worlds_per_rank`, the largest allocation that fit, GPU
utilization/power/VRAM, and synchronized scene-reset, environment-step,
SmolVLA-inference, and backpropagation times. Select the fastest end-to-end
setting; allocating the most VRAM is not useful if selected actions/s falls.

The conservative checked launch is:

```bash
REPO_ROOT="$PWD" ENV_NAME=cdpr-mjlab \
  CUDA_VISIBLE_DEVICES=0,1 \
  WORLDS_PER_RANK=16 \
  SMOLVLA_MICROBATCH_SIZE=16 \
  MAX_TRAIN_STEPS=2000000 \
  bash scripts/train_cdpr_smolvla_move_to_grpo_mjlab_dual_remote.sh
```

Replace both `WORLDS_PER_RANK=16` and `SMOLVLA_MICROBATCH_SIZE=16` with the
benchmark recommendation. The first four production updates emit
`rl/latest_profile.json` with synchronized component times. Timing barriers
then disable automatically, leaving the remainder of the 2M-step run on the
unsynchronized throughput path.

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

Use the GPU grasp-video trace and inspect bilateral contact, both solved pad
forces, relative-position/orientation slip, target lift, and gripper opening
together. Confirm the RoboCasa visual hash, native primitive sizes, friction,
equality, seven substeps, fitted opening, and support height. Do not mask a
mismatch with a larger success tolerance. There is no contactless fallback and
no pose latch.

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
fallback. The independent CPU trainer and optional parity script are separate
entrypoints and are never called by MJWarp production training.

### Renderer/color/orientation mismatch

Run parity with `MUJOCO_GL=egl`, compare the reported order and shapes, and
inspect selected exported frames outside the training loop. MJWarp output must
be BCHW float32 RGB in `[0,1]`; CPU EGL is HWC uint8 RGB. A vertical flip,
BGR swap, or camera-id reversal is a blocker, even if image means look
reasonable.
