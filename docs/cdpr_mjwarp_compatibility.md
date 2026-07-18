# CDPR MJLab / MuJoCo Warp migration and compatibility report

## Version decision

The deployment lock is:

| Component | Exact version |
|---|---:|
| Python | 3.12 |
| CUDA runtime used by PyTorch | 12.8 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| torchaudio | 2.7.1+cu128 |
| NVIDIA Warp | 1.14.0 |
| MuJoCo | 3.10.0 |
| MuJoCo Warp | 3.10.0.1 |
| MJLab | 1.5.0 |
| LeRobot/SmolVLA | 0.6.0 |
| NumPy | 2.2.6 |

MJLab 1.5.0 declares Python 3.10–3.13, Torch 2.7+, Warp 1.14+, MuJoCo
`~=3.10.0`, and MuJoCo Warp `>=3.10.0.1,~=3.10.0`. LeRobot 0.6.0 requires
Python 3.12+ and Torch 2.7+, which determines the Python choice. PyTorch 2.7
officially provides CUDA 12.8 wheels.

Primary references:

- [MJLab 1.5.0 package metadata](https://raw.githubusercontent.com/mujocolab/mjlab/v1.5.0/pyproject.toml)
- [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [MuJoCo Warp public API](https://mujoco.readthedocs.io/en/stable/mjwarp/api.html)
- [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)
- [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- [LeRobot 0.6.0 package metadata](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/pyproject.toml)

The checked lock is
`requirements/cdpr-mjlab-cu128.lock.txt`. The preflight rejects any version
drift, a non-CUDA Torch build, a CUDA runtime other than 12.8, fewer than two
visible GPUs, non-A40 devices, or an NVIDIA driver older than major version
570.

## MJCF feature audit

The fixed-topology scene statically contains:

| Required feature | Static result |
|---|:---:|
| four spatial tendons | pass |
| eight pulley-routing geom wraps with sidesites | pass |
| joint equality for the fingers | pass |
| free end-effector | pass |
| ball stabilizer | pass |
| four slider + yaw + gripper position actuators | pass |
| overview and body-mounted wrist cameras | pass |
| four camera frame sensors | pass |
| four fixed free-body object slots | pass |

The dependency-free audit reports 4 spatial tendons, 8 tendon wraps, 1 joint
equality, 1 ball joint, 5 free joints, 6 position actuators, 2 cameras, and 4
camera frame sensors. It detects no configured PGS solver,
`implicitfast` integrator, nonzero no-slip post-solver iterations, or plugin
actuators/sensors.

The scanner rejects PGS, nonzero no-slip post-solver iterations, and plugin
actuators/sensors. It accepts `implicitfast` with a warning because MJWarp
supports it but documents numerical differences in midpoint-feature and
fluid-force paths.

Local MuJoCo 3.2.4 compilation of both the original robot MJCF and the fixed
scene passes. The fixed scene dimensions are `nq=46`, `nv=40`, `nu=6`,
`nbody=24`, `ngeom=84`, `ncam=2`, `nmat=8`, and `ntex=7`.

An A40 preflight on 2026-07-18 verified the exact package lock, two visible
44 GiB A40s, `mujoco_warp.put_model`, 16-world allocation, required MJCF
features, and both 320×240 batched cameras. MuJoCo 3.10 enables MULTICCD by
default, so the MJWarp-only wrapper explicitly disables MULTICCD and NATIVECCD
to retain the existing non-zero contact margins. The first backend smoke also
measured 789 constraints/world after partial reset; the checked capacity is
therefore 1024 rather than the insufficient original value of 512.
That smoke exposed a separate preload reset defect: copying calibrated slider
coordinates into `qpos0` redefined the slide-joint reference and left all four
tendons roughly 0.4 m beyond their upper limits. The backend now keeps the
compiled reference pose unchanged and restores calibrated dynamic
`qpos`/`qvel`/`ctrl` tensors after each full or partial reset.
The follow-up smoke showed the preload was correct to `4.8e-7`, but reported
exactly 156 contacts/world and diverged during the first step. Unused fixed-slot
primitives had been placed at local `z=-10`; because the floor is an infinite
plane, this created deep penetrations rather than disabling collisions. Unused
primitives are now transparent, tiny, and placed above the workspace. The
backend smoke explicitly rejects reset contacts deeper than 5 cm.

Full controller/contact smoke, the 8–128-world capacity sweep, parity, two-rank
checkpoint/resume, and end-to-end benchmarks remain unverified until new remote
artifacts pass with these corrections. The default local verification
interpreter has no CUDA, Warp, MJWarp, MJLab, or Torch installation. A separate
existing CPU-only PyTorch 2.0.1 environment was used only for tensor predicate
and Gloo/DDP tests; it is not the pinned deployment stack.

`scripts/preflight_cdpr_mjlab.py` performs those operations, not only imports.
It creates a render context, refits the BVH, renders both cameras, extracts
their tensors, instantiates the full backend, and validates the third image
slot. The backend smoke additionally requires four evolving tendon tensors,
active finger equality with bounded joint error, finite frame-sensor data, and
nonzero contacts after stepping. Both scripts exit nonzero unless every check
passes. Their JSON outputs are the authoritative phase-1 compatibility
evidence.

## Behavior preserved

- five normalized actions and the existing scales/limits;
- seven physics substeps for `hold_steps=6`;
- calibrated four-tendon CDPR controller state in reset/broadcast state;
- two physical cameras at 320×240 and 256×256 SmolVLA input;
- exact third-slot wrist duplication;
- four object slots and the seven enabled object catalog names;
- group-shared instruction, shell, scene, initial physics/controller state;
- stochastic independent candidate actions and deterministic rank/group seeds;
- active eight sparse predicates and thresholds;
- reverse-frontier shell counts, horizons, rehearsal, 50-continuation
  validation quota, promotion/demotion thresholds, minimum update gate, and
  one-shell jumps;
- terminal-continuation group reset semantics;
- BF16 inference, action normalization, residual-head checkpoint bootstrap,
  and optional `torch.compile`.

## Intentional numerical and rendering differences

- MJWarp physics is float32 and parallel; CPU MuJoCo state is commonly
  float64. Bit identity is neither expected nor required.
- MJWarp uses its batched ray renderer rather than EGL/OpenGL. Lighting,
  antialiasing, rasterization, and pixel-level output can differ.
- The fixed object slots use the repository's stable primitive collision packs.
  External high-polygon YCB visual meshes are not present in this checkout and
  are not silently substituted. This must be treated as a camera-distribution
  difference until the remote camera parity artifact is accepted.
- Fixed-topology inactive primitives are transparent and moved far outside the
  working geometry instead of removing bodies.
- Grasp latching uses the active CPU contact gate plus its contactless
  centered-close fallback, and keeps a caught object GPU-resident between
  substeps; raw contact agreement is measured separately.

## Remote acceptance gates

Promotion from the CPU entrypoint is blocked until all of these artifacts pass:

1. `preflight.json`: exact environment, two A40s, `put_model`, allocation,
   required MJCF features, and cameras.
2. `backend_smoke.json`: reset, group broadcast, controller/contact stepping,
   masked worlds, partial reset, RGB range/device/shape, capacity safety.
3. `parity.json` and `parity.md`: end-effector, object, tendon, gripper,
   contact, sparse-success, and camera discrepancies from identical state.
4. `resume_report.json`: two ranks complete reset, rollout, update,
   checkpoint, and resume with simulator metadata.
5. `benchmark.json` and `benchmark.md`: end-to-end results for 8, 16, 32, 64,
   and 128 worlds/rank.

The existing CPU production script is intentionally not redirected to MJWarp.

## Local verification record

The final local run produced these results:

- repository test suite: 367 tests, 6 dependency skips, 0 failures;
- MJWarp migration module in the CPU-only Torch environment: 24 tests,
  3 unavailable-MuJoCo/dependency skips, 0 failures;
- all eight active sparse predicate success/failure fixtures against the CPU
  reference: 1 parametrized fixture test, 0 failures;
- two-rank localhost Gloo smoke: one rank with zero informative records and
  one with 1,025 records completed the same 12 optimizer-level backward
  schedules and ended with identical DDP parameters;
- both MJCFs compile in local MuJoCo 3.2.4, Python sources compile, remote
  shell scripts pass `bash -n`, and all new CLI help paths load.

The checked local preflight artifact is
`docs/artifacts/cdpr_mjwarp_compatibility_local.json`. Its nonzero result is
expected: static MJCF inspection passes, while the exact CUDA package lock,
two A40s, `put_model`, world allocation, and batched renderer checks cannot run
on this host.
