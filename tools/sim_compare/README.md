# Simulator Comparator

Lightweight scripted simulator benchmark scaffold. It does not train OpenVLA,
load OpenVLA checkpoints, or require OpenVLA dependencies.

## Run

```bash
python3 tools/sim_compare/run_benchmark.py
```

Useful quick smoke:

```bash
python3 tools/sim_compare/run_benchmark.py \
  --resets 1 \
  --steps 300 \
  --render-steps 2 \
  --task-objects block \
  --contact-objects block \
  --width 64 \
  --height 64
```

Remote MuJoCo RGB profile example for a Linux/NVIDIA server:

```bash
MUJOCO_GL=egl python tools/sim_compare/run_comparator.py \
  --backend mujoco_raw_cdpr \
  --render \
  --camera-count 2 \
  --width 320 \
  --height 240 \
  --episodes 20 \
  --steps 1000 \
  --out tools/sim_compare/out_remote_mujoco
```

The same command is wrapped by:

```bash
MUJOCO_GL=egl tools/sim_compare/run_remote_mujoco_profile.sh
```

Rendering flags:

- `--render-backend auto|egl|osmesa|glfw`
- `--width`
- `--height`
- `--camera-count`
- `--render` / `--no-render`

## Outputs

The script overwrites:

- `tools/sim_compare/out/backend_summary.csv`
- `tools/sim_compare/out/task_results.csv`
- `tools/sim_compare/out/contact_results.csv`
- `tools/sim_compare/out/render_profile.csv`
- `tools/sim_compare/out/SIMULATOR_COMPARATOR_REPORT.md`

## Notes

- `mujoco_raw_cdpr` is the executable baseline and uses the current CDPR MJCF
  with scripted waypoint/direct-state controllers.
- ManiSkill/SAPIEN, Isaac Lab/Isaac Sim, robosuite, and optional PyBullet are
  probed and skipped cleanly when imports are unavailable.
- RGB rendering is measured separately from physics stepping. If MuJoCo cannot
  create a local offscreen context, the render profile records the attempted
  backend, platform, failure reason, and backend-specific recovery instructions
  instead of failing the whole benchmark.
- On Linux/NVIDIA remote servers, prefer `MUJOCO_GL=egl`. If EGL is unavailable,
  try `MUJOCO_GL=osmesa` for CPU headless rendering. Local macOS uses MuJoCo's
  GLFW/CGL path and requires an interactive CoreGraphics session.
