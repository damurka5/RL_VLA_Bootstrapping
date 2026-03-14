# RL VLA Bootstrapping

`rl_vla_bootstrapping` is an embodiment-first orchestration framework for building language-conditioned visuomotor training stacks around a new robot without starting from demonstrations.

The repo is intentionally centered on separable layers instead of a single PPO or OpenVLA implementation:

- Embodiment specification: MuJoCo XML, controller entrypoint, joint/action metadata, limits, gripper semantics, and a shared action codec.
- Task specification: language instructions, object sets, goal relations, success predicates, and optional dense reward hooks.
- Simulation specification: scene builders, object assets, randomization, cameras, and preview helpers.
- Policy specification: external VLA backbones such as OpenVLA-OFT plus action-head/action-codec metadata.
- Training and evaluation orchestration: zero-demo RL first, SFT refinement later, benchmark hooks for RoboTwin 2.0 / ManiTask-style suites.

The framework does not vendor the VLA model itself. Instead, it expects an external OpenVLA-OFT repo path and generates a consistent training plan around it.

## Layout

- `rl_vla_bootstrapping/core`: config loading, import helpers, shared specs.
- `rl_vla_bootstrapping/policy`: action codec and policy connectors.
- `rl_vla_bootstrapping/pipeline`: stage planning and execution.
- `rl_vla_bootstrapping/cli`: entrypoints for training and preview.
- `robots`: embodiment bundles; the CDPR example now lives under `robots/cdpr/`.
- `assets`: staged YCB/LIBERO asset bundles.
- `benchmarks`: staged RoboTwin 2.0 / ManiTask repos and adapters.
- `environments`: remote conda environment definitions.
- `configs/examples`: example configs, including a CDPR + OpenVLA-OFT bootstrap config.
- `scripts`: thin shell wrappers for preview and training.

## Quick Start

1. Create the remote environment:

```bash
conda env update -n openvla-oft -f environments/openvla-oft-remote.yaml --prune
```

Or do the full remote bootstrap in one step:

```bash
./scripts/setup_remote.sh configs/examples/cdpr_openvla_bootstrap.yaml
```

2. Stage assets into repo-local paths:

```bash
python3 -m rl_vla_bootstrapping.cli.assets \
  --config configs/examples/cdpr_openvla_bootstrap.yaml \
  --stage
```

3. Validate the runtime and robot setup:

```bash
./scripts/doctor_bootstrap.sh configs/examples/cdpr_openvla_bootstrap.yaml
```

4. Run a preview:

```bash
./scripts/preview_bootstrap.sh configs/examples/cdpr_openvla_bootstrap.yaml
```

5. Plan the full pipeline:

```bash
python3 -m rl_vla_bootstrapping.cli.train --config configs/examples/cdpr_openvla_bootstrap.yaml
```

6. Execute the selected stages:

```bash
./scripts/train_bootstrap.sh configs/examples/cdpr_openvla_bootstrap.yaml
```

## Runtime Notes

Preview and training stages use the dependencies required by the vendored CDPR example bundle under `robots/cdpr/` plus the external OpenVLA/OFT repo. For the current stack that means MuJoCo, EGL-capable rendering on Linux, `opencv-python`, and the OpenVLA/OFT training dependencies from the included environment file.

## Assets And Benchmarks

The repo intentionally does not commit large YCB/LIBERO assets or full RoboTwin 2.0 / ManiTask repos.

- YCB and LIBERO are staged under `assets/externals/`.
- RoboTwin 2.0 and ManiTask are staged under `benchmarks/externals/`.
- GitHub stores only the framework, configs, and asset bundle definitions; large asset directories are linked or copied into the local checkout with `rl_vla_bootstrapping.cli.assets`.
- The CDPR example config already includes bundle definitions and disabled benchmark stages.
- Benchmark stages use local wrappers in `rl_vla_bootstrapping/evaluation/` so the evaluation layer is visible in this repo even though the benchmark repos stay external.

## Robot Integration Contract

For a new robot, the smallest useful setup is:

- a MuJoCo XML file,
- a Python controller class,
- a config entry declaring action keys, scaling, and controller method names,
- optionally a scene-builder function and reward/success functions.

The framework normalizes RL and SFT around a shared `ActionCodec`. RL policies operate in the common normalized action space; SFT can quantize exactly the same normalized space later, so the refinement stage remains consistent with the RL stage.
