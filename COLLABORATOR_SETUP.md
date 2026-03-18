# Working With This Repo

This repository is the orchestration layer for CDPR + OpenVLA-OFT training.
It does not contain every dependency needed to run end-to-end training by itself.

## What Lives Where

- `RL_VLA_Bootstrapping/`
  - project config
  - CDPR MuJoCo environment
  - reward function
  - local inference runner
  - training pipeline that launches external stages
- `openvla-oft/`
  - actual OpenVLA-OFT training code
  - PPO trainer used by this repo
  - SFT trainer used by this repo
- `assets/externals/`
  - staged YCB and LIBERO assets used by scene generation
- optional benchmark repos
  - `RoboTwin2.0/`
  - `ManiTask/`

## Important Clarification

The zero-demo RL stage is launched from this repo, but the PPO implementation is external.

Current training entrypoint chain:

- [openvla_oft.py](/Users/damirnurtdinov/Desktop/My Courses/Диплом/RL_VLA_Bootstrapping/RL_VLA_Bootstrapping/rl_vla_bootstrapping/policy/openvla_oft.py#L170) builds the RL stage plan
- [cdpr_openvla_bootstrap.yaml](/Users/damirnurtdinov/Desktop/My Courses/Диплом/RL_VLA_Bootstrapping/RL_VLA_Bootstrapping/configs/examples/cdpr_openvla_bootstrap.yaml#L150) points `policy.rl_script` to `openvla-oft/vla-scripts/ppo_finetune_cdpr.py`

So yes: other users need this repo plus the external `openvla-oft` checkout, and they also need the required assets.

## Minimum Required Pieces

To train the CDPR policy, a collaborator needs:

- this repo
- the `openvla-oft` repo
- MuJoCo and the Python deps used by both repos
- YCB assets
- LIBERO assets
- desk textures if they want texture-randomized PPO scenes to match the provided config

Optional pieces:

- `RoboTwin2.0` for RoboTwin evaluation
- `ManiTask` for ManiTask evaluation
- `robotwin2_assets` if they want extra object pools from RoboTwin assets

## Recommended Directory Layout

The example config in this repo currently expects a sibling layout like this:

```text
<workspace>/
├── RL_VLA_Bootstrapping/
└── openvla-oft/
```

From inside `RL_VLA_Bootstrapping/configs/examples/cdpr_openvla_bootstrap.yaml`, that is referenced as:

- `../../../../openvla-oft`

If someone keeps `openvla-oft` somewhere else, they must update:

- `repos.openvla_oft`
- `embodiment.extra_python_paths`
- `policy.repo_path`
- `policy.rl_script`
- `policy.sft_script`

## Asset Expectations

The example config expects staged assets under this repo:

- `assets/externals/ycb`
- `assets/externals/libero`
- `assets/externals/desk_textures_cache`
- `assets/externals/robotwin2_assets` (optional)

The scene builder and CDPR wrappers read from those local staged paths during preview, training, and inference.

## Minimal Setup Flow

1. Clone this repo.
2. Clone `openvla-oft` next to it, or edit the config paths.
3. Create the runtime environment from [openvla-oft-remote.yaml](/Users/damirnurtdinov/Desktop/My Courses/Диплом/RL_VLA_Bootstrapping/RL_VLA_Bootstrapping/environments/openvla-oft-remote.yaml).
4. Stage YCB and LIBERO assets into this repo’s `assets/externals/` paths.
5. Run validation for paths and environment:

```bash
./scripts/doctor_bootstrap.sh configs/examples/cdpr_openvla_bootstrap_fast.yaml
```

6. Start training:

```bash
./scripts/train_bootstrap.sh configs/examples/cdpr_openvla_bootstrap_fast.yaml
```

The `cdpr_openvla_bootstrap_fast.yaml` config is the recommended remote-server profile. It keeps the PPO logic the same, but uses a local wrapper launcher plus a more throughput-oriented runtime profile for multi-GPU CDPR training.

## What Can Be Run Without `openvla-oft`

Some local CDPR code in this repo can be inspected or unit-tested without the external trainer.
But end-to-end OpenVLA-OFT RL and SFT cannot run without the external `openvla-oft` checkout, because the pipeline imports and executes its training scripts directly.

## Practical Recommendation For Collaborators

When sharing this project with someone else, give them:

- this repo
- the exact `openvla-oft` commit you trained with
- the example config you used
- the staged asset locations or asset sync instructions
- the adapter/action-head outputs after training

That will save them from path mismatches and training-contract mismatches.
