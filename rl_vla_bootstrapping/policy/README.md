# Policy Layer Notes

This directory started as a thin policy integration layer for the broader bootstrap pipeline:

- [`action_codec.py`](./action_codec.py): shared normalized action space and quantization metadata.
- [`openvla_oft.py`](./openvla_oft.py): stage-plan builder for external OpenVLA-OFT RL/SFT scripts.
- [`ppo_finetune_cdpr_fast.py`](./ppo_finetune_cdpr_fast.py): fast wrapper around an external PPO trainer with a few project-specific runtime patches.

The new additions in this directory move part of the RL stack in-tree so the repo can train safe actor-critic variants directly instead of only delegating RL logic to an external PPO implementation.

## What Was Added

### 1. In-tree safe RL baseline on the vector CDPR state

- [`blac_finetune_cdpr.py`](./blac_finetune_cdpr.py)

This is the first self-contained RL trainer added to the repo.

It trains directly on the vector observation exposed by [`robots/cdpr/cdpr_dataset/rl_cdpr_env.py`](../../robots/cdpr/cdpr_dataset/rl_cdpr_env.py) and implements:

- a Barrier-Lyapunov Actor-Critic inspired by Zhao et al., [A Barrier-Lyapunov-Actor-Critic Reinforcement Learning Approach for Safe and Stable Control](https://arxiv.org/pdf/2304.04066)
- a learned dynamics model for one-step constraint evaluation
- a Lyapunov network for stability shaping
- workspace barrier constraints for CDPR end-effector safety
- a simple backup controller / projection step for obviously unsafe XYZ proposals

It also exposes several compatible off-policy baselines behind the same entrypoint:

- `blac`
- `bac`
- `sac`
- `td3`
- `redq`

Why this exists:

- the repo originally had no in-tree actor-critic trainer to extend
- the external PPO wrapper was useful for speed and compatibility, but it was not a good place to implement BLAC-style constraints
- the CDPR environment already exposes exactly the structured state needed for safe RL: end-effector position, goal position, object positions, masks, instruction encoding, and workspace safety signals

### 2. In-tree OpenVLA actor-critic stack

- [`openvla_actor_critic.py`](./openvla_actor_critic.py)

This module is the reusable bridge between the repo’s OpenVLA inference path and new in-tree RL trainers.

It extracts the OpenVLA-specific logic that was previously only embedded inside the policy runner:

- prompt formatting
- wrapped-model traversal
- `llm_dim` resolution
- image-count control for wrapped OpenVLA backbones
- multimodal batch preparation
- action-token hidden-state extraction
- twin critic heads operating on pooled OpenVLA action-token features

Main entrypoints:

- `load_openvla_runtime(...)`
- `load_generate_config(...)`
- `extract_action_hidden_states(...)`
- `OpenVLAActorCriticStack`

What it is built on top of:

- the existing OpenVLA runtime loading path in [`rl_vla_bootstrapping/cli/run_cdpr_policy.py`](../cli/run_cdpr_policy.py)
- the external OpenVLA/OFT repo expected by [`openvla_oft.py`](./openvla_oft.py)
- the repo’s action-head convention already used for OpenVLA CDPR execution

This is intentionally not a full replacement for OpenVLA-OFT training internals. It is a local RL-facing integration layer that makes image-conditioned actor-critic experiments possible in this repo.

### 3. In-tree OpenVLA BLAC trainer

- [`openvla_blac_finetune_cdpr.py`](./openvla_blac_finetune_cdpr.py)

This trainer wires the reusable OpenVLA actor-critic stack into the CDPR safe-RL loop.

It does the following:

1. Loads bootstrap config via [`rl_vla_bootstrapping/core/config.py`](../core/config.py)
2. Builds a `CDPRVisionLanguageEnv`
3. Loads OpenVLA base model, LoRA adapter, processor, and action head
4. Wraps them in `OpenVLAActorCriticStack`
5. Collects replay using image-conditioned OpenVLA chunked actions
6. Trains:
   - the action head as the actor
   - twin critics on pooled OpenVLA features
   - a vector-state dynamics model
   - a vector-state Lyapunov network
7. Applies BLAC-style barrier and Lyapunov penalties using the same CDPR workspace constraint logic as the vector baseline

Important scope note:

- this trainer is image-conditioned and uses OpenVLA features
- the safety and stability constraints are still evaluated on the vector CDPR state
- by default the VLA backbone is frozen and the action head + critics are optimized

That design is deliberate. It gives the repo a practical, testable in-tree OpenVLA RL trainer without first re-implementing the full external OpenVLA distributed finetuning stack.

## Architectural Relationship

The additions form a layered progression:

1. Existing external planning/inference layer:
   - [`openvla_oft.py`](./openvla_oft.py)
   - [`ppo_finetune_cdpr_fast.py`](./ppo_finetune_cdpr_fast.py)
   - [`rl_vla_bootstrapping/cli/run_cdpr_policy.py`](../cli/run_cdpr_policy.py)

2. New self-contained safe RL baseline:
   - [`blac_finetune_cdpr.py`](./blac_finetune_cdpr.py)

3. New reusable OpenVLA RL building block:
   - [`openvla_actor_critic.py`](./openvla_actor_critic.py)

4. New image-conditioned safe RL trainer:
   - [`openvla_blac_finetune_cdpr.py`](./openvla_blac_finetune_cdpr.py)

In short:

- `blac_finetune_cdpr.py` proves the BLAC-style safe RL path in-tree on vector observations
- `openvla_actor_critic.py` makes OpenVLA features reusable by local trainers
- `openvla_blac_finetune_cdpr.py` combines those two ideas into an OpenVLA-conditioned trainer

## How The Safety Terms Work

Both BLAC-style trainers use the repo-local CDPR structure rather than a generic robotics safety abstraction.

Constraint sources:

- end-effector workspace bounds
- minimum Z floor / safe vertical workspace
- optional object-clearance margin

State sources:

- `ee_position`
- `target_object_position`
- `all_object_positions`
- `object_position_mask`
- `goal_direction`

These come from [`robots/cdpr/cdpr_dataset/rl_cdpr_env.py`](../../robots/cdpr/cdpr_dataset/rl_cdpr_env.py).

The resulting penalties are used in two ways:

- barrier penalties discourage predicted next states from leaving the safe set
- Lyapunov penalties discourage unstable progress relative to the goal state

## Why Two Trainers Exist

### `blac_finetune_cdpr.py`

Use this when you want:

- a self-contained local safe RL baseline
- easy iteration on barrier/Lyapunov logic
- no dependency on OpenVLA runtime internals during the algorithm design loop

### `openvla_blac_finetune_cdpr.py`

Use this when you want:

- OpenVLA image-conditioned policies in the loop
- action-head RL updates driven by VLA features
- a local trainer that can evolve toward deeper OpenVLA RL finetuning

## Related Files Outside This Directory

- Environment:
  - [`robots/cdpr/cdpr_dataset/rl_cdpr_env.py`](../../robots/cdpr/cdpr_dataset/rl_cdpr_env.py)
- CDPR control contract:
  - [`robots/cdpr/cdpr_mujoco/policy_control.py`](../../robots/cdpr/cdpr_mujoco/policy_control.py)
- Policy execution / original OpenVLA runtime path:
  - [`rl_vla_bootstrapping/cli/run_cdpr_policy.py`](../cli/run_cdpr_policy.py)
- Stage planning:
  - [`openvla_oft.py`](./openvla_oft.py)
- Top-level project overview:
  - [`README.md`](../../README.md)

## External References

- BLAC paper:
  - [A Barrier-Lyapunov-Actor-Critic Reinforcement Learning Approach for Safe and Stable Control](https://arxiv.org/pdf/2304.04066)
- Paper code release:
  - [LiqunZhao/A-Barrier-Lyapunov-Actor-Critic-Reinforcement-Learning-Approach-for-Safe-and-Stable-Control](https://github.com/LiqunZhao/A-Barrier-Lyapunov-Actor-Critic-Reinforcement-Learning-Approach-for-Safe-and-Stable-Control)

## Current Practical Status

What is fully in-tree now:

- vector-state BLAC/BAC/SAC/TD3/REDQ baseline
- reusable OpenVLA actor-critic feature stack
- first OpenVLA-conditioned BLAC trainer
- unit tests for parser/helper/safety integration points

What is not yet fully in-tree:

- full OpenVLA distributed RL finetuning stack equivalent to the external PPO trainer
- end-to-end LoRA/backbone RL updates with the same maturity as the external OpenVLA-OFT training code

That boundary is important: the repo now has a real in-tree actor-critic extension point, but the external PPO path is still the more production-shaped route for large-scale OpenVLA RL runs until the in-tree trainer is expanded further.
