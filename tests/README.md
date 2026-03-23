# Test Notes For Added RL Policy Code

This folder now includes targeted tests for the new in-tree safe RL and OpenVLA actor-critic additions.

## Added Test Files

### [`test_blac_finetune_cdpr.py`](./test_blac_finetune_cdpr.py)

Covers the vector-state BLAC trainer in [`rl_vla_bootstrapping/policy/blac_finetune_cdpr.py`](../rl_vla_bootstrapping/policy/blac_finetune_cdpr.py).

Focus:

- CLI compatibility with the bootstrap stage-planner style arguments
- observation flattening / layout assumptions
- workspace backup projection behavior
- run-directory creation

This test is intentionally lightweight and does not require MuJoCo rollout execution.

### [`test_openvla_actor_critic.py`](./test_openvla_actor_critic.py)

Covers reusable helpers from [`rl_vla_bootstrapping/policy/openvla_actor_critic.py`](../rl_vla_bootstrapping/policy/openvla_actor_critic.py).

Focus:

- prompt formatting
- wrapped-model traversal
- `llm_dim` resolution
- image-count configuration for wrapped OpenVLA models
- fallback generate-config loading

These tests protect the in-tree OpenVLA RL integration layer from regressions in helper behavior.

### [`test_openvla_blac_finetune_cdpr.py`](./test_openvla_blac_finetune_cdpr.py)

Covers the new OpenVLA-conditioned trainer in [`rl_vla_bootstrapping/policy/openvla_blac_finetune_cdpr.py`](../rl_vla_bootstrapping/policy/openvla_blac_finetune_cdpr.py).

Focus:

- CLI parsing
- workspace safety-cost calculation
- trainer run-directory creation

The goal here is to keep import-time and helper-level behavior stable even in environments where the full OpenVLA runtime is not installed.

## Existing Related Tests

### [`test_ppo_finetune_cdpr_fast.py`](./test_ppo_finetune_cdpr_fast.py)

This already existed for the fast PPO wrapper path in [`rl_vla_bootstrapping/policy/ppo_finetune_cdpr_fast.py`](../rl_vla_bootstrapping/policy/ppo_finetune_cdpr_fast.py), but it was slightly adjusted so local Torch stubs do not leak across unrelated tests.

### [`test_cdpr_policy_runner.py`](./test_cdpr_policy_runner.py)

This remains the main coverage point for the original OpenVLA policy runner in [`rl_vla_bootstrapping/cli/run_cdpr_policy.py`](../rl_vla_bootstrapping/cli/run_cdpr_policy.py).

It is still important because the new in-tree OpenVLA actor-critic stack was extracted from that runner’s loading and action-token path.

## Suggested Test Command

The focused regression suite for the newly added policy code is:

```bash
python -m unittest \
  tests.test_openvla_blac_finetune_cdpr \
  tests.test_openvla_actor_critic \
  tests.test_blac_finetune_cdpr \
  tests.test_ppo_finetune_cdpr_fast \
  tests.test_cdpr_policy_runner
```

## Scope Reminder

These tests are intentionally narrow:

- they validate parser/helper/integration assumptions
- they do not attempt full MuJoCo training
- they do not require a full OpenVLA runtime checkout for the lightweight paths

That keeps the local suite fast while still protecting the interfaces that the new in-tree RL work depends on.
