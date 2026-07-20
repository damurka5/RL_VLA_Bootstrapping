# SmolVLA LoRA fine-tune path (design)

Goal: let RL gradients reach SmolVLA's action-expert stream so the prior becomes
conditioned on vision + language in this embodiment's action frame — the thing
the frozen prior cannot do (diagnostic: `lang spread ~0.15`, `mean prior_z
+0.45`, `prior_target_cosine` ~0). Vision encoder stays frozen; only LoRA
adapters on the action-expert attention train, with a KL/trust-region leash to
the frozen prior.

Status: LoRA core (`rl_vla_bootstrapping/policy/lora.py`, tested) and the
target-discovery tool (`scripts/count_smolvla_parameters.py --dump-linear-names`)
are done. The trainer/update wiring below is specified but **needs the module-name
dump + a GPU smoke test before it is trustworthy**.

## Step 0 — resolve LoRA targets (blocking input)

Run on the server:

```
python scripts/count_smolvla_parameters.py --base-checkpoint lerobot/smolvla_base --dump-linear-names
```

From the output we need: (a) the attention leaf names (expected `q_proj, k_proj,
v_proj, o_proj`), and (b) the **qualified-path prefix that selects the action
expert but not the VLM** (99.6% of params sit in `model.vlm_with_expert`; the
expert stream is a submodule of it — likely `...expert...` or `...lm_expert...`).
Those feed `attach_lora(target_leaf_names=..., name_contains=(<expert prefix>,))`.

## Step 1 — attach LoRA in the trainer

In `SmolVLAGRPOTrainer.__init__` (or a post-load hook), when `--train-vla-lora`:
- `attach_lora(runtime.policy, target_leaf_names=<attn>, name_contains=<expert>,
  rank, alpha, dropout)`, then `freeze_all_but_lora(runtime.policy)`.
- Keep a frozen reference: the rollout already stores the detached prior
  (`records["prior"]`) — that is the KL reference, no second model copy needed.
- Optimizer gets **two param groups**: residual (`lr=learning_rate`, as today)
  and LoRA (`lr=vla_lr ~1e-5`). Under DDP, wrap so both sets sync.

## Step 2 — rollout is unchanged, but store VLA inputs

Rollouts still run the prior under `no_grad` for throughput. To recompute it with
grad in the update, `collect_round` must additionally store per selected record:
- `overview` and `wrist` images **at the policy input size** (256x256 uint8),
- `state` (already stored), `instruction` id (map the string to an int id).

Memory: this is the crux. 1024 target records/update x 2 x 256x256x3 uint8
≈ 400 MB/update of images — acceptable if we (a) store uint8 not float, (b) cap
records fed to the VLA update (`vla_update_max_records`, e.g. 256), and (c) keep
them on GPU only for the current update. Do **not** store every sampled action's
images; store only the selected/informative records.

## Step 3 — grad-through-VLA GRPO update

Replace, for LoRA microbatches, the "use stored detached prior" path with:
- Recompute `prior_grad = runtime.forward(images, state, instr)` **with grad**
  (LoRA active), microbatch `vla_microbatch_size` small (~16–32; the frozen VLA
  forward + activations dominate memory).
- `mean = residual(state, prior_grad)`; GRPO ratio/clip/advantage exactly as now.
- **Trust-region term**: `+ vla_kl_coef * mean_over_records || prior_grad -
  prior_ref ||^2` where `prior_ref` is the stored detached rollout prior. Keeps
  LoRA from yanking the prior far from the pretrained manifold (RL-stability).
- Backprop updates residual (all its records) + LoRA (the VLA-grad microbatches).
  Keep the DDP-equal-schedule invariant: every rank runs the same number of
  VLA-grad microbatches (pad like `synchronize_equal_ddp_schedule` already does).

## Step 4 — batch / VRAM recomputation (2xA40, 48 GB)

- `worlds_per_rank`: unchanged for rollout (inference, no_grad) — keep 512.
- `vla_microbatch_size`: NEW, ~16–32 (was residual `microbatch_size: 512`).
  The residual-only update keeps its large microbatch; only the VLA-grad path
  shrinks.
- `vla_update_max_records`: NEW, cap ~256 informative records/update through the
  VLA to bound image memory and update time.
- Expect throughput to drop (VLA backward is the new cost); measure with the
  existing `profile/backpropagation_*` timers.

## Step 5 — config / args

New `training.rl.args` (default off, so current runs are unchanged):
```
train_vla_lora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
lora_target_leaves: [q_proj, k_proj, v_proj, o_proj]
lora_expert_name_contains: [<from step 0>]
vla_lr: 1.0e-5
vla_kl_coef: 0.1
vla_microbatch_size: 16
vla_update_max_records: 256
```

## Step 6 — smoke test before a real run

1. `python -m unittest tests.test_lora` in `cdpr-mjlab` (LoRA core).
2. A 2–5 update `mjwarp_max_updates` run with `train_vla_lora: true`: confirm it
   steps without OOM, `count_trainable` matches expectation, and both
   `prior_target_cosine_mean` (should start moving) and the loss are finite.
3. Only then launch a full run.

## Sequencing note

Run the residual baseline first (already launched). If `policy_target_cosine_mean`
climbs while `prior_target_cosine_mean` stays flat, that confirms the residual is
carrying all the direction and the LoRA path is what unlocks *vision-driven*
generalization. If even the residual can't close XY, debug that before investing
in the VLA fine-tune.
