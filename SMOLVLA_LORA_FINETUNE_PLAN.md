# SmolVLA LoRA fine-tune path (design)

Goal: let RL gradients reach SmolVLA's action-expert stream so the prior becomes
conditioned on vision + language in this embodiment's action frame — the thing
the frozen prior cannot do (diagnostic: `lang spread ~0.15`, `mean prior_z
+0.45`, `prior_target_cosine` ~0). Vision encoder stays frozen; only LoRA
adapters on the action-expert attention train, with a KL/trust-region leash to
the frozen prior.

Status: **implemented end-to-end, off by default, pending a GPU smoke test.**
Confirmed targets from the module dump: the action expert is
`model.vlm_with_expert.lm_expert` (16 layers, `q/k/v/o_proj` + `gate/up/down_proj`)
and the frozen VLM is `model.vlm_with_expert.vlm` — so `lora_expert_name_contains:
lm_expert` selects the expert and excludes the VLM. LoRA r=16 attention-only ≈
1.3M trainable; +MLP ≈ 3.4M.

Implemented pieces:
- `rl_vla_bootstrapping/policy/lora.py` (tested) — LoRALinear + attach_lora.
- Runtime grad-forward: `SmolVLARuntime.sample_cdpr_chunks_from_tensors(...,
  enable_grad=True)` (skips inference_mode; compile auto-disabled when training).
- Trainer: `attach_vla_lora(runtime)` (separate AdamW at `--vla-lr`) and
  `update_vla_lora(records)` (grad-through-VLA PPO clip + KL-to-frozen-prior,
  manual LoRA-grad all-reduce for DDP).
- Collector: capped decision-0 subsample of SmolVLA inputs (`store_vla_records`,
  `vla_update_max_records`, fp16 images) on `CollectorRound.vla_records`.
- `main()` wires attach + the per-update `update_vla_lora` call; config knobs in
  the move-to YAML (`train_vla_lora: false`).

### Smoke test (run this first, 1 GPU is fine)

```
# LoRA core
conda run -n cdpr-mjlab python -m unittest tests.test_lora
# 3-update integration: edit the config to train_vla_lora: true, then
CONFIG=configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml
RLVLA_MJWARP_WORLDS_PER_RANK=64 \
  conda run -n cdpr-mjlab python -m rl_vla_bootstrapping.cli.train \
  --config "$CONFIG" --stage rl --run-name lora_smoke --execute
# (add mjwarp_max_updates: 3, validation_every_steps: 0 to the config args for
#  the smoke run; watch vla_lora/modules>0, vla_lora/grad_norm finite, no OOM)
```

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

## Step 4 — batch / VRAM, MEASURED (2xA40, ~46 GB usable)

Sweep at 512 worlds / inference microbatch 512, LoRA attn+MLP r=16 (3.42M
trainable), with the grad-through-VLA backward confirmed live
(`vla_lora/grad_norm` 1.3-2.2, `vla_lora/records` 256):

| VLA microbatch | VRAM MiB | headroom | selected actions/s |
|---:|---:|---:|---:|
| 16 | 37714 | ~8.3 GB | 152.84 |
| 32 | 41094 | ~4.9 GB | 152.45 |
| 64 | 45488 | ~0.6 GB (unsafe) | 152.46 |
| 128 | - | OOM (extrapolated ~50 GB) | - |

- `worlds_per_rank`: unchanged for rollout (inference, no_grad) - keep 512.
- `vla_microbatch_size`: **16**. Throughput is identical across 16/32/64 (the
  LoRA update is negligible next to ~95 s of SmolVLA rollout inference) and the
  gradient is identical too (all 256 records are accumulated regardless of
  chunk size), so larger microbatches buy nothing and cost 3.4-4.4 GB per
  doubling. At 16 the backward fits under the rollout's existing allocator
  high-water mark, costing only ~0.3 GB.
- `vla_update_max_records`: 128. Peak memory is bounded by the microbatch, not
  by this, so raising it costs update time (more chunks), not VRAM.
- Adapter cost in the rollout forward, present even with the backward off:
  attention-only -3.6% selected actions/s, attn+MLP -9.2%.

Caveat that produced one wasted sweep: with the backward accidentally disabled,
VRAM was FLAT across microbatches. A flat memory curve over a swept batch
dimension means the swept thing is not running - treat it as a bug signal, not
as "the knob is free".

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
