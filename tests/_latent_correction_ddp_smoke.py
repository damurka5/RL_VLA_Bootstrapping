"""Two-rank DDP update of the correction actor against a one-rank full batch.

CPU/gloo only: it checks the DDP reducer, the frozen/trainable partition and
latent-record padding across ranks, NOT MJWarp or GPU execution. Run with

    torchrun --standalone --nproc_per_node 2 tests/_latent_correction_ddp_smoke.py

from the repository root with tests/ and the repository on PYTHONPATH.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from rl_vla_bootstrapping.policy.latent_correction_policy import POLICY_ARCHITECTURE_CORRECTION
from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule, synchronize_equal_ddp_schedule
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import DistributedContext
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import SmolVLAGRPOTrainer
from test_latent_correction_policy import (
    ACTION_DIM, CHUNK, STATE_DIM, converted, inputs, latent_args, latent_records, write_legacy_checkpoint,
)


def main():
    rank = int(os.environ["RANK"])
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        legacy_path, _ = write_legacy_checkpoint(root, with_lora=False)
        source, _ = converted(root, POLICY_ARCHITECTURE_CORRECTION, legacy_path, with_lora=False)
        # Move the correction off zero so hidden layers also receive gradient.
        with torch.no_grad():
            source._unwrap(source.actor).actor.correction_output_layer().weight.normal_(
                0.0, 0.01, generator=torch.Generator().manual_seed(1))
        initial = {k: v.clone() for k, v in source._unwrap(source.actor).state_dict().items()}
        states, priors = inputs(6, 4)
        records = latent_records(source, states, priors, seed=2, advantage_seed=3)
        records["credit_stage"] = torch.tensor([0, 0, 1, 1, 2, 2]).repeat(4)
        records["candidate_id"] = torch.arange(6).repeat(4)
        n = int(records["advantage"].shape[0])  # 24 rows: 4 slots x 6 worlds
        args = latent_args(POLICY_ARCHITECTURE_CORRECTION, "--microbatch-size", "4", "--minibatch-size", "16")

        def build(distributed):
            trainer = SmolVLAGRPOTrainer(args=args, state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK,
                                         run_dir=root, device=torch.device("cpu"), distributed=distributed)
            trainer._unwrap(trainer.actor).load_state_dict(initial)
            trainer._arm_reference_guard()
            return trainer

        reference = build(DistributedContext(device="cpu"))
        torch.manual_seed(0)
        # One optimizer step over all 24 rows, matching the two-rank run where
        # each rank pads to one 16-row minibatch.
        reference.update_tensor_records(records, loss_mask=torch.ones(n),
                                        schedule=EqualDDPSchedule(32, 1, n))
        expected = torch.cat([p.detach().flatten() for p in reference._unwrap(reference.actor).parameters()])

        dist.init_process_group("gloo")
        try:
            local_rows = torch.arange(n)[torch.arange(n) % 6 < 4] if rank == 0 else torch.arange(n)[torch.arange(n) % 6 >= 4]
            local = {k: v.index_select(0, local_rows) for k, v in records.items()}
            trainer = build(DistributedContext(rank=rank, local_rank=rank, world_size=2, enabled=True))
            schedule = synchronize_equal_ddp_schedule(local_informative_records=int(local_rows.numel()),
                                                      records_per_minibatch=16, ppo_epochs=1,
                                                      device=torch.device("cpu"))
            torch.manual_seed(0)
            metrics = trainer.update_tensor_records(local, loss_mask=torch.ones(int(local_rows.numel())),
                                                    schedule=schedule)
            after = torch.cat([p.detach().flatten() for p in trainer._unwrap(trainer.actor).parameters()])
            gathered = [torch.zeros_like(after) for _ in range(2)]
            dist.all_gather(gathered, after)
            assert torch.equal(gathered[0], gathered[1]), "ranks diverged"
            assert metrics["correction/reference_max_abs_change"] == 0.0
            policy = trainer._unwrap(trainer.actor)
            for name, value in policy.actor.reference_net.state_dict().items():
                assert torch.equal(value, initial[f"actor.reference_net.{name}"]), name
            moved = float((after - torch.cat([v.flatten() for v in initial.values()])).abs().max())
            assert moved > 0.0, "nothing trained"
            # Stage-mass normalization makes the two-rank and one-rank updates
            # the same objective; the difference is summation order only.
            error = float((after - expected).abs().max()) if after.numel() == expected.numel() else float("inf")
            assert error < 1e-6, error
            if rank == 0:
                print(json.dumps({"world_size": 2, "ranks_identical": True, "reference_unchanged": True,
                                  "max_param_change": moved, "max_error_vs_full_batch": error,
                                  "optimizer_steps": metrics["optimizer_steps"]}))
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
