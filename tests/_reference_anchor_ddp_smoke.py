"""Two-rank DDP update with the reference anchor on: no reducer fault, ranks agree.

The anchor is added inside the last micro-batch's backward through the
unwrapped module, so DDP must still see exactly one backward per micro-batch
and every parameter's gradient in it. Run with:

    cd tests && python -m torch.distributed.run --nproc_per_node 2 _reference_anchor_ddp_smoke.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from rl_vla_bootstrapping.policy.rank_local_grpo import synchronize_equal_ddp_schedule
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import DistributedContext
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import SmolVLAGRPOTrainer
from test_grpo_episode_offset_exploration import _args, _trainer
from test_reference_anchor import build, perturb, write_bank, zero_advantage_records


def main():
    rank = int(os.environ["RANK"])
    torch.set_num_threads(1)
    root = Path(tempfile.gettempdir()) / f"anchor_ddp_smoke_{os.environ.get('MASTER_PORT', '0')}"
    root.mkdir(exist_ok=True)
    args = _args("--microbatch-size", "2", "--entropy-coef", "0", "--action-l2", "0", "--learning-rate", "1e-2")
    torch.manual_seed(5)
    seed_trainer = _trainer(args, root)
    initial = {k: v.clone() for k, v in seed_trainer._unwrap(seed_trainer.actor).state_dict().items()}
    if rank == 0:
        torch.save({"policy": initial}, root / "reference.pt")
        write_bank(root / "bank.npz", seed_trainer._unwrap(seed_trainer.actor))
    dist.init_process_group("gloo")
    try:
        dist.barrier()
        trainer = SmolVLAGRPOTrainer(args=args, state_dim=6, action_dim=5, chunk_size=2,
            run_dir=root, device=torch.device("cpu"),
            distributed=DistributedContext(rank=rank, local_rank=rank, world_size=2, enabled=True))
        base = trainer._unwrap(trainer.actor)
        base.load_state_dict(initial)
        anchor = build(trainer, root, coef=1.0)
        anchor.generator.manual_seed(100 + rank)  # different rows per rank, as in training
        trainer.reference_anchor = anchor
        perturb(base, scale=0.1)  # same perturbation on both ranks
        before = anchor.full_kl(base)
        records = zero_advantage_records(trainer, n=6 if rank == 0 else 3)
        n = len(records["advantage"])
        for _ in range(3):
            schedule = synchronize_equal_ddp_schedule(local_informative_records=n,
                records_per_minibatch=4, ppo_epochs=1, device=torch.device("cpu"))
            metrics = trainer.update_tensor_records(records, loss_mask=torch.ones(n), schedule=schedule)
        flat = torch.cat([p.detach().flatten() for p in base.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        rank_gap = float((gathered[0] - gathered[1]).abs().max())
        assert rank_gap == 0.0, rank_gap
        assert metrics["anchor/kl_bank"] < before, (metrics["anchor/kl_bank"], before)
        if rank == 0:
            print(json.dumps({"world_size": 2, "rank_parameter_gap": rank_gap,
                              "kl_before": before, "kl_after": metrics["anchor/kl_bank"],
                              "optimizer_steps": metrics["optimizer_steps"]}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
