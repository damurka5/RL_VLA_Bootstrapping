from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rl_vla_bootstrapping.policy.rank_local_grpo import (
    synchronize_equal_ddp_schedule,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise the rank-local fixed DDP schedule with two CPU ranks."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    try:
        schedule = synchronize_equal_ddp_schedule(
            local_informative_records=0 if rank == 0 else 1025,
            records_per_minibatch=512,
            ppo_epochs=4,
            device=torch.device("cpu"),
        )

        torch.manual_seed(41)
        module = DistributedDataParallel(
            torch.nn.Linear(2, 1, bias=False)
        )
        optimizer = torch.optim.SGD(module.parameters(), lr=1.0e-3)
        backward_calls = 0
        for _ in range(schedule.backward_collectives):
            optimizer.zero_grad(set_to_none=True)
            graph_loss = module(torch.ones((4, 2))).sum()
            # Rank 0 represents an update with no informative records. It still
            # traverses the same DDP graph and participates in every collective.
            loss = graph_loss * (0.0 if rank == 0 else 1.0)
            loss.backward()
            optimizer.step()
            backward_calls += 1

        flattened = torch.cat(
            [parameter.detach().reshape(-1) for parameter in module.parameters()]
        )
        minimum = flattened.clone()
        maximum = flattened.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        max_parameter_mismatch = float((maximum - minimum).abs().max().item())

        calls = torch.tensor([backward_calls], dtype=torch.int64)
        min_calls = calls.clone()
        max_calls = calls.clone()
        dist.all_reduce(min_calls, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_calls, op=dist.ReduceOp.MAX)

        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "world_size": world_size,
                        "global_max_records": schedule.global_max_records,
                        "padded_records_per_rank": (
                            schedule.padded_records_per_rank
                        ),
                        "backward_collectives": (
                            schedule.backward_collectives
                        ),
                        "minimum_backward_calls": int(min_calls.item()),
                        "maximum_backward_calls": int(max_calls.item()),
                        "max_parameter_mismatch": max_parameter_mismatch,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
