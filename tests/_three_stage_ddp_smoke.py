"""Compare real two-rank staged updates against one-rank full-batch updates."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from rl_vla_bootstrapping.policy.rank_local_grpo import EqualDDPSchedule, synchronize_equal_ddp_schedule
from rl_vla_bootstrapping.policy.smolvla_finetune_cdpr import DistributedContext
from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import SmolVLAGRPOTrainer
from rl_vla_bootstrapping.policy.smolvla_grpo_mjwarp_cdpr import _synchronize_update_metrics_once
from test_grpo_episode_offset_exploration import _args, _trainer


def main():
    rank = int(os.environ['RANK'])
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        args = _args('--microbatch-size', '2', '--entropy-coef', '0', '--action-l2', '0')
        torch.manual_seed(31)
        reference = _trainer(args, Path(directory))
        initial = {k: v.clone() for k, v in reference.actor.state_dict().items()}
        states, priors = torch.zeros(6, 6), torch.zeros(6, 2, 5)
        actions, probs, _ = reference.sample_action_chunks_tensor(
            states=states, priors=priors, action_count=2, generator=torch.Generator().manual_seed(4))
        records = dict(state=states, prior=priors, action=actions[:, 0],
                       action_index=torch.zeros(6, dtype=torch.long), old_log_prob=probs[:, 0],
                       advantage=torch.tensor([1., -1., 1., -1., 2.646, 2.646]),
                       credit_stage=torch.tensor([0, 0, 0, 0, 1, 2]))
        # Both disjoint-stage ranks and an entirely empty rank must match.
        references = []
        for mask in (torch.ones(6), torch.tensor([0., 0., 0., 0., 1., 1.])):
            ref = _trainer(args, Path(directory))
            ref.actor.load_state_dict(initial)
            ref.update_tensor_records(records, loss_mask=mask, schedule=EqualDDPSchedule(4, 1, int(mask.sum())))
            references.append(torch.cat([p.detach().flatten() for p in ref.actor.parameters()]))
        dist.init_process_group('gloo')
        errors = []
        try:
            local_slice = slice(0, 4) if rank == 0 else slice(4, 6)
            local = {k: v[local_slice] for k, v in records.items()}
            for case in range(2):
                trainer = SmolVLAGRPOTrainer(args=args, state_dim=6, action_dim=5, chunk_size=2,
                    run_dir=Path(directory), device=torch.device('cpu'),
                    distributed=DistributedContext(rank=rank, local_rank=rank, world_size=2, enabled=True))
                trainer._unwrap(trainer.actor).load_state_dict(initial)
                n = len(local['advantage'])
                mask = torch.zeros(n) if case == 1 and rank == 0 else torch.ones(n)
                schedule = synchronize_equal_ddp_schedule(local_informative_records=int(mask.sum()),
                    records_per_minibatch=4, ppo_epochs=1, device=torch.device('cpu'))
                result = trainer.update_tensor_records(local, loss_mask=mask, schedule=schedule)
                flat = torch.cat([p.detach().flatten() for p in trainer.actor.parameters()])
                error = float((flat - references[case]).abs().max())
                assert error < 2e-6, (case, rank, error)
                assert result['optimizer_steps'] == 1
                errors.append(error)
                empty = trainer.update_tensor_records(local, loss_mask=torch.zeros(n), schedule=EqualDDPSchedule(4, 1, 0))
                after = torch.cat([p.detach().flatten() for p in trainer.actor.parameters()])
                assert torch.equal(flat, after)
                assert empty['optimizer_steps'] == 0
            metrics = _synchronize_update_metrics_once({
                'filtered_record_fraction': .25 if rank == 0 else .75,
                'three_stage/pickup_count': float(rank), 'three_stage/episodes': 2.,
                'three_stage/pickup_loss_mass': 0. if rank == 0 else 2.,
            }, device=torch.device('cpu'))
            assert metrics['filtered_record_fraction'] == .5
            assert metrics['three_stage/pickup_rate'] == .25
            assert metrics['three_stage/pickup_loss_mass'] == 1.
            if rank == 0:
                print(json.dumps({'world_size': 2, 'max_parameter_errors': errors,
                                  'empty_update_unchanged': True, 'metric_reduction_correct': True}))
        finally:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
