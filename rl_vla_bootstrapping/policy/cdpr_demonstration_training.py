"""Bounded demonstration-start curriculum feeding the existing GRPO optimizers.

No teacher action is an optimization record. Every update pairs an ordinary
rollout with a verified, freshly sampled assisted rollout (when replay passes).
The ratio is one attempt of each, not a promise of equal active group counts.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from tools.audit.probe_cdpr_demonstration_handoff import (
    collect_suffix_once, run_job, sha256,
)

FAMILIES = ('pick_up', 'put_into_bowl', 'pick_up', 'put_into_plate')


def scheduled_family(update_index):
    """Which family this update's assisted attempt comes from.

    FAMILIES lists pick_up twice, so it takes half of the assisted attempts
    while every ordinary batch still carries all four instructions.
    """
    return FAMILIES[int(update_index) % len(FAMILIES)]


def scheduled_stage(update_index, global_step, budget):
    """0 keeps the easier handoff; 1 is the earlier boundary.

    Nothing moves before half the selected-action budget is spent, and after it
    only every other cycle moves, so the earlier grasp and carry boundaries are
    introduced alongside the easier ones rather than replacing them.
    """
    if int(budget) < 1:
        raise ValueError('Selected-action budget must be positive')
    return int(int(global_step) >= int(budget) // 2 and (int(update_index) // 4) % 2 == 1)


def scheduled_job_index(update_index, rank, count):
    """Walk the bank's jobs, with the two ranks on different scenes."""
    if int(count) < 1:
        raise ValueError('No jobs to choose from')
    return ((int(update_index) // 4) * 2 + int(rank)
            + int(int(update_index) % 4 == 2)) % int(count)


def reset_seed(base, rank, update, round_index):
    return int(base) + int(rank) * 1_000_003 + int(update) * 10_000_019 + int(round_index) * 100_003


def validate_seed_partition(bank_seed, bank_rounds, train_seed, validation_seed, max_updates=5000):
    reserved = {reset_seed(bank_seed, 0, 0, r) for r in bank_rounds}
    ordinary = {reset_seed(train_seed, rank, u, 0) for rank in range(2) for u in range(max_updates)}
    evaluation = {reset_seed(validation_seed, rank, 0, r) for rank in range(2) for r in range(32)}
    if reserved & (ordinary | evaluation):
        raise ValueError('Training demonstration reset seeds overlap ordinary/evaluation seeds')


def validate_bank(path, *, config=None):
    path = Path(path).resolve()
    bank = json.loads(path.read_text())
    if bank.get('schema') != 'cdpr_training_handoffs_v1' or bank.get('split') != 'training_only':
        raise ValueError('A training-only handoff bank is required; evaluation prototype refused')
    if config is not None and sha256(config) != bank['config_sha256']:
        raise ValueError('Bank/training config hash mismatch')
    for source in bank['sources']:
        if sha256(source['path']) != source['sha256']:
            raise ValueError('Training source recording hash mismatch')
    if sha256(bank['checkpoint']) != bank['checkpoint_sha256']:
        raise ValueError('Demonstration donor hash mismatch')
    validate_seed_partition(bank['reset_seed'], bank['source_rounds'], bank['train_seed'], bank['validation_seed'])
    if bank['worlds'] != 512 or bank['group_size'] != 8:
        raise ValueError('This bounded pilot requires 512 worlds and groups of eight')
    for family in set(FAMILIES):
        if not any(j['family'] == family and j['stage'] == 0 for j in bank['jobs']):
            raise ValueError(f'No initial handoff jobs for {family}')
    return bank


def quarantine_suffix(item, report, group_size=8):
    """Exclude whole divergent groups from both update paths and gate metrics."""
    import torch
    groups = item.candidate_success.shape[0]
    valid = torch.zeros(groups, dtype=torch.bool, device=item.loss_mask.device)
    for group in report.get('groups', []):
        if not group['contains_divergence']:
            valid[group['group']] = True
    worlds = groups * group_size
    if item.loss_mask.numel() % worlds:
        raise ValueError('Unexpected collector record layout')
    mask = valid.repeat_interleave(group_size).repeat(item.loss_mask.numel() // worlds)
    item.loss_mask.mul_(mask)
    if item.usable_groups is not None:
        item.usable_groups.logical_and_(valid)
    vla = item.vla_records
    if vla is not None:
        batch = vla
        keep = valid[batch['world_index'] // group_size] & (batch['advantage'].abs() > 1e-6)
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        vla = {k: ([v[i] for i in indices.tolist()] if k == 'instruction' else v[indices])
                            for k, v in batch.items()}
    # Absent groups are not failures of a sampled instruction. All assisted
    # groups skip some approach; never feed them to an approach promotion gate.
    ids = item.group_instruction_ids.clone()
    ids[~valid] = -1
    return replace(item, group_instruction_ids=ids, group_skips_approach=torch.ones_like(valid),
                   vla_records=vla)


def balanced_vla_batches(batches, cap, group_size=8):
    """Reserve complete candidate groups across rollout arms before concatenation.

    Zero-advantage groups consume no budget. Equal arm opportunity avoids the
    first ordinary rollout exhausting the cap before the assisted rollout.
    """
    import torch
    queues = []
    for b in batches:
        if b is None:
            continue
        n = int(b['advantage'].numel())
        if n % group_size:
            raise ValueError('LoRA batch does not contain complete groups')
        live = (b['advantage'].reshape(-1, group_size).abs() > 1e-6).any(dim=1)
        queues.append((b, torch.nonzero(live, as_tuple=False).flatten().tolist()))
    chosen = [[] for _ in queues]
    budget = cap // group_size
    while budget and any(q for _, q in queues):
        for i, (_, q) in enumerate(queues):
            if q and budget:
                g = q.pop(0); chosen[i].extend(range(g * group_size, (g + 1) * group_size)); budget -= 1
    result = []
    for (b, _), rows in zip(queues, chosen):
        if rows:
            idx = torch.tensor(rows, device=b['advantage'].device)
            result.append({k: ([v[i] for i in rows] if k == 'instruction' else v[idx]) for k, v in b.items()})
    return result


class DemonstrationTraining:
    def __init__(self, bank_path, *, collector, args, task_metadata, run_dir, rank):
        from rl_vla_bootstrapping.policy.mjwarp_rank_local_collector import BatchedReverseFrontierResetter
        from tools.audit.xy_approach_probe import _restore_approach_curriculum, _load_checkpoint
        self.bank_path = str(Path(bank_path).resolve())
        self.bank_hash = sha256(self.bank_path)
        self.bank = validate_bank(self.bank_path, config=args.config)
        if args.resume_checkpoint:
            raise ValueError('Bounded demo pilot uses a fresh optimizer; resume is not supported yet')
        if int(args.seed) != self.bank['train_seed'] or int(args.validation_seed) != self.bank['validation_seed']:
            raise ValueError('Training seeds differ from the declared bank partition')
        if int(args.worlds_per_rank) != self.bank['worlds'] or int(args.grpo_group_size) != 8:
            raise ValueError('Training layout differs from source replay layout')
        self.collector, self.rank = collector, int(rank)
        self.log = Path(run_dir) / f'demonstration_rank{rank}.jsonl'
        self.cache = OrderedDict()
        self.misses = {f: 0 for f in set(FAMILIES)}
        self.total_attempts = {f: 0 for f in set(FAMILIES)}
        self.total_usable = {f: 0 for f in set(FAMILIES)}
        source_resetter = BatchedReverseFrontierResetter(
            backend=collector.backend, layout=collector.layout,
            curriculum=collector.resetter.curriculum, rank=0, base_seed=self.bank['reset_seed'],
            instruction_types=args.instruction_types, allowed_objects=args.allowed_objects,
            frontier_probability=1., rehearsal_probability=0., balanced_target_catalogs=True,
            task_metadata=task_metadata,
        )
        donor = _load_checkpoint(Path(self.bank['checkpoint']))
        _restore_approach_curriculum(source_resetter, args=args, task_metadata=task_metadata,
                                     extra_state=dict(donor.get('extra_state') or {}))
        self.world = SimpleNamespace(torch=collector.torch, backend=collector.backend,
                                     collector=collector, resetter=source_resetter, device=collector.device)
        collector.active_only_inference = True
        self.budget = int(args.max_train_steps)

    def _recording(self, index):
        from tools.audit.sil_record import _Recording
        if index not in self.cache:
            self.cache[index] = _Recording.from_npz(Path(self.bank['sources'][index]['path']))
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def collect_update(self, update_index, global_step):
        import torch
        if update_index >= 5000:
            raise RuntimeError('Bounded pilot exceeded 5000 updates; inspect selected-action progress')
        # Equal rollout-attempt allocation, with pickup receiving half of the
        # assisted attempts. Keep all four families in every ordinary batch.
        ordinary = self.collector.collect_round(update_index=update_index, round_index=0)
        family = scheduled_family(update_index)
        # After half the budget, retain half of the assisted attempts at the
        # easier stage and move half earlier. Distances and terminal success
        # predicates stay fixed.
        stage = scheduled_stage(update_index, global_step, self.budget)
        jobs = [j for j in self.bank['jobs'] if j['family'] == family and j['stage'] == stage]
        if not jobs:
            raise ValueError(f'Bank lacks requested stage {stage} for {family}')
        job = jobs[scheduled_job_index(update_index, self.rank, len(jobs))]
        recording = self._recording(job['source_index'])
        collected = []
        def consume(reset):
            # Replay seeds are for the source physics only. Restore training RNG
            # before VLA sampling, and vary the collector's independent offset RNG.
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.collector.device)
            result = collect_suffix_once(self.collector, reset, round_index=1, update_index=update_index)
            collected.append(result)
            return result
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.collector.device)
        try:
            report = run_job(self.world, recording, job['job'],
                             'pick_up' if family == 'pick_up' else 'placement',
                             group_size=8, position_tolerance=.002, opening_tolerance=.03,
                             suffix_decisions=40, seed_torch=self.bank['torch_seed'],
                             suffix_consumer=consume)
        finally:
            if not collected:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, self.collector.device)
        rounds = [ordinary]
        usable = 0
        clean = [g for g in report.get('groups', []) if not g['contains_divergence']]
        if collected:
            assisted = quarantine_suffix(collected[0], report)
            usable = (0 if assisted.usable_groups is None
                      else int(assisted.usable_groups.sum()))
            rounds.append(assisted)
        self.total_attempts[family] += 1
        self.total_usable[family] += usable
        self.misses[family] = 0 if usable else self.misses[family] + 1
        report.update(update_index=update_index, global_step_before_update=global_step,
                      family=family, stage=stage, source_index=job['source_index'],
                      collection_only=True, bank_sha256=self.bank_hash,
                      accepted_clean_groups=len(clean), usable_clean_groups=usable)
        with self.log.open('a') as f:
            f.write(json.dumps(report, default=str) + '\n')
        metrics = {
            'demo/attempts': 1., 'demo/accepted_groups': float(len(clean)),
            'demo/variable_groups': float(usable),
            'demo/successes': float(sum(g['successes'] for g in clean)),
            'demo/candidates': float(len(clean) * 8),
            'demo/prefix_actions': float(report['prefix_replayed_active_actions']),
            'demo/prefix_physics_world_steps': float(report['prefix_physics_world_steps']),
            'demo/suffix_seconds': float(report.get('suffix_wall_seconds', 0)),
            'demo/teacher_loss_rows': 0.,
            'demo/ordinary_groups': float(ordinary.candidate_success.shape[0]),
            'demo/ordinary_loss_rows': float(ordinary.loss_mask.sum()),
            'demo/assisted_loss_rows': float(rounds[-1].loss_mask.sum()) if collected else 0.,
            'demo/assisted_lora_rows': float(rounds[-1].vla_records['advantage'].numel()) if collected and rounds[-1].vla_records else 0.,
            'demo/stalled_ranks': float(self.misses[family] >= 12),
        }
        for f in set(FAMILIES):
            metrics[f'demo/attempts/{f}'] = float(f == family)
            metrics[f'demo/usable_groups/{f}'] = float(usable if f == family else 0)
        return rounds, metrics
