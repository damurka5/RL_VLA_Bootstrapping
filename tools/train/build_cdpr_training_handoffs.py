"""Prepare training-only, hashed replay jobs from freshly reserved source rounds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.audit.probe_cdpr_demonstration_handoff import plan_boundaries, sha256
from tools.audit.extract_cdpr_transition_demonstrations import select_episodes
from tools.audit.sil_record import _Recording
from rl_vla_bootstrapping.policy.cdpr_demonstration_training import validate_seed_partition


def build_bank(*, manifest, checkpoint, config, output, first_round=1000, rounds=12, min_scenes=8):
    import torch
    import yaml
    try:
        donor = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        donor = torch.load(checkpoint, map_location='cpu')
    raw = yaml.safe_load(Path(config).read_text())
    args = raw['training']['rl']['args']
    reset_seed = int(donor['args']['validation_seed'])
    source_rounds = list(range(first_round, first_round + rounds))
    validate_seed_partition(reset_seed, source_rounds, args['seed'], args['validation_seed'])
    extracted = json.loads(Path(manifest).read_text())
    if sorted(s['round_index'] for s in extracted['sources']) != source_rounds:
        raise ValueError('Source round list is not the reserved training partition')
    sources, jobs = [], []
    counts = {(f, st): set() for f in ('pick_up', 'put_into_plate', 'put_into_bowl') for st in (0, 1)}
    for source in extracted['sources']:
        if sha256(source['path']) != source['sha256']:
            raise ValueError('Source changed after extraction')
        r = _Recording.from_npz(Path(source['path']))
        if r.worlds != 512 or r.actions_per_decision != 4:
            raise ValueError('Source layout must be 512 worlds, eight candidates, four actions/decision')
        if r.reset_object_xyz is None or r.reset_ee_xyz is None or r.diverged_world_mask is None:
            raise ValueError('Fresh sources must record pre-action poses and per-world divergence')
        episodes, _ = select_episodes(r)
        source_index = len(sources); sources.append(source)
        for family in ('pick_up', 'put_into_plate', 'put_into_bowl'):
            selected = [e for e in episodes if family == 'pick_up' or e['source_instruction'] == family]
            for e in selected:
                e['episode_uid'] = f"{source['sha256']}_w{e['world']}"
            for stage in (0, 1):
                plans, _ = plan_boundaries(r, selected, 'pick_up' if family == 'pick_up' else 'placement',
                    max_boundaries=40, max_groups=64, boundary_backoff=(2 if family == 'pick_up' else 4 * stage),
                    before_grasp=bool(stage and family == 'pick_up'))
                for job in plans:
                    jobs.append(dict(family=family, stage=stage, source_index=source_index, job=job))
                    counts[family, stage].update((r.round_index, e['world'] // 8) for e in job['episodes'])
    for (family, stage), scenes in counts.items():
        minimum = min_scenes if stage == 0 else max(2, min_scenes // 2)
        if len(scenes) < minimum:
            raise ValueError(f'Insufficient training sources: {family} stage {stage}: {len(scenes)} < {minimum}; increase DEMO_ROUNDS')
    random.Random(17000000).shuffle(jobs)
    bank = dict(schema='cdpr_training_handoffs_v1', split='training_only',
                checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint),
                config=str(Path(config).resolve()), config_sha256=sha256(config),
                extraction_manifest=str(Path(manifest).resolve()), extraction_sha256=sha256(manifest),
                reset_seed=reset_seed, torch_seed=0, train_seed=int(args['seed']),
                validation_seed=int(args['validation_seed']), source_rounds=source_rounds,
                worlds=512, group_size=8, sources=sources, jobs=jobs,
                unique_scene_counts={f'{f}/stage{st}':len(v) for (f,st),v in counts.items()},
                limitations=['Replay validation is still required before each suffix.',
                             'Distinct reset seeds do not imply distinct object categories.',
                             'Stage 1 moves pickup before grasp and placement earlier along held carry.'])
    Path(output).write_text(json.dumps(bank, indent=2)+'\n')
    print(json.dumps({'jobs':len(jobs), 'unique_scene_counts':bank['unique_scene_counts'], 'bank':str(output)}, indent=2))
    return bank


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','checkpoint','config','output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--first-round', type=int, default=1000)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--min-scenes', type=int, default=8)
    a=p.parse_args(argv)
    build_bank(**vars(a))
if __name__ == '__main__':
    main()
