"""Reproduce the attachment audit; NumPy only, no simulator or weight loading.

Run from the repository root:
python3 docs/artifacts/demo_grpo_pilot_20260910/analyze.py /path/to/attachments
The directory must contain baseline/, final_eval/, and train.log.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(source):
    output = {'source_directory': str(source), 'sha256': {}, 'arms': {},
              'recorded_reset_mismatches': [], 'reset_max_abs_difference_m': {}}
    arrays = {}
    for arm in ('baseline', 'final_eval'):
        counts = {}
        arrays[arm] = []
        for path in sorted((source / arm).glob('record_*.npz')):
            with np.load(path, allow_pickle=False) as archive:
                a = {key: archive[key] for key in archive.files}
            arrays[arm].append(a)
            with path.with_name(path.name.replace('record_', 'episodes_')).with_suffix('.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            success = (a['success'] & a['active']).any(axis=0)
            held = a['caught_target'] & (a['gripper_opening'] <= .94) & a['active']
            world = np.arange(len(rows))
            target = a['target_slots']
            reset_target = a['reset_object_xyz'][world, target]
            lift = a['object_xyz'][:, world, target, 2] - reset_target[None, :, 2]
            # Descriptive measurement, not a replacement task-success predicate.
            lifted = (held & (lift >= .05)).any(axis=0)
            for w, row in enumerate(rows):
                assert int(row['world']) == w
                assert (row['success'] == 'True') == bool(success[w])
                name = row['instruction']
                d = counts.setdefault(name, dict(episodes=0, successes=0, reset_groups=0,
                    ever_held=0, held_lift_5cm_from_pre_action_reset=0,
                    success_with_held_lift_5cm=0, starts_caught_first_recorded_step=0,
                    diverged_episodes=0, inside_goal_episodes=0, outside_goal_episodes=0,
                    outside_goal_successes=0, outside_goal_reset_groups=0,
                    outside_goal_success_with_held_lift_5cm=0,
                    reset_ee_target_xy_max_m=0.0, horizons={}))
                d['episodes'] += 1
                d['successes'] += int(success[w])
                d['ever_held'] += int(held[:, w].any())
                d['held_lift_5cm_from_pre_action_reset'] += int(lifted[w])
                d['success_with_held_lift_5cm'] += int(success[w] and lifted[w])
                d['starts_caught_first_recorded_step'] += int(a['caught_target'][0, w])
                d['diverged_episodes'] += int(a['diverged_world_mask'][w])
                d['reset_ee_target_xy_max_m'] = max(d['reset_ee_target_xy_max_m'],
                    float(np.linalg.norm(a['reset_ee_xyz'][w, :2] - reset_target[w, :2])))
                horizon = str(int(a['horizons'][w]))
                d['horizons'][horizon] = d['horizons'].get(horizon, 0) + 1
                if w % 8 == 0:
                    assert len({r['instruction'] for r in rows[w:w+8]}) == 1
                    d['reset_groups'] += 1
                if name.startswith('put_into_'):
                    distance = np.linalg.norm(reset_target[w, :2] -
                        a['reset_object_xyz'][w, a['reference_slots'][w], :2])
                    outside = distance > (.091 if name == 'put_into_plate' else .057)
                    d['inside_goal_episodes'] += int(not outside)
                    d['outside_goal_episodes'] += int(outside)
                    d['outside_goal_successes'] += int(outside and success[w])
                    d['outside_goal_reset_groups'] += int(outside and w % 8 == 0)
                    d['outside_goal_success_with_held_lift_5cm'] += int(outside and success[w] and lifted[w])
        assert len(arrays[arm]) == 3
        summary = json.loads((source / arm / 'summary.json').read_text())
        for name, d in counts.items():
            assert d['successes'] == summary['by_instruction'][name]['successes']
            assert d['episodes'] == summary['by_instruction'][name]['episodes']
            d['rate'] = d['successes'] / d['episodes']
            d['outside_goal_rate'] = (d['outside_goal_successes'] / d['outside_goal_episodes']
                                      if d['outside_goal_episodes'] else None)
        output['arms'][arm] = counts
        for path in sorted((source / arm).iterdir()):
            if path.is_file():
                output['sha256'][str(path.relative_to(source))] = digest(path)
    discrete = ('round_index', 'instruction_ids', 'target_slots', 'reference_slots',
                'second_reference_slots', 'horizons', 'instructions', 'target_catalog_ids',
                'actions_per_decision')
    continuous = ('reset_object_xyz', 'reset_ee_xyz', 'initial_target_xyz',
                  'support_surface_z', 'release_threshold', 'target_rest_height')
    for a, b in zip(arrays['baseline'], arrays['final_eval']):
        for key in discrete:
            if not np.array_equal(a[key], b[key]):
                output['recorded_reset_mismatches'].append([int(a['round_index']), key])
        for key in continuous:
            delta = float(np.max(np.abs(a[key] - b[key])))
            output['reset_max_abs_difference_m'][key] = max(
                output['reset_max_abs_difference_m'].get(key, 0), delta)
            if not np.allclose(a[key], b[key], rtol=0, atol=1e-6):
                output['recorded_reset_mismatches'].append([int(a['round_index']), key])
    log = (source / 'train.log').read_text()
    output['sha256']['train.log'] = digest(source / 'train.log')
    output['validation_lines'] = re.findall(r'\[smolvla-mjwarp\] validation[^\r\n]+', log)
    output['last_update'] = int(re.findall(r'update=(\d+)', log)[-1])
    output['checkpoint_steps'] = [int(s) for s in re.findall(r'checkpoint=[^\r\n]+/step_(\d+)/', log)]
    output['notes'] = [
        'Reset identity covers recorded fields only; full orientations/velocities are absent.',
        'Eight candidate worlds share a reset group; episode counts are not independent trials.',
        'Held lift uses pre-action target height and opening <=0.94; it is a descriptive trace metric.',
        'First recorded caught state is post-action, not a serialized pre-action grasp state.',
        'Outside-goal uses XY distance > configured success radius without an extra transport margin.',
        'All reported raw scores retain diverged worlds, matching supplied summaries.',
    ]
    destination = Path(__file__).with_name('attachment_analysis.json')
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).expanduser().resolve())
