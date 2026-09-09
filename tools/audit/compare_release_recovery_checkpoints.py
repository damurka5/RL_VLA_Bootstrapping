"""Compare the completed continuation with both placement peaks on fixed scenes.

Run from the repository's cdpr-mjlab environment after training has exited.
No training, checkpoint promotion, or checkpoint writes are performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASKS = ('move_to_object', 'pick_up', 'put_into_plate', 'put_into_bowl')


def resolve_run(run: Path | None, root: Path = ROOT) -> Path:
    if run is not None:
        result = run.expanduser().resolve()
        if not (result / 'rl').is_dir():
            raise ValueError(f'Expected a run directory containing rl/: {result}')
        return result
    candidates = sorted(p for p in (root / 'runs').glob('release_recovery_continue_3m_*')
                        if (p / 'rl').is_dir())
    if len(candidates) != 1:
        raise ValueError('Pass --run-dir explicitly; expected exactly one continuation, found: '
                         + ', '.join(str(p) for p in candidates))
    return candidates[0]


def select_checkpoints(run: Path, plate_step: int, bowl_step: int,
                       min_final_step: int) -> dict[str, Path]:
    numbered = {}
    for path in (run / 'rl').glob('step_*/smolvla_grpo_adapter.pt'):
        match = re.fullmatch(r'step_(\d+)', path.parent.name)
        if match:
            step = int(match[1])
            if step in numbered:
                raise ValueError(f'Ambiguous checkpoint names at step {step}')
            numbered[step] = path
    if not numbered:
        raise ValueError(f'No numbered checkpoints in {run / "rl"}')
    final_step = max(numbered)
    if final_step < min_final_step:
        raise ValueError(f'Latest step {final_step:,} is below {min_final_step:,}; '
                         'wait for the planned training to finish.')
    for step in (plate_step, bowl_step):
        if step not in numbered:
            raise ValueError(f'Missing peak checkpoint at step {step:,} in {run}')
    return {'final': numbered[final_step], 'plate_peak': numbered[plate_step],
            'bowl_peak': numbered[bowl_step]}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def pairing_issues(a, b) -> list[str]:
    """Conservatively gate pairing on available fields, not seed labels alone."""
    issues = []
    for name in ('round_index', 'actions_per_decision', 'pick_lift_success_height',
                 'instruction_ids', 'instructions', 'target_slots', 'reference_slots',
                 'second_reference_slots', 'horizons', 'starts_grasped',
                 'target_catalog_ids', 'release_threshold', 'target_rest_height',
                 'support_surface_z'):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            issues.append(name)
    # This is the earliest stored layout, AFTER the first action. It can differ
    # legitimately between policies; mismatch is not proof of different resets.
    # The older initial_target_xyz predates composed-object repositioning, so
    # neither can certify a true pre-action reset for these legacy recordings.
    aa, bb = a.object_xyz[0], b.object_xyz[0]
    if aa.shape != bb.shape or not np.allclose(aa, bb, atol=1e-6, rtol=0, equal_nan=False):
        issues.append('object_xyz_at_step_0')
    return issues


def inspect_existing(output: Path) -> int:
    """Explain an existing comparison's pairing gate without GPU work or writes."""
    from tools.audit.sil_record import _Recording
    output = output.expanduser().resolve()
    report = json.loads((output / 'comparison.json').read_text())
    print('Evaluation artifacts already exist. This inspection does not rerun evaluation.')
    for peak, check in report['pairing'].items():
        print(f'{peak}: recorded_scene_identity_matches={check["recorded_scene_identity_matches"]}')
        for row in check['rounds']:
            print(f'  round {row["round"]}: {row["issues"] or "no recorded-field mismatch"}')
            if 'object_xyz_at_step_0' in row['issues']:
                file = f'record_{row["round"]:02d}.npz'
                a = _Recording.from_npz(output / 'final' / file)
                b = _Recording.from_npz(output / peak / file)
                if a.object_xyz.shape[1:] == b.object_xyz.shape[1:]:
                    delta = np.abs(a.object_xyz[0].astype(np.float64) - b.object_xyz[0].astype(np.float64))
                    print(f'    first POST-ACTION coordinate max difference: {delta.max():.9g} m')
    print('Step-0 object poses were logged AFTER the first policy action. Differences can be '
          'policy effects; these recordings cannot certify pre-action reset identity. '
          'Do not reinterpret the gate as a training crash or relax its tolerance blindly.')
    return 0


def comparison_exit_code(report: dict) -> int:
    # Different first-action outcomes alone are not a failed comparison job.
    # Keep pairing unverified and paired_verdicts=None; do not waive metadata
    # mismatches such as different instructions, horizons or success thresholds.
    issues = [issue for check in report['pairing'].values()
              for row in check['rounds'] for issue in row['issues']]
    return 2 if any(issue != 'object_xyz_at_step_0' for issue in issues) else 0


def outcome_counts(recording, instruction_name) -> dict:
    result = {}
    for instruction_id in np.unique(recording.instruction_ids):
        name = instruction_name(instruction_id)
        mask = recording.instruction_ids == instruction_id
        if name not in TASKS:
            raise ValueError(f'Unexpected instruction: {name}')
        if name.startswith('put_into_'):
            if np.any(recording.starts_grasped[mask]) or np.any(recording.horizons[mask] != 40):
                raise ValueError('Comparison requires uncaught containers with 40 decisions')
        result[name] = {'episodes': int(mask.sum()),
                        'successes': int(recording.episode_success[mask].sum())}
    return result


def summarize(output: Path, rounds: int, first_round: int) -> dict:
    from tools.audit.sil_record import _Recording, _instruction_name

    report = {'arms': {}, 'pairing': {}, 'final_minus_peak': {}}
    for arm in ('final', 'plate_peak', 'bowl_peak'):
        totals = {name: {'episodes': 0, 'successes': 0} for name in TASKS}
        files = sorted((output / arm).glob('record_*.npz'))
        if len(files) != rounds:
            raise ValueError(f'{arm}: expected {rounds} recordings, found {len(files)}')
        for index in range(first_round, first_round + rounds):
            recording = _Recording.from_npz(output / arm / f'record_{index:02d}.npz')
            if recording.round_index != index:
                raise ValueError(f'{arm}: unexpected round identity in record_{index:02d}')
            for name, counts in outcome_counts(recording, _instruction_name).items():
                for key, value in counts.items():
                    totals[name][key] += value
        for name, counts in totals.items():
            if not counts['episodes']:
                raise ValueError(f'{arm}: no evaluation episodes for {name}')
            counts['rate'] = counts['successes'] / counts['episodes']
        report['arms'][arm] = totals
    for peak in ('plate_peak', 'bowl_peak'):
        checks = []
        flips = {name: {'final_only': 0, 'peak_only': 0} for name in TASKS}
        for index in range(first_round, first_round + rounds):
            a = _Recording.from_npz(output / 'final' / f'record_{index:02d}.npz')
            b = _Recording.from_npz(output / peak / f'record_{index:02d}.npz')
            issues = pairing_issues(a, b)
            checks.append({'round': index, 'issues': issues})
            if not issues:
                for instruction_id in np.unique(a.instruction_ids):
                    mask = a.instruction_ids == instruction_id
                    counts = flips[_instruction_name(instruction_id)]
                    counts['final_only'] += int((mask & a.episode_success & ~b.episode_success).sum())
                    counts['peak_only'] += int((mask & b.episode_success & ~a.episode_success).sum())
        paired = all(not check['issues'] for check in checks)
        report['pairing'][peak] = {'recorded_scene_identity_matches': paired, 'rounds': checks}
        report['final_minus_peak'][peak] = {
            name: {'delta': report['arms']['final'][name]['rate'] - report['arms'][peak][name]['rate'],
                   'paired_verdicts': flips[name] if paired else None} for name in TASKS}
    report['interpretation'] = (
        'Development comparison; no automatic promotion. The earliest object positions '
        'were recorded after the first action, not at the true reset. A pairing mismatch '
        'does not by itself establish different initial scenes. Matching fields likewise '
        'does not establish identical simulator hidden state. Candidates sharing a reset '
        'group are correlated; these are descriptive rates, not significance claims. '
        'Confirm the selected shared checkpoint on fresh scene seeds.')
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Final versus placement peaks', '',
             '| Instruction | Final | Plate peak | Bowl peak |', '|---|---:|---:|---:|']
    for name in TASKS:
        cells = []
        for arm in ('final', 'plate_peak', 'bowl_peak'):
            c = report['arms'][arm][name]
            cells.append(f"{c['successes']}/{c['episodes']} = {c['rate']:.4f}")
        lines.append('| ' + ' | '.join([name] + cells) + ' |')
    lines.extend(['', report['interpretation'], ''])
    for peak, check in report['pairing'].items():
        lines.append(f"Recorded scene identity, final vs {peak}: {check['recorded_scene_identity_matches']}")
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return report


def run_logged(command: list[str], log: Path) -> None:
    print(shlex.join(command), flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Auto-resolve only if exactly one continuation exists')
    parser.add_argument('--plate-step', type=int, default=1505251)
    parser.add_argument('--bowl-step', type=int, default=2117145)
    parser.add_argument('--min-final-step', type=int, default=3527307)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--round-index', type=int, default=0)
    parser.add_argument('--seed-torch', type=int, default=0)
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--inspect-existing', type=Path,
                        help='Explain a saved comparison and its pairing gate, CPU only, without writes')
    args = parser.parse_args(argv)
    if args.inspect_existing is not None:
        return inspect_existing(args.inspect_existing)
    if args.rounds <= 0 or args.round_index < 0 or args.min_final_step <= 0:
        parser.error('rounds/min-final-step must be positive; round-index must be nonnegative')
    try:
        run = resolve_run(args.run_dir)
        checkpoints = select_checkpoints(run, args.plate_step, args.bowl_step, args.min_final_step)
    except ValueError as error:
        parser.error(str(error))
    output = (args.output or run / 'eval' / ('final_vs_peaks_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    config = ROOT / 'configs/examples/cdpr_smolvla_release_recovery_pilot.yaml'
    eval_config = output / 'evaluation_config.yaml'
    commands = {}
    for arm, checkpoint in checkpoints.items():
        commands[arm] = [sys.executable, str(ROOT / 'tools/audit/sil_record.py'), '--mode', 'record',
                         '--checkpoint', str(checkpoint), '--config', str(config),
                         '--worlds', '512', '--group-size', '8', '--rounds', str(args.rounds),
                         '--round-index', str(args.round_index), '--devices', args.devices,
                         '--seed-torch', str(args.seed_torch), '--output', str(output / arm)]
        print(f'{arm}: {checkpoint}', flush=True)
    if args.dry_run:
        for command in commands.values():
            print(shlex.join(command))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    eval_config.write_bytes(config.read_bytes())
    manifest = {'run': str(run), 'config': str(config), 'config_sha256': sha256(eval_config),
                'checkpoints': {arm: {'path': str(path), 'sha256': sha256(path)}
                                for arm, path in checkpoints.items()},
                'rounds': list(range(args.round_index, args.round_index + args.rounds)),
                'seed_torch': args.seed_torch, 'devices': args.devices,
                'worlds_per_round': 512, 'group_size': 8,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'commands': commands}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'tracked_changes.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    for arm, command in commands.items():
        # Config paths are relative to its original directory. Keep the copy
        # for provenance; moving the live config would change asset resolution.
        if sha256(config) != manifest['config_sha256']:
            raise RuntimeError('Evaluation config changed between arms')
        run_logged(command, output / f'{arm}.log')
        if sha256(checkpoints[arm]) != manifest['checkpoints'][arm]['sha256']:
            raise RuntimeError(f'{arm} checkpoint changed during evaluation')
    report = summarize(output, args.rounds, args.round_index)
    if sha256(config) != manifest['config_sha256']:
        raise RuntimeError('Evaluation config changed during comparison')
    for arm in checkpoints:
        run_logged([sys.executable, str(ROOT / 'tools/audit/placement_failure_decomposition.py'),
                    '--recordings', str(output / arm / 'record_*.npz'), '--config', str(config),
                    '--output', str(output / arm / 'decomposition')], output / f'{arm}_decomposition.log')
    print(f'Comparison saved to {output / "comparison.json"}', flush=True)
    if not all(v['recorded_scene_identity_matches'] for v in report['pairing'].values()):
        code = comparison_exit_code(report)
        print(f'Evaluation completed. Pairing is unverified; exit status {code}. '
              'Post-action pose differences alone do not fail the job; metadata mismatches return 2. '
              'Rates and decompositions are saved; paired verdict claims remain suppressed. '
              'Use --inspect-existing with this output directory before making paired claims.', flush=True)
        return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
