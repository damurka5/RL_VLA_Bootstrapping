"""Verify demonstrated prefixes, clone live states, collect fresh GRPO suffixes.

This is an inference/collection audit, with ZERO optimizer updates. It replays
the original round at a common decision boundary: inactive actions do not freeze
MJWarp physics, so different prefix lengths cannot share a replay batch. Legacy
NPZs are action traces, not simulator snapshots. Matching their replay supports
the handoff but cannot certify unrecorded pre-action or hidden state identity.
"""
from __future__ import annotations

import argparse
import collections
import copy
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def plan_boundaries(recording, episodes, task, *, group_size=8,
                    max_boundaries=2, max_groups=8):
    """One representative per original scene group; never eight demo candidates.

    prefix_steps is a COUNT. Its last recorded state is prefix_steps - 1 and
    its next action is prefix_steps. Pick-up uses the last held boundary before
    its first lift success. Placement uses the last held boundary before first
    threshold-qualified release in a verified complete placement.
    """
    if recording.worlds % group_size:
        raise ValueError('Recording does not contain complete scene groups')
    if max_boundaries < 1 or max_groups < 1:
        raise ValueError('Audit limits must be positive')
    stride = int(recording.actions_per_decision)
    if stride < 1:
        raise ValueError('Invalid action chunk size')
    buckets = collections.defaultdict(dict)
    rejected = collections.Counter()
    for episode in episodes:
        w = int(episode['world'])
        group = w // group_size
        held = (recording.caught_target[:, w] & recording.active[:, w]
                & (recording.gripper_opening[:, w] <= .94))
        if task == 'pick_up':
            event = int(episode['pickup_success_env_step'])
        elif task == 'placement':
            stop = episode['placement_success_env_step']
            if stop is None:
                rejected['no_complete_placement'] += 1
                continue
            # Require a release after a real grasp, under the recorded opening
            # threshold. This is a landmark, not a replacement success rule.
            releases = np.flatnonzero(
                recording.active[:, w]
                & (recording.gripper_opening[:, w]
                   >= max(.55, float(recording.release_threshold[w])))
                & (np.arange(held.size) > int(episode['first_grasp_env_step']))
                & (np.arange(held.size) <= int(stop)))
            if not releases.size:
                rejected['no_release_landmark'] += 1
                continue
            event = int(releases[0])
        else:
            raise ValueError(f'Unknown task: {task}')
        boundaries = [b for b in range(stride, event + 1, stride)
                      if held[b - 1]
                      and recording.active[:b, w].all()
                      and not recording.terminated[:b, w].any()
                      and b // stride < int(recording.horizons[w])]
        if not boundaries:
            rejected['no_held_decision_boundary_before_event'] += 1
            continue
        b = boundaries[-1]
        start, end = group * group_size, (group + 1) * group_size
        if not np.all(recording.instruction_ids[start:end] == recording.instruction_ids[w]):
            raise ValueError('A source group mixes instruction identities')
        # Deterministic selection, avoiding a claim that repeated candidates
        # are new independent demonstration scenes.
        previous = buckets[b].get(group)
        if previous is None or w < int(previous['world']):
            buckets[b][group] = dict(episode)
    jobs = []
    for boundary in sorted(buckets, key=lambda b: (-len(buckets[b]), b))[:max_boundaries]:
        # Alternate source families when both are available.
        by_family = collections.defaultdict(list)
        for group, ep in sorted(buckets[boundary].items()):
            by_family[ep['source_instruction']].append(ep)
        chosen = []
        while len(chosen) < max_groups and any(by_family.values()):
            for family in sorted(by_family):
                if by_family[family] and len(chosen) < max_groups:
                    chosen.append(by_family[family].pop(0))
        jobs.append({'prefix_steps': boundary, 'episodes': chosen})
    return jobs, dict(rejected)


def clone_reset_groups(reset, representatives, group_size, torch):
    """Clone task/contact history alongside the backend's full-state broadcast."""
    base = torch.as_tensor(representatives, dtype=torch.long,
                           device=reset.horizons.device)
    destinations = ((base // group_size)[:, None] * group_size
                    + torch.arange(group_size, device=base.device)[None, :]).reshape(-1)
    sources = base.repeat_interleave(group_size)
    worlds = int(reset.horizons.shape[0])
    for owner in (reset, reset.task_state):
        for field in fields(owner):
            value = getattr(owner, field.name)
            if torch.is_tensor(value) and value.ndim and value.shape[0] == worlds:
                value.index_copy_(0, destinations, value.index_select(0, sources))
    instructions = list(reset.instructions)
    for source, destination in zip(sources.tolist(), destinations.tolist()):
        instructions[destination] = instructions[source]
    return replace(reset, instructions=tuple(instructions))


def collect_suffix_once(collector, reset, *, round_index):
    """The teacher prefix never goes through the collector's record writer."""
    resetter = collector.resetter
    owned = 'reset' in vars(resetter)
    original = resetter.reset
    called = False

    def prepared_reset(**kwargs):
        nonlocal called
        if called:
            raise RuntimeError('Prepared handoff may only be consumed once')
        called = True
        return reset

    resetter.reset = prepared_reset
    try:
        return collector.collect_round(update_index=0, round_index=round_index)
    finally:
        if owned:
            resetter.reset = original
        else:
            vars(resetter).pop('reset', None)


def _host(tensor):
    return tensor.detach().cpu().numpy()


def _score(collector, reset, low, caught, active, diagnostics, *, state=None):
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import evaluate_active_sparse_tasks
    return evaluate_active_sparse_tasks(
        state=reset.task_state if state is None else state,
        ee_position=low.ee_position, object_positions=low.object_positions,
        gripper_opening=low.gripper_opening, caught_target=caught,
        active_mask=active, max_steps=10_000,
        thresholds=collector._task_thresholds(),
        move_to_distance_reward=collector.move_to_distance_reward,
        catch_release_dense_reward=collector.catch_release_dense_reward,
        bilateral_contact=diagnostics['bilateral_contact'])


def run_job(world, recording, job, task, *, group_size, position_tolerance,
            opening_tolerance, suffix_decisions, seed_torch,
            lift_datum_tolerance=.02):
    """Replay source actions, validate landmarks, broadcast, collect (not train)."""
    from tools.audit.sil_record import _apply_determinism
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID

    torch, backend, collector = world.torch, world.backend, world.collector
    _apply_determinism(torch_seed=seed_torch, deterministic_kernels=False)
    reset = world.resetter.reset(update_index=0, round_index=recording.round_index,
                                 allow_prelifted=False)
    stride = int(collector.actions_per_policy_decision)
    if stride != int(recording.actions_per_decision):
        raise ValueError('Replay action chunk size differs from the recording')
    for name, actual in (
        ('instruction_ids', reset.task_state.instruction_ids),
        ('target_slots', reset.task_state.target_slots),
        ('reference_slots', reset.task_state.reference_slots),
        ('horizons', reset.horizons),
    ):
        if not np.array_equal(_host(actual), getattr(recording, name)):
            raise ValueError(f'Reset metadata mismatch: {name}')
    if tuple(reset.instructions) != tuple(recording.instructions):
        raise ValueError('Reset prompt mismatch')
    if recording.target_catalog_ids is not None:
        catalogs = _host(reset.group_target_catalog_ids).repeat(group_size)
        if not np.array_equal(catalogs, recording.target_catalog_ids):
            raise ValueError('Reset object catalogs differ')
    # Actual pre-action datum from this reconstructed reset, retained for the
    # relabelled pick-up. The source's stale container datum is not reused.
    rows = torch.arange(recording.worlds, device=world.device)
    low = backend.low_dim_observations()
    initial_target = low.object_positions[rows, reset.task_state.target_slots].clone()
    backend.pop_nonfinite_world_events()
    boundary = int(job['prefix_steps'])
    representatives = [int(e['world']) for e in job['episodes']]
    max_errors = {w: {'object_m': 0., 'ee_m': 0., 'opening': 0.,
                      'grasp_mismatch': False, 'early_termination': False}
                  for w in representatives}
    prefix_actions = 0
    for step in range(boundary):
        active = torch.as_tensor(recording.active[step], device=world.device, dtype=torch.bool)
        low = backend.step(torch.as_tensor(recording.actions[step],
                           dtype=torch.float32, device=world.device), active)
        low, caught, diagnostics = collector._update_physical_grasp(reset, low, active)
        result = _score(collector, reset, low, caught, active, diagnostics)
        prefix_actions += int(active.sum().item())
        poses, ee, opening, held, done = map(_host, (
            low.object_positions, low.ee_position, low.gripper_opening,
            caught, result.terminated))
        for w in representatives:
            error = max_errors[w]
            for name, observed, expected in (
                ('object_m', poses[w], recording.object_xyz[step, w]),
                ('ee_m', ee[w], recording.ee_xyz[step, w]),
                ('opening', opening[w], recording.gripper_opening[step, w]),
            ):
                delta = np.abs(observed - expected)
                error[name] = max(error[name], float(delta.max()) if np.isfinite(delta).all() else float('inf'))
            error['grasp_mismatch'] |= bool(held[w] != recording.caught_target[step, w])
            error['early_termination'] |= bool(done[w])
    events, divergence = backend.pop_nonfinite_world_report()
    if divergence is None and events:
        raise RuntimeError('Replay divergence cannot be attributed to worlds')
    divergence = np.zeros(recording.worlds, dtype=bool) if divergence is None else np.asarray(divergence)
    report = {'prefix_steps': boundary, 'prefix_replayed_active_actions': prefix_actions,
              'prefix_physics_world_steps': boundary * recording.worlds,
              'prefix_divergence_events': int(events), 'episodes': [],
              'optimizer_updates': 0, 'teacher_prefix_loss_rows': 0}
    accepted = []
    for episode in job['episodes']:
        w = int(episode['world'])
        errors = max_errors[w]
        reasons = []
        if divergence[w]:
            reasons.append('replay_divergence')
        if errors['object_m'] > position_tolerance or errors['ee_m'] > position_tolerance:
            reasons.append('replay_pose_mismatch')
        if errors['opening'] > opening_tolerance:
            reasons.append('replay_opening_mismatch')
        if errors['grasp_mismatch']:
            reasons.append('replay_grasp_mismatch')
        if errors['early_termination']:
            reasons.append('replay_terminated_before_handoff')
        if not bool(reset.task_state.grasped[w]):
            reasons.append('handoff_not_held')
        # The recording stores a row per PREDICATE call, so object_xyz[0] is
        # the pose after the first env step, while initial_target is the pose
        # at reset. Their difference measures the object's settle during that
        # step; it is not replay error, and the 2 mm replay tolerance does not
        # apply to it. Only pick_up consumes this datum's Z (target_lift ->
        # pick_success), and the live pre-action value below REPLACES the
        # source's, so what remains is a scene-sanity ceiling: a datum off by
        # a large fraction of the 5 cm lift would mean the reconstructed scene
        # is not the recorded one. Placement success reads no Z datum.
        recorded_z = float(recording.object_xyz[0, w, int(recording.target_slots[w]), 2])
        reset_z = float(_host(initial_target)[w, 2])
        baseline_error = abs(reset_z - recorded_z)
        if task == 'pick_up' and baseline_error > lift_datum_tolerance:
            reasons.append('reset_vs_first_post_action_lift_datum')
        entry = {'world': w, 'group': w // group_size,
                 'source_instruction': episode['source_instruction'],
                 'target_instruction': 'pick_up' if task == 'pick_up' else episode['source_instruction'],
                 'source_episode_uid': episode.get('episode_uid'),
                 'errors': errors, 'baseline_error_m': baseline_error,
                 'reset_lift_datum_z_m': reset_z,
                 'recorded_first_post_action_z_m': recorded_z,
                 'rejected': reasons}
        report['episodes'].append(entry)
        if not reasons:
            accepted.append(episode)
    if not accepted:
        report['status'] = 'no_verified_handoffs'
        return report

    representatives = [int(e['world']) for e in accepted]
    # This backend operation copies qpos/qvel, activation/history/warmstart,
    # time/equality state, controller state, catalogs and per-world model data.
    backend.broadcast_group_state(torch.tensor(representatives, device=world.device))
    reset = clone_reset_groups(reset, representatives, group_size, torch)
    selected = torch.zeros(recording.worlds, dtype=torch.bool, device=world.device)
    instructions = list(reset.instructions)
    horizons = torch.zeros_like(reset.horizons)
    prelifted = torch.zeros_like(selected)
    for episode in accepted:
        w = int(episode['world'])
        start, end = w // group_size * group_size, (w // group_size + 1) * group_size
        selected[start:end] = True
        horizons[start:end] = min(suffix_decisions, int(recording.horizons[w]) - boundary // stride)
        if task == 'pick_up':
            reset.task_state.instruction_ids[start:end] = INSTRUCTION_TO_ID['pick_up']
            reset.group_instruction_ids[w // group_size] = INSTRUCTION_TO_ID['pick_up']
            reset.task_state.initial_target_positions[start:end] = initial_target[w]
            # Historical peak in source placement uses a different datum.
            if reset.task_state.peak_lift is not None:
                z = backend.low_dim_observations().object_positions[w, int(recording.target_slots[w]), 2]
                reset.task_state.peak_lift[start:end] = (z - initial_target[w, 2]).clamp_min(0.)
            instructions[start:end] = [episode['pickup_prompt']] * group_size
            prelifted[start:end] = True
    reset = replace(reset, instructions=tuple(instructions), horizons=horizons,
                    prelifted=prelifted)
    # Verify actual backend state broadcast, not merely identical scene labels.
    live = {'qpos': backend._qpos, 'qvel': backend._qvel,
            **backend.controller_state()}
    for name, value in live.items():
        value = torch.as_tensor(value, device=world.device)
        for w in representatives:
            start = w // group_size * group_size
            group = value[start:start + group_size]
            if not torch.isfinite(group).all() or not torch.equal(group, group[:1].expand_as(group)):
                raise RuntimeError(f'Candidate state broadcast failed: {name}')
    # Re-score the destination task on a COPY: merely checking admission must
    # not advance its contact/reward histories. No already-successful reset.
    low = backend.low_dim_observations()
    admission = _score(collector, reset, low, reset.physical_grasp, selected,
                       diagnostics, state=copy.deepcopy(reset.task_state))
    if bool(((admission.success | admission.terminated) & selected).any()):
        terminal = _host((admission.success | admission.terminated) & selected)
        report['destination_terminal_worlds'] = np.flatnonzero(terminal).tolist()
        report['status'] = 'destination_task_already_terminal'
        return report
    started = time.monotonic()
    # Fresh sampling, rendering, state encoding and prompt priors happen inside
    # collect_round. No runtime calls from the teacher prefix enter its records.
    suffix = collect_suffix_once(collector, reset, round_index=recording.round_index)
    report['suffix_wall_seconds'] = time.monotonic() - started
    suffix_events, suffix_divergence = backend.pop_nonfinite_world_report()
    report['suffix_divergence_events'] = int(suffix_events)
    report['suffix_diverged_worlds'] = (None if suffix_divergence is None
                                       else np.flatnonzero(suffix_divergence).tolist())
    report['suffix_metrics'] = {k: float(v) for k, v in suffix.metrics.items()}
    report['suffix_record_rows'] = int(suffix.loss_mask.numel())
    report['suffix_loss_rows'] = int((suffix.loss_mask > 0).sum().item())
    report['suffix_vla_record_rows'] = (0 if suffix.vla_records is None
                                       else int(suffix.vla_records['advantage'].numel()))
    report['groups'] = []
    for episode in accepted:
        g = int(episode['world']) // group_size
        rewards = _host(suffix.candidate_rewards[g])
        successes = _host(suffix.candidate_success[g])
        bad = suffix_divergence is None and bool(suffix_events)
        if suffix_divergence is not None:
            bad = bool(np.asarray(suffix_divergence)[g * group_size:(g + 1) * group_size].any())
        report['groups'].append({'source_world': int(episode['world']), 'group': g,
                                 'successes': int(successes.sum()), 'candidates': group_size,
                                 'rewards': rewards.tolist(),
                                 'reward_std': float(rewards.std()),
                                 'contains_divergence': bad,
                                 'remaining_decisions': int(horizons[g * group_size])})
    report['status'] = 'fresh_suffixes_collected_no_training'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pilot-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=('pick_up', 'placement'), default='pick_up')
    parser.add_argument('--source-round', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-boundaries', type=int, default=2)
    parser.add_argument('--max-groups', type=int, default=8)
    parser.add_argument('--suffix-decisions', type=int, default=40)
    parser.add_argument('--position-tolerance-m', type=float, default=.002)
    parser.add_argument('--opening-tolerance', type=float, default=.03)
    # Settle between reset and the recording's first stored pose, not replay
    # drift. Bounded below the 5 cm lift so a datum gap can never be most of
    # an earned pick-up lift.
    parser.add_argument('--lift-datum-tolerance-m', type=float, default=.02)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.suffix_decisions < 1 or not 0 < args.position_tolerance_m < .01 or not 0 < args.opening_tolerance < .1:
        parser.error('Require positive suffix budget and strict replay tolerances')
    if not 0 < args.lift_datum_tolerance_m < .05:
        parser.error('Lift datum tolerance must be positive and below the lift success height')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    from tools.audit.sil_record import _Recording
    from tools.audit.extract_cdpr_transition_demonstrations import select_episodes
    manifest = json.loads(args.manifest.read_text())
    pilot = json.loads((args.pilot_run / 'pilot_manifest.json').read_text())
    checkpoint, config = Path(pilot['source']), Path(pilot['config'])
    for path, expected in ((checkpoint, pilot['source_sha256']), (config, pilot['config_sha256'])):
        if sha256(path) != expected:
            raise ValueError(f'Provenance hash mismatch: {path}')
    sources = [s for s in manifest['sources'] if int(s['round_index']) == args.source_round]
    if len(sources) != 1:
        raise ValueError('Select one unambiguous source round from the manifest')
    source = sources[0]
    source_path = Path(source['path'])
    if source_path.resolve().parent != (args.pilot_run / 'baseline').resolve():
        raise ValueError('This audit requires the named pilot baseline and its recorded donor/config')
    if sha256(source_path) != source['sha256']:
        raise ValueError('Source recording changed after extraction')
    recording = _Recording.from_npz(source_path)
    revalidated, _ = select_episodes(recording)
    supplied = {int(e['world']): e for e in manifest['episodes']
                if e['source_sha256'] == source['sha256']}
    episodes = []
    for episode in revalidated:
        w = int(episode['world'])
        if w in supplied:
            episode['episode_uid'] = supplied[w]['episode_uid']
            episodes.append(episode)
    jobs, rejected = plan_boundaries(recording, episodes, args.task,
                                     max_boundaries=args.max_boundaries, max_groups=args.max_groups)
    plan = {'task': args.task, 'source': source, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': pilot['source_sha256'], 'config': str(config),
            'config_sha256': pilot['config_sha256'], 'manifest_sha256': sha256(args.manifest),
            'source_round': args.source_round, 'group_size': 8,
            'position_tolerance_m': args.position_tolerance_m,
            'opening_tolerance': args.opening_tolerance,
            'lift_datum_tolerance_m': args.lift_datum_tolerance_m,
            'rejected_landmarks': rejected,
            'jobs': jobs, 'optimizer_updates': 0,
            'limitations': ['Evaluation scenes are prototype only; collect training-only scenes before learning.',
                            'Legacy recordings lack pre-action/full-state snapshots; the reset lift datum is measured live, not read from the source.',
                            'Assisted suffix success is not ordinary-start success.',
                            'Source placement geometry is legacy; this is not an outside-goal benchmark.']}
    print(json.dumps({'checkpoint': str(checkpoint), 'task': args.task,
                      'jobs': [{'prefix_steps': j['prefix_steps'], 'groups': len(j['episodes'])} for j in jobs],
                      'rejected_landmarks': rejected}, indent=2), flush=True)
    if not jobs:
        raise ValueError('No admissible decision-boundary handoffs in this source round')
    if args.dry_run:
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    from tools.audit.xy_approach_probe import _build_world
    world = _build_world(checkpoint=checkpoint, config_path=config, device_str=args.device,
                         worlds=recording.worlds, group_size=8, microbatch=32,
                         load_policy=True, run_dir=args.output,
                         start_distance_cap=(float(recording.start_distance_cap)
                                             if np.isfinite(recording.start_distance_cap) else None))
    # The donor checkpoint's sampling settings, explicitly reported. The
    # config names the source scene/reward protocol; using its successor's
    # gate with the donor's saved offset std would silently mix interventions.
    world.collector.episode_offset_after_grasp = bool(
        getattr(world.args, 'episode_offset_after_grasp', False))
    world.collector.store_vla_records = bool(getattr(world.args, 'train_vla_lora', False))
    world.collector.min_group_reward_std = float(getattr(world.args, 'grpo_min_group_reward_std', 0.))
    world.collector.vla_update_max_records = int(getattr(world.args, 'vla_update_max_records', 128))
    plan['suffix_exploration'] = {
        'episode_offset_std': _host(world.trainer.episode_offset_std).tolist(),
        'episode_offset_after_grasp': world.collector.episode_offset_after_grasp,
        'min_group_reward_std': world.collector.min_group_reward_std,
        'store_vla_records': world.collector.store_vla_records}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    reports = []
    for job in jobs:
        result = run_job(world, recording, job, args.task, group_size=8,
                         position_tolerance=args.position_tolerance_m,
                         opening_tolerance=args.opening_tolerance,
                         suffix_decisions=args.suffix_decisions,
                         seed_torch=int(pilot['evaluation_seed_torch']),
                         lift_datum_tolerance=args.lift_datum_tolerance_m)
        reports.append(result)
        (args.output / 'report.json').write_text(json.dumps({'plan': plan, 'results': reports}, indent=2) + '\n')
        print(f"[handoff] prefix={job['prefix_steps']} status={result['status']} groups={result.get('groups', [])}", flush=True)
    print(f"[handoff] wrote {args.output / 'report.json'}; optimizer updates=0", flush=True)
    return 0 if any(r['status'] == 'fresh_suffixes_collected_no_training' for r in reports) else 2


if __name__ == '__main__':
    raise SystemExit(main())
