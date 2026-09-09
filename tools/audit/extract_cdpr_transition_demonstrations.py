"""Extract real pick-up prefixes and complete placement clips from recordings.

CPU only (NumPy + PyTorch); does not train or replay. Output clips are a
demonstration archive, NOT a ready-to-load SIL dataset or simulator snapshot.
The pick-up verdict is recomputed with the production task evaluator.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pickup_success_mask(recording):
    """Rescore each recorded transition under pick_up, including active masking."""
    import torch
    from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import (
        BatchedCatchReleaseDenseReward, BatchedTaskState, INSTRUCTION_TO_ID,
        evaluate_active_sparse_tasks,
    )
    steps, worlds = recording.active.shape
    size = steps * worlds

    def repeated(array, dtype):
        a = np.asarray(array)
        return torch.as_tensor(np.tile(a, (steps,) + (1,) * (a.ndim - 1)), dtype=dtype)

    state = BatchedTaskState(
        instruction_ids=torch.full((size,), INSTRUCTION_TO_ID['pick_up'], dtype=torch.int64),
        target_slots=repeated(recording.target_slots, torch.int64),
        reference_slots=repeated(recording.reference_slots, torch.int64),
        second_reference_slots=repeated(recording.second_reference_slots, torch.int64),
        initial_target_positions=repeated(recording.initial_target_xyz, torch.float32),
        ever_grasped=torch.zeros(size, dtype=torch.bool), grasped=torch.zeros(size, dtype=torch.bool),
        step_count=torch.zeros(size, dtype=torch.int64),
        release_threshold=repeated(recording.release_threshold, torch.float32),
        support_surface_z=repeated(recording.support_surface_z, torch.float32),
        target_rest_height=repeated(recording.target_rest_height, torch.float32),
    )
    # pick_up success uses current grasp/lift, not a trajectory-history latch.
    # The flattening is intentionally limited to that instruction's predicate.
    with torch.inference_mode():
        result = evaluate_active_sparse_tasks(
            state=state,
            ee_position=torch.as_tensor(recording.ee_xyz.reshape(size, 3), dtype=torch.float32),
            object_positions=torch.as_tensor(recording.object_xyz.reshape(size, -1, 3), dtype=torch.float32),
            gripper_opening=torch.as_tensor(recording.gripper_opening.reshape(size), dtype=torch.float32),
            caught_target=torch.as_tensor(recording.caught_target.reshape(size), dtype=torch.bool),
            active_mask=torch.as_tensor(recording.active.reshape(size), dtype=torch.bool),
            max_steps=steps + 1,
            catch_release_dense_reward=BatchedCatchReleaseDenseReward(
                pick_lift_success_height=float(recording.pick_lift_success_height)),
        )
    return result.success.cpu().numpy().reshape(steps, worlds)


def select_episodes(recording, *, max_start_clearance=.01):
    from tools.audit.sil_record import _instruction_name

    if recording.diverged_worlds:
        return [], {'recording_reported_divergence': recording.worlds}
    scored = pickup_success_mask(recording)
    selected, rejected = [], {}

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for world in range(recording.worlds):
        name = _instruction_name(recording.instruction_ids[world])
        if name not in ('put_into_plate', 'put_into_bowl'):
            reject('not_a_container_source')
            continue
        target, ref = int(recording.target_slots[world]), int(recording.reference_slots[world])
        if not 0 <= ref < recording.object_xyz.shape[2] or ref == target:
            reject('missing_distinct_receptacle')
            continue
        if recording.starts_grasped[world]:
            reject('starts_grasped')
            continue
        start_z = float(recording.object_xyz[0, world, target, 2])
        desk_z = float(recording.support_surface_z[world] + recording.target_rest_height[world])
        if not np.isfinite(start_z) or abs(start_z - desk_z) > max_start_clearance:
            reject('not_a_desk_start')
            continue
        # The production lift baseline predates composed repositioning in old
        # recordings. It must still be consistent with the actual desk height.
        if not np.isfinite(recording.initial_target_xyz[world, 2]) or abs(
                float(recording.initial_target_xyz[world, 2]) - desk_z) > max_start_clearance:
            reject('inconsistent_lift_baseline')
            continue
        hits = np.flatnonzero(scored[:, world])
        if not hits.size:
            reject('no_pickup_success')
            continue
        stop = int(hits[0])
        own_hits = np.flatnonzero(recording.success[:, world] & recording.active[:, world])
        full_stop = int(own_hits[0]) if own_hits.size else None
        if full_stop is not None and full_stop < stop:
            reject('pickup_after_placement_terminal')
            continue
        end = max(stop, full_stop if full_stop is not None else stop) + 1
        if not recording.active[:end, world].all():
            reject('inactive_gap_before_endpoint')
            continue
        arrays = (recording.actions[:end, world], recording.ee_xyz[:end, world],
                  recording.object_xyz[:end, world], recording.gripper_opening[:end, world])
        if not all(np.isfinite(a).all() for a in arrays):
            reject('nonfinite_clip')
            continue
        match = re.fullmatch(r'put (.+) into (plate|bowl)', str(recording.instructions[world]).strip())
        if match is None or name != 'put_into_' + match[2]:
            reject('unrecognized_source_prompt')
            continue
        grasps = np.flatnonzero(recording.caught_target[:stop + 1, world]
                               & (recording.gripper_opening[:stop + 1, world] <= .94))
        selected.append(dict(world=world, source_instruction=name,
                             source_prompt=str(recording.instructions[world]),
                             pickup_prompt=f'pick up {match[1]}', target_slot=target,
                             receptacle_slot=ref, first_grasp_env_step=int(grasps[0]),
                             pickup_success_env_step=stop, placement_success_env_step=full_stop,
                             placement_prefix_only=full_stop is None,
                             start_ee_to_target_distance_m=float(np.linalg.norm(
                                 recording.ee_xyz[0, world] - recording.object_xyz[0, world, target])),
                             start_object_to_receptacle_xy_m=float(np.linalg.norm(
                                 recording.object_xyz[0, world, target, :2] - recording.object_xyz[0, world, ref, :2])),
                             lift_object_to_receptacle_xy_m=float(np.linalg.norm(
                                 recording.object_xyz[stop, world, target, :2] - recording.object_xyz[stop, world, ref, :2]))))
    return selected, rejected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', nargs='+', required=True, help='Source record_*.npz paths or quoted globs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-start-clearance', type=float, default=.01)
    args = parser.parse_args(argv)
    if not 0 < args.max_start_clearance < .05:
        parser.error('max-start-clearance must be positive and below the 5 cm lift threshold')
    paths = []
    for pattern in args.recordings:
        matches = sorted(Path(p).resolve() for p in glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            parser.error(f'No recordings matched {pattern}')
        paths.extend(matches)
    paths = sorted(set(paths))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    from tools.audit.sil_record import _Recording
    manifest = {'schema': 'cdpr_transition_clips_v1', 'sources': [], 'episodes': [],
                'status': 'offline_demonstrations_not_on_policy_training_records',
                'limitations': [
                    'Positions/actions are not complete restorable simulator/controller states.',
                    'Relabelled prompts need fresh images/prior inference before any imitation loss.',
                    'Landmarks are env-step observations, not guaranteed decision-boundary reset states.',
                    'Composed starts may skip approach; desk-start here means uncaught object on support.',
                    'Reported divergence is quarantined; unrecorded contained resets cannot be certified absent.',
                    'Source episodes and relabelled views are not independent demonstration scenes.',
                ]}
    seen = set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        recording = _Recording.from_npz(path)
        episodes, rejected = select_episodes(recording, max_start_clearance=args.max_start_clearance)
        manifest['sources'].append(dict(path=str(path), sha256=digest,
                                        round_index=recording.round_index, rejected=rejected))
        for episode in episodes:
            world = episode['world']
            uid = f'{digest}_w{world}'
            episode.update(source_sha256=digest, source_recording=str(path), episode_uid=uid,
                           round_index=recording.round_index,
                           actions_per_decision=recording.actions_per_decision,
                           lift_threshold_m=recording.pick_lift_success_height)
            def save_clip(kind, stop):
                relative = f'{kind}/{uid}.npz'
                destination = output / relative
                destination.parent.mkdir(exist_ok=True)
                end = stop + 1
                np.savez_compressed(destination,
                    executed_actions=recording.actions[:end, world],
                    ee_xyz=recording.ee_xyz[:end, world], object_xyz=recording.object_xyz[:end, world],
                    gripper_opening=recording.gripper_opening[:end, world],
                    caught_target=recording.caught_target[:end, world],
                    source_instruction=episode['source_instruction'],
                    demonstration_instruction='pick_up' if kind == 'pickup' else episode['source_instruction'],
                    instruction_text=episode['pickup_prompt'] if kind == 'pickup' else episode['source_prompt'],
                    target_slot=episode['target_slot'], receptacle_slot=episode['receptacle_slot'],
                    source_episode_uid=uid, success_env_step=stop,
                    actions_per_decision=recording.actions_per_decision)
                return relative
            episode['pickup_clip'] = save_clip('pickup', episode['pickup_success_env_step'])
            if episode['placement_success_env_step'] is not None:
                episode['placement_clip'] = save_clip('placement', episode['placement_success_env_step'])
            manifest['episodes'].append(episode)
        print(f'[demos] {path.name}: {len(episodes)} lift prefixes; '
              f'{sum(e["placement_success_env_step"] is not None for e in episodes)} complete placements', flush=True)
    manifest['counts'] = {
        'pickup_clips': len(manifest['episodes']),
        'complete_placement_clips': sum('placement_clip' in e for e in manifest['episodes']),
        'lift_prefixes_from_failed_placements': sum(e['placement_prefix_only'] for e in manifest['episodes'])}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['counts'], indent=2))
    print(f'[demos] wrote {output / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
