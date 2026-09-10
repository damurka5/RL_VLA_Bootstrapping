"""Remote orchestration: fresh bank -> baseline -> bounded GRPO -> final evaluation."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.audit.probe_cdpr_demonstration_handoff import sha256


def command(command, log, *, env=None):
    print('[demo-pilot] '+ ' '.join(map(str,command)), flush=True)
    with Path(log).open('w') as stream:
        proc = subprocess.Popen(list(map(str,command)), cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='', flush=True); stream.write(line); stream.flush()
        if proc.wait():
            raise RuntimeError(f'Command failed ({proc.returncode}); see {log}')


def compare_evaluations(run, checkpoint):
    import numpy as np
    from tools.audit.sil_record import _Recording, _instruction_name
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'arms':{},
            'reset_identity_checked':True, 'interpretation':'Ordinary-start before/after; no matched ordinary-only training control.'}
    recordings={}
    for arm in ('baseline','final_eval'):
        paths=sorted((run/arm).glob('record_*.npz'))
        if len(paths)!=3:
            raise ValueError(f'{arm}: expected exactly three rounds')
        recordings[arm]=[_Recording.from_npz(p) for p in paths]
        counts={}
        for r in recordings[arm]:
            if r.reset_object_xyz is None or r.reset_ee_xyz is None:
                raise ValueError('Evaluation lacks pre-action reset poses')
            for w, task in enumerate(r.instruction_ids):
                name=_instruction_name(task)
                row=counts.setdefault(name,dict(successes=0,episodes=0,outside_goal_successes=0,outside_goal_episodes=0))
                row['episodes']+=1; row['successes']+=int(r.episode_success[w])
                if name.startswith('put_into_'):
                    # starts_grasped is caught_target[0]; physical_grasp_at_reset
                    # is live state read after the round, so it reports the FINAL
                    # grasp and would pass this check on every recording.
                    if r.starts_grasped[w] or r.horizons[w]!=40:
                        raise ValueError('Container evaluation is not ordinary uncaught / 40 decisions')
                    pos=r.reset_object_xyz[w]
                    distance=np.linalg.norm(pos[r.target_slots[w],:2]-pos[r.reference_slots[w],:2])
                    if distance>(.091 if name=='put_into_plate' else .057):
                        row['outside_goal_episodes']+=1; row['outside_goal_successes']+=int(r.episode_success[w])
        for row in counts.values():
            row['rate']=row['successes']/row['episodes']
            row['outside_goal_rate']=(row['outside_goal_successes']/row['outside_goal_episodes'] if row['outside_goal_episodes'] else None)
        result['arms'][arm]=counts
    mismatches=[]
    for a,b in zip(recordings['baseline'],recordings['final_eval']):
        for name in ('round_index','instruction_ids','target_slots','reference_slots','horizons','instructions','target_catalog_ids'):
            if not np.array_equal(getattr(a,name),getattr(b,name)):
                mismatches.append(f'round {a.round_index}: {name}')
        for name in ('reset_object_xyz','reset_ee_xyz'):
            if not np.allclose(getattr(a,name),getattr(b,name),rtol=0,atol=1e-6):
                mismatches.append(f'round {a.round_index}: {name}')
    result['reset_identity_checked']=not mismatches
    result['reset_mismatches']=mismatches
    (run/'pilot_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for family,before in result['arms']['baseline'].items():
        after=result['arms']['final_eval'][family]
        print(f"[demo-pilot] {family}: {before['successes']}/{before['episodes']}={before['rate']:.4f} -> {after['successes']}/{after['episodes']}={after['rate']:.4f}")
    if mismatches:
        raise ValueError('Evaluation reset identity differs; saved raw scores, do not call them matched')
    return result


def donor_provenance(checkpoint, config):
    """Both evaluation arms must share one reset distribution, checked FIRST.

    sil_record seeds its resetter from the CHECKPOINT's saved validation_seed,
    not from the config, so a donor carrying a different seed makes the
    baseline and the final evaluation different episode sets -- which
    compare_evaluations can only discover after the training has been paid for.
    The approach caps are safe by construction here (the pilot config's ladders
    are single-rung, so load_state_dict snaps any restored cap onto that rung),
    and this asserts the one thing that is not.
    """
    import torch
    import yaml
    try:
        donor = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        donor = torch.load(checkpoint, map_location='cpu')
    saved = dict(donor.get('args') or {})
    if not saved:
        raise ValueError(f'{checkpoint} carries no training arguments; its resets cannot be reproduced')
    wanted = yaml.safe_load(Path(config).read_text())['training']['rl']['args']
    if int(saved['validation_seed']) != int(wanted['validation_seed']):
        raise ValueError(
            f"Donor validation_seed {saved['validation_seed']} differs from the pilot config's "
            f"{wanted['validation_seed']}. The baseline and final evaluations would score different "
            'episodes, and the demonstration bank would be cut from a third set. Use a donor trained '
            'under the same validation seed, or re-record the baseline with the trained checkpoint.')
    print(f"[demo-pilot] donor validation_seed={saved['validation_seed']} matches the config", flush=True)
    return saved


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/release_recovery_continue_3m_20260908_102004/rl/step_3540208/smolvla_grpo_adapter.pt')
    p.add_argument('--config',type=Path,default=ROOT/'configs/examples/cdpr_smolvla_demo_grpo_pilot.yaml')
    p.add_argument('--steps',type=int,default=1000000)
    p.add_argument('--demo-rounds',type=int,default=12)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.steps<1 or a.demo_rounds<3 or a.demo_rounds>100:
        p.error('Require positive steps and 3–100 demo rounds')
    a.checkpoint=a.checkpoint.expanduser().resolve(); a.config=a.config.expanduser().resolve()
    if not a.checkpoint.is_file() or not a.config.is_file():
        p.error('Checkpoint and config must exist; set WARMSTART_CHECKPOINT explicitly if needed')
    run=(a.run_dir or ROOT/'runs'/datetime.datetime.now().strftime('demo_grpo_pilot_%Y%m%d_%H%M%S')).resolve()
    if run.exists():
        p.error('Run directory already exists; choose a new run (no silent overwrite/resume)')
    # The launcher derives the training run name from this directory, so it has
    # to sit where the launcher will look. Checked HERE, not after the bank and
    # the baseline have been paid for.
    if run.parent != ROOT/'runs':
        p.error(f'Run directory must be directly under {ROOT/"runs"}')
    print(f'[demo-pilot] output={run}\n[demo-pilot] selected_action_budget={a.steps}\n[demo-pilot] donor={a.checkpoint}',flush=True)
    print('[demo-pilot] CPU checks -> reserved training rounds 1000 onward -> bank -> ordinary baseline -> GRPO -> ordinary final',flush=True)
    if a.dry_run:
        return 0
    saved=donor_provenance(a.checkpoint,a.config)
    run.mkdir(parents=True)
    env=os.environ.copy()
    env.pop('RLVLA_CDPR_DEMO_BANK',None)
    env.pop('RLVLA_SMOLVLA_RESUME_CHECKPOINT',None)
    env.update(PYTHONUNBUFFERED='1',MUJOCO_GL='egl')
    config_hash=sha256(a.config)
    manifest=dict(schema='cdpr_demo_pilot_v1',source=str(a.checkpoint),source_sha256=sha256(a.checkpoint),
                  config=str(a.config),config_sha256=config_hash,max_selected_actions=a.steps,
                  reset_base_seed=int(saved['validation_seed']),
                  bank_rounds=list(range(1000,1000+a.demo_rounds)),evaluation_rounds=[0,1,2],
                  torch_seed=0,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (run/'pilot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (run/'config_snapshot.yaml').write_bytes(a.config.read_bytes())
    (run/'tracked_changes.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
    py=[sys.executable]
    # File existence prevents unittest discovery from returning a false zero-test pass.
    for pattern in ('test_cdpr_demo_training.py','test_cdpr_demonstration_handoff.py','test_extract_cdpr_transition_demonstrations.py','test_fixed_approach_pilot.py'):
        if not (ROOT/'tests'/pattern).is_file():
            raise FileNotFoundError(pattern)
        command(py+['-m','unittest','discover','-s','tests','-p',pattern],run/(pattern+'.log'),env=env)
    def record(checkpoint,output,round_index,rounds):
        command(py+['tools/audit/sil_record.py','--mode','record','--checkpoint',checkpoint,'--config',a.config,
                    '--worlds','512','--group-size','8','--round-index',str(round_index),'--rounds',str(rounds),
                    '--devices','cuda:0,cuda:1','--seed-torch','0','--output',output],run/(output.name+'.log'),env=env)
    record(a.checkpoint,run/'training_sources',1000,a.demo_rounds)
    command(py+['tools/audit/extract_cdpr_transition_demonstrations.py','--recordings',str(run/'training_sources'/'record_*.npz'),
                '--output',run/'demonstrations'],run/'extraction.log',env=env)
    command(py+['tools/train/build_cdpr_training_handoffs.py','--manifest',run/'demonstrations'/'manifest.json',
                '--checkpoint',a.checkpoint,'--config',a.config,'--rounds',str(a.demo_rounds),
                '--output',run/'training_bank.json'],run/'bank.log',env=env)
    record(a.checkpoint,run/'baseline',0,3)
    if sha256(a.config)!=config_hash:
        raise ValueError('Config changed during source collection')
    train_env=env | dict(CONFIG=str(a.config),WARMSTART_CHECKPOINT=str(a.checkpoint),MAX_TRAIN_STEPS=str(a.steps),
                         REPO_ROOT=str(ROOT),RUN_NAME=run.name,WORLDS_PER_RANK='512',
                         RLVLA_CDPR_DEMO_BANK=str(run/'training_bank.json'))
    command(['bash','scripts/train_cdpr_phase7_sparse_joint_remote.sh'],run/'pilot_train.log',env=train_env)
    paths=list((run/'rl').glob('step_*/smolvla_grpo_adapter.pt'))
    if not paths:
        raise ValueError('No training checkpoints produced')
    checkpoint=max(paths,key=lambda p:int(p.parent.name.split('_')[1]))
    if sha256(a.config)!=config_hash:
        raise ValueError('Config changed during training')
    record(checkpoint,run/'final_eval',0,3)
    compare_evaluations(run,checkpoint)
    # Small artifact bundle suitable for sending back; excludes recordings/weights.
    import tarfile
    with tarfile.open(run/'review_logs.tar.gz','w:gz') as archive:
        for path in sorted(run.rglob('*')):
            if path.is_file() and (path.suffix in ('.json','.jsonl','.log','.yaml','.patch') or 'tfevents' in path.name):
                archive.add(path,arcname=str(path.relative_to(run)))
    print(f'[demo-pilot] finished; send {run}/review_logs.tar.gz; all checkpoints retained, no automatic promotion',flush=True)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
