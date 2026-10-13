"""E05: fixed external-method and unseen-corruption experiment, no outcome selection."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONFIRM = REPO / 'task_restore_confirm_v1'
METHOD = REPO / 'task_restore_method_v1'
CORE = REPO / 'task_restore_v1'
sys.path.insert(0, str(CONFIRM))
import contract as OLD

TASKS, SEEDS = OLD.TASKS, OLD.SEEDS
START, BANK_N, HORIZON, SHARD, UPDATES = 30000, 100, 400, 5, 2000
COUNTS = dict(clean=50, lowlight=100, exposure_shift=50, sensor_fpn=50)
NOISE = {
    'clean': dict(exposure=1., gain=0., read=0., fpn_std=0.),
    'lowlight': dict(exposure=.25, gain=.004, read=.008, fpn_std=0.),
    'exposure_shift': dict(exposure=.125, gain=.004, read=.008, fpn_std=0.),
    'sensor_fpn': dict(exposure=.25, gain=.004, read=.008, fpn_std=.004),
}
ARMS = ('image', 'feature', 'decision', 'retinex_finetune')
NEW_EPISODES = 4000
read, sha, write, tag, group_path, paired = OLD.read, OLD.sha, OLD.write, OLD.tag, OLD.group_path, OLD.paired

def sources():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and (p.suffix == '.py' or p.parent.name == 'vendor')}

def groups():
    base = [(s,t,'retinex_finetune',c) for s in SEEDS for t in TASKS for c in ('clean','lowlight')]
    base += [(None,t,'retinex_pretrained',c) for t in TASKS for c in ('clean','lowlight')]
    for c in ('exposure_shift','sensor_fpn'):
        base += [(s,t,a,c) for s in SEEDS for t in TASKS for a in ARMS]
        base += [(None,t,a,c) for t in TASKS for a in ('original','retinex_pretrained')]
    return base

def jobs():
    return [(g,o,min(SHARD,COUNTS[g[3]]-o)) for o in range(0,100,SHARD)
            for g in groups() if o < COUNTS[g[3]]]

def asset_manifest():
    m = read(HERE / 'assets/manifest.json')
    if m['commit'] != '1e9a0efce4b306b6701b824768370ff26066c32a' or m['gt_mean'] or m['self_ensemble']:
        raise RuntimeError('External asset provenance changed')
    for name,entry in m['files'].items():
        if sha(HERE / name) != entry['sha256']:
            raise RuntimeError('Official asset checksum mismatch: ' + name)
    return m

def reference(root):
    """Read-only audit: retain the completed E04 and all 18 frozen models."""
    if not (root / 'finished.json').exists():
        raise RuntimeError('E04 must finish first')
    lock = read(root / 'model_lock.json')
    if read(root / 'finished.json')['model_lock_sha256'] != sha(root / 'model_lock.json'):
        raise RuntimeError('E04 lock changed')
    if lock['confirm_sources'] != OLD.sources():
        raise RuntimeError('E04 source revision changed')
    from run_confirm import frozen_inputs
    settings = lock['settings']
    if frozen_inputs(Path(settings['source']),Path(settings['repeat'])) != settings:
        raise RuntimeError('Original training/policy/data inputs changed')
    if len(lock['adapters']) != 18:
        raise RuntimeError('Retain all 18 models, never omit seed3073')
    for entry in lock['adapters'].values():
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Frozen model changed')
    for task in TASKS:
        if sha(root / 'banks' / (task + '.json')) != lock['banks'][task]:
            raise RuntimeError('State bank changed')
    count = 0
    for job in OLD.jobs():
        count += len(OLD.verify_shard(OLD.group_path(root,job[0])/'shards'/f'{job[1]:05d}',job,lock))
    if count != 3200:
        raise RuntimeError('E04 is incomplete')
    return lock

def adapter(lock, group):
    seed,task,arm,_ = group
    if arm == 'original':
        return None
    if arm == 'retinex_pretrained':
        return dict(path=str(HERE / 'assets/LOL_v1.pth'),sha256=lock['settings']['assets']['files']['assets/LOL_v1.pth']['sha256'])
    return lock['adapters'][f'{seed}/{task}/{arm}']

def verify_shard(target, job, lock):
    group,offset,count = job
    seed,task,arm,condition = group
    result = read(target / 'result.json')
    entry = adapter(lock, group)
    expected = dict(task=task,arm=arm,condition=condition,training_seed=seed,
                    bank_episodes=BANK_N,episodes=COUNTS[condition],start_seed=START,
                    offset=offset,count=count,max_steps=HORIZON,noise=NOISE[condition],
                    external_sources=lock['settings']['external_sources'],
                    sources=lock['base_sources'],confirm_sources=lock['confirm_sources'],
                    adapter_sha256=entry['sha256'] if entry else None,
                    state_bank_sha256=lock['banks'][task],policy_sha256=lock['policies'][task])
    if any(result['protocol'].get(k) != v for k,v in expected.items()):
        raise RuntimeError('Wrong shard protocol: ' + str(target))
    rows = result['rows']
    if [r['seed'] for r in rows] != list(range(START+offset,START+offset+count)):
        raise RuntimeError('Missing or duplicated states')
    if rows != [json.loads(l) for l in (target/'episodes.jsonl').read_text().splitlines()]:
        raise RuntimeError('JSON/JSONL disagree')
    for r in rows:
        if (r['task'],r['arm'],r['condition'],r['training_seed']) != group[1:3]+(condition,seed):
            raise RuntimeError('Wrong episode group')
        if r['steps'] != HORIZON or not all(isinstance(r[k],bool) for k in ('success_ever','success_final')):
            raise RuntimeError('Only full400-step Boolean simulator success is formal')
        if r['success_final'] and not r['success_ever']:
            raise RuntimeError('Inconsistent final success')
        first = r['first_success_step']
        if (first is not None) != r['success_ever'] or (first is not None and not 0 <= first <= HORIZON):
            raise RuntimeError('Invalid first success')
        if not all(re.fullmatch('[0-9a-f]{64}',r[k]) for k in ('initial_state_sha256','initial_model_sha256','state_bank_entry_sha256')):
            raise RuntimeError('Invalid simulator fingerprints')
        if not all(math.isfinite(r[k]) and r[k]>=0 for k in ('wall_seconds','inference_seconds')):
            raise RuntimeError('Invalid timing')
    if result['episodes']!=count or result['successes']!=sum(r['success_ever'] for r in rows) or result['final_successes']!=sum(r['success_final'] for r in rows):
        raise RuntimeError('Invalid aggregate counts')
    return rows
