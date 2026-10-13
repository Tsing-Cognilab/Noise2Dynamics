"""Repeat the immutable 3072 method protocol; only adapter training seed changes."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
METHOD = REPO / 'task_restore_method_v1'
CORE = REPO / 'task_restore_v1'
sys.path.insert(0, str(METHOD))
from run_method import GROUPS, IMAGE, assert_gpu_free, container_path, idle_gpus, sha, verify_shard, write_json

LEARNED = [(a, c) for a, c in GROUPS if a in ('image', 'decision')]
EXPECTED = dict(tasks=['can', 'square'], seed=3072, updates=2000, batch_size=16,
                train_demos=24, valid_demos=4, stride=4, episodes=50,
                start_seed=20000, max_steps=400, shard_size=5, smoke=False,
                groups=[list(g) for g in GROUPS])


def read(path):
    return json.loads(path.read_text())


def valid_tag(tag):
    if not re.fullmatch(r'[A-Za-z0-9._-]+', tag) or tag in ('.', '..'):
        raise ValueError('Invalid output tag')
    return tag


def load_reference(source):
    if not (source / 'finished.json').exists():
        raise RuntimeError('3072 reference run is not complete')
    protocol = read(source / 'suite_protocol.json')
    for key, value in EXPECTED.items():
        if protocol.get(key) != value:
            raise RuntimeError(f'Reference is not the agreed fixed protocol: {key}')
    sources = {p.name: sha(p) for p in sorted(METHOD.glob('*.py'))}
    if sources != protocol['sources']:
        raise RuntimeError('Original method code changed; refusing a non-matched repeat')
    for task in protocol['tasks']:
        bank = CORE / 'assets/initial_states_v1/paired_xml_seed20000_n50' / f'{task}.json'
        if sha(bank) != protocol['banks'][task]:
            raise RuntimeError('Initial-state bank changed')
        ready_path = source / 'data' / task / 'ready.json'
        ready = read(ready_path)
        if ready['spec']['method_code'] != sources or ready['spec']['task'] != task:
            raise RuntimeError('Dataset source/task changed')
        for split in ('train', 'valid'):
            if sha(ready_path.parent / f'{split}.npz') != ready['files'][split]:
                raise RuntimeError('Dataset checksum changed')
        for arm in ('image', 'decision'):
            finished = read(source / 'train' / task / arm / 'finished.json')
            spec = finished['spec']
            wanted = dict(task=task, arm=arm, seed=3072, updates=2000, batch_size=16,
                          decision_weight=.1, learning_rate=.0003, width=24,
                          data_sha256=sha(ready_path), sources=sources, parameters=267195,
                          noise=dict(exposure=.25, gain=.004, read=.008, correlated=False),
                          identity_probability=.2, teacher_frozen=True,
                          selection='last fixed-budget checkpoint; no closed-loop selection')
            if spec != wanted or finished['checkpoint_sha256'] != protocol['adapters'][f'{task}/{arm}']:
                raise RuntimeError('Reference training configuration does not match')
    return protocol


def verify_training(target, source, task, arm, seed):
    finished = read(target / 'finished.json')
    expected = dict(read(source / 'train' / task / arm / 'finished.json')['spec'], seed=seed)
    if finished['spec'] != expected or sha(target / 'model.pt') != finished['checkpoint_sha256']:
        raise RuntimeError(f'Training configuration/weights mismatch: {target}')
    return finished['checkpoint_sha256']


def collect(root, protocol, task, arm, condition):
    rows, specs = [], []
    for offset in range(0, protocol['episodes'], protocol['shard_size']):
        count = min(protocol['shard_size'], protocol['episodes'] - offset)
        target = root / 'eval' / task / f'{arm}_{condition}' / 'shards' / f'{offset:05d}'
        record = verify_shard(target, (task, arm, condition, offset, count), protocol)
        rows.extend(record['rows'])
        specs.append({k: v for k, v in record['protocol'].items() if k not in ('offset', 'count')})
    if any(s != specs[0] for s in specs) or any(r['steps'] != protocol['max_steps'] for r in rows):
        raise RuntimeError('Inconsistent shards or shortened formal episode')
    return rows


def paired(image, decision, field):
    gains = sum(d[field] and not i[field] for i, d in zip(image, decision))
    losses = sum(i[field] and not d[field] for i, d in zip(image, decision))
    n = gains + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n) if n else 1.
    return dict(gains=gains, losses=losses, delta_pp=100 * (gains - losses) / len(image), exact_two_sided_p=p)


def report(output, source, reference, seeds):
    tables, comparisons, aggregate = [], [], []
    text = ['# Fixed-budget training-seed replication', '',
            'Only restoration-adapter training seed changes. The pretrained policy, paired data, noise, loss weights, optimizer, 2000 updates, final-checkpoint rule, and 50 full 400-step evaluation states stay fixed.',
            'Original-policy and exposure-oracle controls are shared from seed3072: they have no adapter-training seed. They are not rerun or counted as new independent evidence.',
            'Evaluation states are reused to test training repeatability, not a new held-out generalization test. Report all seeds, not the best seed.', '',
            '|Training seed|Task|Arm|Condition|Ever success|Final success|', '|---|---|---|---|---|---|']
    for task in reference['tasks']:
        anchor = collect(source, reference, task, 'original', 'clean')
        for seed in [reference['seed'], *seeds]:
            root = source if seed == reference['seed'] else output / f'seed{seed}'
            protocol = reference if seed == reference['seed'] else read(root / 'suite_protocol.json')
            expected = {k: v for k, v in dict(reference, seed=seed).items() if k != 'adapters'}
            if {k: v for k, v in protocol.items() if k != 'adapters'} != expected:
                raise RuntimeError('Repeat suite drifted from reference')
            groups = {}
            for arm, condition in GROUPS:
                reused = arm not in ('image', 'decision')
                rows = collect(source if reused else root, reference if reused else protocol, task, arm, condition)
                for baseline, row in zip(anchor, rows):
                    for key in ('seed', 'initial_state_sha256', 'initial_model_sha256', 'state_bank_entry_sha256'):
                        if baseline[key] != row[key]:
                            raise RuntimeError('Actual initial states are not matched')
                groups[(arm, condition)] = rows
                if reused and seed != reference['seed']:
                    continue
                ever = sum(r['success_ever'] for r in rows)
                final = sum(r['success_final'] for r in rows)
                entry = dict(seed=None if reused else seed, task=task, arm=arm, condition=condition,
                             episodes=len(rows), ever=ever, final=final, success_rate=ever/len(rows),
                             final_success_rate=final/len(rows), shared_control=reused)
                tables.append(entry)
                label = 'shared' if reused else str(seed)
                text.append(f'|{label}|{task}|{arm}|{condition}|{ever}/{len(rows)}|{final}/{len(rows)}|')
            for condition in ('clean', 'lowlight'):
                image, decision = groups[('image', condition)], groups[('decision', condition)]
                comparisons.append(dict(seed=seed, task=task, condition=condition,
                                        ever=paired(image, decision, 'success_ever'),
                                        final=paired(image, decision, 'success_final')))
    text += ['', '## Decision vs image-only: paired lowlight differences', '',
             '|Training seed|Task|Gains|Losses|Ever delta (pp)|Exact paired p|Final delta (pp)|', '|---|---|---|---|---|---|---|']
    for c in comparisons:
        if c['condition'] == 'lowlight':
            e, f = c['ever'], c['final']
            text.append(f'|{c["seed"]}|{c["task"]}|{e["gains"]}|{e["losses"]}|{e["delta_pp"]:+.1f}|{e["exact_two_sided_p"]:.4g}|{f["delta_pp"]:+.1f}|')
    text += ['', '## Across-training-seed summary', '',
             'Mean and sample SD use the three adapter training seeds, not 150 pooled independent episodes. Three seeds do not establish general superiority across policies, tasks, or real cameras.', '',
             '|Task|Condition|Metric|Image mean +/- SD (%)|Decision mean +/- SD (%)|Paired delta mean +/- SD (pp)|', '|---|---|---|---|---|---|']
    for task in reference['tasks']:
        for condition in ('clean', 'lowlight'):
            for field in ('success_rate', 'final_success_rate'):
                image = [100*r[field] for r in tables if r['task']==task and r['condition']==condition and r['arm']=='image']
                decision = [100*r[field] for r in tables if r['task']==task and r['condition']==condition and r['arm']=='decision']
                delta = [d-i for i, d in zip(image, decision)]
                entry = dict(task=task, condition=condition, metric=field, seeds=[reference['seed'], *seeds],
                             image_mean_pct=statistics.mean(image), image_sample_sd_pct=statistics.stdev(image),
                             decision_mean_pct=statistics.mean(decision), decision_sample_sd_pct=statistics.stdev(decision),
                             paired_delta_mean_pp=statistics.mean(delta), paired_delta_sample_sd_pp=statistics.stdev(delta))
                aggregate.append(entry)
                text.append(f'|{task}|{condition}|{field}|{entry["image_mean_pct"]:.2f} +/- {entry["image_sample_sd_pct"]:.2f}|{entry["decision_mean_pct"]:.2f} +/- {entry["decision_sample_sd_pct"]:.2f}|{entry["paired_delta_mean_pp"]:+.2f} +/- {entry["paired_delta_sample_sd_pp"]:.2f}|')
    write_json(output / 'summary.json', dict(source=str(source), results=tables, paired_comparisons=comparisons, across_training_seeds=aggregate))
    (output / 'RESULTS.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('preflight', 'full', 'summary'), default='preflight')
    p.add_argument('--gpus', default='auto')
    p.add_argument('--seeds', default='3073,3074')
    p.add_argument('--source-tag', default='decision_pilot_v1_20261003')
    p.add_argument('--tag', default='decision_repeat_v1_20261003')
    p.add_argument('--workers-per-gpu', type=int, default=4)
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    if seeds != [3073, 3074] or not 1 <= a.workers_per_gpu <= 4:
        raise ValueError('This matched replication is fixed to seeds3073,3074; at most 4 evaluators/GPU')
    source = REPO / 'outputs/task_restore_method_v1' / valid_tag(a.source_tag)
    output = REPO / 'outputs/task_restore_repeat_v1' / valid_tag(a.tag)
    reference = load_reference(source)
    settings = dict(reference=reference, source=str(source), seeds=seeds, shared_controls=True,
                    data_ready_sha256={t:sha(source/'data'/t/'ready.json') for t in reference['tasks']},
                    launcher_sha256=sha(Path(__file__)))
    if (output / 'settings.json').exists() and read(output / 'settings.json') != settings:
        raise RuntimeError('Existing output tag uses a different protocol; refusing overwrite')
    if a.stage == 'summary':
        if not (output / 'settings.json').exists():
            raise RuntimeError('No repeat experiment to summarize')
        report(output, source, reference, seeds)
        return
    gpus = idle_gpus() if a.gpus == 'auto' else a.gpus.split(',')
    if not gpus or len(set(gpus)) != len(gpus) or not all(re.fullmatch(r'[0-7]', g) for g in gpus):
        raise RuntimeError('No valid idle GPU selection')
    for gpu in gpus:
        assert_gpu_free(gpu)
    subprocess.run(['docker', 'image', 'inspect', IMAGE], check=True, stdout=subprocess.DEVNULL)
    for task in reference['tasks']:
        for arm, condition in GROUPS:
            if arm not in ('image', 'decision'):
                collect(source, reference, task, arm, condition)
    print(f'REPEAT_PREFLIGHT_OK seeds={seeds} updates=2000 tasks={reference["tasks"]} gpus={gpus} new_adapters=8 new_episodes=800 train_parallel={min(8,len(gpus))} eval_parallel={len(gpus)*a.workers_per_gpu}', flush=True)
    if a.stage == 'preflight':
        return
    output.mkdir(parents=True, exist_ok=True)
    locks, active = [], set()
    active_lock, stop = threading.Lock(), threading.Event()

    def halt_owned():
        with active_lock:
            names = list(active)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(names))) as pool:
            list(pool.map(lambda name:subprocess.run(['docker','stop','--time','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL), names))

    def execute(script, args, target, gpu, slot):
        if stop.is_set():
            raise RuntimeError('Repeat suite stopped')
        target.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(str(target).encode()).hexdigest()[:12]
        name = f'n2d-repeat-{identity}-{Path(script).stem}-s{slot}'
        cmd = ['docker','run','--rm','--init','--name',name,'--shm-size','2g','--gpus',f'device={gpu}',
               '-e','CUDA_VISIBLE_DEVICES=0','-e','PYTHONPATH=','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
               '-e','OPENBLAS_NUM_THREADS=1','-e','NUMEXPR_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
               '-e','TORCH_HOME=/workspace/task_restore_v1/cache/torch','-v',f'{REPO}:/workspace',
               '-w','/workspace',IMAGE,'/opt/taskrestore/bin/python','-u',script,*args]
        with active_lock:
            active.add(name)
        try:
            with (target/'console.log').open('a') as log:
                process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(('METHOD_','TRAIN_PROGRESS','DECISION_')):
                        print(f'[g{gpu} {target.relative_to(output)}] {line.strip()}', flush=True)
                if process.wait():
                    raise RuntimeError(f'Container failed: {target/"console.log"}')
        finally:
            with active_lock:
                active.discard(name)

    def pool_jobs(jobs, workers, function):
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)
        def worker(index):
            while not stop.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    function(job,gpus[index%len(gpus)],index//len(gpus))
                except BaseException:
                    stop.set()
                    raise
                finally:
                    pending.task_done()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = [pool.submit(worker,i) for i in range(workers)]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            halt_owned()
            raise
        finally:
            pool.shutdown(wait=True)

    start = time.perf_counter()
    try:
        for path in [output/'.suite.lock']+[CORE/f'.gpu_{g}.lock' for g in gpus]:
            handle = path.open('a')
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            locks.append(handle)
        write_json(output/'settings.json',settings)
        write_json(output/'dispatch.json',dict(gpus=gpus,workers_per_gpu=a.workers_per_gpu,eval_workers=len(gpus)*a.workers_per_gpu))
        def train(job, gpu, slot):
            seed, task, arm = job
            target = output/f'seed{seed}'/'train'/task/arm
            execute('task_restore_method_v1/train.py', ['--task',task,'--arm',arm,'--seed',str(seed),
                    '--updates','2000','--batch-size','16','--decision-weight','0.1',
                    '--data',container_path(source/'data'/task),'--output',container_path(target)],target,gpu,slot)
            verify_training(target,source,task,arm,seed)
        train_jobs = [(s,t,arm) for s in seeds for arm in ('image','decision') for t in reference['tasks']]
        pool_jobs(train_jobs,min(len(train_jobs),len(gpus)),train)
        protocols = {}
        for seed in seeds:
            root = output/f'seed{seed}'
            adapters = {f'{t}/{arm}':verify_training(root/'train'/t/arm,source,t,arm,seed)
                        for t in reference['tasks'] for arm in ('image','decision')}
            protocols[seed] = dict(reference,seed=seed,adapters=adapters)
            write_json(root/'suite_protocol.json',protocols[seed])
        jobs = [(s,t,arm,c,offset,min(reference['shard_size'],reference['episodes']-offset))
                for offset in range(0,reference['episodes'],reference['shard_size'])
                for s in seeds for arm,c in LEARNED for t in reference['tasks']]
        def evaluate(job, gpu, slot):
            seed,t,arm,c,offset,count = job
            root = output/f'seed{seed}'
            target = root/'eval'/t/f'{arm}_{c}'/'shards'/f'{offset:05d}'
            shard = (t,arm,c,offset,count)
            if (target/'result.json').exists():
                verify_shard(target,shard,protocols[seed])
                print(f'REPEAT_SHARD_CACHED seed={seed} task={t} arm={arm} condition={c} offset={offset}',flush=True)
                return
            bank = CORE/'assets/initial_states_v1/paired_xml_seed20000_n50'/f'{t}.json'
            args = ['--task',t,'--arm',arm,'--condition',c,'--episodes','50','--start-seed','20000',
                    '--offset',str(offset),'--count',str(count),'--max-steps','400','--state-bank',container_path(bank),
                    '--checkpoint',container_path(root/'train'/t/arm/'model.pt'),'--output',container_path(target)]
            execute('task_restore_method_v1/evaluate_method.py',args,target,gpu,slot)
            verify_shard(target,shard,protocols[seed])
            print(f'REPEAT_SHARD_DONE seed={seed} task={t} arm={arm} condition={c} offset={offset}',flush=True)
        print(f'REPEAT_CLOSED_LOOP_START episodes=800 steps_per_episode=400 workers={len(gpus)*a.workers_per_gpu}',flush=True)
        pool_jobs(jobs,len(gpus)*a.workers_per_gpu,evaluate)
        report(output,source,reference,seeds)
        write_json(output/'finished.json',dict(seconds=time.perf_counter()-start,seeds=seeds,new_episodes=800,settings=settings))
        print(f'REPEAT_FINISHED outputs={output} seconds={time.perf_counter()-start:.1f}',flush=True)
    except KeyboardInterrupt:
        stop.set()
        halt_owned()
        print('Interrupted: only this repeat suite containers stopped; checkpoints and episode results retained.',flush=True)
        raise
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
