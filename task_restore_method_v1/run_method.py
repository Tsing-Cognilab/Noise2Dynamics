"""User-run matched pilot: replay -> parallel training -> paired closed-loop table."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORE = REPO/'task_restore_v1'
sys.path.insert(0,str(CORE))
from run import assert_gpu_free,IMAGE
from run_parallel import idle_gpus,write_json,sha


def container_path(path):
    return str(Path('/workspace')/Path(path).relative_to(REPO))


def verify_shard(target,job,protocol):
    task,arm,condition,offset,count = job
    record = json.loads((target/'result.json').read_text())
    spec = record['protocol']
    expected = dict(task=task,arm=arm,condition=condition,offset=offset,count=count,episodes=protocol['episodes'],
                    start_seed=protocol['start_seed'],max_steps=protocol['max_steps'],sources=protocol['sources'],
                    policy_sha256=protocol['policies'][task],state_bank_sha256=protocol['banks'][task])
    if any(spec.get(k)!=v for k,v in expected.items()):
        raise RuntimeError(f'Shard protocol mismatch {target}')
    expected_adapter = protocol['adapters'].get(f'{task}/{arm}')
    if spec.get('adapter_sha256') != expected_adapter:
        raise RuntimeError('Shard uses a different adapter checkpoint')
    rows = record['rows']
    wanted = list(range(protocol['start_seed']+offset,protocol['start_seed']+offset+count))
    if [r['seed'] for r in rows]!=wanted or record['episodes']!=count:
        raise RuntimeError('Missing or duplicated evaluation episodes')
    if rows != [json.loads(l) for l in (target/'episodes.jsonl').read_text().splitlines()]:
        raise RuntimeError('Shard JSON and JSONL disagree')
    for r in rows:
        if (r['task'],r['arm'],r['condition'])!=(task,arm,condition) or not 1<=r['steps']<=protocol['max_steps']:
            raise RuntimeError('Wrong task or invalid horizon')
        if not isinstance(r['success_ever'],bool) or not isinstance(r['success_final'],bool):
            raise RuntimeError('Invalid success values')
    if record['successes']!=sum(r['success_ever'] for r in rows) or record['final_successes']!=sum(r['success_final'] for r in rows):
        raise RuntimeError('Success count inconsistent')
    return record


GROUPS = [('original','clean'),('original','lowlight'),('exposure_oracle','lowlight'),
          ('image','clean'),('image','lowlight'),('decision','clean'),('decision','lowlight')]


def summary(output,protocol):
    table = []
    merged = {}
    report = ['# Matched restoration method pilot','',
              f'One adapter training seed; fixed {protocol["updates"]}-update budget. Not training-seed replication or proof of publication readiness.',
              'Exposure oracle uses known synthetic exposure 0.25. It is a diagnostic upper-information control, not blind restoration.',
              ('SMOKE ONLY: shortened episodes do not measure learned operation or method effectiveness.' if protocol['smoke'] else
               f'Primary: real environment task success, full {protocol["max_steps"]}-step episodes. Fresh test states, paired across all arms.'),'',
              '|Task|Arm|Condition|Ever successes|Final successes|','|---|---|---|---|---|']
    for task in protocol['tasks']:
        base_rows = None
        for arm,condition in GROUPS:
            rows = []
            specs = []
            for offset in range(0,protocol['episodes'],protocol['shard_size']):
                count = min(protocol['shard_size'],protocol['episodes']-offset)
                job = (task,arm,condition,offset,count)
                target = output/'eval'/task/f'{arm}_{condition}'/'shards'/f'{offset:05d}'
                r = verify_shard(target,job,protocol)
                specs.append({k:v for k,v in r['protocol'].items() if k not in ['offset','count']})
                rows.extend(r['rows'])
            if any(s!=specs[0] for s in specs):
                raise RuntimeError('Adapter/policy differs across shards')
            if base_rows is None:
                base_rows = rows
            for baseline,row in zip(base_rows,rows):
                for key in ['seed','initial_state_sha256','initial_model_sha256','state_bank_entry_sha256']:
                    if baseline[key]!=row[key]:
                        raise RuntimeError('Actual initial states not paired across methods')
            n = len(rows)
            ever = sum(r['success_ever'] for r in rows)
            final = sum(r['success_final'] for r in rows)
            entry = dict(task=task,arm=arm,condition=condition,episodes=n,ever=ever,final=final,
                         success_rate=ever/n,final_success_rate=final/n,
                         mean_inference_seconds=sum(r['inference_seconds'] for r in rows)/n)
            table.append(entry)
            merged[(task,arm,condition)] = rows
            target = output/'eval'/task/f'{arm}_{condition}'
            write_json(target/'merged.json',dict(protocol=specs[0],rows=rows,summary=entry))
            report.append(f'|{task}|{arm}|{condition}|{ever}/{n} ({100*ever/n:.1f}%)|{final}/{n} ({100*final/n:.1f}%)|')
    comparisons = []
    report.extend(['','## Paired decision vs image-only (lowlight)',''])
    for task in protocol['tasks']:
        image = merged[(task,'image','lowlight')]
        decision = merged[(task,'decision','lowlight')]
        gains = sum(d['success_ever'] and not i['success_ever'] for i,d in zip(image,decision))
        losses = sum(i['success_ever'] and not d['success_ever'] for i,d in zip(image,decision))
        discordant = gains+losses
        p = min(1.,2*sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1.
        comparisons.append(dict(task=task,gains=gains,losses=losses,delta_pp=100*(gains-losses)/len(image),exact_two_sided_p=p))
        report.append(f'- {task}: gained {gains}, lost {losses}, delta {100*(gains-losses)/len(image):+.1f} pp; exact paired p={p:.4g}.')
    report.extend(['','Do not choose best checkpoint/seed/test state. Offline image/decision losses are diagnostics, not closed-loop benefit.',
                   'If ordinary recovery or oracle exposure matches decision recovery, added decision supervision is not supported by this pilot.'])
    write_json(output/'summary.json',dict(results=table,paired_comparisons=comparisons))
    (output/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=['preflight','smoke','full','summary'],default='preflight')
    p.add_argument('--gpus',default='auto')
    p.add_argument('--tasks',default='can,square')
    p.add_argument('--tag',default='decision_pilot_v1_20261003')
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--train-demos',type=int,default=24)
    p.add_argument('--valid-demos',type=int,default=4)
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--start-seed',type=int,default=20000)
    p.add_argument('--max-steps',type=int,default=400)
    p.add_argument('--workers-per-gpu',type=int,default=4)
    p.add_argument('--shard-size',type=int,default=5)
    a = p.parse_args()
    if not re.fullmatch('[A-Za-z0-9._-]+',a.tag) or a.tag in ['.','..']:
        raise ValueError('Invalid tag')
    tasks = a.tasks.split(',')
    if not set(tasks)<={'lift','can','square'} or len(set(tasks))!=len(tasks):
        raise ValueError('Invalid tasks')
    if not 1<=a.workers_per_gpu<=4 or min(a.updates,a.episodes,a.max_steps,a.shard_size,a.train_demos,a.valid_demos,a.stride,a.batch_size)<1:
        raise ValueError('Invalid budget')
    output = REPO/'outputs/task_restore_method_v1'/a.tag
    if a.stage=='summary':
        summary(output,json.loads((output/'suite_protocol.json').read_text()))
        return
    gpus = idle_gpus() if a.gpus=='auto' else a.gpus.split(',')
    if not gpus or len(set(gpus))!=len(gpus) or not all(re.fullmatch('[0-7]',g) for g in gpus):
        raise RuntimeError('No idle GPU')
    for gpu in gpus:
        assert_gpu_free(gpu)
    subprocess.run(['docker','image','inspect',IMAGE],check=True,stdout=subprocess.DEVNULL)
    for task in tasks:
        if not (CORE/'assets/diffusion_policy_transformer'/task/'original_demo.hdf5').exists():
            raise RuntimeError('Missing demonstrations')
    print(f'METHOD_PREFLIGHT_OK tasks={tasks} gpus={gpus} train_parallel={min(len(gpus),2*len(tasks))} eval_parallel={len(gpus)*a.workers_per_gpu}',flush=True)
    if a.stage=='preflight':
        return
    if a.stage=='smoke':
        a.updates,a.train_demos,a.valid_demos,a.episodes,a.max_steps,a.shard_size = 100,1,1,1,8,1
    output.mkdir(parents=True,exist_ok=True)
    locks = []
    active = set()
    active_lock = threading.Lock()
    stop = threading.Event()
    def halt_owned():
        with active_lock:
            names = list(active)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(names))) as pool:
            list(pool.map(lambda name:subprocess.run(['docker','stop','--time','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL),names))
    def execute(script,arguments,target,gpu=None,slot=0):
        if stop.is_set():
            raise RuntimeError('Suite already stopped')
        target.mkdir(parents=True,exist_ok=True)
        identity = hashlib.sha256(str(target).encode()).hexdigest()[:12]
        name = f'n2d-method-{identity}-{Path(script).stem}-s{slot}'
        cmd = ['docker','run','--rm','--init','--name',name,'--shm-size','2g']
        if gpu is not None:
            cmd += ['--gpus',f'device={gpu}','-e','CUDA_VISIBLE_DEVICES=0']
        cmd += ['-e','PYTHONPATH=','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
                '-e','NUMEXPR_NUM_THREADS=1','-e','LP_NUM_THREADS=1','-e','TORCH_HOME=/workspace/task_restore_v1/cache/torch',
                '-v',f'{REPO}:/workspace','-w','/workspace',IMAGE,'/opt/taskrestore/bin/python','-u',script,*arguments]
        with active_lock:
            active.add(name)
        try:
            with (target/'console.log').open('a') as log:
                process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(('METHOD_','TRAIN_PROGRESS','DATA_REPLAY','DATA_TEACHER','DECISION_','STATE_BANK_')):
                        print(f'[g{gpu}] {line.strip()}',flush=True)
                if process.wait():
                    raise RuntimeError(f'Container failed; see {target/"console.log"}')
        finally:
            with active_lock:
                active.discard(name)
    def pool_jobs(jobs,worker_count,function):
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
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
        futures = [pool.submit(worker,i) for i in range(worker_count)]
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
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks.append(handle)
        sources = {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
        settings = dict(tasks=tasks,seed=a.seed,updates=a.updates,batch_size=a.batch_size,train_demos=a.train_demos,
                        valid_demos=a.valid_demos,stride=a.stride,episodes=a.episodes,start_seed=a.start_seed,max_steps=a.max_steps,
                        shard_size=a.shard_size,sources=sources,smoke=a.stage=='smoke',groups=GROUPS)
        if (output/'settings.json').exists() and json.loads((output/'settings.json').read_text())!=json.loads(json.dumps(settings)):
            raise RuntimeError('Existing tag is a different method protocol; no overwrite')
        write_json(output/'settings.json',settings)
        write_json(output/'dispatch.json',dict(gpus=gpus,workers_per_gpu=a.workers_per_gpu))
        def prepare(task,gpu,slot):
            target = output/'data'/task
            args = ['--task',task,'--output',container_path(target),'--train-demos',str(a.train_demos),
                    '--valid-demos',str(a.valid_demos),'--stride',str(a.stride)]
            if a.stage=='smoke':
                args += ['--limit-samples','4']
            execute('task_restore_method_v1/prepare_data.py',args,target,gpu,slot)
        pool_jobs(tasks,min(len(tasks),len(gpus)),prepare)
        def train(job,gpu,slot):
            task,arm = job
            target = output/'train'/task/arm
            execute('task_restore_method_v1/train.py',['--task',task,'--arm',arm,'--seed',str(a.seed),'--updates',str(a.updates),
                    '--batch-size',str(a.batch_size),'--data',container_path(output/'data'/task),'--output',container_path(target)],target,gpu,slot)
        pool_jobs([(t,arm) for arm in ['image','decision'] for t in tasks],min(2*len(tasks),len(gpus)),train)
        bankdir = CORE/'assets/initial_states_v1'/f'paired_xml_seed{a.start_seed}_n{a.episodes}'
        execute('task_restore_v1/states.py',['--tasks',a.tasks,'--episodes',str(a.episodes),'--start-seed',str(a.start_seed),
                '--directory',container_path(bankdir)],output/'state_prepare')
        records = {r['task']:r for r in json.loads((CORE/'assets/manifest.json').read_text())['tasks']}
        protocol = dict(settings,policies={t:records[t]['checkpoint_sha256'] for t in tasks},banks={t:sha(bankdir/f'{t}.json') for t in tasks},
                        adapters={f'{t}/{arm}':sha(output/'train'/t/arm/'model.pt') for t in tasks for arm in ['image','decision']})
        write_json(output/'suite_protocol.json',protocol)
        jobs = [(t,arm,c,offset,min(a.shard_size,a.episodes-offset))
                for offset in range(0,a.episodes,a.shard_size) for arm,c in GROUPS for t in tasks]
        def evaluate(job,gpu,slot):
            t,arm,c,offset,count = job
            target = output/'eval'/t/f'{arm}_{c}'/'shards'/f'{offset:05d}'
            if (target/'result.json').exists():
                verify_shard(target,job,protocol)
                print(f'METHOD_SHARD_CACHED task={t} arm={arm} condition={c} offset={offset}',flush=True)
                return
            args = ['--task',t,'--arm',arm,'--condition',c,'--episodes',str(a.episodes),'--start-seed',str(a.start_seed),
                    '--offset',str(offset),'--count',str(count),'--max-steps',str(a.max_steps),'--state-bank',container_path(bankdir/f'{t}.json'),
                    '--output',container_path(target)]
            if arm in ['image','decision']:
                args += ['--checkpoint',container_path(output/'train'/t/arm/'model.pt')]
            execute('task_restore_method_v1/evaluate_method.py',args,target,gpu,slot)
            verify_shard(target,job,protocol)
        print(f'METHOD_CLOSED_LOOP_START episodes={len(tasks)*len(GROUPS)*a.episodes} max_steps={a.max_steps} workers={len(gpus)*a.workers_per_gpu}',flush=True)
        pool_jobs(jobs,len(gpus)*a.workers_per_gpu,evaluate)
        summary(output,protocol)
        write_json(output/'finished.json',dict(seconds=time.perf_counter()-start,settings=settings,
                   warning='smoke, not effectiveness evidence' if a.stage=='smoke' else 'one adapter training seed pilot'))
        print(f'METHOD_PILOT_FINISHED outputs={output} seconds={time.perf_counter()-start:.1f}',flush=True)
    except KeyboardInterrupt:
        stop.set()
        halt_owned()
        print('Interrupted: stopped only owned method containers; checkpoints and episodes retained.',flush=True)
        raise
    finally:
        for handle in locks:
            handle.close()


if __name__=='__main__':
    main()
