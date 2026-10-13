"""User-run frozen-model confirmation + train-only feature control; all seeds."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
import contract as C
from report import report

HERE, REPO = C.HERE, C.REPO
METHOD, CORE = REPO / 'task_restore_method_v1', REPO / 'task_restore_v1'
sys.path.insert(0, str(REPO / 'task_restore_repeat_v1'))
from run_repeat import load_reference, verify_training
from run_method import IMAGE, assert_gpu_free, container_path, idle_gpus


def frozen_inputs(source, repeat):
    reference = load_reference(source)
    if not (repeat / 'finished.json').exists():
        raise RuntimeError('Both repeat seeds must finish before confirmation')
    prior = {}
    for seed in C.SEEDS:
        root = source if seed == 3072 else repeat / f'seed{seed}'
        for task in C.TASKS:
            for arm in ('image', 'decision'):
                target = root / 'train' / task / arm
                digest = verify_training(target, source, task, arm, seed)
                prior[f'{seed}/{task}/{arm}'] = dict(path=str(target / 'model.pt'), sha256=digest,
                                                    spec=C.read(target / 'finished.json')['spec'])
    records = {r['task']: r for r in C.read(CORE / 'assets/manifest.json')['tasks']}
    for task in C.TASKS:
        if C.sha(CORE / records[task]['checkpoint']) != reference['policies'][task]:
            raise RuntimeError('Frozen original policy weights changed')
    return dict(source=str(source), repeat=str(repeat), tasks=list(C.TASKS), seeds=list(C.SEEDS),
                counts=C.COUNTS, start_seed=C.START, bank_episodes=C.BANK_N, max_steps=C.HORIZON,
                shard_size=C.SHARD, new_models=6, new_episodes=3200, updates=C.UPDATES,
                calibration=C.CALIBRATION, base_sources=reference['sources'], confirm_sources=C.sources(),
                core_sources={name:C.sha(CORE / name) for name in ('evaluate.py', 'states.py', 'legacy_compat.py')},
                policies=reference['policies'], prior_adapters=prior,
                data_sha256={t:C.sha(source / 'data' / t / 'ready.json') for t in C.TASKS},
                selection='all fixed seeds, final checkpoint, no state exclusion, no significance stopping')


def lock_models(output, settings):
    adapters = dict(settings['prior_adapters'])
    calibrations = {}
    for task in C.TASKS:
        path = output / 'calibration' / f'{task}.json'
        cal = C.read(path)
        expected = dict(task=task, calibration=C.CALIBRATION, data_sha256=settings['data_sha256'][task],
                        base_sources=settings['base_sources'], confirm_sources=settings['confirm_sources'],
                        width=24, initialization_seed=3072, policy_sha256=settings['policies'][task])
        if cal['spec'] != expected or not 0 < cal['weight'] < 1e8:
            raise RuntimeError('Feature calibration does not match the agreed train-only rule')
        calibrations[task] = dict(sha256=C.sha(path), weight=cal['weight'])
        for seed in C.SEEDS:
            target = output / f'seed{seed}' / 'train' / task / 'feature'
            done = C.read(target / 'finished.json')
            spec = done['spec']
            wanted = dict(task=task, arm='feature', seed=seed, updates=C.UPDATES, batch_size=16,
                          feature_weight=cal['weight'], calibration_sha256=C.sha(path), learning_rate=3e-4,
                          weight_decay=1e-4, gradient_clip=1., width=24, sources=settings['base_sources'],
                          confirm_sources=settings['confirm_sources'], data_sha256=settings['data_sha256'][task],
                          parameters=267195, noise=dict(exposure=.25, gain=.004, read=.008, correlated=False),
                          identity_probability=.2, teacher_frozen=True,
                          selection='last fixed-budget checkpoint; no closed-loop selection')
            if spec != wanted or C.sha(target / 'model.pt') != done['checkpoint_sha256']:
                raise RuntimeError('Feature model configuration or hash changed')
            if [r['update'] for r in done['feature_checks']] != [1,500,1000,1500,2000] or any(r['feature_gradient'] <= 0 for r in done['feature_checks']):
                raise RuntimeError('Missing effective-feature-gradient checks')
            adapters[f'{seed}/{task}/feature'] = dict(path=str(target / 'model.pt'), sha256=done['checkpoint_sha256'], spec=spec)
    banks = {}
    for task in C.TASKS:
        path = output / 'banks' / f'{task}.json'
        bank = C.read(path)
        if (bank['task'], bank['start_seed'], bank['episodes']) != (task, C.START, C.BANK_N):
            raise RuntimeError('Wrong confirmation state bank')
        if [e['seed'] for e in bank['entries']] != list(range(C.START, C.START + C.BANK_N)):
            raise RuntimeError('Missing or replaced confirmation states')
        banks[task] = C.sha(path)
    locked = dict(settings=settings, adapters=adapters, banks=banks, policies=settings['policies'],
                  base_sources=settings['base_sources'], confirm_sources=settings['confirm_sources'], calibrations=calibrations)
    path = output / 'model_lock.json'
    if path.exists() and C.read(path) != locked:
        raise RuntimeError('Models/states changed after evaluation was locked')
    C.write(path, locked)
    return locked


class Dispatcher:
    def __init__(self, output, gpus):
        self.output, self.gpus = output, gpus
        self.active, self.guard, self.stop = set(), threading.Lock(), threading.Event()

    def halt(self):
        with self.guard:
            names = list(self.active)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
            list(pool.map(lambda n:subprocess.run(['docker','stop','--time','5',n],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), names))

    def execute(self, script, arguments, target, gpu=None, slot=0):
        if self.stop.is_set():
            raise RuntimeError('Confirmation dispatch stopped')
        target.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(str(target).encode()).hexdigest()[:12]
        name = f'n2d-confirm-{identity}-{Path(script).stem}-s{slot}'
        cmd = ['docker','run','--rm','--init','--name',name,'--shm-size','2g']
        if gpu is not None:
            cmd += ['--gpus',f'device={gpu}','-e','CUDA_VISIBLE_DEVICES=0']
        cmd += ['-e','PYTHONPATH=','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
                '-e','OPENBLAS_NUM_THREADS=1','-e','NUMEXPR_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
                '-e','TORCH_HOME=/workspace/task_restore_v1/cache/torch','-v',f'{REPO}:/workspace',
                '-w','/workspace',IMAGE,'/opt/taskrestore/bin/python','-u',script,*arguments]
        with self.guard:
            if self.stop.is_set():
                raise RuntimeError('Suite stopped before container launch')
            self.active.add(name)
            process = None
        try:
            with (target / 'console.log').open('a') as log:
                # Holding the guard during creation closes the interrupt/start race.
                with self.guard:
                    if self.stop.is_set():
                        raise RuntimeError('Suite stopped before process launch')
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(('FEATURE_', 'METHOD_', 'STATE_BANK_')):
                        print(f'[g{gpu} {target.relative_to(self.output)}] {line.strip()}', flush=True)
                if process.wait():
                    raise RuntimeError(f'Container failed; inspect {target / "console.log"}')
        finally:
            with self.guard:
                self.active.discard(name)

    def pool(self, jobs, workers, function):
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)
        def worker(index):
            while not self.stop.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    function(job, self.gpus[index % len(self.gpus)], index // len(self.gpus))
                except BaseException:
                    self.stop.set()
                    raise
                finally:
                    pending.task_done()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = [pool.submit(worker, i) for i in range(workers)]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            self.stop.set()
            self.halt()
            raise
        finally:
            pool.shutdown(wait=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('preflight','full','summary'), default='preflight')
    p.add_argument('--gpus', default='auto')
    p.add_argument('--workers-per-gpu', type=int, default=4)
    p.add_argument('--source-tag', default='decision_pilot_v1_20261003')
    p.add_argument('--repeat-tag', default='decision_repeat_v1_20261003')
    p.add_argument('--tag', default='decision_confirm_v1_20261004')
    a = p.parse_args()
    if not 1 <= a.workers_per_gpu <= 4:
        p.error('One to four evaluation workers per GPU')
    source = REPO / 'outputs/task_restore_method_v1' / C.tag(a.source_tag)
    repeat = REPO / 'outputs/task_restore_repeat_v1' / C.tag(a.repeat_tag)
    output = REPO / 'outputs/task_restore_confirm_v1' / C.tag(a.tag)
    settings = frozen_inputs(source, repeat)
    if (output / 'settings.json').exists() and C.read(output / 'settings.json') != settings:
        raise RuntimeError('Output already belongs to a different protocol; no overwrite')
    if a.stage == 'summary':
        locked = lock_models(output, settings)
        report(output, locked)
        return
    gpus = idle_gpus() if a.gpus == 'auto' else a.gpus.split(',')
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise RuntimeError('No distinct idle GPUs')
    for gpu in gpus:
        assert_gpu_free(gpu)
    subprocess.run(['docker','image','inspect',IMAGE], check=True, stdout=subprocess.DEVNULL)
    print(f'CONFIRM_PREFLIGHT_OK seeds={C.SEEDS} new_models=6 episodes=3200 steps=400 gpus={gpus} eval_workers={len(gpus)*a.workers_per_gpu}', flush=True)
    if a.stage == 'preflight':
        return
    output.mkdir(parents=True, exist_ok=True)
    locks, runner, start = [], Dispatcher(output, gpus), time.perf_counter()
    try:
        for path in [output / '.suite.lock'] + [CORE / f'.gpu_{g}.lock' for g in gpus]:
            handle = path.open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            locks.append(handle)
        for gpu in gpus:
            assert_gpu_free(gpu)
        C.write(output / 'settings.json', settings)
        C.write(output / 'dispatch.json', dict(gpus=gpus, workers_per_gpu=a.workers_per_gpu))
        runner.execute('task_restore_v1/states.py', ['--tasks',','.join(C.TASKS),'--episodes','100',
                       '--start-seed','30000','--directory',container_path(output / 'banks')], output / 'banks')
        def calibration(task, gpu, slot):
            runner.execute('task_restore_confirm_v1/feature_train.py', ['--mode','calibrate','--task',task,
                           '--data',container_path(source / 'data' / task),
                           '--calibration',container_path(output / 'calibration' / f'{task}.json')],
                           output / 'calibration' / task, gpu, slot)
        runner.pool(list(C.TASKS), min(2, len(gpus)), calibration)
        def train(job, gpu, slot):
            seed, task = job
            target = output / f'seed{seed}' / 'train' / task / 'feature'
            runner.execute('task_restore_confirm_v1/feature_train.py', ['--mode','train','--task',task,
                           '--seed',str(seed),'--data',container_path(source / 'data' / task),
                           '--calibration',container_path(output / 'calibration' / f'{task}.json'),
                           '--output',container_path(target)], target, gpu, slot)
        runner.pool([(s,t) for s in C.SEEDS for t in C.TASKS], min(6,len(gpus)), train)
        locked = lock_models(output, settings)
        print('CONFIRM_ALL_MODELS_LOCKED new_models=6 retained_models=12; starting 3200 full simulator rollouts', flush=True)
        completed = 0
        progress = threading.Lock()
        def evaluate(job, gpu, slot):
            nonlocal completed
            group, offset, count = job
            seed, task, arm, condition = group
            target = C.group_path(output, group) / 'shards' / f'{offset:05d}'
            if not (target / 'result.json').exists():
                args = ['--task',task,'--arm',arm,'--condition',condition,'--episodes',str(C.COUNTS[condition]),
                        '--start-seed','30000','--bank-episodes','100','--offset',str(offset),'--count',str(count),
                        '--max-steps','400','--state-bank',container_path(output / 'banks' / f'{task}.json'),
                        '--output',container_path(target)]
                if seed is not None:
                    args += ['--training-seed',str(seed),'--checkpoint',container_path(Path(locked['adapters'][f'{seed}/{task}/{arm}']['path']))]
                runner.execute('task_restore_confirm_v1/evaluate_confirm.py', args, target, gpu, slot)
            C.verify_shard(target, job, locked)
            with progress:
                completed += count
                print(f'CONFIRM_PROGRESS complete_episodes={completed}/3200 group={group} offset={offset}', flush=True)
        runner.pool(C.jobs(), len(gpus) * a.workers_per_gpu, evaluate)
        report(output, locked)
        C.write(output / 'finished.json', dict(seconds=time.perf_counter()-start, episodes=3200,
                                               model_lock_sha256=C.sha(output / 'model_lock.json')))
        print(f'CONFIRM_FINISHED outputs={output} seconds={time.perf_counter()-start:.1f}', flush=True)
    except BaseException:
        runner.stop.set()
        runner.halt()
        print('Confirmation stopped. Only owned n2d-confirm containers stopped; rerun the SAME full command to resume. No tmux exit/kill.', flush=True)
        raise
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
