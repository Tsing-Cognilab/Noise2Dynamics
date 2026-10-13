"""Server-only E05: reuse E04, six external adaptations, 4000 new full rollouts."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import queue
import subprocess
import threading
import time
from pathlib import Path
import protocol as P
from run_confirm import Dispatcher as BaseDispatcher
from run_method import IMAGE, assert_gpu_free, container_path, idle_gpus
from analysis import report

class Dispatcher(BaseDispatcher):
    def execute(self, script, arguments, target, gpu=None, slot=0):
        if self.stop.is_set():
            raise RuntimeError('E05 stopped')
        target.mkdir(parents=True,exist_ok=True)
        name = 'n2d-external-'+hashlib.sha256(str(target).encode()).hexdigest()[:12]+f'-s{slot}'
        cmd = ['docker','run','--rm','--init','--name',name,'--shm-size','2g']
        if gpu is not None:
            cmd += ['--gpus',f'device={gpu}','-e','CUDA_VISIBLE_DEVICES=0']
        cmd += ['-e','PYTHONPATH=','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
                '-e','OPENBLAS_NUM_THREADS=1','-e','NUMEXPR_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
                '-e','TORCH_HOME=/workspace/task_restore_v1/cache/torch','-v',f'{P.REPO}:/workspace',
                '-w','/workspace',IMAGE,'/opt/taskrestore/bin/python','-u',script,*arguments]
        with self.guard:
            if self.stop.is_set():
                raise RuntimeError('Interrupted before launch')
            self.active.add(name)
        try:
            with (target/'console.log').open('a') as log:
                with self.guard:
                    if self.stop.is_set():
                        raise RuntimeError('Interrupted before process creation')
                    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(('EXTERNAL_','COMPAT_','TEST_')):
                        print(f'[g{gpu} {target.relative_to(self.output)}] {line.strip()}',flush=True)
                if process.wait():
                    raise RuntimeError('Worker failed; inspect '+str(target/'console.log'))
        finally:
            with self.guard:
                self.active.discard(name)

def settings_for(reference):
    old = P.reference(reference)
    return dict(reference=str(reference),reference_lock=old,reference_lock_sha256=P.sha(reference/'model_lock.json'),
        external_sources=P.sources(),assets=P.asset_manifest(),tasks=list(P.TASKS),seeds=list(P.SEEDS),
        noise=P.NOISE,counts=P.COUNTS,new_episodes=4000,reused_E04_episodes=3200,new_models=6,
        updates=2000,batch_size=16,learning_rate=3e-4,weight_decay=1e-4,gradient_clip=1.,
        identity_probability=.2,horizon=400,start_seed=30000,bank_episodes=100,
        selection='all fixed seeds and last fixed-budget models; no validation/closed-loop selection',
        unseen='corruption profiles only; physical states reused from E04',
        inference='DDPM100; 2 observations; 8 actions; absolute10D to7D; cameras84x84',
        statistics='ever primary:6 external tests Holm;36 unseen tests Holm; final separate secondary family')

def lock_models(output, settings):
    old = settings['reference_lock']
    adapters = dict(old['adapters'])
    for seed in P.SEEDS:
        for task in P.TASKS:
            target = output/f'seed{seed}'/'train'/task/'retinex_finetune'
            done = P.read(target/'finished.json')
            spec = done['spec']
            expected = dict(task=task,arm='retinex_finetune',seed=seed,updates=2000,batch_size=16,
                            learning_rate=3e-4,weight_decay=1e-4,gradient_clip=1.,identity_probability=.2,
                            noise=P.NOISE['lowlight'],external_sources=settings['external_sources'],
                            assets=settings['assets'],sources=old['base_sources'],
                            data_sha256=old['settings']['data_sha256'][task])
            if any(spec.get(k)!=v for k,v in expected.items()) or P.sha(target/'model.pt')!=done['checkpoint_sha256'] or done['max_gradient']<=0:
                raise RuntimeError('Wrong external fixed-budget model')
            adapters[f'{seed}/{task}/retinex_finetune'] = dict(path=str(target/'model.pt'),sha256=done['checkpoint_sha256'],spec=spec)
    locked = dict(settings=settings,adapters=adapters,banks=old['banks'],policies=old['policies'],
                  base_sources=old['base_sources'],confirm_sources=old['confirm_sources'])
    path = output/'model_lock.json'
    if path.exists() and P.read(path)!=locked:
        raise RuntimeError('Evaluation models changed')
    P.write(path,locked)
    return locked

def eval_args(group, offset, count, output, settings, locked=None):
    seed,task,arm,condition = group
    args = ['--task',task,'--arm',arm,'--condition',condition,'--episodes',str(P.COUNTS[condition]),
        '--start-seed','30000','--bank-episodes','100','--offset',str(offset),'--count',str(count),
        '--max-steps','400','--state-bank',container_path(Path(settings['reference'])/'banks'/f'{task}.json'),
        '--output',container_path(output)]
    if seed is not None:
        args += ['--training-seed',str(seed),'--checkpoint',container_path(Path(P.adapter(locked,group)['path']))]
    return args

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('preflight','compatibility','full','summary'),default='preflight')
    p.add_argument('--gpus',default='auto')
    p.add_argument('--workers-per-gpu',type=int,default=4)
    p.add_argument('--reference-tag',default='decision_confirm_v1_20261004')
    p.add_argument('--tag',default='external_sensor_v1_20261004')
    a = p.parse_args()
    if not 1<=a.workers_per_gpu<=4:
        p.error('One to four evaluation workers per GPU')
    reference = P.REPO/'outputs/task_restore_confirm_v1'/P.tag(a.reference_tag)
    output = P.REPO/'outputs/task_restore_external_v1'/P.tag(a.tag)
    settings = settings_for(reference)
    if (output/'settings.json').exists() and P.read(output/'settings.json') != settings:
        raise RuntimeError('Output protocol changed; refusing overwrite')
    if a.stage=='summary':
        report(output,lock_models(output,settings))
        return
    gpus = idle_gpus() if a.gpus=='auto' else a.gpus.split(',')
    if not gpus or len(gpus)!=len(set(gpus)) or not all(g.isdigit() for g in gpus):
        raise RuntimeError('No distinct idle GPUs')
    for gpu in gpus:
        assert_gpu_free(gpu)
    subprocess.run(['docker','image','inspect',IMAGE],check=True,stdout=subprocess.DEVNULL)
    if sum(j[2] for j in P.jobs()) != 4000:
        raise RuntimeError('Budget changed')
    print(f'EXTERNAL_PREFLIGHT_OK retained_models=18 new_models=6 new_full_episodes=4000 reused=3200 gpus={gpus} eval_workers={len(gpus)*a.workers_per_gpu}',flush=True)
    if a.stage=='preflight':
        return
    output.mkdir(parents=True,exist_ok=True)
    runner,locks,start = Dispatcher(output,gpus),[],time.perf_counter()
    try:
        for path in [output/'.suite.lock']+[P.CORE/f'.gpu_{g}.lock' for g in gpus]:
            handle = path.open('a')
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            locks.append(handle)
        for gpu in gpus:
            assert_gpu_free(gpu)
        if a.stage=='compatibility':
            runner.execute('task_restore_external_v1/test_external.py',[],output/'compatibility'/'tests',gpus[0])
            def smoke(task,gpu,slot):
                group = None,task,'retinex_pretrained','lowlight'
                target = output/'compatibility'/task
                runner.execute('task_restore_external_v1/evaluate_external.py',eval_args(group,0,1,target,settings),target,gpu,slot)
                done = P.read(target/'result.json')
                if done['episodes']!=1 or done['rows'][0]['steps']!=400:
                    raise RuntimeError('Compatibility rollout not400 steps')
            runner.pool(list(P.TASKS),min(2,len(gpus)),smoke)
            P.write(output/'compatibility/finished.json',dict(external_sources=settings['external_sources'],
                    actual_full_rollouts=2,formal=False,seconds=time.perf_counter()-start))
            print('EXTERNAL_COMPATIBILITY_OK two actual400-step episodes; NOT formal evidence or hyperparameter selection',flush=True)
            return
        P.write(output/'settings.json',settings)
        P.write(output/'dispatch.json',dict(gpus=gpus,workers_per_gpu=a.workers_per_gpu))
        def train(job,gpu,slot):
            seed,task = job
            target = output/f'seed{seed}'/'train'/task/'retinex_finetune'
            runner.execute('task_restore_external_v1/train_external.py',[
                '--task',task,'--seed',str(seed),'--data',container_path(Path(settings['reference_lock']['settings']['source'])/'data'/task),
                '--output',container_path(target)],target,gpu,slot)
        runner.pool([(s,t) for s in P.SEEDS for t in P.TASKS],min(6,len(gpus)),train)
        locked = lock_models(output,settings)
        print('EXTERNAL_MODELS_LOCKED retained=18 new=6; starting 4000 full400-step rollouts',flush=True)
        completed,guard = 0,threading.Lock()
        def evaluate(job,gpu,slot):
            nonlocal completed
            group,offset,count = job
            target = P.group_path(output,group)/'shards'/f'{offset:05d}'
            if not (target/'result.json').exists():
                runner.execute('task_restore_external_v1/evaluate_external.py',eval_args(group,offset,count,target,settings,locked),target,gpu,slot)
            P.verify_shard(target,job,locked)
            with guard:
                completed += count
                print(f'EXTERNAL_PROGRESS complete_episodes={completed}/4000 group={group} offset={offset}',flush=True)
        runner.pool(P.jobs(),len(gpus)*a.workers_per_gpu,evaluate)
        report(output,locked)
        P.write(output/'finished.json',dict(new_episodes=4000,reused_E04_episodes=3200,
            model_lock_sha256=P.sha(output/'model_lock.json'),seconds=time.perf_counter()-start))
        print('EXTERNAL_FINISHED outputs='+str(output),flush=True)
    except BaseException:
        runner.stop.set()
        runner.halt()
        print('Stopped ONLY owned n2d-external containers. Rerun SAME full command to resume. No tmux exit/kill.',flush=True)
        raise
    finally:
        for handle in locks:
            handle.close()

if __name__=='__main__':
    main()
