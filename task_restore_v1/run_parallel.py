"""Dynamic episode shards: multiple isolated evaluators per idle GPU."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from run import ROOT, REPO, IMAGE, ALL_CONDITIONS, assert_gpu_free, summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def idle_gpus():
    indices = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], text=True).splitlines()
    available = []
    for gpu in indices:
        gpu = gpu.strip()
        try:
            assert_gpu_free(gpu)
        except RuntimeError as error:
            print(f'SKIP_BUSY_GPU {error}', flush=True)
        else:
            available.append(gpu)
    return available


def build_jobs(tasks, conditions, episodes, shard_size):
    # Interleave tasks/conditions, not 50-episode blocks; idle workers steal work.
    return [(task, condition, offset, min(shard_size, episodes-offset))
            for offset in range(0, episodes, shard_size)
            for condition in conditions for task in tasks]


def shard_directory(output, job):
    task, condition, offset, count = job
    return output/task/condition/'shards'/f'{offset:05d}_{count:05d}'


def verify_result(path, job, suite):
    task, condition, offset, count = job
    record = json.loads(path.read_text())
    protocol = record['protocol']
    expected = dict(task=task, condition=condition, episode_offset=offset, episode_count=count,
                    episodes=suite['episodes'], start_seed=suite['start_seed'], max_steps=suite['max_steps'],
                    checkpoint_sha256=suite['source_checkpoints'][task], state_bank_sha256=suite['state_banks'][task],
                    evaluator_sha256=suite['evaluator_sha256'], parallel_wrapper_sha256=suite['parallel_wrapper_sha256'])
    if any(protocol.get(k) != v for k,v in expected.items()):
        raise RuntimeError(f'Shard protocol mismatch: {path}')
    rows = record['rows']
    wanted = list(range(suite['start_seed']+offset, suite['start_seed']+offset+count))
    if record['episodes'] != count or [r['seed'] for r in rows] != wanted:
        raise RuntimeError(f'Missing, duplicate or out-of-range shard episodes: {path}')
    logged = [json.loads(line) for line in (path.parent/'episodes.jsonl').read_text().splitlines()]
    if logged != rows:
        raise RuntimeError(f'Shard JSONL and result disagree: {path}')
    for row in rows:
        if row['task'] != task or row['condition'] != condition or not isinstance(row['success_ever'], bool) or not isinstance(row['success_final'], bool):
            raise RuntimeError(f'Invalid task/success episode record: {path}')
        if row['steps'] < 1 or row['steps'] > suite['max_steps'] or not math.isfinite(row['wall_seconds']) or row['wall_seconds'] <= 0:
            raise RuntimeError(f'Invalid episode length/time: {path}')
        for key in ['initial_state_sha256', 'initial_model_sha256', 'state_bank_entry_sha256']:
            if not re.fullmatch('[a-f0-9]{64}', row[key]):
                raise RuntimeError(f'Invalid fingerprint: {path}')
    if record['successes'] != sum(r['success_ever'] for r in rows):
        raise RuntimeError(f'Shard success count mismatch: {path}')
    if record['success_rate'] != sum(r['success_ever'] for r in rows)/count or record['final_success_rate'] != sum(r['success_final'] for r in rows)/count:
        raise RuntimeError(f'Shard success rate mismatch: {path}')
    return record


def merge_results(output, suite):
    for task in suite['tasks']:
        for condition in suite['conditions']:
            records = []
            for job in build_jobs([task], [condition], suite['episodes'], suite['shard_size']):
                path = shard_directory(output, job)/'result.json'
                if not path.exists():
                    raise RuntimeError(f'Incomplete suite: missing {path}')
                records.append(verify_result(path, job, suite))
            protocol = dict(records[0]['protocol'])
            protocol.pop('episode_offset')
            protocol.pop('episode_count')
            protocol['parallel_shards'] = len(records)
            for record in records[1:]:
                other = dict(record['protocol'])
                other.pop('episode_offset')
                other.pop('episode_count')
                other['parallel_shards'] = len(records)
                if other != protocol:
                    raise RuntimeError(f'Underlying environment/policy differs across shards: {task} {condition}')
            rows = sorted([row for record in records for row in record['rows']], key=lambda r:r['seed'])
            if [r['seed'] for r in rows] != list(range(suite['start_seed'], suite['start_seed']+suite['episodes'])):
                raise RuntimeError('Merged episodes are missing or duplicated')
            n = len(rows)
            success = sum(r['success_ever'] for r in rows)
            result = dict(protocol=protocol, rows=rows, episodes=n, successes=success,
                          success_rate=success/n, final_success_rate=sum(r['success_final'] for r in rows)/n,
                          mean_episode_seconds=sum(r['wall_seconds'] for r in rows)/n)
            target = output/task/condition
            write_json(target/'result.json', result)
            write_json(target/'protocol.json', protocol)
            temporary = target/'episodes.jsonl.tmp'
            temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            temporary.replace(target/'episodes.jsonl')
    return summarize(output)  # Strict actual state/model/bank-entry pairing remains mandatory.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'baseline', 'summary'], default='preflight')
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--shard-size', type=int, default=5)
    parser.add_argument('--tasks', default='lift,can,square')
    parser.add_argument('--conditions', default='clean,shot_read_medium,lowlight')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--start-seed', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--video-episodes', type=int, default=2)
    parser.add_argument('--tag', default='sensor_baseline_parallel_v1_20261003')
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9._-]+', args.tag) or args.tag in {'.','..'}:
        raise ValueError('Invalid output tag')
    tasks, conditions = args.tasks.split(','), args.conditions.split(',')
    if len(set(tasks)) != len(tasks) or len(set(conditions)) != len(conditions) or not set(tasks) <= {'lift','can','square'} or not set(conditions) <= set(ALL_CONDITIONS):
        raise ValueError('Invalid or repeated tasks/conditions')
    if not 1 <= args.workers_per_gpu <= 4 or args.shard_size < 1 or args.episodes < 1 or args.max_steps < 1:
        raise ValueError('Workers must be 1..4 and sizes/steps positive')
    output = REPO/'outputs'/'task_restore_v1'/args.tag
    if args.stage == 'summary':
        suite = json.loads((output/'suite_protocol.json').read_text())
        merge_results(output, suite)
        print(f'TASK_RESTORE_PARALLEL_SUMMARY_FINISHED outputs={output}', flush=True)
        return
    gpus = idle_gpus() if args.gpus == 'auto' else args.gpus.split(',')
    if not gpus or len(set(gpus)) != len(gpus) or not all(re.fullmatch('[0-7]',g) for g in gpus):
        raise RuntimeError('No idle GPUs or invalid GPU list')
    for gpu in gpus:
        assert_gpu_free(gpu)
    subprocess.run(['docker','image','inspect',IMAGE], check=True, stdout=subprocess.DEVNULL)
    manifest = json.loads((ROOT/'assets'/'manifest.json').read_text())
    records = {r['task']:r for r in manifest['tasks']}
    for task in tasks:
        for key in ['checkpoint','metadata_hdf5','config']:
            if not (ROOT/records[task][key]).is_file():
                raise RuntimeError(f'Missing {key} for {task}')
    total_workers = len(gpus)*args.workers_per_gpu
    print(f'TASK_RESTORE_PARALLEL_PREFLIGHT_OK gpus={gpus} workers_per_gpu={args.workers_per_gpu} total_workers={total_workers} shard_size={args.shard_size}', flush=True)
    if args.stage == 'preflight':
        return
    output.mkdir(parents=True, exist_ok=True)
    locks = []
    try:
        for path in [output/'.suite.lock']+[ROOT/f'.gpu_{gpu}.lock' for gpu in gpus]:
            handle = path.open('a')
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_directory = ROOT/'assets'/'initial_states_v1'/f'paired_xml_seed{args.start_seed}_n{args.episodes}'
        subprocess.run(['docker','run','--rm','-e','PYTHONPATH=','-e','OMP_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
                        '-v',f'{REPO}:/workspace','-w','/workspace',IMAGE,'/opt/taskrestore/bin/python','-u','task_restore_v1/states.py',
                        '--tasks',args.tasks,'--episodes',str(args.episodes),'--start-seed',str(args.start_seed),
                        '--directory',str(Path('/workspace')/state_directory.relative_to(REPO))],check=True)
        suite = dict(version=2, tasks=tasks, conditions=conditions, episodes=args.episodes, start_seed=args.start_seed,
                     max_steps=args.max_steps, shard_size=args.shard_size, video_episodes=args.video_episodes, source_archive_sha256=manifest['source_archive_sha256'],
                     source_checkpoints={t:records[t]['checkpoint_sha256'] for t in tasks},
                     evaluator_sha256=sha(ROOT/'evaluate.py'), parallel_wrapper_sha256=sha(ROOT/'evaluate_parallel.py'),
                     launcher_sha256=sha(Path(__file__)), state_banks={t:sha(state_directory/f'{t}.json') for t in tasks})
        saved = output/'suite_protocol.json'
        if saved.exists() and json.loads(saved.read_text()) != suite:
            raise RuntimeError('Output tag already belongs to a different protocol; no overwriting')
        write_json(saved,suite)
        write_json(output/'dispatch.json',dict(gpus=gpus,workers_per_gpu=args.workers_per_gpu,total_workers=total_workers,
                                              cpu_threads_per_process=1,mesa_threads_per_process=1,video_episodes=args.video_episodes))
        jobs = queue.Queue()
        planned = build_jobs(tasks,conditions,args.episodes,args.shard_size)
        resumed = 0
        for job in planned:
            result_path = shard_directory(output,job)/'result.json'
            if result_path.exists():
                verify_result(result_path,job,suite)
                resumed += 1
            else:
                jobs.put(job)
        print(f'SHARD_SCHEDULE total_shards={len(planned)} already_complete={resumed} remaining={jobs.qsize()} total_episodes={len(tasks)*len(conditions)*args.episodes}',flush=True)
        stop = threading.Event()
        active, errors = set(), []
        active_lock = threading.Lock()
        def stop_owned():
            with active_lock:
                names = list(active)
            def halt(name):
                subprocess.run(['docker','stop','--time','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(names))) as stopping:
                list(stopping.map(halt,names))
        def worker(gpu,slot):
            while not stop.is_set():
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    return
                task,condition,offset,count = job
                target = shard_directory(output,job)
                target.mkdir(parents=True,exist_ok=True)
                name = f'n2d-taskrestore-p-{args.tag}-{task}-{condition}-o{offset}-g{gpu}-w{slot}'
                cmd = ['docker','run','--rm','--init','--name',name,'--gpus',f'device={gpu}','--shm-size','2g',
                       '-e','CUDA_VISIBLE_DEVICES=0','-e','PYTHONPATH=','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
                       '-e','OPENBLAS_NUM_THREADS=1','-e','NUMEXPR_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
                       '-e','TORCH_HOME=/workspace/task_restore_v1/cache/torch','-v',f'{REPO}:/workspace','-w','/workspace',IMAGE,
                       '/opt/taskrestore/bin/python','-u','task_restore_v1/evaluate_parallel.py',
                       '--task',task,'--condition',condition,'--episodes',str(args.episodes),'--start-seed',str(args.start_seed),
                       '--episode-offset',str(offset),'--episode-count',str(count),'--max-steps',str(args.max_steps),
                       '--video-episodes',str(args.video_episodes),'--state-bank',str(Path('/workspace')/(state_directory/f'{task}.json').relative_to(REPO)),
                       '--output',str(Path('/workspace')/target.relative_to(REPO))]
                with active_lock:
                    active.add(name)
                print(f'SHARD_START gpu={gpu} slot={slot} task={task} condition={condition} offset={offset} count={count} log={target/"console.log"}',flush=True)
                try:
                    with (target/'console.log').open('a') as log:
                        process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            if line.startswith(('STEP_PROGRESS','EPISODE_DONE','TASK_RESTORE_EVAL_FINISHED')):
                                print(f'[g{gpu}/w{slot}] {line.strip()}',flush=True)
                        status = process.wait()
                    if status:
                        raise RuntimeError(f'Container exit {status}; inspect {target/"console.log"}')
                    verify_result(target/'result.json',job,suite)
                except Exception as error:
                    errors.append(dict(job=job,error=str(error)))
                    stop.set()
                    return
                finally:
                    with active_lock:
                        active.discard(name)
                    jobs.task_done()
        start = time.perf_counter()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=total_workers)
        futures = [pool.submit(worker,gpu,slot) for slot in range(args.workers_per_gpu) for gpu in gpus]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
                if errors:
                    stop.set()
                    stop_owned()
                    break
        except KeyboardInterrupt:
            stop.set()
            print('INTERRUPTED: stopping only this launcher containers; episode records retained.',flush=True)
            stop_owned()
            raise
        finally:
            pool.shutdown(wait=True)
        if errors:
            print(json.dumps(errors,indent=2),flush=True)
            raise SystemExit(1)
        summary = merge_results(output,suite)
        elapsed = time.perf_counter()-start
        write_json(output/'throughput.json',dict(worker_pool_seconds=elapsed,total_workers=total_workers,
                                                completed_episodes=sum(r['episodes'] for r in summary),resumed_shards=resumed))
        print(f'TASK_RESTORE_PARALLEL_FINISHED outputs={output} worker_pool_seconds={elapsed:.1f}',flush=True)
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
