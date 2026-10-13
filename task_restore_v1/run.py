"""Server launcher; one container per GPU, resumable episode records, no tmux commands."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
IMAGE = 'noise2dynamics:taskrestore-v1'
ALL_CONDITIONS = ['clean', 'shot_read_mild', 'shot_read_medium', 'shot_read_strong', 'lowlight', 'isp_local']

def memory_used(gpu):
    result = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    return int(result.strip())

def assert_gpu_free(gpu):
    used = memory_used(gpu)
    processes = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    utilization = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
    if used > 2048 or processes or utilization > 15:
        raise RuntimeError(f'GPU {gpu} is busy ({used} MiB, utilization {utilization}%, compute PIDs {processes or "none"}). Choose free GPU IDs.')

def summarize(output):
    found = {}
    for path in output.glob('*/*/result.json'):
        record = json.loads(path.read_text())
        found[(record['protocol']['task'], record['protocol']['condition'])] = record
    rows = []
    for (task, condition), record in sorted(found.items()):
        clean = found.get((task, 'clean'))
        paired_drop = None
        if clean is not None:
            clean_seeds = {r['seed']: r for r in clean['rows']}
            matched = []
            for row in record['rows']:
                if row['seed'] not in clean_seeds:
                    continue
                baseline = clean_seeds[row['seed']]
                for key in ['initial_state_sha256', 'initial_model_sha256', 'state_bank_entry_sha256']:
                    if key not in baseline or key not in row or baseline[key] != row[key]:
                        raise RuntimeError(f'Initial states are not paired: {task} seed={row["seed"]} field={key}')
                matched.append(int(baseline['success_ever']) - int(row['success_ever']))
            paired_drop = sum(matched)/len(matched) if matched else None
        n, p = record['episodes'], record['success_rate']
        z = 1.96
        center = (p + z*z/(2*n))/(1+z*z/n)
        half = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
        rows.append(dict(task=task, condition=condition, successes=record['successes'], episodes=n, success_rate=p, final_success_rate=record['final_success_rate'], wilson95=[center-half, center+half], paired_drop_from_clean=paired_drop, mean_episode_seconds=record['mean_episode_seconds']))
    (output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    report = ['# Published policy sensor-noise baseline', '', 'This measures fixed policy robustness, not a proposed method improvement. Single underlying policy training run per task; episode intervals do not imply training-seed replication.', '', '|Task|Condition|Successes|Ever SR|Final SR|95% Wilson interval|Drop from clean|', '|---|---|---|---|---|---|---|']
    for r in rows:
        lo, hi = r['wilson95']
        drop = '—' if r['paired_drop_from_clean'] is None else f'{100*r["paired_drop_from_clean"]:.1f} pp'
        report.append(f'|{r["task"]}|{r["condition"]}|{r["successes"]}/{r["episodes"]}|{100*r["success_rate"]:.1f}%|{100*r["final_success_rate"]:.1f}%|[{100*lo:.1f}, {100*hi:.1f}]%|{drop}|')
    (output/'RESULTS.md').write_text('\n'.join(report)+'\n')
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'smoke', 'baseline', 'summary'], default='preflight')
    parser.add_argument('--gpus', default='2,3,6,7')
    parser.add_argument('--tasks', default='lift,can,square')
    parser.add_argument('--conditions', default=','.join(ALL_CONDITIONS))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--smoke-condition', choices=ALL_CONDITIONS, default='clean')
    parser.add_argument('--tag', default='sensor_baseline_v1_20261003')
    parser.add_argument('--start-seed', type=int, default=10000)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.tag) or args.tag in {'.', '..'}:
        raise ValueError('Invalid output tag')
    tasks = args.tasks.split(',')
    conditions = args.conditions.split(',')
    gpus = args.gpus.split(',')
    if not set(tasks) <= {'lift', 'can', 'square'} or not set(conditions) <= set(ALL_CONDITIONS):
        raise ValueError('Task or condition outside protocol')
    if len(set(gpus)) != len(gpus) or not all(re.fullmatch('[0-7]', g) for g in gpus):
        raise ValueError('GPU IDs must be distinct integers in 0..7')
    output = REPO/'outputs'/'task_restore_v1'/args.tag
    if args.stage == 'summary':
        summarize(output)
        return
    manifest_file = ROOT/'assets'/'manifest.json'
    if not manifest_file.exists():
        raise RuntimeError('Missing assets. Run task_restore_v1/prepare.py first.')
    manifest = json.loads(manifest_file.read_text())
    records = {r['task']: r for r in manifest['tasks']}
    for task in tasks:
        if task not in records or not all((ROOT/records[task][key]).exists() for key in ['checkpoint', 'metadata_hdf5']):
            raise RuntimeError(f'Incomplete assets for {task}')
    subprocess.run(['docker', 'image', 'inspect', IMAGE], stdout=subprocess.DEVNULL, check=True)
    for gpu in gpus:
        assert_gpu_free(gpu)
    print(f'TASK_RESTORE_PREFLIGHT_OK tasks={tasks} gpus={gpus} image={IMAGE}', flush=True)
    if args.stage == 'preflight':
        return
    episodes = 1 if args.stage == 'smoke' else args.episodes
    if episodes < 1:
        raise ValueError('episodes must be positive')
    if args.stage == 'smoke':
        conditions = [args.smoke_condition]
    output.mkdir(parents=True, exist_ok=True)
    state_directory = ROOT/'assets'/'initial_states_v1'/f'paired_xml_seed{args.start_seed}_n{episodes}'
    subprocess.run(['docker', 'run', '--rm', '-e', 'PYTHONPATH=', '-e', 'OMP_NUM_THREADS=4', '-v', f'{REPO}:/workspace', '-w', '/workspace', IMAGE, '/opt/taskrestore/bin/python', '-u', 'task_restore_v1/states.py', '--tasks', args.tasks, '--episodes', str(episodes), '--start-seed', str(args.start_seed), '--directory', str(Path('/workspace')/state_directory.relative_to(REPO))], check=True)
    protocol = dict(stage=args.stage, tasks=tasks, conditions=conditions, episodes=episodes, start_seed=args.start_seed, source_archive_sha256=manifest['source_archive_sha256'], source_checkpoints={t:records[t]['checkpoint_sha256'] for t in tasks}, evaluator_sha256=hashlib.sha256((ROOT/'evaluate.py').read_bytes()).hexdigest(), launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), state_banks={t:hashlib.sha256((state_directory/f'{t}.json').read_bytes()).hexdigest() for t in tasks})
    saved = output/'suite_protocol.json'
    if saved.exists() and json.loads(saved.read_text()) != protocol:
        raise RuntimeError('Output tag already belongs to a different suite')
    saved.write_text(json.dumps(protocol, indent=2)+'\n')
    locks = []
    for gpu in gpus:
        handle = (ROOT/f'.gpu_{gpu}.lock').open('w')
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.append(handle)
    stop = threading.Event()
    active = set()
    active_lock = threading.Lock()
    jobs = queue.Queue()
    for condition in conditions:
        for task in tasks:
            jobs.put((task, condition))
    errors = []
    def worker(gpu):
        while not stop.is_set():
            try:
                task, condition = jobs.get_nowait()
            except queue.Empty:
                return
            target = output/task/condition
            target.mkdir(parents=True, exist_ok=True)
            name = f'n2d-taskrestore-{args.tag}-{task}-{condition}-g{gpu}'
            cmd = ['docker', 'run', '--rm', '--name', name, '--gpus', f'device={gpu}', '--shm-size', '2g', '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'OMP_NUM_THREADS=4', '-e', 'PYTHONPATH=', '-e', 'TORCH_HOME=/workspace/task_restore_v1/cache/torch', '-v', f'{REPO}:/workspace', '-w', '/workspace', IMAGE, '/opt/taskrestore/bin/python', '-u', 'task_restore_v1/evaluate.py', '--task', task, '--condition', condition, '--episodes', str(episodes), '--start-seed', str(args.start_seed), '--state-bank', str(Path('/workspace')/(state_directory/f'{task}.json').relative_to(REPO)), '--output', str(Path('/workspace')/target.relative_to(REPO))]
            with active_lock:
                active.add(name)
            print(f'START gpu={gpu} task={task} condition={condition} episodes={episodes} log={target/"console.log"}', flush=True)
            try:
                with (target/'console.log').open('a') as log:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith(('STEP_PROGRESS', 'EPISODE_DONE', 'TASK_RESTORE_EVAL_FINISHED')):
                            print(line.strip(), flush=True)
                    status = process.wait()
                if status:
                    errors.append(dict(task=task, condition=condition, exit_code=status, log=str(target/'console.log')))
                    stop.set()
                    return
            except Exception as error:
                errors.append(dict(task=task, condition=condition, error=str(error)))
                stop.set()
                return
            finally:
                with active_lock:
                    active.discard(name)
                jobs.task_done()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus))
    futures = [pool.submit(worker, g) for g in gpus]
    try:
        for f in futures:
            f.result()
    except KeyboardInterrupt:
        stop.set()
        print('INTERRUPTED: stopping only containers created by this launcher; completed episode records remain.', flush=True)
        with active_lock:
            names = list(active)
        for name in names:
            subprocess.run(['docker', 'stop', '--time', '5', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    finally:
        pool.shutdown(wait=True)
        for handle in locks:
            handle.close()
    if errors:
        print(json.dumps(errors, indent=2), flush=True)
        raise SystemExit(1)
    summarize(output)
    marker = 'TASK_RESTORE_SMOKE_FINISHED' if args.stage == 'smoke' else 'TASK_RESTORE_BASELINE_FINISHED'
    print(f'{marker} outputs={output}', flush=True)

if __name__ == '__main__':
    main()
