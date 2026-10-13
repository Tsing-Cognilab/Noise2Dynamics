"""Shared, immutable simulator XML + state banks for paired evaluations."""
import argparse
import fcntl
import hashlib
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_SEEDS = {'lift': 30000000, 'can': 30000001, 'square': 30000002}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def state_digest(state):
    return digest(np.asarray(state, dtype=np.float64).tobytes())


def entry_digest(entry):
    return digest(json.dumps(entry, sort_keys=True, separators=(',', ':')).encode())


def load_bank(path, task, start_seed, episodes):
    bank = json.loads(Path(path).read_text())
    if (bank['version'], bank['task'], bank['start_seed'], bank['episodes']) != (1, task, start_seed, episodes):
        raise RuntimeError('Initial-state bank does not match the evaluation protocol')
    if [e['seed'] for e in bank['entries']] != list(range(start_seed, start_seed + episodes)):
        raise RuntimeError('Initial-state bank has missing or duplicate episode IDs')
    for entry in bank['entries']:
        if state_digest(entry['states']) != entry['state_sha256'] or digest(entry['model'].encode()) != entry['model_sha256']:
            raise RuntimeError('Initial-state bank has a corrupt entry')
    return bank


def restore_initial(env, entry):
    """Restore full physics; allow only measured float64 forward normalization.

    MuJoCo forward can alternate the lowest quaternion bits. Bank-to-restored
    state is checked at eight float64 eps; BETWEEN CONDITIONS actual hashes
    remain strictly bitwise paired, without rounding.
    """
    raw = env.reset_to({'model': entry['model'], 'states': np.asarray(entry['states'], dtype=np.float64)})
    actual = env.get_state()
    expected = np.asarray(entry['states'], dtype=np.float64)
    restored = np.asarray(actual['states'], dtype=np.float64)
    if expected.shape != restored.shape or not np.isfinite(restored).all():
        raise RuntimeError('Explicit simulator-state restoration has an invalid shape or value')
    error = np.abs(restored - expected)
    bound = 8*np.finfo(np.float64).eps*np.maximum(1., np.abs(expected))
    if np.any(error > bound):
        raise RuntimeError(f'Explicit simulator-state restoration exceeds float64 precision: {error.max()}')
    return raw, dict(initial_state_sha256=state_digest(restored), initial_model_sha256=digest(actual['model'].encode()), state_bank_entry_sha256=entry_digest(entry), initial_state_bank_max_abs_error=float(error.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='lift,can,square')
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--start-seed', type=int, default=10000)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError('episodes must be positive')
    from omegaconf import OmegaConf
    from evaluate import create_env
    records = {r['task']: r for r in json.loads((ROOT/'assets'/'manifest.json').read_text())['tasks']}
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for task in args.tasks.split(','):
            record = records[task]
            spec = dict(version=1, task=task, start_seed=args.start_seed, episodes=args.episodes, model_seed=MODEL_SEEDS[task], metadata_sha256=digest((ROOT/record['metadata_hdf5']).read_bytes()), config_sha256=digest((ROOT/record['config']).read_bytes()), generation_script_sha256=digest(Path(__file__).read_bytes()), environment_adapter_sha256=digest((ROOT/'evaluate.py').read_bytes()))
            path = args.directory/f'{task}.json'
            if path.exists():
                bank = load_bank(path, task, args.start_seed, args.episodes)
                if {k: bank[k] for k in spec} != spec:
                    raise RuntimeError('Existing state bank uses different environment code or assets; do not overwrite it')
                print(f'STATE_BANK_READY task={task} episodes={args.episodes} sha256={digest(path.read_bytes())}', flush=True)
                continue
            # Geometry is fixed once per task, independently of episode/condition.
            random.seed(MODEL_SEEDS[task])
            np.random.seed(MODEL_SEEDS[task])
            cfg = OmegaConf.load(ROOT/record['config'])
            env, meta = create_env(record, cfg)
            entries = []
            try:
                for seed in range(args.start_seed, args.start_seed + args.episodes):
                    random.seed(seed)
                    np.random.seed(seed)
                    env.reset()
                    state = env.get_state()
                    entry = dict(seed=seed, model=state['model'], states=state['states'].tolist(), state_sha256=state_digest(state['states']), model_sha256=digest(state['model'].encode()))
                    entries.append(entry)
            finally:
                env.env.close()
            bank = dict(**spec, env_metadata=meta, geometry_scope='one fixed canonical geometry per task; independent robot/object reset states', entries=entries)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(bank, indent=2)+'\n')
            temporary.replace(path)
            print(f'STATE_BANK_READY task={task} episodes={args.episodes} sha256={digest(path.read_bytes())}', flush=True)
    print('TASK_RESTORE_STATE_BANK_FINISHED', flush=True)


if __name__ == '__main__':
    main()
