"""Dependency-free immutable confirmation protocol and result validators."""
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TASKS = ('can', 'square')
SEEDS = (3072, 3073, 3074)
ARMS = ('image', 'feature', 'decision')
START, BANK_N, HORIZON, SHARD, UPDATES = 30000, 100, 400, 5, 2000
COUNTS = {'clean': 50, 'lowlight': 100}
CALIBRATION = dict(seed=880031, batches=4, batch_size=16, decision_weight=0.1,
                   rule='median initial-gradient ratio to 0.1 * decision loss; train split only')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sources():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}


def tag(value):
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value) or value in ('.', '..'):
        raise ValueError('Invalid output tag')
    return value


def groups():
    controls = [(None, t, a, c) for t in TASKS for a, c in
                [('original', 'clean'), ('original', 'lowlight'), ('exposure_oracle', 'lowlight')]]
    learned = [(s, t, a, c) for s in SEEDS for t in TASKS for a in ARMS for c in COUNTS]
    return controls + learned


def group_path(root, group):
    seed, task, arm, condition = group
    return Path(root) / ('shared' if seed is None else f'seed{seed}') / 'eval' / task / f'{arm}_{condition}'


def jobs():
    return [(g, offset, min(SHARD, COUNTS[g[3]] - offset))
            for offset in range(0, BANK_N, SHARD) for g in groups() if offset < COUNTS[g[3]]]


def paired(a, b, field):
    if not a or len(a) != len(b) or [r['seed'] for r in a] != [r['seed'] for r in b]:
        raise RuntimeError('Unpaired comparison')
    gains = sum(y[field] and not x[field] for x, y in zip(a, b))
    losses = sum(x[field] and not y[field] for x, y in zip(a, b))
    n = gains + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n) if n else 1.
    return dict(gains=gains, losses=losses, delta_pp=100 * (gains - losses) / len(a), exact_two_sided_p=p)


def verify_shard(target, job, locked):
    group, offset, count = job
    seed, task, arm, condition = group
    record = read(Path(target) / 'result.json')
    expected = dict(task=task, arm=arm, condition=condition, training_seed=seed,
                    start_seed=START, bank_episodes=BANK_N, episodes=COUNTS[condition],
                    offset=offset, count=count, max_steps=HORIZON,
                    state_bank_sha256=locked['banks'][task], policy_sha256=locked['policies'][task],
                    sources=locked['base_sources'], confirm_sources=locked['confirm_sources'],
                    adapter_sha256=None if seed is None else locked['adapters'][f'{seed}/{task}/{arm}']['sha256'])
    if any(record['protocol'].get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'Shard protocol mismatch: {target}')
    rows = record['rows']
    if [r['seed'] for r in rows] != list(range(START + offset, START + offset + count)):
        raise RuntimeError('Missing, duplicate, or unexpected states')
    if rows != [json.loads(line) for line in (Path(target) / 'episodes.jsonl').read_text().splitlines()]:
        raise RuntimeError('Result JSON and episode JSONL disagree')
    for row in rows:
        if (row['task'], row['arm'], row['condition'], row['training_seed']) != (task, arm, condition, seed):
            raise RuntimeError('Wrong evaluation group')
        if row['steps'] != HORIZON or not all(isinstance(row[k], bool) for k in ('success_ever', 'success_final')):
            raise RuntimeError('Formal evaluation requires full 400 steps and Boolean success')
        if row['success_final'] and not row['success_ever']:
            raise RuntimeError('Final success cannot occur without ever success')
        first = row['first_success_step']
        if (first is not None) != row['success_ever'] or (first is not None and not 0 <= first <= HORIZON):
            raise RuntimeError('Invalid first-success step')
        for key in ('initial_state_sha256', 'initial_model_sha256', 'state_bank_entry_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', row[key]):
                raise RuntimeError('Invalid simulator fingerprint')
        if not math.isfinite(row['wall_seconds']) or row['wall_seconds'] <= 0:
            raise RuntimeError('Invalid timing')
    if record['episodes'] != count or record['successes'] != sum(r['success_ever'] for r in rows) or record['final_successes'] != sum(r['success_final'] for r in rows):
        raise RuntimeError('Success count mismatch')
    return rows
