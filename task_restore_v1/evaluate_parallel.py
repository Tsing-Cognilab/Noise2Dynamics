"""Parallel shard evaluator; physics, sensing and policy use unchanged evaluate.py."""
import argparse
import collections
import hashlib
import json
import os
import random
import time
from pathlib import Path

# Importing the verified core sets the legacy environment before other libraries.
from evaluate import ROOT, CONDITIONS, load_policy, create_env, absolute_action, corrupt
from states import MODEL_SEEDS, digest, load_bank, restore_initial
import imageio.v2 as imageio
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['lift', 'can', 'square'], required=True)
    parser.add_argument('--condition', choices=list(CONDITIONS), required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--start-seed', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--video-episodes', type=int, default=2)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--state-bank', type=Path, required=True)
    parser.add_argument('--episode-offset', type=int, default=0)
    parser.add_argument('--episode-count', type=int)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError('episodes must be positive')
    count = args.episodes if args.episode_count is None else args.episode_count
    if args.episode_offset < 0 or count < 1 or args.episode_offset + count > args.episodes:
        raise ValueError('Shard outside the full episode protocol')
    indices = range(args.episode_offset, args.episode_offset + count)
    torch.set_num_threads(1)
    record = next(x for x in json.loads((ROOT / 'assets' / 'manifest.json').read_text())['tasks'] if x['task'] == args.task)
    bank = load_bank(args.state_bank, args.task, args.start_seed, args.episodes)
    policy, cfg = load_policy(record, args.device)
    random.seed(MODEL_SEEDS[args.task])
    np.random.seed(MODEL_SEEDS[args.task])
    env, meta = create_env(record, cfg)
    if bank['env_metadata'] != meta:
        raise RuntimeError('State-bank environment metadata does not match the actor environment')
    horizon = int(cfg.task.env_runner.max_steps)
    if args.max_steps is not None:
        horizon = args.max_steps
    args.output.mkdir(parents=True, exist_ok=True)
    contract = dict(task=args.task, condition=args.condition, noise=CONDITIONS[args.condition], episodes=args.episodes, start_seed=args.start_seed, max_steps=horizon, checkpoint_sha256=record['checkpoint_sha256'], n_obs_steps=int(policy.n_obs_steps), n_action_steps=int(policy.n_action_steps), inference_steps=int(policy.num_inference_steps), env_metadata=meta, source=f'published_{record["method"]}_train_0', sensor_claim='synthetic sensor-inspired, not calibrated RAW', evaluator_sha256=hashlib.sha256((ROOT/'evaluate.py').read_bytes()).hexdigest())
    contract.update(initialization='explicit shared XML + float64 simulator state', state_bank_sha256=digest(args.state_bank.read_bytes()), state_bank_generator_sha256=bank['generation_script_sha256'], geometry_scope=bank['geometry_scope'])
    contract.update(episode_offset=args.episode_offset, episode_count=count, video_episodes=args.video_episodes, parallel_wrapper_sha256=digest(Path(__file__).read_bytes()))
    config_file = args.output / 'protocol.json'
    if config_file.exists() and json.loads(config_file.read_text()) != contract:
        raise RuntimeError('Existing output uses a different protocol; choose a new tag')
    config_file.write_text(json.dumps(contract, indent=2) + '\n')
    done = {}
    episode_path = args.output / 'episodes.jsonl'
    if episode_path.exists():
        for line in episode_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row['seed'] in done or row['seed'] not in {args.start_seed+i for i in indices}:
                    raise RuntimeError('Duplicate or out-of-shard episode in existing log')
                done[row['seed']] = row
            except json.JSONDecodeError:
                raise RuntimeError('Incomplete episode log: do not silently skip or overwrite')
    wall_start = time.perf_counter()
    shape = cfg.shape_meta.obs
    for index in indices:
        seed = args.start_seed + index
        if seed in done:
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # A separate generator keeps noise draws independent of robot reset RNG.
        rng = np.random.default_rng(seed + 20_000_000)
        policy.reset()
        raw, fingerprints = restore_initial(env, bank['entries'][index])
        # reset_to may consume reset RNG; reseed policy/environment only after restoration.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        history = {k: collections.deque(maxlen=policy.n_obs_steps) for k in shape}
        def ingest(obs):
            for key, attr in shape.items():
                value = np.asarray(obs[key], dtype=np.float32)
                if tuple(value.shape) != tuple(attr['shape']):
                    raise RuntimeError(f'Observation shape mismatch {key}: {value.shape}')
                if attr.get('type') == 'rgb':
                    if value.min() < 0 or value.max() > 1:
                        raise RuntimeError('Image must be normalized to [0,1]')
                    value = corrupt(value, args.condition, rng)
                history[key].append(value)
        ingest(raw)
        for values in history.values():
            while len(values) < policy.n_obs_steps:
                values.appendleft(values[0].copy())
        ever = bool(env.is_success()['task'])
        first_success = 0 if ever else None
        final = ever
        steps = 0
        inference_seconds = 0.
        episode_start = time.perf_counter()
        video = None
        if index < args.video_episodes:
            video = imageio.get_writer(args.output / f'episode_{seed}.mp4', fps=10)
        try:
            while steps < horizon:
                inputs = {k: torch.from_numpy(np.stack(v)[None]).to(args.device) for k, v in history.items()}
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    predicted = policy.predict_action(inputs)['action'][0].cpu().numpy()
                inference_seconds += time.perf_counter() - t
                if not np.isfinite(predicted).all():
                    raise RuntimeError('Non-finite policy action')
                for action in absolute_action(predicted):
                    raw, reward, terminated, info = env.step(action)
                    steps += 1
                    ingest(raw)
                    final = bool(env.is_success()['task'])
                    if final and first_success is None:
                        first_success = steps
                    ever |= final
                    if video is not None and steps % 2 == 0:
                        clean = (np.moveaxis(raw['agentview_image'], 0, -1) * 255).astype(np.uint8)
                        noisy = (np.moveaxis(history['agentview_image'][-1], 0, -1) * 255).astype(np.uint8)
                        video.append_data(np.concatenate([clean, noisy], axis=1))
                    if steps % 80 == 0:
                        print(f'STEP_PROGRESS task={args.task} condition={args.condition} episode={index+1}/{args.episodes} steps={steps}/{horizon} success_ever={int(ever)} seconds={time.perf_counter()-episode_start:.1f}', flush=True)
                    if steps >= horizon or terminated:
                        break
                if terminated:
                    break
        finally:
            if video is not None:
                video.close()
        row = dict(seed=seed, success_ever=ever, success_final=final, first_success_step=first_success, steps=steps, wall_seconds=time.perf_counter()-episode_start, inference_seconds=inference_seconds, task=args.task, condition=args.condition, **fingerprints)
        with episode_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
            f.flush()
            os.fsync(f.fileno())
        done[seed] = row
        print(f'EPISODE_DONE task={args.task} condition={args.condition} episode={index+1}/{args.episodes} seed={seed} success={int(ever)} seconds={row["wall_seconds"]:.1f}', flush=True)
    rows = [done[args.start_seed+index] for index in indices]
    result = dict(protocol=contract, success_rate=sum(r['success_ever'] for r in rows)/len(rows), final_success_rate=sum(r['success_final'] for r in rows)/len(rows), successes=sum(r['success_ever'] for r in rows), episodes=len(rows), mean_episode_seconds=float(np.mean([r['wall_seconds'] for r in rows])), current_run_seconds=time.perf_counter()-wall_start, rows=rows)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'TASK_RESTORE_EVAL_FINISHED task={args.task} condition={args.condition} successes={result["successes"]}/{len(rows)}', flush=True)
    env.env.close()

if __name__ == '__main__':
    main()
