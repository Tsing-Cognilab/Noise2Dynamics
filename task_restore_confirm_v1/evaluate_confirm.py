"""Actual 400-step closed-loop method evaluation on fresh paired simulator states."""
import argparse
import collections
import os
import random
import time
from pathlib import Path
import sys
import contract as C
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'task_restore_method_v1'))
from common import *
import imageio.v2 as imageio


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task',choices=['lift','can','square'],required=True)
    p.add_argument('--arm',choices=['original','exposure_oracle','image','feature','decision'],required=True)
    p.add_argument('--condition',choices=['clean','lowlight'],required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--training-seed',type=int)
    p.add_argument('--bank-episodes',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state-bank',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--start-seed',type=int,default=20000)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--count',type=int,default=5)
    p.add_argument('--max-steps',type=int,default=400)
    p.add_argument('--video-episodes',type=int,default=1)
    a = p.parse_args()
    if a.max_steps != C.HORIZON or a.start_seed != C.START or a.bank_episodes != C.BANK_N or a.episodes != C.COUNTS[a.condition]:
        raise ValueError('Confirmation budget and states are fixed')
    if (a.arm in C.ARMS) != (a.training_seed in C.SEEDS):
        raise ValueError('Wrong training-seed/group')
    if a.offset<0 or a.count<1 or a.offset+a.count>a.episodes or a.max_steps<1:
        raise ValueError('Invalid shard')
    if a.arm=='exposure_oracle' and a.condition!='lowlight':
        raise ValueError('Exposure diagnostic only for lowlight')
    torch.set_num_threads(1)
    record = record_for(a.task)
    policy,cfg = load_policy(record,'cuda:0')
    bank = load_bank(a.state_bank,a.task,a.start_seed,a.bank_episodes)
    random.seed(MODEL_SEEDS[a.task])
    np.random.seed(MODEL_SEEDS[a.task])
    env,meta = create_env(record,cfg)
    if bank['env_metadata']!=meta:
        raise RuntimeError('State bank metadata mismatch')
    model = None
    adapter_hash = None
    adapter_spec = None
    if a.arm in C.ARMS:
        if a.checkpoint is None:
            raise ValueError('Learned arm needs checkpoint')
        saved = torch.load(a.checkpoint,weights_only=False,map_location='cpu')
        adapter_spec = saved['spec']
        if adapter_spec['task']!=a.task or adapter_spec['arm']!=a.arm or adapter_spec['sources']!=source_hashes():
            raise RuntimeError('Wrong or stale adapter')
        if adapter_spec['seed'] != a.training_seed:
            raise RuntimeError('Wrong training seed')
        if a.arm == 'feature' and adapter_spec['confirm_sources'] != C.sources():
            raise RuntimeError('Stale feature control')
        model = Restorer(adapter_spec['width']).cuda().eval()
        model.load_state_dict(saved['model'],strict=True)
        adapter_hash = digest(a.checkpoint.read_bytes())
    spec = dict(task=a.task,arm=a.arm,condition=a.condition,training_seed=a.training_seed,
                bank_episodes=a.bank_episodes,confirm_sources=C.sources(),noise=CONDITIONS[a.condition],
                episodes=a.episodes,start_seed=a.start_seed,offset=a.offset,count=a.count,max_steps=a.max_steps,
                adapter_sha256=adapter_hash,adapter_spec=adapter_spec,policy_sha256=record['checkpoint_sha256'],
                state_bank_sha256=digest(a.state_bank.read_bytes()),sources=source_hashes(),
                n_obs_steps=policy.n_obs_steps,n_action_steps=policy.n_action_steps,inference_steps=policy.num_inference_steps,
                exposure_control='known synthetic exposure oracle; not blind or proposed method' if a.arm=='exposure_oracle' else None)
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'protocol.json').exists() and json.loads((a.output/'protocol.json').read_text())!=spec:
        raise RuntimeError('Evaluation protocol differs; no overwrite')
    save_json(a.output/'protocol.json',spec)
    path = a.output/'episodes.jsonl'
    done = {}
    wanted = list(range(a.start_seed+a.offset,a.start_seed+a.offset+a.count))
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['seed'] not in wanted or row['seed'] in done:
                raise RuntimeError('Duplicate or invalid resumed episode')
            done[row['seed']] = row
    for index in range(a.offset,a.offset+a.count):
        seed = a.start_seed+index
        if seed in done:
            continue
        raw,fingerprint = restore_initial(env,bank['entries'][index])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        policy.reset()
        rng = np.random.default_rng(seed+20000000)
        history = {k:collections.deque(maxlen=policy.n_obs_steps) for k in cfg.shape_meta.obs}
        def ingest(observation):
            for k in history:
                value = np.asarray(observation[k],dtype=np.float32)
                if k in RGB:
                    value = corrupt(value,a.condition,rng)
                history[k].append(value)
        ingest(raw)
        for values in history.values():
            while len(values)<policy.n_obs_steps:
                values.appendleft(values[0].copy())
        ever = bool(env.is_success()['task'])
        final = ever
        first = 0 if ever else None
        steps = 0
        inference_seconds = 0.
        start = time.perf_counter()
        video = imageio.get_writer(a.output/f'episode_{seed}.mp4',fps=5) if index<a.video_episodes else None
        try:
            while steps<a.max_steps:
                inputs = {k:torch.from_numpy(np.stack(v)[None]).cuda() for k,v in history.items()}
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.inference_mode():
                    if model is not None:
                        inputs = restore_observations(model,inputs)
                    elif a.arm=='exposure_oracle':
                        for k in RGB:
                            inputs[k] = exposure_compensate(inputs[k])
                    predicted = policy.predict_action(inputs)['action'][0].cpu().numpy()
                inference_seconds += time.perf_counter()-tick
                if not np.isfinite(predicted).all():
                    raise RuntimeError('Non-finite action')
                for action in absolute_action(predicted):
                    raw,_,terminated,_ = env.step(action)
                    steps += 1
                    ingest(raw)
                    final = bool(env.is_success()['task'])
                    if final and first is None:
                        first = steps
                    ever |= final
                    if video is not None and steps%4==0:
                        clean = torch.from_numpy(raw[RGB[0]][None,None]).cuda()
                        noisy = torch.from_numpy(np.stack(history[RGB[0]])[None]).cuda()
                        with torch.inference_mode():
                            actual = model(noisy)[:,-1:] if model is not None else exposure_compensate(noisy[:,-1:]) if a.arm=='exposure_oracle' else noisy[:,-1:]
                        images = [clean[0,0],noisy[0,-1],actual[0,0]]
                        video.append_data(np.concatenate([(x.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8) for x in images],axis=1))
                    if steps>=a.max_steps or terminated:
                        break
                if terminated:
                    break
        finally:
            if video is not None:
                video.close()
        row = dict(task=a.task,arm=a.arm,condition=a.condition,training_seed=a.training_seed,seed=seed,steps=steps,success_ever=bool(ever),
                   success_final=bool(final),first_success_step=first,wall_seconds=time.perf_counter()-start,
                   inference_seconds=inference_seconds,**fingerprint)
        with path.open('a') as f:
            f.write(json.dumps(row)+'\n')
            f.flush()
            os.fsync(f.fileno())
        done[seed] = row
        print(f'METHOD_EPISODE_DONE task={a.task} arm={a.arm} condition={a.condition} seed={seed} ever={int(ever)} final={int(final)} seconds={row["wall_seconds"]:.1f}',flush=True)
    rows = [done[s] for s in wanted]
    save_json(a.output/'result.json',dict(protocol=spec,rows=rows,successes=sum(r['success_ever'] for r in rows),
                                          final_successes=sum(r['success_final'] for r in rows),episodes=len(rows)))
    env.env.close()
    print(f'METHOD_EVAL_SHARD_FINISHED task={a.task} arm={a.arm}',flush=True)


if __name__=='__main__':
    main()
