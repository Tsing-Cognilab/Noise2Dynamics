"""Replay whole demonstration splits; cache clean frozen-policy teacher trajectories."""
import argparse
import time
from pathlib import Path
from common import *
import h5py
from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task',choices=['lift','can','square'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-demos',type=int,default=24)
    p.add_argument('--valid-demos',type=int,default=4)
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--teacher-batch',type=int,default=32)
    p.add_argument('--limit-samples',type=int,default=0,help='smoke only; never used in formal pilot')
    a = p.parse_args()
    if min(a.train_demos,a.valid_demos,a.stride,a.teacher_batch)<1:
        raise ValueError('Positive data sizes required')
    record = record_for(a.task)
    cfg = OmegaConf.load(CORE/record['config'])
    source = CORE/record['metadata_hdf5']
    spec = dict(task=a.task,train_demos=a.train_demos,valid_demos=a.valid_demos,stride=a.stride,
                limit_samples=a.limit_samples,source_hdf5_sha256=digest(source.read_bytes()),
                teacher_checkpoint_sha256=record['checkpoint_sha256'],method_code=source_hashes(),
                supervision='paired simulator RGB; frozen clean teacher; no evaluation episodes used')
    if (a.output/'ready.json').exists():
        saved = json.loads((a.output/'ready.json').read_text())
        if saved['spec'] != spec:
            raise RuntimeError('Data protocol differs: use a new output tag')
        for split in ['train','valid']:
            if digest((a.output/f'{split}.npz').read_bytes()) != saved['files'][split]:
                raise RuntimeError('Data checksum mismatch')
        print(f'DATA_READY task={a.task} cached=1',flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=True)
    env,meta = create_env(record,cfg)
    arrays = {}
    ids = {}
    start = time.perf_counter()
    try:
        with h5py.File(source) as f:
            order = lambda x:int(x.split('_')[-1])
            for split,number in [('train',a.train_demos),('valid',a.valid_demos)]:
                names = sorted([x.decode() if isinstance(x,bytes) else str(x) for x in f['mask'][split][:]],key=order)[:number]
                if len(names)!=number:
                    raise RuntimeError('Not enough complete demonstrations in official split')
                ids[split] = names
                observations = {k:[] for k in cfg.shape_meta.obs}
                episode_ids,frame_ids = [],[]
                for name in names:
                    d = f['data'][name]
                    states = d['states'][:]
                    env.reset_to({'model':d.attrs['model_file'],'states':states[0]})
                    for index in range(0,len(states),a.stride):
                        pair = [env.reset_to({'states':states[max(0,index-1)]}),env.reset_to({'states':states[index]})]
                        for key,attr in cfg.shape_meta.obs.items():
                            value = np.stack([o[key] for o in pair]).astype(np.float32)
                            if list(value.shape[1:]) != list(attr['shape']) or not np.isfinite(value).all():
                                raise RuntimeError('Bad replay observation')
                            if key in RGB:
                                value = np.rint(value*255).clip(0,255).astype(np.uint8)
                            observations[key].append(value)
                        episode_ids.append(name)
                        frame_ids.append(index)
                        if a.limit_samples and len(episode_ids)>=a.limit_samples:
                            break
                    print(f'DATA_REPLAY task={a.task} split={split} demo={name} samples={len(episode_ids)} seconds={time.perf_counter()-start:.1f}',flush=True)
                    if a.limit_samples and len(episode_ids)>=a.limit_samples:
                        break
                arrays[split] = {k:np.stack(v) for k,v in observations.items()}
                arrays[split].update(episode_id=np.asarray(episode_ids),frame_id=np.asarray(frame_ids))
    finally:
        env.env.close()
    if set(ids['train']) & set(ids['valid']):
        raise RuntimeError('Whole-demo split leakage')
    torch.set_num_threads(1)
    policy,_ = load_policy(record,'cuda:0')
    if not policy.obs_as_cond or policy.n_obs_steps!=2:
        raise RuntimeError('Unsupported policy for decision matching')
    for split,data in arrays.items():
        predictions,features = [],[]
        for offset in range(0,len(data['episode_id']),a.teacher_batch):
            obs = {k:torch.from_numpy(v[offset:offset+a.teacher_batch]).to('cuda:0').float()/(255 if k in RGB else 1)
                   for k,v in data.items() if k in cfg.shape_meta.obs}
            # Fixed dataset-teacher randomness independent of adapter training/evaluation RNG.
            torch.manual_seed(70000000+offset+(0 if split=='train' else 100000))
            with torch.inference_mode():
                features.append(condition_features(policy,obs).cpu().numpy())
                predictions.append(policy.predict_action(obs)['action_pred'].cpu().numpy())
            print(f'DATA_TEACHER task={a.task} split={split} samples={min(offset+a.teacher_batch,len(data["episode_id"]))}/{len(data["episode_id"])}',flush=True)
        data['teacher_action'] = np.concatenate(predictions)
        data['teacher_features'] = np.concatenate(features)
        temporary = a.output/f'{split}.tmp.npz'
        np.savez(temporary,**data)
        temporary.replace(a.output/f'{split}.npz')
    ready = dict(spec=spec,split_demonstrations=ids,samples={s:len(d['episode_id']) for s,d in arrays.items()},
                 files={s:digest((a.output/f'{s}.npz').read_bytes()) for s in arrays},env_metadata=meta,seconds=time.perf_counter()-start)
    save_json(a.output/'ready.json',ready)
    print(f'METHOD_DATA_FINISHED task={a.task} seconds={ready["seconds"]:.1f}',flush=True)


if __name__ == '__main__':
    main()
