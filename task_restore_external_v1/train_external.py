"""Fixed-budget L1 adaptation of official pretrained RetinexFormer, not LOL retraining."""
import argparse
import random
import time
from frontend import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task',choices=P.TASKS,required=True)
    p.add_argument('--seed',type=int,choices=P.SEEDS,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    model = ExternalRestorer().cuda().train()
    ready = P.read(a.data/'ready.json')
    if ready['spec']['task'] != a.task or ready['spec']['method_code'] != source_hashes():
        raise RuntimeError('Wrong training data revision')
    data = {}
    for split in ('train','valid'):
        path = a.data / (split+'.npz')
        if P.sha(path) != ready['files'][split]:
            raise RuntimeError('Training data checksum mismatch')
        with np.load(path) as z:
            data[split] = {k:torch.from_numpy(z[k].copy()) for k in RGB+['frame_id']}
    spec = dict(task=a.task,arm='retinex_finetune',seed=a.seed,updates=2000,batch_size=16,
                learning_rate=3e-4,weight_decay=1e-4,gradient_clip=1.,identity_probability=.2,
                noise=P.NOISE['lowlight'],data_sha256=P.sha(a.data/'ready.json'),
                sources=source_hashes(),external_sources=P.sources(),assets=P.asset_manifest(),
                parameters=sum(v.numel() for v in model.parameters()),loss='mean per-camera image L1',
                history='independent per-frame network; same frozen policy temporal observations',
                selection='last2000-update checkpoint; no policy loss, no closed-loop/model selection',
                pretraining='official LOL-v1; extra external data/capacity, not matched initialization')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'protocol.json').exists() and P.read(a.output/'protocol.json') != spec:
        raise RuntimeError('Training protocol changed; no overwrite')
    P.write(a.output/'protocol.json',spec)
    if (a.output/'finished.json').exists():
        result = P.read(a.output/'finished.json')
        if result['spec'] != spec or P.sha(a.output/'model.pt') != result['checkpoint_sha256']:
            raise RuntimeError('Finished training mismatch')
        print('EXTERNAL_TRAIN_READY',a.task,a.seed,flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sample_rng = np.random.default_rng(a.seed+1000000)
    sensor_rng = torch.Generator(device='cuda').manual_seed(a.seed+2000000)
    resume = a.output/'resume.pt'
    first,max_grad,previous = 0,0.,0.
    if resume.exists():
        saved = torch.load(resume,map_location='cpu',weights_only=False)
        if saved['spec'] != spec:
            raise RuntimeError('Resume revision changed')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        sample_rng.bit_generator.state = saved['sample_rng']
        sensor_rng.set_state(saved['sensor_rng'])
        first,max_grad,previous = saved['update'],saved['max_gradient'],saved['seconds']
    def objective(split, index, generator):
        d = data[split]
        clean = {k:d[k][index].cuda().float()/255 for k in RGB}
        images = torch.stack([clean[k] for k in RGB],dim=2)
        noisy = noise_tensor(images,generator)
        duplicate = (d['frame_id'][index] == 0).cuda()
        noisy[duplicate,0] = noisy[duplicate,1]
        restored = restore_observations(model,{k:noisy[:,:,i] for i,k in enumerate(RGB)})
        return sum(F.l1_loss(restored[k],clean[k]) for k in RGB)/len(RGB)
    start = time.perf_counter()
    for update in range(first+1,2001):
        index = sample_rng.integers(0,len(data['train']['frame_id']),16)
        optimizer.zero_grad(set_to_none=True)
        loss = objective('train',index,sensor_rng)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite image loss')
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not np.isfinite(grad) or (update==1 and grad<=0):
            raise RuntimeError('Invalid external-model gradient')
        max_grad = max(max_grad,grad)
        optimizer.step()
        seconds = previous+time.perf_counter()-start
        if update==1 or update%100==0:
            row = dict(update=update,image_l1=float(loss),gradient=grad,seconds=seconds)
            with (a.output/'train.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(f'EXTERNAL_TRAIN_PROGRESS task={a.task} seed={a.seed} update={update}/2000 loss={float(loss):.6f} seconds={seconds:.1f}',flush=True)
        if update%100==0:
            torch.save(dict(spec=spec,update=update,model=model.state_dict(),optimizer=optimizer.state_dict(),
                            sample_rng=sample_rng.bit_generator.state,sensor_rng=sensor_rng.get_state(),
                            seconds=seconds,max_gradient=max_grad),a.output/'resume.tmp.pt')
            (a.output/'resume.tmp.pt').replace(resume)
    model.eval()
    validation_rng = torch.Generator(device='cuda').manual_seed(880051)
    with torch.no_grad():
        loss = objective('valid',np.arange(min(64,len(data['valid']['frame_id']))),validation_rng)
    torch.save(dict(spec=spec,model=model.state_dict()),a.output/'model.tmp.pt')
    (a.output/'model.tmp.pt').replace(a.output/'model.pt')
    P.write(a.output/'finished.json',dict(spec=spec,checkpoint_sha256=P.sha(a.output/'model.pt'),
            validation_image_l1=float(loss),max_gradient=max_grad,seconds=previous+time.perf_counter()-start))
    print('EXTERNAL_TRAIN_FINISHED',a.task,a.seed,flush=True)

if __name__ == '__main__':
    main()
