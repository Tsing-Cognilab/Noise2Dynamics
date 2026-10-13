"""Matched image-only vs frozen diffusion-decision consistency training."""
import argparse
import random
import time
from pathlib import Path
from common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task',choices=['lift','can','square'],required=True)
    p.add_argument('--arm',choices=['image','decision'],required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=3072)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--decision-weight',type=float,default=.1)
    a = p.parse_args()
    if min(a.updates,a.batch_size)<1 or a.decision_weight<=0:
        raise ValueError('Invalid fixed training budget')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    model = Restorer().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    policy,cfg = load_policy(record_for(a.task),'cuda:0')
    if not policy.obs_as_cond:
        raise RuntimeError('Only observation-conditioned policy supported')
    data = {}
    ready = json.loads((a.data/'ready.json').read_text())
    if ready['spec']['task'] != a.task or ready['spec']['method_code'] != source_hashes():
        raise RuntimeError('Stale/cross-task data, prepare this revision first')
    for split in ['train','valid']:
        path = a.data/f'{split}.npz'
        if digest(path.read_bytes()) != ready['files'][split]:
            raise RuntimeError('Dataset checksum mismatch')
        with np.load(path) as z:
            data[split] = {k:torch.from_numpy(z[k].copy()).cuda() for k in z.files if k != 'episode_id'}
    spec = dict(task=a.task,arm=a.arm,seed=a.seed,updates=a.updates,batch_size=a.batch_size,
                decision_weight=a.decision_weight,learning_rate=3e-4,width=24,
                data_sha256=digest((a.data/'ready.json').read_bytes()),sources=source_hashes(),
                parameters=sum(p.numel() for p in model.parameters()),noise=dict(CONDITIONS['lowlight']),
                identity_probability=.2,teacher_frozen=True,selection='last fixed-budget checkpoint; no closed-loop selection')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'protocol.json').exists() and json.loads((a.output/'protocol.json').read_text())!=spec:
        raise RuntimeError('Training output uses another protocol')
    save_json(a.output/'protocol.json',spec)
    if (a.output/'finished.json').exists():
        saved = json.loads((a.output/'finished.json').read_text())
        if saved['spec']!=spec or digest((a.output/'model.pt').read_bytes())!=saved['checkpoint_sha256']:
            raise RuntimeError('Training completion mismatch')
        print(f'TRAIN_READY task={a.task} arm={a.arm}',flush=True)
        return
    sample_rng = np.random.default_rng(a.seed+1000000)
    sensor_rng = torch.Generator(device='cuda').manual_seed(a.seed+2000000)
    diffusion_rng = torch.Generator(device='cuda').manual_seed(a.seed+3000000)
    first = 0
    checkpoint = a.output/'resume.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint,weights_only=False)
        if saved['spec']!=spec:
            raise RuntimeError('Resume protocol mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        sample_rng.bit_generator.state = saved['sample_rng']
        sensor_rng.set_state(saved['sensor_rng'])
        diffusion_rng.set_state(saved['diffusion_rng'])
        first = saved['update']
    def batch(split,index):
        d = data[split]
        obs = {k:d[k][index].float()/(255 if k in RGB else 1) for k in cfg.shape_meta.obs}
        images = torch.stack([obs[k] for k in RGB],dim=2)
        # B,T,V,C,H,W; noise drawn once per observed physical frame.
        noisy = noise_tensor(images,sensor_rng)
        duplicate = d['frame_id'][index] == 0
        noisy[duplicate,0] = noisy[duplicate,1]
        corrupted = dict(obs)
        for i,k in enumerate(RGB):
            corrupted[k] = noisy[:,:,i]
        return d,obs,corrupted
    def objective(split,index):
        d,clean,noisy = batch(split,index)
        restored = restore_observations(model,noisy)
        image_loss = sum(F.l1_loss(restored[k],clean[k]) for k in RGB)/len(RGB)
        teacher_actions = policy.normalizer['action'].normalize(d['teacher_action'][index])
        if policy.pred_action_steps_only:
            begin = policy.n_obs_steps-1
            teacher_actions = teacher_actions[:,begin:begin+policy.n_action_steps]
        eps = torch.randn(teacher_actions.shape,device='cuda',generator=diffusion_rng)
        times = torch.randint(0,policy.noise_scheduler.config.num_train_timesteps,(len(index),),device='cuda',generator=diffusion_rng)
        xt = policy.noise_scheduler.add_noise(teacher_actions,eps,times)
        # Same noisy action trajectory and diffusion time for clean teacher and restored student.
        with torch.no_grad():
            target = policy.model(xt,times,d['teacher_features'][index])
        student = policy.model(xt,times,condition_features(policy,restored))
        begin = 0 if policy.pred_action_steps_only else policy.n_obs_steps-1
        decision_loss = F.mse_loss(student[:,begin:begin+policy.n_action_steps],target[:,begin:begin+policy.n_action_steps])
        weight = a.decision_weight if a.arm=='decision' else 0.
        return image_loss+weight*decision_loss,image_loss,decision_loss
    start = time.perf_counter()
    max_grad = 0.
    for update in range(first+1,a.updates+1):
        index = sample_rng.integers(0,len(data['train']['frame_id']),a.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss,image_loss,decision_loss = objective('train',index)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training objective')
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not np.isfinite(grad) or (update==1 and grad<=0):
            raise RuntimeError('Restoration gradient is invalid')
        max_grad = max(max_grad,grad)
        if update==1 and a.arm=='decision':
            # Dedicated input-gradient check ensures frozen policy still supervises the adapter.
            saved_sensor_rng = sensor_rng.get_state()
            _,clean,noisy = batch('valid',np.arange(min(2,len(data['valid']['frame_id']))))
            sensor_rng.set_state(saved_sensor_rng)
            restored = restore_observations(model,noisy)
            student_feat = condition_features(policy,restored)
            xt = torch.zeros((len(student_feat),policy.horizon,policy.action_dim),device='cuda')
            output = policy.model(xt,torch.zeros(len(xt),dtype=torch.long,device='cuda'),student_feat)
            gradient = torch.autograd.grad(output.square().mean(),model.out.weight)[0]
            if not torch.isfinite(gradient).all() or gradient.abs().sum()<=0:
                raise RuntimeError('Decision supervision has no differentiable path')
            if any(p.grad is not None for p in policy.parameters()):
                raise RuntimeError('Frozen policy received gradients')
            print(f'DECISION_GRADIENT_OK gradient_sum={float(gradient.abs().sum()):.6g}',flush=True)
        optimizer.step()
        if update==1 or update%100==0 or update==a.updates:
            row = dict(update=update,loss=float(loss),image_l1=float(image_loss),decision_mse=float(decision_loss),
                       gradient=grad,seconds=time.perf_counter()-start)
            with (a.output/'train.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(f'TRAIN_PROGRESS task={a.task} arm={a.arm} update={update}/{a.updates} image={row["image_l1"]:.5f} decision={row["decision_mse"]:.5f} seconds={row["seconds"]:.1f}',flush=True)
        if update%500==0 or update==a.updates:
            saved = dict(spec=spec,update=update,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         sample_rng=sample_rng.bit_generator.state,sensor_rng=sensor_rng.get_state(),diffusion_rng=diffusion_rng.get_state())
            torch.save(saved,a.output/'resume.tmp.pt')
            (a.output/'resume.tmp.pt').replace(checkpoint)
    model.eval()
    with torch.no_grad():
        index = np.arange(min(64,len(data['valid']['frame_id'])))
        _,image_loss,decision_loss = objective('valid',index)
    torch.save(dict(spec=spec,model=model.state_dict()),a.output/'model.tmp.pt')
    (a.output/'model.tmp.pt').replace(a.output/'model.pt')
    result = dict(spec=spec,checkpoint_sha256=digest((a.output/'model.pt').read_bytes()),validation_image_l1=float(image_loss),
                  validation_decision_mse=float(decision_loss),max_gradient=max_grad,seconds=time.perf_counter()-start,
                  interpretation='offline validation only; method utility determined by paired closed-loop success')
    save_json(a.output/'finished.json',result)
    print(f'METHOD_TRAIN_FINISHED task={a.task} arm={a.arm} seconds={result["seconds"]:.1f}',flush=True)


if __name__=='__main__':
    main()
