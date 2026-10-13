"""Compatibility tests only; do not select methods using simulator outcomes."""
import copy
import tempfile
import time
from frontend import *
from analysis import holm

def main():
    torch.set_num_threads(1)
    torch.manual_seed(880061)
    assert len(P.groups())==72 and len(P.jobs())==800
    assert sum(j[2] for j in P.jobs())==4000
    assert P.SEEDS==(3072,3073,3074)
    assert holm([.01,.04,.2])==[.03,.08,.2]
    image = np.random.default_rng(880061).random((3,84,84),dtype=np.float32)
    for condition in P.COUNTS:
        a,b = Sensor(condition,30000),Sensor(condition,30000)
        for _ in range(2):
            for camera in RGB:
                x,y = a(image,camera),b(image,camera)
                assert np.array_equal(x,y) and np.isfinite(x).all() and 0<=x.min()<=x.max()<=1
    a,b = Sensor('lowlight',30000),np.random.default_rng(30000+20000000)
    for _ in range(2):
        for camera in RGB:
            assert np.array_equal(a(image,camera),corrupt(image,'lowlight',b))
    fpn = Sensor('sensor_fpn',30000)
    bias = fpn.maps[RGB[0]].copy()
    fpn(image,RGB[0])
    assert np.array_equal(bias,fpn.maps[RGB[0]])
    assert not np.array_equal(fpn.maps[RGB[0]],fpn.maps[RGB[1]])
    e = Sensor('exposure_shift',30000)(image,RGB[0]).mean()
    l = Sensor('lowlight',30000)(image,RGB[0]).mean()
    assert e<l
    print('TEST_SENSOR_OK repeatable, existing lowlight exact, persistent per-camera bias, no global structure',flush=True)
    model = ExternalRestorer().cuda().eval()
    x = torch.from_numpy(np.stack([image,image])[None]).cuda()
    with torch.no_grad():
        a = model(x)
        altered = x.clone()
        altered[:,1] = .5
        b = model(altered)
        assert a.shape==x.shape and torch.isfinite(a).all()
        torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)
    # Real cached training batch and exact augmentation path, but no model saved.
    data_path = P.REPO/'outputs/task_restore_method_v1/decision_pilot_v1_20261003/data/can/train.npz'
    with np.load(data_path) as z:
        frame_id = torch.from_numpy(z['frame_id'][:16].copy()).cuda()
        clean = {k:torch.from_numpy(z[k][:16].copy()).cuda().float()/255 for k in RGB}
    generator = torch.Generator(device='cuda').manual_seed(3072+2000000)
    noisy = noise_tensor(torch.stack([clean[k] for k in RGB],dim=2),generator)
    duplicate = frame_id==0
    noisy[duplicate,0] = noisy[duplicate,1]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    before = next(model.parameters()).detach().clone()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    restored = restore_observations(model,{k:noisy[:,:,i] for i,k in enumerate(RGB)})
    loss = sum(F.l1_loss(restored[k],clean[k]) for k in RGB)/2
    loss.backward()
    grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    assert np.isfinite(grad) and grad>0 and torch.isfinite(loss)
    optimizer.step()
    assert not torch.equal(before,next(model.parameters()))
    torch.cuda.synchronize()
    print(f'COMPAT_TRAIN_OK parameters={sum(v.numel() for v in model.parameters())} batch=16 loss={float(loss):.6f} gradient={grad:.6f} seconds={time.perf_counter()-begin:.3f} peak_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}',flush=True)
    model.eval()
    record = record_for('can')
    policy,cfg = load_policy(record,'cuda:0')
    assert (policy.n_obs_steps,policy.n_action_steps,policy.num_inference_steps)==(2,8,100)
    assert all(not v.requires_grad for v in policy.parameters())
    print('COMPAT_POLICY_OK frozen DDPM100 obs2 action8 absolute10D; native84x84',flush=True)
    # Validator regression: full400 accepted, shortened/bad row rejected.
    group = None,'can','original','sensor_fpn'
    lock = dict(settings=dict(external_sources=P.sources()),banks=dict(can='b'),policies=dict(can='p'),
                base_sources={},confirm_sources={})
    spec = dict(task='can',arm='original',condition='sensor_fpn',training_seed=None,
                bank_episodes=100,episodes=50,start_seed=30000,offset=0,count=1,max_steps=400,
                noise=P.NOISE['sensor_fpn'],external_sources=P.sources(),sources={},confirm_sources={},
                adapter_sha256=None,state_bank_sha256='b',policy_sha256='p')
    row = dict(task='can',arm='original',condition='sensor_fpn',training_seed=None,seed=30000,
               steps=400,success_ever=False,success_final=False,first_success_step=None,
               wall_seconds=1.,inference_seconds=.5,initial_state_sha256='a'*64,
               initial_model_sha256='c'*64,state_bank_entry_sha256='d'*64)
    with tempfile.TemporaryDirectory(prefix='n2d_external_test_') as td:
        target = Path(td)
        def put(r):
            P.write(target/'result.json',dict(protocol=spec,rows=[r],episodes=1,successes=0,final_successes=0))
            (target/'episodes.jsonl').write_text(json.dumps(r)+'\n')
        put(row)
        P.verify_shard(target,(group,0,1),lock)
        for key,value in [('steps',10),('seed',30001),('initial_state_sha256','bad'),('success_final',1)]:
            bad = dict(row,**{key:value})
            put(bad)
            try:
                P.verify_shard(target,(group,0,1),lock)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Validator accepted invalid '+key)
    print('TEST_EXTERNAL_OK budget, causality, pretrained strict load, gradients, frozen policy, result rejection',flush=True)

if __name__=='__main__':
    main()
