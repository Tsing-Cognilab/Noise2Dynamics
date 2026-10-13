"""Encoder-feature control, calibrated only on training data before any rollout."""
import argparse
import random
import sys
import time
from pathlib import Path
import contract as C

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'task_restore_method_v1'))
from common import (CONDITIONS, RGB, Restorer, condition_features, load_policy,
                    noise_tensor, record_for, restore_observations, source_hashes, np, torch, F)


def load_data(path, task, device):
    ready = C.read(path / 'ready.json')
    if ready['spec']['task'] != task or ready['spec']['method_code'] != source_hashes():
        raise RuntimeError('Wrong or stale cached data')
    data = {}
    for split in ('train', 'valid'):
        if C.sha(path / f'{split}.npz') != ready['files'][split]:
            raise RuntimeError('Data checksum changed')
        with np.load(path / f'{split}.npz') as z:
            data[split] = {k: torch.from_numpy(z[k].copy()).to(device) for k in z.files if k != 'episode_id'}
    return data


def mixed_batch(data, cfg, index, generator):
    clean = {k: data[k][index].float() / (255 if k in RGB else 1) for k in cfg.shape_meta.obs}
    noisy = noise_tensor(torch.stack([clean[k] for k in RGB], dim=2), generator)
    duplicate = data['frame_id'][index] == 0
    noisy[duplicate, 0] = noisy[duplicate, 1]
    corrupted = dict(clean)
    for i, k in enumerate(RGB):
        corrupted[k] = noisy[:, :, i]
    return clean, corrupted


def losses(model, policy, clean, noisy, target_features):
    restored = restore_observations(model, noisy)
    image = sum(F.l1_loss(restored[k], clean[k]) for k in RGB) / len(RGB)
    features = condition_features(policy, restored)
    return image, F.mse_loss(features, target_features), features


def decision_loss(policy, data, index, features, generator):
    action = policy.normalizer['action'].normalize(data['teacher_action'][index])
    if policy.pred_action_steps_only:
        begin = policy.n_obs_steps - 1
        action = action[:, begin:begin + policy.n_action_steps]
    eps = torch.randn(action.shape, device=action.device, generator=generator)
    times = torch.randint(0, policy.noise_scheduler.config.num_train_timesteps,
                          (len(index),), device=action.device, generator=generator)
    xt = policy.noise_scheduler.add_noise(action, eps, times)
    with torch.no_grad():
        target = policy.model(xt, times, data['teacher_features'][index])
    student = policy.model(xt, times, features)
    begin = 0 if policy.pred_action_steps_only else policy.n_obs_steps - 1
    return F.mse_loss(student[:, begin:begin + policy.n_action_steps], target[:, begin:begin + policy.n_action_steps])


def gradient_norm(loss, model):
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    norm = sum(g.square().sum() for g in gradients if g is not None).sqrt()
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError('Auxiliary gradient must be finite and nonzero')
    return float(norm)


def calibration_spec(task, data):
    return dict(task=task, calibration=C.CALIBRATION, data_sha256=C.sha(data / 'ready.json'),
                base_sources=source_hashes(), confirm_sources=C.sources(), width=24,
                initialization_seed=3072, policy_sha256=record_for(task)['checkpoint_sha256'])


def calibrate(task, path, output, device='cuda:0'):
    spec = calibration_spec(task, path)
    if output.exists():
        existing = C.read(output)
        if existing['spec'] != spec or not np.isfinite(existing['weight']) or existing['weight'] <= 0:
            raise RuntimeError('Calibration changed; do not overwrite')
        return existing
    torch.manual_seed(3072)
    model = Restorer().to(device)
    policy, cfg = load_policy(record_for(task), device)
    data = load_data(path, task, device)['train']
    sample = np.random.default_rng(C.CALIBRATION['seed'])
    sensor = torch.Generator(device=device).manual_seed(C.CALIBRATION['seed'] + 1)
    diffusion = torch.Generator(device=device).manual_seed(C.CALIBRATION['seed'] + 2)
    records = []
    for _ in range(C.CALIBRATION['batches']):
        index = sample.integers(0, len(data['frame_id']), C.CALIBRATION['batch_size'])
        clean, noisy = mixed_batch(data, cfg, index, sensor)
        image, feature, feat = losses(model, policy, clean, noisy, data['teacher_features'][index])
        decision = decision_loss(policy, data, index, feat, diffusion)
        gi, gf, gd = [gradient_norm(x, model) for x in (image, feature, decision)]
        records.append(dict(image_gradient=gi, feature_gradient=gf, decision_gradient=gd,
                            candidate_weight=.1 * gd / gf))
    weight = float(np.median([r['candidate_weight'] for r in records]))
    if not 1e-8 < weight < 1e8 or any(p.grad is not None or p.requires_grad for p in policy.parameters()):
        raise RuntimeError('Invalid scale or unfrozen teacher')
    result = dict(spec=spec, weight=weight, batches=records,
                  gradient_ratio_to_image_median=float(np.median([weight * r['feature_gradient'] / r['image_gradient'] for r in records])),
                  interpretation='train-only initial gradient scale; not selected using closed-loop success')
    C.write(output, result)
    print(f'FEATURE_CALIBRATION_OK task={task} weight={weight:.8g}', flush=True)
    return result


def train(task, data_path, output, seed, calibration_path):
    if seed not in C.SEEDS:
        raise ValueError('All agreed training seeds must be retained')
    calibration = C.read(calibration_path)
    if calibration['spec'] != calibration_spec(task, data_path):
        raise RuntimeError('Stale calibration')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = Restorer().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    policy, cfg = load_policy(record_for(task), 'cuda:0')
    data = load_data(data_path, task, 'cuda:0')
    weight = calibration['weight']
    spec = dict(task=task, arm='feature', seed=seed, updates=C.UPDATES, batch_size=16,
                feature_weight=weight, calibration_sha256=C.sha(calibration_path),
                learning_rate=3e-4, weight_decay=1e-4, gradient_clip=1., width=24,
                sources=source_hashes(), confirm_sources=C.sources(),
                data_sha256=C.sha(data_path / 'ready.json'), parameters=sum(p.numel() for p in model.parameters()),
                noise=dict(CONDITIONS['lowlight']), identity_probability=.2, teacher_frozen=True,
                selection='last fixed-budget checkpoint; no closed-loop selection')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists() and C.read(output / 'protocol.json') != spec:
        raise RuntimeError('Training protocol changed; no overwrite')
    C.write(output / 'protocol.json', spec)
    if (output / 'finished.json').exists():
        done = C.read(output / 'finished.json')
        if done['spec'] != spec or C.sha(output / 'model.pt') != done['checkpoint_sha256']:
            raise RuntimeError('Completed model changed')
        print(f'FEATURE_TRAIN_CACHED task={task} seed={seed}', flush=True)
        return
    sample = np.random.default_rng(seed + 1000000)
    sensor = torch.Generator(device='cuda').manual_seed(seed + 2000000)
    first, max_gradient, feature_checks = 0, 0., []
    resume = output / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, weights_only=False)
        if saved['spec'] != spec:
            raise RuntimeError('Resume spec changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        sample.bit_generator.state = saved['sample_rng']
        sensor.set_state(saved['sensor_rng'])
        first, max_gradient, feature_checks = saved['update'], saved['max_gradient'], saved['feature_checks']
    start = time.perf_counter()
    for update in range(first + 1, C.UPDATES + 1):
        index = sample.integers(0, len(data['train']['frame_id']), 16)
        clean, noisy = mixed_batch(data['train'], cfg, index, sensor)
        image, feature, _ = losses(model, policy, clean, noisy, data['train']['teacher_features'][index])
        loss = image + weight * feature
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite feature training objective')
        if update == 1 or update % 500 == 0:
            gi, gf = gradient_norm(image, model), gradient_norm(feature, model)
            feature_checks.append(dict(update=update, image_gradient=gi, feature_gradient=gf, weighted_feature_ratio=weight * gf / gi))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not np.isfinite(grad) or grad <= 0 or any(p.grad is not None for p in policy.parameters()):
            raise RuntimeError('Invalid adapter gradient or unfrozen teacher')
        max_gradient = max(max_gradient, grad)
        optimizer.step()
        if update == 1 or update % 100 == 0:
            row = dict(update=update, loss=float(loss), image_l1=float(image), feature_mse=float(feature),
                       feature_weight=weight, gradient=grad, seconds=time.perf_counter() - start)
            with (output / 'train.jsonl').open('a') as f:
                f.write(C.json.dumps(row) + '\n')
            print(f'FEATURE_PROGRESS task={task} seed={seed} update={update}/2000 image={float(image):.5f} feature={float(feature):.6f}', flush=True)
        if update % 500 == 0:
            torch.save(dict(spec=spec, update=update, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            sample_rng=sample.bit_generator.state, sensor_rng=sensor.get_state(),
                            max_gradient=max_gradient, feature_checks=feature_checks), output / 'resume.tmp.pt')
            (output / 'resume.tmp.pt').replace(resume)
    model.eval()
    with torch.no_grad():
        index = np.arange(min(64, len(data['valid']['frame_id'])))
        clean, noisy = mixed_batch(data['valid'], cfg, index, sensor)
        image, feature, _ = losses(model, policy, clean, noisy, data['valid']['teacher_features'][index])
    torch.save(dict(spec=spec, model=model.state_dict()), output / 'model.tmp.pt')
    (output / 'model.tmp.pt').replace(output / 'model.pt')
    C.write(output / 'finished.json', dict(spec=spec, checkpoint_sha256=C.sha(output / 'model.pt'),
            validation_image_l1=float(image), validation_feature_mse=float(feature), feature_checks=feature_checks,
            max_gradient=max_gradient, seconds=time.perf_counter() - start,
            interpretation='offline mixed-condition diagnostic only; simulator success determines utility'))
    print(f'FEATURE_TRAIN_FINISHED task={task} seed={seed}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('calibrate', 'train'), required=True)
    p.add_argument('--task', choices=C.TASKS, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.mode == 'calibrate':
        calibrate(a.task, a.data, a.calibration)
    else:
        if a.output is None or a.seed is None:
            p.error('Training requires output and seed')
        train(a.task, a.data, a.output, a.seed, a.calibration)


if __name__ == '__main__':
    main()
