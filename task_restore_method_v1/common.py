"""Isolated method pilot; reuse, never modify, the verified simulator contract."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORE = REPO / 'task_restore_v1'
sys.path.insert(0, str(CORE))
from evaluate import CONDITIONS, absolute_action, corrupt, create_env, load_policy
from states import MODEL_SEEDS, digest, load_bank, restore_initial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RGB = ['agentview_image', 'robot0_eye_in_hand_image']


def record_for(task):
    return next(r for r in json.loads((CORE/'assets/manifest.json').read_text())['tasks'] if r['task'] == task)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def source_hashes():
    return {p.name: digest(p.read_bytes()) for p in sorted(HERE.glob('*.py'))}


def condition_features(policy, obs):
    normalized = policy.normalizer.normalize(obs)
    batch = next(iter(obs.values())).shape[0]
    flat = {k: v[:, :policy.n_obs_steps].reshape(-1, *v.shape[2:]) for k,v in normalized.items()}
    return policy.obs_encoder(flat).reshape(batch, policy.n_obs_steps, -1)


def exposure_compensate(x):
    """Oracle diagnostic only: known synthetic exposure 0.25, not blind restoration."""
    linear = torch.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)
    linear = (linear/.25).clamp(0,1)
    return torch.where(linear <= .0031308, linear*12.92, 1.055*linear.clamp_min(1e-8)**(1/2.4)-.055).clamp(0,1)


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1), nn.GroupNorm(4, outgoing), nn.SiLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.GroupNorm(4, outgoing), nn.SiLU())


class Restorer(nn.Module):
    """Shared-camera two-frame causal U-Net. Both arms have identical parameters."""
    def __init__(self, width=24):
        super().__init__()
        self.a = Block(6,width)
        self.b = Block(width,width*2)
        self.c = Block(width*2,width*4)
        self.u = Block(width*6,width*2)
        self.v = Block(width*3,width)
        self.out = nn.Conv2d(width,3,1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # B,T=2,C,H,W. Each output reads only its own frame and its predecessor.
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError('Expected B,2,3,H,W causal history')
        past = torch.cat([x[:,:1], x[:,:-1]], dim=1)
        y = torch.cat([past,x],dim=2).flatten(0,1)
        a = self.a(y)
        b = self.b(F.avg_pool2d(a,2))
        c = self.c(F.avg_pool2d(b,2))
        u = self.u(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],dim=1))
        v = self.v(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],dim=1))
        logits = torch.logit(x.flatten(0,1).clamp(.001,.999))
        return torch.sigmoid(logits+self.out(v)).reshape_as(x)


def restore_observations(model, observations):
    out = dict(observations)
    # Cameras are separate streams; no accidental time/camera swapping.
    images = torch.cat([observations[k] for k in RGB],dim=0)
    restored = model(images)
    for k,v in zip(RGB,restored.chunk(len(RGB),dim=0)):
        out[k] = v
    return out


def noise_tensor(clean, generator):
    """Same lowlight model as core; 20% identity samples protect clean operation."""
    b = clean.shape[0]
    identity = torch.rand((b,1,1,1,1,1),device=clean.device,generator=generator) < .2
    linear = torch.where(clean <= .04045, clean/12.92, ((clean+.055)/1.055)**2.4)
    signal = linear*.25
    sensor = torch.poisson(signal/.004,generator=generator)*.004
    sensor += torch.randn(clean.shape,device=clean.device,generator=generator)*.008
    sensor = sensor.clamp(0,1)
    out = torch.where(sensor <= .0031308,sensor*12.92,1.055*sensor.clamp_min(1e-8)**(1/2.4)-.055)
    out = (out.clamp(0,1)*255).round()/255
    return torch.where(identity,clean,out)
