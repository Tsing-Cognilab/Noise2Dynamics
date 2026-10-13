"""Official per-frame RetinexFormer, plus fixed sensor-inspired RGB corruption."""
import sys
from pathlib import Path
import protocol as P
sys.path.insert(0, str(P.METHOD))
from common import *
sys.path.insert(0, str(P.HERE / 'vendor'))
from retinexformer_arch import RetinexFormer

class ExternalRestorer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = RetinexFormer(**P.asset_manifest()['architecture'])
        if pretrained:
            ckpt = torch.load(P.HERE/'assets/LOL_v1.pth',map_location='cpu',weights_only=False)
            state = ckpt['params']
            if state and all(k.startswith('module.') for k in state):
                state = {k[7:]:v for k,v in state.items()}
            self.net.load_state_dict(state,strict=True)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:3] != (2,3) or x.shape[-2:] != (84,84):
            raise ValueError('Expected native 84x84 B,2,3,H,W; no rescaling/GT mean')
        y = self.net(x.flatten(0,1)).reshape_as(x)
        return y if self.training else y.clamp(0,1)

class Sensor:
    """One corruption per physical frame; bias map is fixed per episode/camera.

    Maps use separate streams from shot/read RNG. Shared seed gives identical
    corruption on identical observations across methods, not identical trajectories.
    This is linear-light RGB emulation, NOT a calibrated physical RAW camera.
    """
    def __init__(self, condition, episode_seed):
        self.condition = condition
        self.params = P.NOISE[condition]
        self.rng = np.random.default_rng(episode_seed+20000000)
        self.maps = {key:np.random.default_rng(episode_seed+31000000+i).normal(
                     0,self.params['fpn_std'],(3,84,84)).astype(np.float32)
                     for i,key in enumerate(RGB)}

    def __call__(self, image, camera):
        if self.condition in ('clean','lowlight'):
            return corrupt(image,self.condition,self.rng)
        x = np.asarray(image,dtype=np.float32)
        if x.shape != (3,84,84):
            raise ValueError('Camera shape changed')
        linear = np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
        signal = linear*self.params['exposure']
        sensor = self.rng.poisson(signal/self.params['gain']).astype(np.float32)*self.params['gain']
        sensor += self.rng.normal(0,self.params['read'],x.shape).astype(np.float32)
        sensor += self.maps[camera]
        sensor = sensor.clip(0,1)
        y = np.where(sensor<=.0031308,sensor*12.92,1.055*sensor**(1/2.4)-.055)
        return (np.rint(y.clip(0,1)*255)/255).astype(np.float32)
