"""Read-only diagnostic of seeded versus explicit-state simulator resets."""
import json
import hashlib
import numpy as np
from omegaconf import OmegaConf
from evaluate import ROOT, create_env

record = next(r for r in json.loads((ROOT/'assets'/'manifest.json').read_text())['tasks'] if r['task']=='lift')
cfg = OmegaConf.load(ROOT/record['config'])
states = []
for index in range(2):
    env, meta = create_env(record, cfg)
    np.random.seed(10000)
    env.reset()
    state = env.get_state()
    vector = state['states'].copy()
    print('RESET',index,hashlib.sha256(vector.tobytes()).hexdigest(),vector.shape,flush=True)
    states.append(state)
    if index == 1:
        difference = np.abs(states[0]['states']-vector)
        print('MAX_DIFFERENCE',float(difference.max()),'INDICES',np.flatnonzero(difference).tolist(),flush=True)
        for i in np.flatnonzero(difference):
            print('DELTA',int(i),float(states[0]['states'][i]),float(vector[i]),flush=True)
        env.reset_to(states[0])
        restored = env.get_state()['states']
        print('EXPLICIT_RESET_MAX_DIFF',float(np.abs(restored-states[0]['states']).max()),flush=True)
    env.env.close()
