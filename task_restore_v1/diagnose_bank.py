"""Read-only bounded diagnosis of simulator forward normalization."""
import json
import numpy as np
from omegaconf import OmegaConf
from evaluate import ROOT, create_env
from states import load_bank, state_digest

for record in json.loads((ROOT/'assets'/'manifest.json').read_text())['tasks']:
    task = record['task']
    entry = load_bank(ROOT/'assets'/'initial_states_v1'/'seed10000_n1'/f'{task}.json', task, 10000, 1)['entries'][0]
    np.random.seed(30000000)
    env, meta = create_env(record, OmegaConf.load(ROOT/record['config']))
    expected = np.asarray(entry['states'])
    for i in range(6):
        env.reset_to({'model': entry['model'], 'states': expected})
        actual = env.get_state()['states']
        changed = np.flatnonzero(actual != expected)
        print('FORWARD_DIAGNOSTIC', task, i, 'max_difference', np.abs(actual-expected).max(), 'indices', changed.tolist(), 'hash', state_digest(actual), flush=True)
        for index in changed:
            print('VALUE', int(index), repr(expected[index]), repr(actual[index]), flush=True)
        expected = actual.copy()
    env.env.close()
