"""Bounded real-simulator check: two independently constructed envs per task."""
import json
import random

import numpy as np
from omegaconf import OmegaConf
from evaluate import ROOT, create_env
from states import MODEL_SEEDS, digest, load_bank, restore_initial


def main():
    records = json.loads((ROOT/'assets'/'manifest.json').read_text())['tasks']
    for record in records:
        task = record['task']
        bank = load_bank(ROOT/'assets'/'initial_states_v1'/'paired_xml_seed10000_n1'/f'{task}.json', task, 10000, 1)
        observed = []
        for repeat in range(2):
            # Deliberately change construction RNG: XML restoration must still pair.
            random.seed(MODEL_SEEDS[task] + repeat*1000)
            np.random.seed(MODEL_SEEDS[task] + repeat*1000)
            env, meta = create_env(record, OmegaConf.load(ROOT/record['config']))
            try:
                raw, fingerprint = restore_initial(env, bank['entries'][0])
                fingerprint['rgb_sha256'] = {k:digest(np.asarray(v).tobytes()) for k,v in raw.items() if k.endswith('_image')}
                observed.append(fingerprint)
            finally:
                env.env.close()
        if observed[0] != observed[1]:
            raise RuntimeError(f'Restored model/state/RGB mismatch: {task} {observed}')
        print(f'EXPLICIT_PAIRING_OK task={task} independent_envs=2 state={observed[0]["initial_state_sha256"]} model={observed[0]["initial_model_sha256"]}', flush=True)
    print('TASK_RESTORE_REAL_STATE_TEST_FINISHED', flush=True)


if __name__ == '__main__':
    main()
