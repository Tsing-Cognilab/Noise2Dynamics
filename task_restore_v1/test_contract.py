"""Meaningful checks for action geometry, camera noise and paired comparisons."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from evaluate import absolute_action, corrupt
from run import summarize
from states import digest, load_bank, restore_initial, state_digest

class ContractTests(unittest.TestCase):
    def test_absolute_rotations_roundtrip(self):
        rotations = Rotation.from_euler('xyz', [[0,0,0],[.4,-.7,1.2],[3.1,.2,-2.]])
        matrix = rotations.as_matrix()
        six = matrix[:, :2, :].reshape(-1,6)
        actions = np.concatenate([np.ones((3,3))*.12, six, np.ones((3,1))], axis=-1)
        converted = absolute_action(actions)
        np.testing.assert_allclose(Rotation.from_rotvec(converted[:,3:6]).as_matrix(), matrix, atol=1e-6)
        np.testing.assert_allclose(converted[:, :3], actions[:, :3])
        np.testing.assert_allclose(converted[:, -1], actions[:, -1])

    def test_clean_identity_and_repeatable_noise(self):
        x = np.full((3,84,84), .6, np.float32)
        np.testing.assert_array_equal(corrupt(x,'clean',np.random.default_rng(7)),x)
        a = corrupt(x,'shot_read_medium',np.random.default_rng(7))
        b = corrupt(x,'shot_read_medium',np.random.default_rng(7))
        np.testing.assert_array_equal(a,b)
        self.assertGreater(np.std(a-x),.01)
        self.assertTrue(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1)
        dark = corrupt(x,'lowlight',np.random.default_rng(7))
        self.assertLess(dark.mean(),x.mean()*.8)

    def test_pairing_rejects_different_initial_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition, fingerprint in [('clean','a'),('shot_read_medium','b')]:
                target = root/'lift'/condition
                target.mkdir(parents=True)
                result = dict(protocol=dict(task='lift',condition=condition),episodes=1,success_rate=1.,successes=1,final_success_rate=1.,mean_episode_seconds=1.,rows=[dict(seed=10000,success_ever=True,initial_state_sha256=fingerprint)])
                (target/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError,'not paired'):
                summarize(root)

    def test_explicit_restoration_checks_exact_state_and_model(self):
        class FakeEnv:
            def reset_to(self, state):
                self.state = state
                return {'observation': 1}
            def get_state(self):
                return self.state
        entry = dict(seed=10000, model='<mujoco/>', states=[0., .123456789], state_sha256=state_digest([0., .123456789]), model_sha256=digest(b'<mujoco/>'))
        raw, fingerprints = restore_initial(FakeEnv(), entry)
        self.assertEqual(raw['observation'], 1)
        self.assertEqual(fingerprints['initial_state_sha256'], entry['state_sha256'])
        self.assertEqual(fingerprints['initial_model_sha256'], entry['model_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bank.json'
            bank = dict(version=1, task='lift', start_seed=10000, episodes=1, entries=[entry])
            path.write_text(json.dumps(bank))
            load_bank(path, 'lift', 10000, 1)
            bank['entries'][0]['states'][1] += 1e-10
            path.write_text(json.dumps(bank))
            with self.assertRaisesRegex(RuntimeError, 'corrupt entry'):
                load_bank(path, 'lift', 10000, 1)

    def test_restoration_allows_lowest_float_bits_but_rejects_physical_change(self):
        class ForwardEnv:
            def __init__(self, delta):
                self.delta = delta
            def reset_to(self, state):
                self.state = {'model':state['model'], 'states':state['states'].copy()}
                self.state['states'][1] += self.delta
                return {}
            def get_state(self):
                return self.state
        entry = dict(seed=10000, model='<mujoco/>', states=[0., .99])
        raw, fingerprint = restore_initial(ForwardEnv(2.22e-16), entry)
        self.assertGreater(fingerprint['initial_state_bank_max_abs_error'], 0.)
        with self.assertRaisesRegex(RuntimeError, 'exceeds float64 precision'):
            restore_initial(ForwardEnv(6.74e-5), entry)

if __name__ == '__main__':
    unittest.main()
