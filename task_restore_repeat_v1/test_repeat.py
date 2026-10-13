"""Launcher-only checks; not training/effectiveness evidence."""
import copy
import json
import unittest
from unittest.mock import patch
import run_repeat as r


class RepeatProtocolTests(unittest.TestCase):
    def test_only_learned_groups_are_new(self):
        self.assertEqual(r.LEARNED, [('image','clean'),('image','lowlight'),('decision','clean'),('decision','lowlight')])
        self.assertEqual(2*len(r.EXPECTED['tasks'])*len(r.LEARNED)*r.EXPECTED['episodes'],800)

    def test_config_fixed(self):
        self.assertEqual((r.EXPECTED['updates'],r.EXPECTED['batch_size'],r.EXPECTED['max_steps']),(2000,16,400))
        self.assertEqual((r.EXPECTED['train_demos'],r.EXPECTED['valid_demos'],r.EXPECTED['stride']),(24,4,4))

    def test_paired_first_seed_signal(self):
        image = [{'success_ever':i<43} for i in range(50)]
        decision = [{'success_ever':i<48} for i in range(50)]
        self.assertEqual(r.paired(image,decision,'success_ever'),
                         dict(gains=5,losses=0,delta_pp=10.,exact_two_sided_p=.0625))

    def test_paired_tie(self):
        rows = [{'success_final':True}]*50
        self.assertEqual(r.paired(rows,rows,'success_final')['exact_two_sided_p'],1.)

    def test_tags_cannot_escape_output(self):
        for tag in ('..','../other','a/b','/tmp','x;echo'):
            with self.assertRaises(ValueError):
                r.valid_tag(tag)

    def test_reference_source_has_no_changes(self):
        source = r.REPO/'outputs/task_restore_method_v1/decision_pilot_v1_20261003'
        protocol = r.load_reference(source)
        self.assertEqual(protocol['seed'],3072)

    def test_finished_spec_allows_only_seed(self):
        source = r.REPO/'outputs/task_restore_method_v1/decision_pilot_v1_20261003'
        original = r.read(source/'train/square/decision/finished.json')
        repeated = copy.deepcopy(original)
        repeated['spec']['seed'] = 3073
        with patch.object(r,'read',side_effect=[repeated,original]), patch.object(r,'sha',return_value=repeated['checkpoint_sha256']):
            self.assertEqual(r.verify_training(r.Path('/not-written'),source,'square','decision',3073),original['checkpoint_sha256'])

    def test_changed_budget_rejected(self):
        source = r.REPO/'outputs/task_restore_method_v1/decision_pilot_v1_20261003'
        original = r.read(source/'train/square/decision/finished.json')
        repeated = copy.deepcopy(original)
        repeated['spec'].update(seed=3073,updates=2001)
        with patch.object(r,'read',side_effect=[repeated,original]):
            with self.assertRaises(RuntimeError):
                r.verify_training(r.Path('/not-written'),source,'square','decision',3073)


if __name__ == '__main__':
    unittest.main()
