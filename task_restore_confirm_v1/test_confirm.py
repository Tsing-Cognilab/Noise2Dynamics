"""Protocol and paired-statistics checks; no training or effectiveness claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import contract as C
from report import holm, cluster_interval


class ConfirmationTests(unittest.TestCase):
    def test_fixed_counts(self):
        self.assertEqual(C.SEEDS, (3072,3073,3074))
        self.assertEqual(len(C.groups()), 42)
        self.assertEqual(len(C.jobs()), 640)
        self.assertEqual(sum(n for _,_,n in C.jobs()), 3200)
        self.assertEqual(sum(C.COUNTS[g[3]] for g in C.groups() if g[0] is None), 500)
        self.assertEqual(sum(C.COUNTS[g[3]] for g in C.groups() if g[2]=='feature'), 900)
        self.assertEqual((C.START,C.BANK_N,C.HORIZON,C.UPDATES),(30000,100,400,2000))

    def test_clean_is_prespecified_subset(self):
        for group, offset, count in C.jobs():
            self.assertLessEqual(offset+count, C.COUNTS[group[3]])
        self.assertEqual(C.COUNTS['clean'],50)

    def test_paired_signal(self):
        a = [dict(seed=i,success_ever=i<43) for i in range(50)]
        b = [dict(seed=i,success_ever=i<48) for i in range(50)]
        self.assertEqual(C.paired(a,b,'success_ever'),dict(gains=5,losses=0,delta_pp=10.,exact_two_sided_p=.0625))

    def test_unpaired_rejected(self):
        with self.assertRaises(RuntimeError):
            C.paired([dict(seed=1)],[dict(seed=2)],'success_ever')

    def test_holm(self):
        self.assertEqual(holm([.04,.01,.03]),[.06,.03,.06])

    def test_cluster_interval_deterministic(self):
        for bound in cluster_interval([.1]*10,100):
            self.assertAlmostEqual(bound,10.)
        self.assertEqual(cluster_interval([0.,1.,-1.],100),cluster_interval([0.,1.,-1.],100))

    def test_output_scope(self):
        for value in ('..','../old','/tmp','bad/name','x;exit'):
            with self.assertRaises(ValueError):
                C.tag(value)

    def example(self):
        group = (3073,'square','feature','lowlight')
        h = 'a'*64
        locked = dict(banks={'square':h},policies={'square':h},base_sources={},confirm_sources={},
                      adapters={'3073/square/feature':dict(sha256=h)})
        spec = dict(task='square',arm='feature',condition='lowlight',training_seed=3073,
                    start_seed=30000,bank_episodes=100,episodes=100,offset=0,count=5,max_steps=400,
                    state_bank_sha256=h,policy_sha256=h,sources={},confirm_sources={},adapter_sha256=h)
        rows = [dict(task='square',arm='feature',condition='lowlight',training_seed=3073,seed=30000+i,
                     steps=400,success_ever=True,success_final=True,first_success_step=10,
                     initial_state_sha256=h,initial_model_sha256=h,state_bank_entry_sha256=h,
                     wall_seconds=10.) for i in range(5)]
        return (group,0,5), locked, dict(protocol=spec,rows=rows,episodes=5,successes=5,final_successes=5)

    def verify(self, change=None):
        job, locked, data = self.example()
        if change:
            change(data)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data['rows']))
            C.write(target/'result.json',data)
            return C.verify_shard(target,job,locked)

    def test_valid_shard(self):
        self.assertEqual(len(self.verify()),5)

    def test_shortened_simulator_episode_rejected(self):
        with self.assertRaises(RuntimeError):
            self.verify(lambda d:d['rows'][0].update(steps=200))

    def test_checkpoint_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            self.verify(lambda d:d['protocol'].update(adapter_sha256='b'*64))

    def test_duplicate_state_rejected(self):
        with self.assertRaises(RuntimeError):
            self.verify(lambda d:d['rows'][1].update(seed=30000))

    def test_success_count_rejected(self):
        with self.assertRaises(RuntimeError):
            self.verify(lambda d:d.update(successes=4))


if __name__ == '__main__':
    unittest.main()
