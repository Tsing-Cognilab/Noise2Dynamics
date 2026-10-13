"""No CUDA needed: exact shard coverage, merging, corruption and pairing checks."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from run_parallel import build_jobs, merge_results, shard_directory, verify_result


def fingerprint(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class ParallelTests(unittest.TestCase):
    def suite(self):
        return dict(tasks=['lift'], conditions=['clean','shot_read_medium'], episodes=2, start_seed=10000,
                    max_steps=80, shard_size=1, source_checkpoints={'lift':fingerprint('policy')},
                    state_banks={'lift':fingerprint('bank')}, evaluator_sha256=fingerprint('core'),
                    parallel_wrapper_sha256=fingerprint('wrapper'))

    def fixture(self, root, suite):
        for job in build_jobs(suite['tasks'],suite['conditions'],suite['episodes'],suite['shard_size']):
            task,condition,offset,count = job
            seed = suite['start_seed']+offset
            protocol = dict(task=task,condition=condition,episodes=suite['episodes'],start_seed=suite['start_seed'],
                            max_steps=suite['max_steps'],episode_offset=offset,episode_count=count,
                            checkpoint_sha256=suite['source_checkpoints'][task],state_bank_sha256=suite['state_banks'][task],
                            evaluator_sha256=suite['evaluator_sha256'],parallel_wrapper_sha256=suite['parallel_wrapper_sha256'])
            success = condition == 'clean' or offset == 0
            row = dict(seed=seed,task=task,condition=condition,steps=80,wall_seconds=2.,success_ever=success,success_final=success,
                       initial_state_sha256=fingerprint(seed),initial_model_sha256=fingerprint('model'),state_bank_entry_sha256=fingerprint(('entry',seed)))
            record = dict(protocol=protocol,rows=[row],episodes=1,successes=int(success),success_rate=float(success),final_success_rate=float(success))
            target = shard_directory(root,job)
            target.mkdir(parents=True)
            (target/'result.json').write_text(json.dumps(record))
            (target/'episodes.jsonl').write_text(json.dumps(row)+'\n')

    def test_all_450_episodes_exactly_once(self):
        jobs = build_jobs(['lift','can','square'],['clean','shot_read_medium','lowlight'],50,5)
        episodes = [(task,condition,10000+i) for task,condition,offset,count in jobs for i in range(offset,offset+count)]
        self.assertEqual(len(jobs),90)
        self.assertEqual(len(episodes),450)
        self.assertEqual(len(set(episodes)),450)
        self.assertTrue(all(0 <= offset < 50 and count == 5 for _,_,offset,count in jobs))

    def test_tail_shard_and_merge(self):
        self.assertEqual(build_jobs(['lift'],['clean'],7,5),[('lift','clean',0,5),('lift','clean',5,2)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = self.suite()
            self.fixture(root,suite)
            summary = merge_results(root,suite)
            self.assertEqual(len(summary),2)
            self.assertEqual(summary[0]['success_rate'],1.)
            self.assertEqual(summary[1]['success_rate'],.5)
            self.assertEqual(summary[1]['paired_drop_from_clean'],.5)
            for condition in suite['conditions']:
                self.assertEqual(len((root/'lift'/condition/'episodes.jsonl').read_text().splitlines()),2)

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'Incomplete suite'):
                merge_results(Path(directory),self.suite())

    def test_wrong_seed_and_changed_policy_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = self.suite()
            self.fixture(root,suite)
            job = ('lift','clean',0,1)
            path = shard_directory(root,job)/'result.json'
            record = json.loads(path.read_text())
            record['rows'][0]['seed'] = 10001
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'Missing, duplicate'):
                verify_result(path,job,suite)
            record['rows'][0]['seed'] = 10000
            record['protocol']['checkpoint_sha256'] = fingerprint('other_policy')
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'protocol mismatch'):
                verify_result(path,job,suite)

    def test_pairing_remains_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = self.suite()
            self.fixture(root,suite)
            path = shard_directory(root,('lift','shot_read_medium',0,1))/'result.json'
            record = json.loads(path.read_text())
            record['rows'][0]['initial_state_sha256'] = fingerprint('wrong_state')
            path.write_text(json.dumps(record))
            (path.parent/'episodes.jsonl').write_text(json.dumps(record['rows'][0])+'\n')
            with self.assertRaisesRegex(RuntimeError,'not paired'):
                merge_results(root,suite)

    def test_log_result_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = self.suite()
            self.fixture(root,suite)
            job = ('lift','clean',0,1)
            path = shard_directory(root,job)/'result.json'
            (path.parent/'episodes.jsonl').write_text('')
            with self.assertRaisesRegex(RuntimeError,'disagree'):
                verify_result(path,job,suite)


if __name__ == '__main__':
    unittest.main()
