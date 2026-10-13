"""Complete-table reporting: all seeds, paired simulator outcomes, no selective claims."""
import statistics
import protocol as P
from report import holm, cluster_interval

def report(output, lock):
    root = Path(lock['settings']['reference'])
    reference_lock = lock['settings']['reference_lock']
    merged,entries, fingerprints = {},[],{}
    reused,new = 0,0
    for group in P.OLD.groups()+P.groups():
        is_old = group in P.OLD.groups()
        n = P.OLD.COUNTS[group[3]] if is_old else P.COUNTS[group[3]]
        rows = []
        for offset in range(0,n,P.SHARD):
            job = group,offset,min(P.SHARD,n-offset)
            target = P.group_path(root if is_old else output,group)/'shards'/f'{offset:05d}'
            rows += (P.OLD.verify_shard(target,job,reference_lock) if is_old else P.verify_shard(target,job,lock))
        if is_old:
            reused += len(rows)
        else:
            new += len(rows)
        for row in rows:
            key = row['task'],row['seed']
            fp = tuple(row[k] for k in ('initial_state_sha256','initial_model_sha256','state_bank_entry_sha256'))
            if key in fingerprints and fingerprints[key] != fp:
                raise RuntimeError('Actual initial states not paired across models/conditions')
            fingerprints[key] = fp
        merged[group] = rows
        seed,task,arm,condition = group
        entry = dict(task=task,training_seed=seed,arm=arm,condition=condition,episodes=n,
                     ever=sum(r['success_ever'] for r in rows),final=sum(r['success_final'] for r in rows),
                     inference_seconds=statistics.mean(r['inference_seconds'] for r in rows),reused_E04=is_old)
        entries.append(entry)
        if not is_old:
            P.write(P.group_path(output,group)/'merged.json',dict(summary=entry,rows=rows))
    if (new,reused) != (4000,3200):
        raise RuntimeError('Wrong pre-fixed budget')
    tests = []
    for task in P.TASKS:
        for seed in P.SEEDS:
            for condition,controls,family in [('lowlight',('retinex_finetune',),'external_primary')]+[
                 (c,('image','feature','retinex_finetune'),'unseen_secondary') for c in ('exposure_shift','sensor_fpn')]:
                for control in controls:
                    for field in ('success_ever','success_final'):
                        a,b = merged[(seed,task,control,condition)],merged[(seed,task,'decision',condition)]
                        tests.append(dict(task=task,training_seed=seed,condition=condition,control=control,field=field,
                                          family=family,**P.paired(a,b,field)))
    for family in ('external_primary','unseen_secondary'):
        for field in ('success_ever','success_final'):
            selected = [t for t in tests if t['family']==family and t['field']==field]
            adjusted = holm([t['exact_two_sided_p'] for t in selected])
            for t,value in zip(selected,adjusted):
                t['holm_adjusted_p'] = value
    aggregates = []
    for task in P.TASKS:
        for condition in P.COUNTS:
            for control in ('image','feature','retinex_finetune'):
                for field in ('success_ever','success_final'):
                    delta,paired_states = [],[]
                    for seed in P.SEEDS:
                        a,b = merged[(seed,task,control,condition)],merged[(seed,task,'decision',condition)]
                        delta.append(P.paired(a,b,field)['delta_pp'])
                        paired_states.append([int(y[field])-int(x[field]) for x,y in zip(a,b)])
                    per_state = [sum(v)/len(P.SEEDS) for v in zip(*paired_states)]
                    aggregates.append(dict(task=task,condition=condition,control=control,field=field,
                        seed_deltas_pp=delta,mean_delta_pp=statistics.mean(delta),sample_sd_pp=statistics.stdev(delta),
                        conditional_state_cluster_interval_pp=cluster_interval(per_state)))
    lines = ['# E05 external restoration and unseen corruption results','',
             '4000 NEW full400-step rollouts; 3200 E04 records reused, not recounted as new. All 3 seeds retained.',
             'Primary outcome: env.is_success()[task] ever success; final success secondary.',
             'E05 reuses E04 physical states. Only corruption profiles are unseen; this is NOT unseen-state generalization.',
             'RetinexFormer official LOL-v1 weights: pretrained zero-shot and fixed2000-update L1 adaptation are distinct baselines.',
             'External pretrained data, model capacity and per-frame information differ. This is not a pure decision-supervision ablation.',
             'Sensor-inspired RGB emulation is not calibrated real RAW sensor data. No GT mean or true exposure supplied to blind methods.',
             '', '|Task|Training seed|Method|Corruption|Ever|Final|Mean full-episode inference seconds|Reused E04|',
             '|---|---|---|---|---|---|---|---|']
    for e in entries:
        lines.append(f'|{e["task"]}|{e["training_seed"]}|{e["arm"]}|{e["condition"]}|{e["ever"]}/{e["episodes"]}|{e["final"]}/{e["episodes"]}|{e["inference_seconds"]:.2f}|{e["reused_E04"]}|')
    lines += ['', '## Pre-fixed paired comparisons',
              'Exact two-sided McNemar; Holm across 6 external-primary tests and 36 unseen-secondary tests, separately for ever/final.',
              'Zero-shot external is one shared checkpoint, NOT three independent training seeds. Clean and shared-controls are descriptive.',
              '', '|Task|Seed|Condition|Decision vs|Outcome|Gained|Lost|Delta pp|Exact p|Holm p|',
              '|---|---|---|---|---|---|---|---|---|---|']
    for t in tests:
        lines.append(f'|{t["task"]}|{t["training_seed"]}|{t["condition"]}|{t["control"]}|{t["field"]}|{t["gains"]}|{t["losses"]}|{t["delta_pp"]:+.2f}|{t["exact_two_sided_p"]:.5g}|{t["holm_adjusted_p"]:.5g}|')
    lines += ['', 'Conditional cluster bootstrap resamples physical states shared across 3 fixed trained models; it does not establish training-population uncertainty.',
              'Do not stop or exclude seeds after seeing outcomes. Failure or no advantage is reportable. No publication-readiness claim follows automatically.']
    P.write(output/'summary.json',dict(new_episodes=new,reused_E04_episodes=reused,table=entries,
            comparisons=tests,aggregates=aggregates,model_lock_sha256=P.sha(output/'model_lock.json')))
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('EXTERNAL_REPORT_READY new=4000 reused=3200 all_seeds=3072,3073,3074',flush=True)

from pathlib import Path
