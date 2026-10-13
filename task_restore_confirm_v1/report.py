"""Full simulator outcomes, paired tests, and state-cluster uncertainty."""
import random
import statistics
import contract as C


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted, running = [0.] * len(values), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(values) - rank) * values[i]))
        adjusted[i] = running
    return adjusted


def cluster_interval(deltas, repetitions=10000):
    # Each element is one state's delta averaged across the SAME three trained models.
    # This is conditional on those weights, not uncertainty over new training runs.
    rng = random.Random(610031)
    n = len(deltas)
    samples = sorted(100 * sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(repetitions))
    return [samples[int(.025 * (repetitions - 1))], samples[int(.975 * (repetitions - 1))]]


def report(root, locked):
    merged, fingerprints, table = {}, {}, []
    for group in C.groups():
        seed, task, arm, condition = group
        rows = []
        for offset in range(0, C.COUNTS[condition], C.SHARD):
            count = min(C.SHARD, C.COUNTS[condition] - offset)
            target = C.group_path(root, group) / 'shards' / f'{offset:05d}'
            rows.extend(C.verify_shard(target, (group, offset, count), locked))
        for row in rows:
            key = (task, row['seed'])
            fp = tuple(row[k] for k in ('initial_state_sha256', 'initial_model_sha256', 'state_bank_entry_sha256'))
            if key in fingerprints and fingerprints[key] != fp:
                raise RuntimeError('Actual physical states differ across methods or conditions')
            fingerprints[key] = fp
        merged[group] = rows
        n = len(rows)
        first = [r['first_success_step'] for r in rows if r['success_ever']]
        entry = dict(training_seed=seed, task=task, arm=arm, condition=condition, episodes=n,
                     ever=sum(r['success_ever'] for r in rows), final=sum(r['success_final'] for r in rows),
                     first_success_mean_conditional=None if not first else statistics.mean(first),
                     mean_inference_seconds=statistics.mean(r['inference_seconds'] for r in rows),
                     mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rows), shared_control=seed is None)
        table.append(entry)
        C.write(C.group_path(root, group) / 'merged.json', dict(summary=entry, rows=rows))
    if sum(t['episodes'] for t in table) != 3200:
        raise RuntimeError('Confirmation requires 3200 complete simulator episodes')
    paired = []
    for seed in C.SEEDS:
        for task in C.TASKS:
            for condition in C.COUNTS:
                for baseline in ('image', 'feature'):
                    a, b = merged[(seed, task, baseline, condition)], merged[(seed, task, 'decision', condition)]
                    paired.append(dict(training_seed=seed, task=task, condition=condition, baseline=baseline,
                                       ever=C.paired(a, b, 'success_ever'), final=C.paired(a, b, 'success_final')))
    primary = [p for p in paired if p['condition'] == 'lowlight']
    for field in ('ever', 'final'):
        for p, value in zip(primary, holm([p[field]['exact_two_sided_p'] for p in primary])):
            p[field]['holm_p_across_12_lowlight_tests'] = value
    aggregate = []
    for task in C.TASKS:
        for condition in C.COUNTS:
            for field in ('success_ever', 'success_final'):
                rates = {arm: [100 * statistics.mean(r[field] for r in merged[(s, task, arm, condition)])
                               for s in C.SEEDS] for arm in C.ARMS}
                contrasts = {}
                for baseline in ('image', 'feature'):
                    delta = [d - a for a, d in zip(rates[baseline], rates['decision'])]
                    by_state = [statistics.mean(int(merged[(s, task, 'decision', condition)][i][field]) -
                                                int(merged[(s, task, baseline, condition)][i][field]) for s in C.SEEDS)
                                for i in range(C.COUNTS[condition])]
                    contrasts[baseline] = dict(delta_mean_pp=statistics.mean(delta), delta_sample_sd_pp=statistics.stdev(delta),
                                               state_cluster_percentile95_conditional=cluster_interval(by_state))
                aggregate.append(dict(task=task, condition=condition, metric=field,
                                      rates={a: dict(mean_pct=statistics.mean(v), sample_sd_pct=statistics.stdev(v)) for a, v in rates.items()},
                                      decision_contrasts=contrasts))
    text = ['# New-state confirmation and supervision-form control', '',
            'Real MuJoCo / robosuite closed-loop task success; all episodes last 400 control steps. No seed or state exclusion.',
            '100 lowlight states (30000..30099), first 50 of the same bank for clean. Six new feature models; twelve prior image/decision models frozen.',
            'Encoder-feature loss was scaled using training-only initial gradients, never evaluation success. This is a supervision ablation, not an official SR4IR/EDTR replication.', '',
            '|Training seed|Task|Arm|Condition|Ever success|Final success|', '|---|---|---|---|---|---|']
    for r in table:
        label = 'shared' if r['training_seed'] is None else str(r['training_seed'])
        text.append(f'|{label}|{r["task"]}|{r["arm"]}|{r["condition"]}|{r["ever"]}/{r["episodes"]}|{r["final"]}/{r["episodes"]}|')
    text += ['', '## Lowlight paired comparisons', '',
             'Primary ever-success: Holm correction across 12 per-seed lowlight tests. Final-success is a separate secondary family. Clean comparisons are descriptive.',
             '|Seed|Task|Decision versus|Gains|Losses|Delta pp|Exact paired p|Holm p|', '|---|---|---|---|---|---|---|---|']
    for p in primary:
        e = p['ever']
        text.append(f'|{p["training_seed"]}|{p["task"]}|{p["baseline"]}|{e["gains"]}|{e["losses"]}|{e["delta_pp"]:+.1f}|{e["exact_two_sided_p"]:.6g}|{e["holm_p_across_12_lowlight_tests"]:.6g}|')
    text += ['', '## Across-training-seed rates', '',
             '|Task|Condition|Metric|Image mean +/- SD %|Feature mean +/- SD %|Decision mean +/- SD %|', '|---|---|---|---|---|---|']
    for r in aggregate:
        cells = [f'{r["rates"][a]["mean_pct"]:.2f} +/- {r["rates"][a]["sample_sd_pct"]:.2f}' for a in C.ARMS]
        text.append(f'|{r["task"]}|{r["condition"]}|{r["metric"]}|' + '|'.join(cells) + '|')
    text += ['', 'The JSON also contains state-cluster percentile intervals, conditional on the three frozen training seeds. Shared controls are counted once; 300 repeated lowlight rollouts are NOT 300 independent training runs.',
             'First-success time is conditional on success and must be interpreted with success rate, not alone. No assumption of significant superiority, real-camera calibration, precision-assembly quality, or hardware transfer.']
    C.write(root / 'summary.json', dict(results=table, paired_comparisons=paired, aggregate=aggregate,
                                      episodes=3200, physical_pairing_verified=True, protocol=locked))
    (root / 'RESULTS.md').write_text('\n'.join(text) + '\n')
    print('\n'.join(text), flush=True)
