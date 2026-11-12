"""Complete-coverage suite aggregation; seed variance and source CIs stay separate."""
from __future__ import annotations
from collections import defaultdict
import math
import statistics

from .suite_runner import require, digest


def components(rows):
    """TempCompass transformed clips share underlying videos, so union them."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for row in rows:
        ids = row.get('underlying_source_video_ids') or [row['source_video_id']]
        require(isinstance(ids, list) and ids and all(isinstance(x, str) and x for x in ids), 'Missing source identities')
        first = find(ids[0])
        for other in ids[1:]:
            a, b = find(first), find(other)
            parent[max(a, b)] = min(a, b)
    return {r['sample_id']: find((r.get('underlying_source_video_ids') or [r['source_video_id']])[0]) for r in rows}


def bootstrap(rows, field='correct', *, repeats=2000, seed=17):
    import numpy as np
    require(repeats >= 100, 'At least 100 source resamples required')
    grouped = defaultdict(list)
    mapping = components(rows)
    for row in rows:
        grouped[mapping[row['sample_id']]].append(float(row[field]))
    values = list(grouped.values())
    require(values and all(math.isfinite(v) for group in values for v in group), 'Nonfinite/empty score')
    point = sum(map(sum, values)) / sum(map(len, values))
    if len(values) < 2:
        return {'estimate': point, 'ci95': None, 'source_components': len(values), 'qa': len(rows)}
    sums, sizes = np.array([sum(x) for x in values]), np.array([len(x) for x in values])
    random = np.random.default_rng(seed)
    observed = []
    for _ in range(repeats):
        selected = random.integers(0, len(values), len(values))
        observed.append(float(sums[selected].sum() / sizes[selected].sum()))
    return {'estimate': point, 'ci95': np.quantile(observed, [.025, .975]).tolist(),
            'source_components': len(values), 'qa': len(rows)}


def paired(current, reference):
    old = {r['sample_id']: r for r in reference}
    require(len(old) == len(reference) and set(old) == {r['sample_id'] for r in current}, 'Paired sample coverage differs')
    out = []
    for row in current:
        ref = old[row['sample_id']]
        # record_sha256 may differ because F4/F16 media paths differ. Pair only
        # after label/source/question identity is independently frozen below.
        for key in ('source_video_id', 'answer', 'metric', 'split', 'question_identity_sha256',
                    'underlying_source_video_ids'):
            require(row.get(key) == ref.get(key), 'Paired benchmark identity differs: ' + key)
        out.append({**row, 'delta': float(row['correct']) - float(ref['correct'])})
    return out


def aggregate(plan, evaluations, *, repeats=2000):
    expected = {t['id']: t for t in plan['tasks'] if t['kind'] == 'evaluate'}
    require(set(evaluations) == set(expected), 'All declared evaluations are required; no dropped failures')
    per_model, scores, raw = {}, defaultdict(list), {}
    models = {m['id']: m for m in plan['models']}
    for tid, task in expected.items():
        value = evaluations[tid]
        require(value.get('status') == 'SUITE_EVALUATION_COMPLETE' and value.get('task_sha256') == digest(task)
                and value.get('failed_units') == 0 and value.get('suite_protocol') == task['protocol'], 'Unaccepted evaluation')
        rows = value['rows']
        count = 2697 if task['benchmark'] == 'videomme899' else 1580
        require(len(rows) == len({r['sample_id'] for r in rows}) == count and value.get('qa') == count, 'Incomplete evaluation')
        require(all(type(r.get('correct')) in (bool, int) and r['correct'] in (0, 1)
                    and r.get('question_identity_sha256') for r in rows), 'Measured binary scores and question identities required')
        result = bootstrap(rows, repeats=repeats)
        mid, benchmark = task['model_id'], task['benchmark']
        per_model.setdefault(mid, {})[benchmark] = result
        raw[(mid, benchmark)] = rows
        if models[mid]['group'] == 'core':
            scores[(models[mid]['family'], benchmark)].append((models[mid]['seed'], result['estimate']))
    seeds = {}
    for (family, benchmark), values in scores.items():
        require(sorted(seed for seed, _ in values) == [17, 29, 43], 'All three declared training seeds required')
        rates = [score for _, score in values]
        seeds.setdefault(family, {})[benchmark] = {'seeds': sorted(values), 'mean_accuracy': statistics.mean(rates),
            'sample_std_across_training_seeds': statistics.stdev(rates), 'num_training_seeds': 3,
            'scope': 'training seed variability; distinct from source bootstrap uncertainty'}
    comparisons = {}
    for model in plan['models']:
        if model['group'] == 'ablation':
            reference = 'ouro_r4_s17'
        elif model['family'] != 'ouro_r1':
            reference = 'ouro_r1_s' + str(model['seed'])
        else:
            continue
        comparisons[model['id']] = {'reference_model': reference, 'benchmarks': {}}
        for benchmark in ('videomme899', 'tempcompass_mc'):
            rows = paired(raw[(model['id'], benchmark)], raw[(reference, benchmark)])
            comparisons[model['id']]['benchmarks'][benchmark] = bootstrap(rows, 'delta', repeats=repeats)
    return {'schema_version': 1, 'status': 'SUITE_DECLARED_RESULTS_COMPLETE', 'models': per_model,
            'training_seed_statistics': seeds, 'paired_accuracy_deltas': comparisons,
            'bootstrap_resamples': repeats, 'bootstrap_seed': 17,
            'bootstrap_unit': 'source video; union of shared underlying sources for TempCompass',
            'no_cross_benchmark_combined_accuracy': True, 'heldout_selection_used': False,
            'conditional_work_not_claimed_complete': plan['scope']['conditional_work_not_claimed_complete']}
