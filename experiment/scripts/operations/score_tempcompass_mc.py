#!/usr/bin/env python3
"""Deterministic TempCompass MC scoring matching the pinned --disable_llm rules.

This independent scorer consumes complete, identity-bound prediction JSONL.
No network/LLM judge, answer repair, dropped failures, or model selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

OFFICIAL_SCORER_SHA = 'ccffa4f717a3e2a29435b2bbe440109bf2def589cea73fde9fd44fd262fb379a'
OFFICIAL_SUMMARY_SHA = '514f728e6a4e273db7db2c3feab316c129ebcc08901e62352d5af71efb4e964c'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def official_rule(prediction, answer):
    if prediction is None:
        return {'correct': False, 'match_success': False}
    if not isinstance(prediction, str):
        raise ValueError('Prediction must be text or null')
    prediction = prediction.replace('</s>', '').strip()
    if prediction == answer:
        return {'correct': True, 'match_success': True}
    if prediction in ('A', 'B', 'C', 'D'):
        return {'correct': prediction == answer[0], 'match_success': True}
    if any(prediction.startswith(x) for x in ('A.', 'B.', 'C.', 'D.')):
        return {'correct': prediction.split('.')[0] == answer[0], 'match_success': True}
    if any(prediction.startswith(x) for x in ('A)', 'B)', 'C)', 'D)')):
        return {'correct': prediction.split(')')[0] == answer[0], 'match_success': True}
    return {'correct': False, 'match_success': False}


def score(records, predictions):
    ids = [r['sample_id'] for r in records]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Manifest sample IDs must be unique and nonempty')
    predicted = {p['sample_id']: p for p in predictions}
    if len(predicted) != len(predictions) or set(predicted) != set(ids):
        raise ValueError('Predictions must cover the exact sample set once')
    rows, dimensions, fine = [], defaultdict(list), defaultdict(list)
    for row in records:
        pred = predicted[row['sample_id']]
        record_sha = hashlib.sha256(canonical(row)).hexdigest()
        if pred.get('manifest_row_sha256') != record_sha or 'prediction' not in pred:
            raise ValueError('Prediction identity is not bound to the complete manifest row')
        result = official_rule(pred['prediction'], row['original_answer'])
        out = {'sample_id': row['sample_id'], 'video_id': row['video_id'],
               'source_video_id': row['source_video_id'], 'underlying_source_video_ids': row['underlying_source_video_ids'],
               'task_type': row['task_type'], 'fine_grained_task_type': row['fine_grained_task_type'],
               'prediction': pred['prediction'], 'answer': row['answer'], 'manifest_row_sha256': record_sha, **result}
        rows.append(out)
        dimensions[row['task_type']].append(out)
        fine[row['fine_grained_task_type']].append(out)

    def metrics(items):
        count = len(items)
        correct, matched = sum(x['correct'] for x in items), sum(x['match_success'] for x in items)
        return {'count': count, 'correct': correct, 'accuracy': correct / count,
                'official_accuracy_percent_1dp': round(correct * 100 / count, 1),
                'rule_match_count': matched, 'rule_match_rate': matched / count}

    groups = {k: metrics(v) for k, v in sorted(dimensions.items())}
    summary = {'protocol': 'tempcompass-official-disable-llm-v1', 'overall': metrics(rows), 'dimensions': groups,
               'fine_grained_dimensions': {k: metrics(v) for k, v in sorted(fine.items())},
               'dimension_macro_accuracy': sum(x['accuracy'] for x in groups.values()) / len(groups),
               'official_avg_is_qa_micro_accuracy': True, 'all_predictions_included': True,
               'official_scorer_sha256': OFFICIAL_SCORER_SHA, 'official_summary_sha256': OFFICIAL_SUMMARY_SHA,
               'source_component_counts': dict(Counter(x['source_video_id'] for x in rows)),
               'model_provenance_admission_performed': False,
               'quality_claim_scope': 'Scores only the SHA-bound predictions. The evaluation runner must separately attest model and inference provenance.'}
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--predictions-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    data = []
    for path, expected in [(args.manifest, args.manifest_sha256), (args.predictions, args.predictions_sha256)]:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Regular input files required')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Pinned input SHA differs')
        data.append([json.loads(s) for s in raw.decode().splitlines()])
    if len(data[0]) != 1580 or len({x['video_id'] for x in data[0]}) != 410:
        raise ValueError('Complete frozen 1580 / 410 set required')
    rows, summary = score(*data)
    summary.update({'manifest_sha256': args.manifest_sha256, 'predictions_sha256': args.predictions_sha256})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'scored_predictions.jsonl').open('x') as f:
        for row in rows:
            f.write(canonical(row).decode() + '\n')
    with (args.output_dir / 'SCORES.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
