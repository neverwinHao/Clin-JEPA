"""Prepare patient-clustered rollout curves; no patient records are exported."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--baselines', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.draws < 1:
        raise ValueError('draws must be positive')
    index = pd.read_parquet(args.rollout / 'index.parquet')
    splits = pd.read_csv(args.splits, usecols=['stay_id', 'subject_id'])
    subjects = index.merge(splits, on='stay_id', how='left', validate='many_to_one')['subject_id']
    if subjects.isna().any():
        raise ValueError('Unmatched ICU stays in rollout index')
    subjects = subjects.to_numpy()
    e = np.load(args.rollout / 'errors.npz')
    extra = np.load(args.baselines)
    matrices = {'predictor': e['model_l1'], 'copy_forward': e['copy_forward_l1'],
                'context_mean': extra['context_l1'], 'train_mean': extra['train_mean_l1']}
    if any(v.shape != matrices['predictor'].shape for v in matrices.values()) or len(subjects) != len(index):
        raise ValueError('Curve inputs and index do not align')
    rng = np.random.default_rng(args.seed)
    records = []
    for h in range(matrices['predictor'].shape[1]):
        valid = np.logical_and.reduce([np.isfinite(v[:, h]) for v in matrices.values()])
        ids, inverse = np.unique(subjects[valid], return_inverse=True)
        if len(ids) == 0:
            continue
        n = np.bincount(inverse)
        draws = rng.integers(0, len(ids), size=(args.draws, len(ids)))
        counts = n[draws].sum(axis=1)
        record = {'horizon': h + 1, 'windows': int(valid.sum()), 'patients': len(ids)}
        means = {}
        for name, matrix in matrices.items():
            value = matrix[valid, h]
            totals = np.bincount(inverse, weights=value)
            means[name] = totals[draws].sum(axis=1) / counts
            record[name + '_l1'] = float(value.mean())
            record[name + '_ci_low'], record[name + '_ci_high'] = map(float, np.quantile(means[name], [.025, .975]))
        ratio = means['predictor'] / means['context_mean']
        record['improvement_vs_context'] = 1 - record['predictor_l1'] / record['context_mean_l1']
        ci = np.quantile(1 - ratio, [.025, .975])
        record['improvement_ci_low'], record['improvement_ci_high'] = map(float, ci)
        records.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output / 'rollout_curves_bootstrap.csv', index=False)
    (args.output / 'rollout_curves_bootstrap.json').write_text(json.dumps({'seed': args.seed, 'draws': args.draws, 'curves': records}, indent=2) + '\n')
    for r in records:
        if r['horizon'] in (1, 8, 24, 48):
            print('h={horizon}: n={windows}, patients={patients}, predictor={predictor_l1:.2f}, context={context_mean_l1:.2f}, relative improvement={improvement_vs_context:.2%}'.format(**r))


if __name__ == '__main__':
    main()
