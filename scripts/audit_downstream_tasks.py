"""Count usable labels and patients for each author's downstream task."""
import argparse
import json
from pathlib import Path

import pandas as pd

from clin_jepa.downstream.tasks import TASKS, CONTEXT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    split_table = pd.read_csv(args.splits, usecols=['stay_id', 'subject_id', 'split'])
    if split_table.stay_id.duplicated().any() or (split_table.groupby('subject_id')['split'].nunique() > 1).any():
        raise ValueError('Invalid patient split')
    hourly = pd.read_parquet(args.labels / 'hourly.parquet')
    stay = pd.read_parquet(args.labels / 'stay.parquet')
    records = []
    for task in TASKS:
        record = {'task': task.name, 'benchmark': task.benchmark, 'horizon': task.horizon, 'hourly': task.hourly}
        for split in ('train', 'val', 'test'):
            index = pd.read_parquet(args.history / split / 'index.parquet')
            if not task.hourly:
                index = index[index.start_step == 0]
            index = index.merge(split_table, on='stay_id', how='left', validate='many_to_one')
            if index.subject_id.isna().any() or (index['split'] != split).any():
                raise ValueError(f'Patient split mismatch: {task.name}/{split}')
            if task.hourly:
                index['hour'] = index.start_step + CONTEXT - 1
                joined = index.merge(hourly[['stay_id', 'hour', task.name]], on=['stay_id', 'hour'], how='left', validate='one_to_one')
            else:
                joined = index.merge(stay[['stay_id', task.name]], on='stay_id', how='left', validate='one_to_one')
            valid = joined[joined[task.name].notna()]
            if not set(valid[task.name].unique()).issubset({0., 1.}):
                raise ValueError(f'Nonbinary labels: {task.name}/{split}')
            positives = valid[valid[task.name] == 1]
            negatives = valid[valid[task.name] == 0]
            record[split] = {'rows': len(valid), 'patients': int(valid.subject_id.nunique()),
                             'positive_rows': len(positives), 'negative_rows': len(negatives),
                             'positive_patients': int(positives.subject_id.nunique()),
                             'negative_patients': int(negatives.subject_id.nunique())}
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n')
    for r in records:
        print(r['benchmark'], r['task'], 'train/val/test positives:', *(r[s]['positive_rows'] for s in ('train','val','test')),
              'test positive patients:', r['test']['positive_patients'])


if __name__ == '__main__':
    main()
