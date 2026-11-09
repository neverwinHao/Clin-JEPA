"""Merge disjoint future-feature shards and verify alignment with historical rows."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--parts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-tasks', type=int, default=3)
    args = p.parse_args()
    if args.num_tasks < 1:
        raise ValueError('num-tasks must be positive')
    for split in ('train', 'val', 'test'):
        original = pd.read_parquet(args.history / split / 'index.parquet')
        if original.row.tolist() != list(range(len(original))):
            raise ValueError(f'Noncontiguous history row indices: {split}')
        folder = args.output / split
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / 'future.npy'
        tmp = folder / 'future.npy.tmp'
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite {dest}')
        target = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(len(original), 3, 3, 4096))
        seen = np.zeros(len(original), dtype=bool)
        for task in range(args.num_tasks):
            part = pd.read_parquet(args.parts / split / f'part_{task}.parquet')
            vals = np.load(args.parts / split / f'part_{task}.npy', mmap_mode='r')
            if vals.shape != (len(part), 3, 3, 4096) or vals.dtype != np.float16:
                raise ValueError(f'Invalid feature shape or dtype: {split}/{task}')
            if part.part_row.tolist() != list(range(len(part))):
                raise ValueError(f'Invalid part row order: {split}/{task}')
            rows = part.row.to_numpy(dtype=np.int64)
            if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(original)) or seen[rows].any():
                raise ValueError(f'Duplicate or invalid rows: {split}/{task}')
            columns = ['row', 'stay_id', 'start_step', 'length', 'shard', 'window_idx']
            if not part[columns].reset_index(drop=True).equals(original.iloc[rows][columns].reset_index(drop=True)):
                raise ValueError(f'Part index does not match histories: {split}/{task}')
            for start in range(0, len(rows), 128):
                block = np.asarray(vals[start:start + 128])
                if not np.isfinite(block).all():
                    raise ValueError(f'Nonfinite future features: {split}/{task}')
                target[rows[start:start + 128]] = block
            seen[rows] = True
        if not seen.all():
            raise ValueError(f'Missing future feature rows: {split}')
        target.flush()
        del target
        tmp.replace(dest)
        print(f'{split}: {int(seen.sum())} future rows aligned and finite', flush=True)


if __name__ == '__main__':
    main()
