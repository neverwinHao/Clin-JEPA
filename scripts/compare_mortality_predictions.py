"""Paired patient-clustered comparisons of held-out mortality predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compare(frame, draws=2000, seed=42):
    methods = ("history_32", "history_64", "history_32_plus_future_32")
    y = frame["label"].to_numpy()
    subjects = frame["subject_id"].to_numpy()
    scores = {m: frame[m].to_numpy() for m in methods}
    groups = [np.flatnonzero(subjects == p) for p in np.unique(subjects)]
    if draws < 1 or len(groups) < 2 or len(np.unique(y)) < 2:
        raise ValueError("Need positive draws, at least two patients and both outcome classes")
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("Nonfinite predictions")
    pairs = ("history_32", "history_64")
    deltas = {base: {"auroc": [], "auprc": []} for base in pairs}
    rng = np.random.default_rng(seed)
    skipped = 0
    for _ in range(draws):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picked])
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        sampled = {}
        for m in methods:
            sampled[m] = (roc_auc_score(y[idx], scores[m][idx]),
                          average_precision_score(y[idx], scores[m][idx]))
        for base in pairs:
            deltas[base]["auroc"].append(sampled[methods[-1]][0] - sampled[base][0])
            deltas[base]["auprc"].append(sampled[methods[-1]][1] - sampled[base][1])
    out = {}
    for base in pairs:
        out[base] = {}
        for metric, func in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            delta = func(y, scores[methods[-1]]) - func(y, scores[base])
            ci = np.quantile(deltas[base][metric], [.025, .975]).tolist() if deltas[base][metric] else None
            out[base][metric] = {"difference": float(delta), "ci95": ci}
    return {"draws": draws, "seed": seed, "skipped": skipped, "comparisons": out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    result = compare(pd.read_parquet(args.predictions), args.draws, args.seed)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
