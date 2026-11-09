"""Patient-clustered uncertainty for the rollout model/copy-forward L1 ratio."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.draws < 1:
        raise ValueError("draws must be positive")
    index = pd.read_parquet(args.rollout / "index.parquet")
    splits = pd.read_csv(args.splits)
    subjects = index.merge(splits[["stay_id", "subject_id"]], on="stay_id", how="left", validate="many_to_one")["subject_id"]
    if subjects.isna().any():
        raise ValueError("Unmatched stays in rollout index")
    subjects = subjects.to_numpy()
    errors = np.load(args.rollout / "errors.npz")
    if len(subjects) != errors["model_l1"].shape[0]:
        raise ValueError("Rollout index and error rows differ")
    rng = np.random.default_rng(args.seed)
    out = {}
    for horizon in (1, 8, 24, 48):
        model = errors["model_l1"][:, horizon - 1]
        baseline = errors["copy_forward_l1"][:, horizon - 1]
        valid = np.isfinite(model) & np.isfinite(baseline)
        ids, inverse = np.unique(subjects[valid], return_inverse=True)
        if len(ids) == 0:
            out[str(horizon)] = {"patients": 0, "windows": 0, "mase_ci95": None}
            continue
        sums_model = np.bincount(inverse, weights=model[valid])
        sums_base = np.bincount(inverse, weights=baseline[valid])
        picks = rng.integers(0, len(ids), size=(args.draws, len(ids)))
        denominator = sums_base[picks].sum(axis=1)
        ratios = np.divide(sums_model[picks].sum(axis=1), denominator,
                           out=np.full(args.draws, np.nan), where=denominator > 0)
        finite = ratios[np.isfinite(ratios)]
        ci = np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None
        out[str(horizon)] = {"patients": len(ids), "windows": int(valid.sum()), "mase_ci95": ci}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "draws": args.draws, "horizons": out}, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
