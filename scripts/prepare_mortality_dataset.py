"""Prepare patient-disjoint mortality features and labels for a downstream probe."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--future", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    mortality = pd.read_parquet(args.labels, columns=["stay_id", "hospital_mortality"])
    splits = pd.read_csv(args.splits, usecols=["stay_id", "subject_id", "split"])
    if splits.stay_id.duplicated().any() or mortality.stay_id.duplicated().any():
        raise ValueError("Duplicate ICU stays in labels or splits")
    info = {}
    subjects = {}
    args.output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        history_index = pd.read_parquet(args.history / split / "index.parquet")
        future_index = pd.read_parquet(args.future / split / "index.parquet")
        if future_index.feature_row.tolist() != list(range(len(future_index))):
            raise ValueError(f"{split} future feature rows are not contiguous")
        if future_index.stay_id.duplicated().any() or (future_index.start_step != 0).any():
            raise ValueError(f"{split} must contain one admission window per stay")
        check = future_index.merge(
            history_index[["row", "stay_id", "start_step"]],
            on=["row", "stay_id", "start_step"], how="left", validate="one_to_one", indicator=True)
        if (check["_merge"] != "both").any():
            raise ValueError(f"{split} histories and futures do not align")
        selected = future_index.merge(mortality, on="stay_id", how="left", validate="one_to_one")
        selected = selected.merge(splits, on="stay_id", how="left", validate="one_to_one")
        if selected["subject_id"].isna().any() or (selected["split"] != split).any():
            raise ValueError(f"{split} does not match the patient split")
        if selected.hospital_mortality.isna().any():
            raise ValueError(f"{split} has missing mortality labels")
        if not set(selected.hospital_mortality.unique()).issubset({0.0, 1.0}):
            raise ValueError(f"{split} has invalid mortality labels")
        history = np.load(args.history / split / "history.npy", mmap_mode="r")
        future = np.load(args.future / split / "future.npy", mmap_mode="r")
        if len(future) != len(future_index):
            raise ValueError(f"{split} future arrays and index differ")
        x_hist = np.ascontiguousarray(history[selected["row"].to_numpy(dtype=np.int64)])
        x_future = np.ascontiguousarray(future[selected["feature_row"].to_numpy(dtype=np.int64)])
        if x_hist.shape[1:] != (7, 4096) or x_future.shape[1:] != (3, 4096):
            raise ValueError(f"{split} has invalid feature shapes")
        if not np.isfinite(x_hist).all() or not np.isfinite(x_future).all():
            raise ValueError(f"{split} has nonfinite features")
        y = selected.hospital_mortality.to_numpy(dtype=np.int8)
        patient_ids = selected.subject_id.to_numpy(dtype=np.int64)
        subjects[split] = set(patient_ids)
        np.savez(args.output / f"{split}.npz", history=x_hist, future=x_future, labels=y,
                 subject_ids=patient_ids, stay_ids=selected.stay_id.to_numpy(dtype=np.int64))
        info[split] = {"stays": len(y), "patients": len(subjects[split]),
                       "positive_stays": int(y.sum()), "negative_stays": int(len(y) - y.sum())}
        print(f"{split}: {info[split]}", flush=True)
    if subjects["train"] & subjects["val"] or subjects["train"] & subjects["test"] or subjects["val"] & subjects["test"]:
        raise ValueError("A patient appears across splits")
    metadata = {"task": "hospital mortality after first 24 hours in ICU",
                "history": "First 24 hours of recorded states/actions and demographics",
                "future": "24-step prediction under synthetic no-interventions after hour 23",
                "warning": "Hypothetical action policy, not a medical recommendation or validated counterfactual.",
                "splits": info}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Patient splits are disjoint; no future observations or actual future treatments used.", flush=True)


if __name__ == "__main__":
    main()
