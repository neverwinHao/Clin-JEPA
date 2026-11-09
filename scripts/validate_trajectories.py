"""Validate patient splits and hourly trajectory shards without printing patient data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    cohort = pd.read_csv(args.data / "cohort/cohort.csv")
    splits = pd.read_csv(args.data / "splits/splits.csv")
    if cohort["stay_id"].duplicated().any() or splits["stay_id"].duplicated().any():
        raise ValueError("Duplicate ICU stays in cohort or split")
    if set(cohort["stay_id"]) != set(splits["stay_id"]):
        raise ValueError("Cohort and split ICU stays do not match")
    if (splits.groupby("subject_id")["split"].nunique() > 1).any():
        raise ValueError("A patient appears in multiple splits")
    stay_split = dict(zip(splits["stay_id"], splits["split"]))
    count = {"train": 0, "val": 0, "test": 0}
    seen_stays = set()
    for split in count:
        files = sorted((args.data / "trajectories" / split).glob("trajectories_*.pt"))
        if not files:
            raise ValueError(f"No trajectory shards for {split}")
        for path in files:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            hour = shard["per_hour"]
            stay = shard["per_stay"]
            length = len(hour["state_texts"])
            if any(len(hour[key]) != length for key in ("action_texts", "stay_ids", "hours", "sofa")):
                raise ValueError(f"Hourly arrays have unequal length in {path.name}")
            if any(len(stay[key]) != len(stay["stay_ids"]) for key in ("ages", "genders", "races")):
                raise ValueError(f"Stay arrays have unequal length in {path.name}")
            if any(stay_split.get(int(s)) != split for s in stay["stay_ids"]):
                raise ValueError(f"Incorrect patient split in {path.name}")
            shard_stays = {int(s) for s in stay["stay_ids"]}
            if len(shard_stays) != len(stay["stay_ids"]) or shard_stays & seen_stays:
                raise ValueError(f"Duplicate stays across shards in {path.name}")
            seen_stays.update(shard_stays)
            for window in shard["windows"]:
                indices = np.asarray(window["hour_indices"])
                n = int(window["length"])
                si = int(window["stay_index"])
                if not 6 <= n <= 72 or len(indices) != n or si < 0 or si >= len(stay["stay_ids"]):
                    raise ValueError(f"Invalid trajectory window in {path.name}")
                if np.any(indices < 0) or np.any(indices >= length):
                    raise ValueError(f"Invalid hour indices in {path.name}")
                if np.any(hour["stay_ids"][indices] != stay["stay_ids"][si]):
                    raise ValueError(f"Window crosses stays in {path.name}")
                if not np.array_equal(hour["hours"][indices], np.arange(window["start_step"], window["start_step"] + n)):
                    raise ValueError(f"Nonconsecutive hours in {path.name}")
                count[split] += 1
        print(f"{split}: {len(files)} shards, {count[split]} windows", flush=True)
    print(f"Cohort: {len(cohort)} stays; trajectory stays: {len(seen_stays)}; windows: {sum(count.values())}", flush=True)
    print("Patient splits, window indices, and hourly consistency: OK", flush=True)


if __name__ == "__main__":
    main()
