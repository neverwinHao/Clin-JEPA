"""Sample patients deterministically from the original split and preserve their trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def subset_shard(shard: dict, selected_stay_ids: set[int]) -> dict | None:
    """Keep selected stays and their original windows; remap stay and hour indices."""
    stays = shard["per_stay"]["stay_ids"]
    selected = np.array([i for i, stay in enumerate(stays) if int(stay) in selected_stay_ids], dtype=np.int64)
    if len(selected) == 0:
        return None
    stay_map = {int(old): new for new, old in enumerate(selected)}
    windows = [dict(w) for w in shard["windows"] if int(w["stay_index"]) in stay_map]
    if not windows or set(int(w["stay_index"]) for w in windows) != set(map(int, selected)):
        raise ValueError("Every selected stay must have at least one window")
    hours = np.unique(np.concatenate([w["hour_indices"] for w in windows]))

    def take(value, indices):
        return [value[int(i)] for i in indices] if isinstance(value, list) else value[indices]

    for w in windows:
        w["stay_index"] = stay_map[int(w["stay_index"])]
        w["hour_indices"] = np.searchsorted(hours, w["hour_indices"]).astype(np.int32)
    return {
        "per_hour": {key: take(value, hours) for key, value in shard["per_hour"].items()},
        "per_stay": {key: take(value, selected) for key, value in shard["per_stay"].items()},
        "windows": windows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Full processed data directory")
    p.add_argument("--output", type=Path, required=True, help="New separate output directory")
    p.add_argument("--train-patients", type=int, default=1000)
    p.add_argument("--val-patients", type=int, default=200)
    p.add_argument("--test-patients", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cohort = pd.read_csv(args.source / "cohort/cohort.csv")
    splits = pd.read_csv(args.source / "splits/splits.csv")
    if (splits.groupby("subject_id")["split"].nunique() > 1).any():
        raise ValueError("Source contains patients in more than one split")
    rng = np.random.default_rng(args.seed)
    selected_split = []
    counts = {}
    for split, requested in (("train", args.train_patients), ("val", args.val_patients), ("test", args.test_patients)):
        part = splits[splits["split"] == split]
        subjects = np.sort(part["subject_id"].unique())
        if requested < 1 or requested > len(subjects):
            raise ValueError(f"Invalid patient count for {split}")
        chosen = rng.choice(subjects, size=requested, replace=False)
        selection = part[part["subject_id"].isin(chosen)].copy()
        selected_split.append(selection)
        stay_ids = set(map(int, selection["stay_id"]))
        seen = set()
        total = {"patients": requested, "stays": len(stay_ids), "shards": 0, "windows": 0,
                 "state_texts": 0, "unique_action_texts": 0}
        for path in sorted((args.source / "trajectories" / split).glob("trajectories_*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            result = subset_shard(shard, stay_ids)
            if result is None:
                continue
            dest = args.output / "trajectories" / split / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(".pt.tmp")
            torch.save(result, temp)
            temp.replace(dest)
            seen.update(map(int, result["per_stay"]["stay_ids"]))
            total["shards"] += 1
            total["windows"] += len(result["windows"])
            total["state_texts"] += len(result["per_hour"]["state_texts"])
            total["unique_action_texts"] += len(set(result["per_hour"]["action_texts"]))
        if seen != stay_ids:
            raise ValueError(f"Selected stays missing from {split} trajectory shards")
        counts[split] = total
        print(f"{split}: {total}", flush=True)
    selected_split = pd.concat(selected_split, ignore_index=True)
    chosen_stays = set(selected_split["stay_id"])
    chosen_cohort = cohort[cohort["stay_id"].isin(chosen_stays)]
    if len(chosen_cohort) != len(chosen_stays):
        raise ValueError("Selected stay mismatch in cohort")
    (args.output / "cohort").mkdir()
    (args.output / "splits").mkdir()
    chosen_cohort.to_csv(args.output / "cohort/cohort.csv", index=False)
    selected_split.to_csv(args.output / "splits/splits.csv", index=False)
    (args.output / "sampling_manifest.json").write_text(json.dumps({"seed": args.seed, "splits": counts}, indent=2) + "\n")
    (args.output / ".progress").mkdir()


if __name__ == "__main__":
    main()
