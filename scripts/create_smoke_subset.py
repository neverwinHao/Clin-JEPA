"""Create a deterministic, isolated small subset of existing trajectory shards."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def select_shard(src, dst, count, seed):
    shard = torch.load(src, map_location="cpu", weights_only=False)
    stay_count = len(shard["per_stay"]["stay_ids"])
    if count < 1 or count > stay_count:
        raise ValueError(f"Requested {count} stays from {stay_count}")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(stay_count, count, replace=False))
    stay_map = {int(old): new for new, old in enumerate(selected)}
    windows = [dict(w) for w in shard["windows"] if int(w["stay_index"]) in stay_map]
    if set(int(w["stay_index"]) for w in windows) != set(map(int, selected)):
        raise ValueError("Every selected stay must have at least one window")
    hours = np.unique(np.concatenate([w["hour_indices"] for w in windows]))
    def take(value, indices):
        if isinstance(value, list):
            return [value[int(i)] for i in indices]
        return value[indices]
    for w in windows:
        w["stay_index"] = stay_map[int(w["stay_index"])]
        w["hour_indices"] = np.searchsorted(hours, w["hour_indices"]).astype(np.int32)
    subset = {
        "per_hour": {key: take(value, hours) for key, value in shard["per_hour"].items()},
        "per_stay": {key: take(value, selected) for key, value in shard["per_stay"].items()},
        "windows": windows,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite {dst}")
    torch.save(subset, dst)
    return {"source_shard": str(src), "stays": count, "windows": len(windows),
            "state_texts": len(hours), "unique_action_texts": len(set(subset["per_hour"]["action_texts"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-stays", type=int, default=32)
    parser.add_argument("--val-stays", type=int, default=16)
    args = parser.parse_args()
    results = {}
    for index, (split, count) in enumerate((("train", args.train_stays), ("val", args.val_stays))):
        files = sorted((args.source / split).glob("trajectories_*.pt"))
        if not files:
            raise FileNotFoundError(f"No shards for {split}")
        results[split] = select_shard(files[0], args.output / split / "trajectories_smoke.pt", count, args.seed + index)
    print(json.dumps({"seed": args.seed, "splits": results}, indent=2))


if __name__ == "__main__":
    main()
