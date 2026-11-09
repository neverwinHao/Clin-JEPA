"""Verify embedding caches against trajectory shards without revealing records."""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--encoder", default="qwen3_base")
    p.add_argument("--dim", type=int, default=4096)
    args = p.parse_args()
    for split in ("train", "val", "test"):
        paths = sorted((args.data / "trajectories" / split).glob("trajectories_*.pt"))
        if not paths:
            raise ValueError(f"No trajectories for {split}")
        windows = 0
        for path in paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            cache_path = args.data / "embeddings" / args.encoder / split / path.name.replace("trajectories_", "embeddings_")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            action_map = {}
            action_of_hour = [action_map.setdefault(text, len(action_map)) for text in shard["per_hour"]["action_texts"]]
            action_of_hour = np.asarray(action_of_hour)
            expected = {"z_state_embeddings": len(shard["per_hour"]["state_texts"]),
                        "z_action_embeddings": len(action_map), "z_statics": len(shard["per_stay"]["stay_ids"])}
            for key, count in expected.items():
                value = cached[key]
                if value.shape != (count, args.dim) or value.dtype != torch.float16 or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid {key} in {cache_path}")
            if len(shard["windows"]) != len(cached["windows"]):
                raise ValueError(f"Window count mismatch in {cache_path}")
            for src, out in zip(shard["windows"], cached["windows"]):
                hours = np.asarray(src["hour_indices"], dtype=np.int64)
                if int(out["length"]) != int(src["length"]) or int(out["start_step"]) != int(src["start_step"]):
                    raise ValueError(f"Window mismatch in {cache_path}")
                if int(out["static_idx"]) != int(src["stay_index"]) or int(out["stay_id"]) != int(shard["per_stay"]["stay_ids"][src["stay_index"]]):
                    raise ValueError(f"Stay mismatch in {cache_path}")
                if not np.array_equal(out["emb_indices_state"].numpy(), hours):
                    raise ValueError(f"State index mismatch in {cache_path}")
                if not np.array_equal(out["emb_indices_action"].numpy(), action_of_hour[hours]):
                    raise ValueError(f"Action index mismatch in {cache_path}")
                windows += 1
        print(f"{split}: {len(paths)} caches, {windows} windows, dimensions/finiteness/indices OK", flush=True)


if __name__ == "__main__":
    main()
