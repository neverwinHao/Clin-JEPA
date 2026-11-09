"""Prepare sharded 8/24/48-hour hypothetical rollouts for all clinical tasks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.rollout import load_predictor
from clin_jepa.utils import load_config

HORIZONS = (8, 24, 48)


@torch.inference_mode()
def rollout_features(model, states, actions, static, policy, starts, device):
    """Use only the first 24 observed hours and a fixed synthetic future policy."""
    states, actions = states.to(device).float(), actions.to(device).float()
    static, policy = static.to(device).float(), policy.to(device).float()
    starts = starts.to(device).long()
    if states.shape[1] < 24 or actions.shape[1] < 24:
        raise ValueError("At least 24 observed hours are required")
    if len(starts) != len(states) or int((starts + 70).max()) >= len(policy):
        raise ValueError("Synthetic policy does not cover every future hour")
    z_hist, a_hist = states[:, :23], actions[:, :23]
    current = states[:, 23]
    generated, snapshots = [], []
    for step in range(max(HORIZONS)):
        next_action = actions[:, 23] if step == 0 else policy[starts + 23 + step]
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            next_state = model.forward_rollout_step(z_hist, static, a_hist, next_action, current)
        next_state = next_state.float()
        generated.append(next_state)
        if step + 1 in HORIZONS:
            window = torch.stack(generated, dim=1)
            snapshots.append(torch.stack((window.mean(dim=1), window.std(dim=1), window[:, -1]), dim=1))
        z_hist = torch.cat((z_hist, current.unsqueeze(1)), dim=1)
        a_hist = torch.cat((a_hist, next_action.unsqueeze(1)), dim=1)
        current = next_state
    return torch.stack(snapshots, dim=1).half().cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="configs/rollout_frozen_base.yaml")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--num-tasks", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.num_tasks < 1 or not 0 <= args.task_id < args.num_tasks or args.batch_size < 1:
        raise ValueError("Invalid worker or batch settings")
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_predictor(args.checkpoint, config["predictor"], device)
    policy = torch.load(args.policy, map_location="cpu", weights_only=False)["embeddings"]
    if policy.shape[1] != 4096 or not torch.isfinite(policy).all():
        raise ValueError("Invalid synthetic action vectors")
    for split in ("train", "val", "test"):
        original = pd.read_parquet(args.history / split / "index.parquet")
        selected = original.iloc[args.task_id::args.num_tasks].copy().reset_index(drop=True)
        if (selected.length < 24).any():
            raise ValueError(f"{split} has short history windows")
        selected.insert(0, "part_row", np.arange(len(selected)))
        folder = args.output / split
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"part_{args.task_id}.npy"
        index_path = folder / f"part_{args.task_id}.parquet"
        if dest.exists() or index_path.exists():
            raise FileExistsError(f"Refusing to overwrite {dest} or its index")
        temp = folder / f"part_{args.task_id}.npy.tmp"
        store = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16,
                                           shape=(len(selected), len(HORIZONS), 3, 4096))
        for shard_name, group in selected.groupby("shard", sort=False):
            shard = torch.load(args.data / "embeddings/qwen3_base" / split / shard_name,
                               map_location="cpu", weights_only=False, mmap=True)
            for start in range(0, len(group), args.batch_size):
                batch = group.iloc[start:start + args.batch_size]
                windows = [shard["windows"][int(i)] for i in batch["window_idx"]]
                state_at = torch.stack([w["emb_indices_state"][:24].long() for w in windows])
                action_at = torch.stack([w["emb_indices_action"][:24].long() for w in windows])
                states = shard["z_state_embeddings"][state_at]
                actions = shard["z_action_embeddings"][action_at]
                static = shard["z_statics"][torch.tensor([int(w["static_idx"]) for w in windows])]
                starts = torch.as_tensor(batch["start_step"].to_numpy())
                future = rollout_features(model, states, actions, static, policy, starts, device)
                if not torch.isfinite(future).all():
                    raise ValueError(f"Nonfinite rollout: {split}/{shard_name}")
                store[batch["part_row"].to_numpy()] = future.numpy()
            print(f"worker {args.task_id}: {split}/{shard_name} ({len(group)} windows)", flush=True)
        store.flush()
        del store
        temp.replace(dest)
        selected.to_parquet(index_path, index=False)
        print(f"worker {args.task_id}: {split} complete ({len(selected)} windows)", flush=True)


if __name__ == "__main__":
    main()
