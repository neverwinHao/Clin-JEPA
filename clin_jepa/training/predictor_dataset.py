"""Trajectory windows as cached embeddings, for training a predictor on a frozen encoder."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PredictorDataset(Dataset):
    """Trajectory windows of one split of an embedding cache."""

    def __init__(
        self,
        embedding_dir: str | Path,
        split: str = "train",
        max_windows: int | None = None,
    ) -> None:
        embedding_dir = Path(embedding_dir) / split
        shard_files = sorted(embedding_dir.glob("*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No .pt shards in {embedding_dir}")

        logger.info("Loading %d embedding shards from %s", len(shard_files), embedding_dir)
        self.shards: list[dict] = []
        self.windows: list[dict] = []

        for f in shard_files:
            shard = torch.load(f, map_location="cpu", weights_only=False)
            shard_idx = len(self.shards)
            self.shards.append(shard)

            for w in shard["windows"]:
                w_copy = dict(w)
                w_copy["_shard_idx"] = shard_idx
                self.windows.append(w_copy)

                if max_windows and len(self.windows) >= max_windows:
                    break

            if max_windows and len(self.windows) >= max_windows:
                break

        logger.info(
            "PredictorDataset: %d windows from %d shards (%s split)",
            len(self.windows), len(self.shards), split,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        shard = self.shards[w["_shard_idx"]]

        z_states = shard["z_state_embeddings"][w["emb_indices_state"]]
        z_actions = shard["z_action_embeddings"][w["emb_indices_action"]]
        z_static = shard["z_statics"][w["static_idx"]]

        return {
            "z_states":  z_states,
            "z_actions": z_actions,
            "z_static":  z_static,
            "length":    int(w["length"]),
        }


def predictor_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length trajectories: pad to batch max + valid mask."""
    B = len(batch)
    T_max = max(item["length"] for item in batch)
    state_dim = batch[0]["z_states"].shape[1]
    action_dim = batch[0]["z_actions"].shape[1]

    z_states = torch.zeros(B, T_max, state_dim, dtype=batch[0]["z_states"].dtype)
    z_actions = torch.zeros(B, T_max, action_dim, dtype=batch[0]["z_actions"].dtype)
    z_static = torch.stack([item["z_static"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    padding_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["length"]
        z_states[i, :L] = item["z_states"]
        z_actions[i, :L] = item["z_actions"]
        padding_mask[i, :L] = True

    return {
        "z_states":     z_states,
        "z_actions":    z_actions,
        "z_static":     z_static,
        "lengths":      lengths,
        "padding_mask": padding_mask,
    }
