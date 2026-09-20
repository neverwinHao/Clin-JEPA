"""Trajectory windows as raw text: per-hour state and action texts plus the demographics text."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


GENDER = {"M": "Male", "F": "Female"}


def demographics_text(age: int, gender: str, race: str) -> str:
    return f"Age: {age} years. Gender: {GENDER[gender]}. Race: {race}."


class TrajectoryWindowDataset(Dataset):
    """Windows of the trajectory shards under ``shard_dir/<split>``."""

    def __init__(
        self,
        shard_dir: str | Path,
        split: str = "train",
        max_windows: int | None = None,
    ) -> None:
        shard_dir = Path(shard_dir) / split
        shard_files = sorted(shard_dir.glob("*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"no .pt shards in {shard_dir}")

        self.shards: list[dict] = []
        self.window_refs: list[dict] = []
        for path in shard_files:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            shard_idx = len(self.shards)
            self.shards.append(shard)
            for w in shard["windows"]:
                self.window_refs.append({
                    "_shard_idx": shard_idx,
                    "stay_index": int(w["stay_index"]),
                    "length": int(w["length"]),
                    "hour_indices": w["hour_indices"],
                })
                if max_windows and len(self.window_refs) >= max_windows:
                    break
            if max_windows and len(self.window_refs) >= max_windows:
                break

        logger.info("%d windows from %d shards (%s)", len(self.window_refs), len(self.shards), shard_dir)

    def __len__(self) -> int:
        return len(self.window_refs)

    def __getitem__(self, idx: int) -> dict:
        w = self.window_refs[idx]
        shard = self.shards[w["_shard_idx"]]
        per_hour, per_stay = shard["per_hour"], shard["per_stay"]
        hours = w["hour_indices"].tolist()
        stay = w["stay_index"]
        return {
            "length": w["length"],
            "state_texts": [per_hour["state_texts"][h] for h in hours],
            "action_texts": [per_hour["action_texts"][h] for h in hours],
            "demographics_text": demographics_text(
                int(round(float(per_stay["ages"][stay]))),
                per_stay["genders"][stay],
                per_stay["races"][stay],
            ),
        }


def trajectory_collate_fn(batch: list[dict]) -> dict:
    return {"batch": batch}
