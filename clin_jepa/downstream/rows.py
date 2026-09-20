"""Evaluation rows of the downstream tasks."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.downstream.tasks import CONTEXT


def scan(trajectory_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windows of at least 24 h, in shard order, and the number of hours the shards hold for every stay."""
    windows, stays = [], []
    for path in sorted((trajectory_dir / split).glob("trajectories_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        stay_ids = np.asarray(shard["per_stay"]["stay_ids"], dtype=np.int64)
        windows += [(int(stay_ids[w["stay_index"]]), int(w["start_step"]))
                    for w in shard["windows"] if int(w["length"]) >= CONTEXT]
        hours = pd.Series(np.asarray(shard["per_hour"]["stay_ids"], dtype=np.int64)).value_counts()
        stays.append(pd.DataFrame({"stay_id": stay_ids, "n_hours": hours.reindex(stay_ids).to_numpy()}))
    windows = pd.DataFrame(windows, columns=["stay_id", "start_step"])
    windows["target_hour"] = windows["start_step"] + CONTEXT - 1
    return windows, pd.concat(stays, ignore_index=True)


def hourly_rows(stays: pd.DataFrame) -> pd.DataFrame:
    """One row per hour of every stay from hour 23 on."""
    stays = stays[stays["n_hours"] >= CONTEXT].sort_values("stay_id")
    target = np.concatenate([np.arange(CONTEXT - 1, n) for n in stays["n_hours"]])
    rows = pd.DataFrame({"stay_id": np.repeat(stays["stay_id"].to_numpy(), stays["n_hours"] - CONTEXT + 1)})
    rows["start_step"], rows["target_hour"] = target - (CONTEXT - 1), target
    return rows


def with_labels(rows: pd.DataFrame, labels_dir: Path, tasks: list[str], hourly: bool) -> pd.DataFrame:
    """Rows with one float column per task; NaN where the label is undefined."""
    if hourly:
        table = pd.read_parquet(labels_dir / "hourly.parquet", columns=["stay_id", "hour"] + tasks)
        return rows.merge(table.rename(columns={"hour": "target_hour"}), on=["stay_id", "target_hour"], how="left",
                          validate="one_to_one")
    table = pd.read_parquet(labels_dir / "stay.parquet", columns=["stay_id"] + tasks)
    return rows.merge(table, on="stay_id", how="left", validate="one_to_one")
