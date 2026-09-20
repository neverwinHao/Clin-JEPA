"""Baseline inputs: the 24 hourly rows of an evaluation row, as summary statistics or as a sequence."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

STATISTICS = ("mean", "last", "min", "max", "std", "trend")
CHUNK = 20000


class HourlyTable:
    """Hourly rows and demographics of every stay of a split, addressed by ``(stay_id, hour)``."""

    def __init__(self, tabular_dir: Path, split: str) -> None:
        shards = [torch.load(p, map_location="cpu", weights_only=False)
                  for p in sorted((tabular_dir / split).glob("tabular_*.pt"))]
        stay_ids = np.concatenate([np.asarray(s["stay_ids"], dtype=np.int64) for s in shards])
        lengths = np.array([len(h) for s in shards for h in s["hourly"]], dtype=np.int64)
        self.values = np.concatenate([np.asarray(h, dtype=np.float32) for s in shards for h in s["hourly"]])
        self.demographics = np.concatenate([np.asarray(s["statics"], dtype=np.float32) for s in shards])
        self.position = pd.Series(np.arange(len(stay_ids)), index=stay_ids)
        self.offsets, self.lengths = np.cumsum(lengths) - lengths, lengths

    def windows(self, rows: pd.DataFrame, context: int) -> np.ndarray:
        """``(n, context, columns)`` float32, NaN in hours after the end of the stay."""
        stay = self.position.loc[rows["stay_id"].to_numpy()].to_numpy()
        hours = rows["start_step"].to_numpy(dtype=np.int64)[:, None] + np.arange(context)
        inside = hours < self.lengths[stay][:, None]
        out = np.full((len(rows), context, self.values.shape[1]), np.nan, dtype=np.float32)
        out[inside] = self.values[(self.offsets[stay][:, None] + hours)[inside]]
        return out

    def statics(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asfortranarray(self.demographics[self.position.loc[rows["stay_id"].to_numpy()].to_numpy()])


def summaries(windows: np.ndarray) -> np.ndarray:
    """Six statistics per column (all of column 0, then column 1, ...) followed by each column's observed hours."""
    n, hours, columns = windows.shape
    observed = np.isfinite(windows)
    ever = observed.any(axis=1)
    first_at, last_at = observed.argmax(axis=1), hours - 1 - observed[:, ::-1].argmax(axis=1)
    first = np.where(ever, np.take_along_axis(windows, first_at[:, None], axis=1)[:, 0], np.nan)
    last = np.where(ever, np.take_along_axis(windows, last_at[:, None], axis=1)[:, 0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = {"mean": np.nanmean(windows, axis=1), "last": last, "min": np.nanmin(windows, axis=1),
                  "max": np.nanmax(windows, axis=1), "std": np.nanstd(windows, axis=1), "trend": last - first}
    statistics = np.stack([values[name] for name in STATISTICS], axis=2).astype(np.float32)
    return np.concatenate([statistics.reshape(n, -1), observed.sum(axis=1).astype(np.float32)], axis=1)


def sequences(windows: np.ndarray) -> np.ndarray:
    return np.concatenate([windows, np.isfinite(windows).astype(np.float32)], axis=2)


def tabular_features(table: HourlyTable, rows: pd.DataFrame, context: int) -> np.ndarray:
    parts = [np.concatenate([summaries(table.windows(chunk, context)), table.statics(chunk)], axis=1)
             for chunk in chunks(rows)]
    return np.concatenate(parts)


def chunks(rows: pd.DataFrame, size: int = CHUNK):
    for start in range(0, len(rows), size):
        yield rows.iloc[start : start + size]


@dataclass
class Standardizer:
    """Column medians for imputation and mean / std for scaling, from the finite training values."""

    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "Standardizer":
        flat = x.reshape(-1, x.shape[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median, mean, std = np.nanmedian(flat, axis=0), np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std > 0), std, 0.0).astype(np.float32) + eps
        return cls(median, mean, std)

    def transform(self, x: np.ndarray, scale: bool) -> np.ndarray:
        x = np.array(x, dtype=np.float32, copy=True)
        missing = ~np.isfinite(x)
        x[missing] = np.broadcast_to(self.median, x.shape)[missing]
        if scale:
            x = (x - self.mean) / self.std
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
