"""Step 06: build per-hour state and action texts, slice trajectory windows and write shards."""

import argparse
import logging
import pickle
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.data.helpers.action_filter import filter_act_events
from clin_jepa.data.helpers.event_ordering import sort_events
from clin_jepa.data.helpers.io_utils import resolve_table_path
from clin_jepa.data.helpers.observation_filter import filter_obs_events
from clin_jepa.data.helpers.text_templates import format_event
from clin_jepa.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

NO_OBSERVATIONS = "No observations recorded."
NO_INTERVENTIONS = "No active interventions."


def hour_text(events: pd.DataFrame, step_idx: int, empty: str) -> str:
    """One hour of events as ``"t=12h | Heart rate: 88 bpm. MAP: 56 mmHg"``."""
    fragments = []
    for _, row in sort_events(events).iterrows():
        fragment = format_event(
            row["source_table"], row["variable_name"],
            row.get("numeric_value"), row.get("text_value"), row.get("unit", ""),
        )
        if fragment is not None:
            fragments.append(fragment)
    return f"t={step_idx}h | {'. '.join(fragments) if fragments else empty}"


def hourly_texts(events: pd.DataFrame, n_steps: int, empty: str) -> list[str]:
    if len(events) == 0:
        return [f"t={s}h | {empty}" for s in range(n_steps)]
    by_step = dict(list(events.groupby("step_idx")))
    none = pd.DataFrame(columns=events.columns)
    return [hour_text(by_step.get(s, none), s, empty) for s in range(n_steps)]


def hourly_sofa(obs_events: pd.DataFrame, n_steps: int, source: dict) -> np.ndarray:
    """SOFA total of every hour as float32, NaN where it was not recorded."""
    sofa = np.full(n_steps, np.nan, dtype=np.float32)
    rows = obs_events.loc[
        (obs_events["variable_name"] == source["variable_name"])
        & (obs_events["source_table"] == source["source_table"])
    ]
    if len(rows) > 0:
        for step_idx, val in rows.groupby("step_idx")["numeric_value"].first().items():
            if 0 <= step_idx < n_steps and pd.notna(val):
                sofa[int(step_idx)] = val
    return sofa


def slice_windows(n_steps: int, config: dict) -> list[tuple[int, int]]:
    """(start, length) of each window: one window for a short stay, strided ``t_max`` windows otherwise."""
    t_max, min_len, stride = config["t_max"], config["min_len"], config["stride"]
    if n_steps < min_len:
        return []
    starts = range(0, n_steps - t_max + 1, stride) if n_steps >= t_max else [0]
    windows = [(int(s), int(min(t_max, n_steps - s))) for s in starts]
    return [w for w in windows if w[1] >= min_len]


def process_stay(args: tuple[Path, dict]) -> dict | None:
    pkl_path, config = args
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    windows = slice_windows(data["n_steps"], config)
    if not windows:
        return None

    n_steps = data["n_steps"]
    obs_events = data["obs_events"]
    if len(obs_events) > 0:
        obs_events = filter_obs_events(obs_events, config["state_text"]).reset_index(drop=True)
    act_events = filter_act_events(data["act_events"])

    metadata = data["metadata"]
    return {
        "stay_id": int(data["stay_id"]),
        "state_texts": hourly_texts(obs_events, n_steps, NO_OBSERVATIONS),
        "action_texts": hourly_texts(act_events, n_steps, NO_INTERVENTIONS),
        "sofa": hourly_sofa(obs_events, n_steps, config["sofa"]),
        "per_stay": {
            "stay_id": int(data["stay_id"]),
            "age": float(metadata["age"]),
            "gender": str(metadata["gender"]),
            "race": str(metadata["race"]),
        },
        "windows": windows,
    }


class Shard:
    """Accumulates stays; every (stay, hour) text is stored once and windows index into it."""

    def __init__(self, shard_idx: int) -> None:
        self.shard_idx = shard_idx
        self.hour_index: dict[tuple[int, int], int] = {}
        self.stay_ids, self.hours, self.state_texts, self.action_texts, self.sofa = [], [], [], [], []
        self.per_stay: list[dict] = []
        self.windows: list[dict] = []

    def add(self, bundle: dict) -> None:
        stay_id = bundle["stay_id"]
        stay_index = len(self.per_stay)
        self.per_stay.append(bundle["per_stay"])
        for start, length in bundle["windows"]:
            hour_indices = np.empty(length, dtype=np.int32)
            for i in range(length):
                hour = start + i
                idx = self.hour_index.get((stay_id, hour))
                if idx is None:
                    idx = self.hour_index[(stay_id, hour)] = len(self.state_texts)
                    self.stay_ids.append(stay_id)
                    self.hours.append(hour)
                    self.state_texts.append(bundle["state_texts"][hour])
                    self.action_texts.append(bundle["action_texts"][hour])
                    self.sofa.append(float(bundle["sofa"][hour]))
                hour_indices[i] = idx
            self.windows.append({
                "stay_index": stay_index, "start_step": start,
                "length": length, "hour_indices": hour_indices,
            })

    def finalize(self) -> dict:
        return {
            "per_hour": {
                "stay_ids": np.asarray(self.stay_ids, dtype=np.int64),
                "hours": np.asarray(self.hours, dtype=np.int32),
                "state_texts": self.state_texts,
                "action_texts": self.action_texts,
                "sofa": np.asarray(self.sofa, dtype=np.float32),
            },
            "per_stay": {
                "stay_ids": np.asarray([r["stay_id"] for r in self.per_stay], dtype=np.int64),
                "ages": np.asarray([r["age"] for r in self.per_stay], dtype=np.float64),
                "genders": [r["gender"] for r in self.per_stay],
                "races": [r["race"] for r in self.per_stay],
            },
            "windows": self.windows,
        }


def write_chunk(
    chunk_idx: int, pkl_files: list[Path], stay_to_split: dict, output_dir: Path,
    config: dict, n_workers: int,
) -> None:
    jobs = [(path, config) for path in pkl_files]
    if n_workers <= 1:
        bundles = [process_stay(job) for job in jobs]
    else:
        with Pool(n_workers) as pool:
            bundles = list(pool.imap_unordered(process_stay, jobs, chunksize=50))

    by_split: dict[str, list[dict]] = defaultdict(list)
    for bundle in bundles:
        if bundle is not None:
            by_split[stay_to_split[bundle["stay_id"]]].append(bundle)

    for split in ("train", "val", "test"):
        shards = [Shard(0)]
        for bundle in sorted(by_split[split], key=lambda b: b["stay_id"]):
            current = shards[-1]
            if current.windows and len(current.windows) + len(bundle["windows"]) > config["shard_size"]:
                current = Shard(len(shards))
                shards.append(current)
            current.add(bundle)
        for shard in shards:
            if shard.windows:
                path = output_dir / split / f"trajectories_c{chunk_idx:02d}_{shard.shard_idx:03d}.pt"
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(shard.finalize(), temporary)
                temporary.replace(path)
        logger.info(
            "chunk %d %s: %d stays, %d windows",
            chunk_idx, split, len(by_split[split]), sum(len(s.windows) for s in shards),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/trajectories.yaml")
    parser.add_argument("--paths_config", default="configs/data/paths.yaml")
    parser.add_argument("--chunk_idx", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    paths_config = load_config(args.paths_config)
    n_workers = args.n_workers or config["n_workers"]

    splits = pd.read_csv(resolve_table_path(paths_config, "output.splits") / "splits.csv")
    stay_to_split = dict(zip(splits["stay_id"], splits["split"]))
    output_dir = resolve_table_path(paths_config, "output.trajectories")
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(resolve_table_path(paths_config, "output.discretized").glob("*.pkl"))
    n_chunks = config["n_chunks"]
    chunk_size = (len(pkl_files) + n_chunks - 1) // n_chunks
    if args.chunk_idx is not None and not 0 <= args.chunk_idx < n_chunks:
        raise ValueError(f"--chunk_idx must be in [0, {n_chunks})")
    chunks = range(n_chunks) if args.chunk_idx is None else [args.chunk_idx]
    for chunk_idx in chunks:
        files = pkl_files[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        write_chunk(chunk_idx, files, stay_to_split, output_dir, config, n_workers)


if __name__ == "__main__":
    main()
