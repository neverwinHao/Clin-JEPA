"""Read-out of CLMBR-T-base representations on the downstream tasks."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.downstream.readout import SPLITS, train_probe
from clin_jepa.downstream.rows import scan, with_labels
from clin_jepa.downstream.tasks import TASKS
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def load_rows(config: dict, kind: str, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Represented rows of a split and their vectors."""
    data_dir = Path(config["data_dir"])
    parts = sorted((Path(config["output_dir"]) / "clmbr" / "features" / kind / split).glob("*.parquet"))
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    stay_hours = pd.read_csv(data_dir / "cohort" / "cohort.csv", usecols=["stay_id", "icu_los_hours"])
    windows = scan(data_dir / "trajectories", split)[0]
    keys = ["stay_id", "target_hour"]
    frame = frame.merge(stay_hours, on="stay_id").merge(windows[keys], on=keys)
    frame = frame[frame["icu_los_hours"] > frame["target_hour"] + 1]
    frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    vectors = frame[[c for c in frame.columns if c[0] == "f" and c[1:].isdigit()]].to_numpy(dtype=np.float32)
    return frame[keys], vectors.astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/clmbr.yaml")
    parser.add_argument("--rows", required=True, choices=["stay", "hourly"])
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    output_dir, spec = Path(config["output_dir"]), config["readout"]
    configurations = [dict(spec["defaults"], **entry) for entry in spec["configurations"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = [task.name for task in TASKS if task.hourly == (args.rows == "hourly")]

    rows, vectors = {}, {}
    for split in SPLITS:
        rows[split], features = load_rows(config, args.rows, split)
        rows[split] = with_labels(rows[split], output_dir / "labels", tasks, args.rows == "hourly")
        vectors[split] = torch.from_numpy(features).to(device)
    for task in tasks:
        path = output_dir / "predictions" / task / "clmbr.parquet"
        if path.exists():
            continue
        labelled = {s: np.flatnonzero(rows[s][task].notna().to_numpy()) for s in SPLITS}
        y = {s: rows[s][task].to_numpy(dtype=np.float32)[labelled[s]] for s in SPLITS}
        if y["train"].sum() < 1 or y["train"].all() or len(np.unique(y["val"])) < 2:
            logger.warning("%s skipped: training or validation labels hold a single class", task)
            continue
        x = {s: vectors[s][torch.as_tensor(labelled[s], device=device)] for s in SPLITS}
        results = [train_probe(configuration, spec["seed"], x, y) for configuration in configurations]
        best = int(np.nanargmax([score for score, _ in results]))
        cell = rows["test"].iloc[labelled["test"]][["stay_id", "target_hour"]]
        cell.assign(label=y["test"], p0=results[best][1]).to_parquet(ensure_dir(path.parent) / path.name, index=False)
        logger.info("%s: configuration %d, validation AUROC %.4f", task, best, results[best][0])


if __name__ == "__main__":
    main()
