"""Stay-clustered bootstrap of AUROC and AUPRC for every prediction cell."""

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def stay_draws(stays: np.ndarray, n_draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique stays and, per draw, how often each was drawn when resampling them with replacement."""
    unique = np.unique(stays)
    generator = np.random.RandomState(seed)
    counts = np.zeros((n_draws, len(unique)), dtype=np.int32)
    for b in range(n_draws):
        counts[b] = np.bincount(np.searchsorted(unique, generator.choice(unique, size=len(unique), replace=True)),
                                minlength=len(unique))
    return unique, counts


def weighted_metrics(group: np.ndarray, n_groups: int, labels: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """AUROC (ties count one half) and step-wise average precision of rows sorted by decreasing score.

    ``group`` numbers the runs of equal scores from the highest score down.
    """
    positives = np.bincount(group, weights=weights * labels, minlength=n_groups)
    negatives = np.bincount(group, weights=weights * (1.0 - labels), minlength=n_groups)
    n_positive, n_negative = positives.sum(), negatives.sum()
    if n_positive <= 0:
        return float("nan"), float("nan")
    true_positives = np.cumsum(positives)
    found = true_positives + np.cumsum(negatives)
    precision = np.divide(true_positives, found, out=np.zeros_like(found), where=found > 0)
    recall = true_positives / n_positive
    auprc = float(np.sum(np.diff(recall, prepend=0.0) * precision))
    if n_negative <= 0:
        return float("nan"), auprc
    below = n_negative - np.cumsum(negatives)
    return float(np.sum(positives * (below + 0.5 * negatives)) / (n_positive * n_negative)), auprc


def bootstrap(labels: np.ndarray, probabilities: np.ndarray, stays: np.ndarray, n_draws: int, seed: int) -> dict:
    """Per-draw AUROC and AUPRC of ``probabilities`` (configurations, rows)."""
    unique, counts = stay_draws(stays, n_draws, seed)
    stay_index = np.searchsorted(unique, stays)
    labels = labels.astype(np.float64)
    values = np.empty((len(probabilities), n_draws, 2))
    for i, scores in enumerate(probabilities.astype(np.float64)):
        order = np.argsort(-scores, kind="stable")
        group = np.cumsum(np.diff(scores[order], prepend=np.inf) != 0) - 1
        sorted_labels, sorted_stays = labels[order], stay_index[order]
        for b in range(n_draws):
            weights = counts[b, sorted_stays].astype(np.float64)
            values[i, b] = weighted_metrics(group, group[-1] + 1, sorted_labels, weights)
    auroc, auprc = values.mean(axis=0).T
    return {"auroc": auroc, "auprc": auprc, "n": len(labels), "n_positive": int(labels.sum())}


def bootstrap_cell(job: tuple[Path, Path, int, int]) -> None:
    source, target, n_draws, seed = job
    cell = pd.read_parquet(source)
    probabilities = cell[[c for c in cell.columns if c[0] == "p" and c[1:].isdigit()]].to_numpy().T
    result = bootstrap(cell["label"].to_numpy(), probabilities, cell["stay_id"].to_numpy(), n_draws, seed)
    np.savez(ensure_dir(target.parent) / target.name, **result)
    logger.info("%s/%s: AUROC %.4f, AUPRC %.4f", target.parent.name, target.stem,
                np.nanmean(result["auroc"]), np.nanmean(result["auprc"]))


def bootstrap_pair(job: tuple[Path, Path, list[str], int, int]) -> None:
    source, target, methods, n_draws, seed = job
    keys = ["stay_id", "target_hour"]
    first, second = (pd.read_parquet(source / f"{method}.parquet") for method in methods)
    rows = first.merge(second[keys + ["p0"]], on=keys, suffixes=("", "_second"))
    labels, stays = rows["label"].to_numpy(), rows["stay_id"].to_numpy()
    for method, column in zip(methods, ("p0", "p0_second")):
        result = bootstrap(labels, rows[[column]].to_numpy().T, stays, n_draws, seed)
        np.savez(ensure_dir(target) / f"{method}_paired.npz", **result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    output_dir, spec = Path(config["output_dir"]), config["bootstrap"]
    jobs = []
    for source in sorted((output_dir / "predictions").glob("*/*.parquet")):
        target = output_dir / "bootstrap" / source.parent.name / f"{source.stem}.npz"
        if not target.exists():
            jobs.append((source, target, spec["draws"], spec["seed"]))
    pairs = []
    for source in sorted((output_dir / "predictions").iterdir()):
        target = output_dir / "bootstrap" / source.name
        ready = all((source / f"{method}.parquet").exists() for method in spec["paired"])
        done = all((target / f"{method}_paired.npz").exists() for method in spec["paired"])
        if ready and not done:
            pairs.append((source, target, spec["paired"], spec["draws"], spec["seed"]))
    with Pool(config["n_workers"]) as pool:
        pool.map(bootstrap_cell, jobs, chunksize=1)
        pool.map(bootstrap_pair, pairs, chunksize=1)


if __name__ == "__main__":
    main()
