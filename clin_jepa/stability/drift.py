"""Rollout drift and its decomposition, stay-clustered bootstrap intervals and the paired Wilcoxon test."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

CURVES = ("mase", "drift", "degradation", "displacement")


def load_rollout(rollout_dir: Path, context: int) -> dict[str, np.ndarray]:
    """Per-step errors of the windows with at least one rollout step."""
    index = pd.read_parquet(rollout_dir / "index.parquet")
    keep = (index["length"] > context).to_numpy()
    errors = np.load(rollout_dir / "errors.npz")
    out = {name: errors[name][keep] for name in errors.files}
    out["stay_id"] = index["stay_id"].to_numpy()[keep]
    out["start_step"] = index["start_step"].to_numpy()[keep]
    return out


def horizon_means(values: np.ndarray) -> np.ndarray:
    return np.array([column[np.isfinite(column)].mean() for column in values.T], dtype=np.float64)


def decompose(l1: np.ndarray, cf: np.ndarray) -> dict[str, np.ndarray]:
    mase = l1 / cf
    return {"mase": mase, "drift": l1 / l1[..., :1], "degradation": mase / mase[..., :1],
            "displacement": cf / cf[..., :1]}


def bootstrap_intervals(
    model: np.ndarray, cf: np.ndarray, stay_id: np.ndarray, n_draws: int, rng: np.random.Generator,
) -> dict[str, list]:
    """95 % intervals of the four curves from one set of stay resamples, each stay drawn with all its windows."""
    stays, stay_of_row = np.unique(stay_id, return_inverse=True)
    draws = rng.integers(0, stays.size, size=(n_draws, stays.size))
    weights = np.stack([np.bincount(d, minlength=stays.size) for d in draws]).astype(np.float64)

    def per_stay(values: np.ndarray) -> np.ndarray:
        return np.stack([
            np.bincount(stay_of_row[ok], weights=column[ok], minlength=stays.size)
            for column, ok in zip(values.T, np.isfinite(model).T)
        ], axis=1)

    count = weights @ per_stay(np.ones_like(model))
    resampled = decompose(weights @ per_stay(model) / count, weights @ per_stay(cf) / count)
    return {name: np.percentile(resampled[name], [2.5, 97.5], axis=0).tolist() for name in CURVES}


def window_degradation(rollout: dict[str, np.ndarray]) -> np.ndarray:
    """MASE(last) / MASE(1) of each window that reaches the last horizon; NaN elsewhere."""
    model, cf = rollout["model_l1"], rollout["copy_forward_l1"]
    with np.errstate(divide="ignore", invalid="ignore"):
        first, last = model[:, 0] / cf[:, 0], model[:, -1] / cf[:, -1]
        ok = np.isfinite(last) & (cf[:, -1] > 0) & (cf[:, 0] > 0) & (model[:, 0] > 0)
        return np.where(ok, last / first, np.nan)


def analyse(rollouts: list[dict], n_draws: int, rng: np.random.Generator) -> dict:
    model = np.concatenate([r["model_l1"] for r in rollouts])
    cf = np.concatenate([r["copy_forward_l1"] for r in rollouts])
    stay_id = np.concatenate([r["stay_id"] for r in rollouts])
    l1, l1_cf = horizon_means(model), horizon_means(cf)
    return {
        "n_windows": np.isfinite(rollouts[0]["model_l1"]).sum(axis=0).tolist(),
        "l1": l1.tolist(),
        "l1_copy_forward": l1_cf.tolist(),
        "l1_linear_extrap": horizon_means(rollouts[0]["linear_extrap_l1"]).tolist(),
        "curves": {name: values.tolist() for name, values in decompose(l1, l1_cf).items()},
        "intervals": bootstrap_intervals(model, cf, stay_id, n_draws, rng),
    }


def wilcoxon(reference: dict, rollouts: list[dict]) -> dict:
    """One-sided paired test that the reference degrades less: share of windows and p-value."""
    x, shares, p_values = window_degradation(reference), [], []
    for r in rollouts:
        if not (np.array_equal(r["stay_id"], reference["stay_id"])
                and np.array_equal(r["start_step"], reference["start_step"])):
            raise ValueError("rollouts are not over the same windows")
        y = window_degradation(r)
        both = np.isfinite(x) & np.isfinite(y)
        shares.append(float((x[both] < y[both]).mean()))
        p_values.append(float(stats.wilcoxon(x[both], y[both], alternative="less").pvalue))
    return {"share_reference_lower": float(np.mean(shares)), "p_value": max(p_values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stability.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    root, boot = Path(config["rollout_dir"]), config["bootstrap"]

    def load(name: str) -> dict:
        return load_rollout(root / name / "test", config["context"])

    results = {"models": {}, "checkpoints": {}}
    reference = load(config["models"][config["reference"]]["rollouts"][0])
    for key, entry in config["models"].items():
        rollouts = [load(name) for name in entry["rollouts"]]
        rng = np.random.default_rng([boot["seed"], entry["bootstrap_seed"]])
        results["models"][key] = analyse(rollouts, boot["draws"], rng)
        if key != config["reference"]:
            results["models"][key]["wilcoxon"] = wilcoxon(reference, rollouts)
        logger.info("%s: degradation %.3f, displacement %.3f", key,
                    results["models"][key]["curves"]["degradation"][-1],
                    results["models"][key]["curves"]["displacement"][-1])
    for entry in config["checkpoints"]:
        rollout = load(entry["rollout"])
        l1, l1_cf = horizon_means(rollout["model_l1"]), horizon_means(rollout["copy_forward_l1"])
        results["checkpoints"][entry["rollout"]] = {
            name: values.tolist() for name, values in decompose(l1, l1_cf).items()
        }

    output = ensure_dir(Path(config["output_dir"])) / "drift.json"
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
