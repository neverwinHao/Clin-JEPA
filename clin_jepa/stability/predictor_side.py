"""Rank of the predicted states against the true states at every rollout horizon."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.stability.cache import load_states
from clin_jepa.stability.spectrum import covariance_spectrum, rankme, subspace_overlap
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def analyse(rollout_dir: Path, cache_dir: Path, config: dict, device: torch.device) -> list[dict]:
    index = pd.read_parquet(rollout_dir / "index.parquet")
    predicted = np.load(rollout_dir / "predicted.npy", mmap_mode="r")
    z, windows, first_window = load_states(cache_dir)
    rows = [windows[first_window[shard] + i] for shard, i in zip(index["shard"], index["window_idx"])]
    n_inside = index["length"].to_numpy() - config["context"]

    per_horizon = []
    for h in range(1, predicted.shape[1] + 1):
        reach = np.flatnonzero(n_inside >= h)
        z_true = z[torch.stack([rows[i][config["context"] - 1 + h] for i in reach])]
        z_pred = torch.from_numpy(np.ascontiguousarray(predicted[reach, h - 1]))
        true = covariance_spectrum(z_true, device, n_basis=config["subspace_dim"])
        pred = covariance_spectrum(z_pred, device, n_basis=config["subspace_dim"])
        per_horizon.append({
            "h": h, "n": len(reach), "rankme_predicted": rankme(pred), "rankme_true": rankme(true),
            "overlap": subspace_overlap(pred.basis, true.basis),
        })
        logger.info("h=%d: RankMe %.1f / %.1f, overlap %.3f", h, per_horizon[-1]["rankme_predicted"],
                    per_horizon[-1]["rankme_true"], per_horizon[-1]["overlap"])
    return per_horizon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stability.yaml")
    parser.add_argument("--rollout", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = [m["encoder"] for m in config["models"].values() if args.rollout in m["rollouts"]]
    if not encoders:
        raise ValueError(f"{args.rollout} is not a rollout of any model in {args.config}")
    encoder = encoders[0]
    cache_dir = Path(config["embedding_dir"]) / encoder / "test"
    result = analyse(Path(config["rollout_dir"]) / args.rollout / "test", cache_dir, config, device)
    output = ensure_dir(Path(config["output_dir"]) / "predictor_side") / f"{args.rollout}.json"
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
