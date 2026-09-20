"""UMAP projection of the 50 + 50 subsample under one encoder, fitted on a random sample of its test-split states."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import umap

from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    cfg, hours = config["umap"], config["hours"]
    output_dir = Path(config["output_dir"])
    subsample = json.loads((output_dir / "cohorts.json").read_text())["subsample"]
    wanted = {stay_id for stays in subsample.values() for stay_id in stays}

    states, trajectories = [], {}
    for path in sorted((Path(config["embedding_dir"]) / args.encoder / "test").glob("embeddings_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        z = shard["z_state_embeddings"].float().numpy()
        states.append(z)
        for w in shard["windows"]:
            if int(w["stay_id"]) in wanted and int(w["start_step"]) == 0 and int(w["length"]) >= hours:
                trajectories[int(w["stay_id"])] = z[w["emb_indices_state"][:hours].long().numpy()]
    states = np.concatenate(states)

    rng = np.random.RandomState(cfg["seed"])
    sample = states[rng.choice(len(states), min(cfg["fit_samples"], len(states)), replace=False)]
    reducer = umap.UMAP(n_components=2, n_neighbors=cfg["n_neighbors"], min_dist=cfg["min_dist"],
                        metric=cfg["metric"], random_state=cfg["seed"], low_memory=False)
    reducer.fit(sample)
    logger.info("fitted on %d of %d states", len(sample), len(states))

    projected = {cohort: np.stack([reducer.transform(trajectories[stay_id]) for stay_id in stays])
                 for cohort, stays in subsample.items()}
    np.savez(ensure_dir(output_dir / "umap") / f"{args.encoder}.npz", **projected)


if __name__ == "__main__":
    main()
