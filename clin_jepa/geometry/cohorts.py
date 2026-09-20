"""Cohorts of the latent-geometry diagnosis, from the SOFA course of every admission window of the test split."""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def eligible_stays(trajectory_dir: Path, hours: int) -> list[dict]:
    """Admission windows with a SOFA total at each of their first ``hours`` hours, in shard order."""
    rows = []
    for path in sorted((trajectory_dir / "test").glob("trajectories_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        sofa_by_hour = shard["per_hour"]["sofa"]
        texts = shard["per_hour"]["state_texts"]
        for w in shard["windows"]:
            if int(w["start_step"]) != 0 or int(w["length"]) < hours:
                continue
            index = w["hour_indices"][:hours]
            sofa = np.asarray(sofa_by_hour[index], dtype=np.float32)
            if not np.all(np.isfinite(sofa)):
                continue
            rows.append({
                "stay_id": int(shard["per_stay"]["stay_ids"][w["stay_index"]]),
                "delta": float(sofa[-1] - sofa[0]),
                "range": float(sofa.max() - sofa.min()),
                "std": float(sofa.std()),
                "statements": [len([s for s in re.split(r"\.\s+", texts[int(index[h])]) if s.strip()])
                               for h in (0, 1, hours - 1)],
            })
    return rows


def assign(rows: list[dict], rule: dict) -> np.ndarray:
    """1 = deteriorating, 0 = stable, -1 = neither; deteriorating is tested first."""
    labels = np.full(len(rows), -1)
    for i, r in enumerate(rows):
        if r["delta"] >= rule["deteriorating_delta"]:
            labels[i] = 1
        elif r["std"] <= rule["stable_std"] and (rule["stable_range"] is None or r["range"] <= rule["stable_range"]):
            labels[i] = 0
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geometry.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    rows = eligible_stays(Path(config["trajectory_dir"]), config["hours"])
    labels = {name: assign(rows, rule) for name, rule in config["definitions"].items()}

    main_labels = labels["main"]
    rng = np.random.RandomState(config["subsample"]["seed"])
    subsample = {}
    for name, value in (("deteriorating", 1), ("stable", 0)):
        members = np.flatnonzero(main_labels == value)
        drawn = rng.choice(len(members), min(config["subsample"]["per_cohort"], len(members)), replace=False)
        subsample[name] = [rows[i]["stay_id"] for i in members[drawn]]

    pool = np.flatnonzero(main_labels >= 0)
    n_deteriorating = int((main_labels[pool] == 1).sum())
    rng = np.random.RandomState(config["null_control"]["seed"])
    null_labels = []
    for _ in range(config["null_control"]["relabelings"]):
        relabelled = np.zeros(len(pool), dtype=int)
        relabelled[rng.permutation(len(pool))[:n_deteriorating]] = 1
        null_labels.append(relabelled.tolist())

    output = ensure_dir(Path(config["output_dir"])) / "cohorts.json"
    output.write_text(json.dumps({
        "eligible": rows,
        "labels": {name: values.tolist() for name, values in labels.items()},
        "subsample": subsample,
        "null_labels": null_labels,
    }))
    for name, values in labels.items():
        logger.info("%s: %d eligible, %d deteriorating, %d stable", name, len(rows),
                    (values == 1).sum(), (values == 0).sum())


if __name__ == "__main__":
    main()
