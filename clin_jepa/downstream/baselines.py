"""Ridge, LightGBM, LSTM and TCN baselines on the tabular view."""

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from clin_jepa.downstream import sequence_models
from clin_jepa.downstream.rows import hourly_rows, scan, with_labels
from clin_jepa.downstream.tasks import BY_NAME, CONTEXT, TASK_SETS
from clin_jepa.downstream.windows import HourlyTable, Standardizer, chunks, sequences, tabular_features
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
SEQUENCE_MODELS = ("lstm", "tcn")


def configurations(spec: dict, seed: int) -> list[dict]:
    keys = list(spec["grid"])
    grid = [dict(zip(keys, values)) for values in itertools.product(*spec["grid"].values())]
    if len(grid) <= spec["n_configurations"]:
        return grid
    chosen = np.random.default_rng(seed).choice(len(grid), size=spec["n_configurations"], replace=False)
    return [grid[i] for i in sorted(chosen)]


def evaluation_rows(config: dict, names: list[str], hourly: bool) -> dict[str, pd.DataFrame]:
    rows = {}
    for split in SPLITS:
        windows, stays = scan(Path(config["data_dir"]) / "trajectories", split)
        if not hourly:
            windows = windows[windows["start_step"] == 0]
        elif split == "test":
            windows = hourly_rows(stays)
        rows[split] = with_labels(windows.reset_index(drop=True), Path(config["output_dir"]) / "labels", names, hourly)
    return rows


def fit_ridge(spec: dict, configuration: dict, x: dict, y: dict) -> np.ndarray:
    model = LogisticRegression(C=configuration["C"], solver="lbfgs", max_iter=spec["max_iter"], class_weight="balanced")
    model.fit(x["train"], y["train"].astype(int))
    return model.predict_proba(x["test"])[:, 1]


def fit_lightgbm(spec: dict, configuration: dict, x: dict, y: dict) -> np.ndarray:
    import lightgbm

    target = y["train"].astype(int)
    model = lightgbm.LGBMClassifier(
        n_estimators=spec["n_estimators"], scale_pos_weight=(target == 0).sum() / max(1, (target == 1).sum()),
        verbose=-1, random_state=spec["seed"], num_leaves=configuration["num_leaves"],
        max_depth=configuration["max_depth"], min_child_samples=configuration["min_data_in_leaf"],
        learning_rate=configuration["learning_rate"], colsample_bytree=configuration["feature_fraction"],
        subsample=configuration["bagging_fraction"], subsample_freq=spec["subsample_freq"],
        reg_lambda=configuration["reg_lambda"], n_jobs=spec["n_jobs"],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        model.fit(x["train"], target, eval_set=[(x["val"], y["val"].astype(int))], eval_metric="auc",
                  callbacks=[lightgbm.early_stopping(spec["early_stopping_rounds"], verbose=False)])
        return model.predict_proba(x["test"])[:, 1]


def write_cell(path: Path, rows: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> None:
    cell = rows[["stay_id", "target_hour"]].assign(label=labels.astype(np.float32))
    columns = {f"p{i}": p.astype(np.float32) for i, p in enumerate(probabilities)}
    pd.concat([cell.reset_index(drop=True), pd.DataFrame(columns)], axis=1).to_parquet(path, index=False)


def run_tabular(model: str, spec: dict, config: dict, rows: dict, tasks: list[str], cells: dict[str, Path]) -> None:
    tabular_dir = Path(config["output_dir"]) / "tabular"
    x = {s: tabular_features(HourlyTable(tabular_dir, s), rows[s], CONTEXT) for s in SPLITS}
    scaler = Standardizer.fit(x["train"])
    x = {s: scaler.transform(x[s], scale=model == "ridge") for s in SPLITS}
    fit = fit_ridge if model == "ridge" else fit_lightgbm
    for task in tasks:
        at = {s: np.flatnonzero(rows[s][task].notna().to_numpy()) for s in SPLITS}
        y = {s: rows[s][task].to_numpy(dtype=np.float32)[at[s]] for s in SPLITS}
        too_few = len(y["train"]) < sequence_models.MIN_ROWS or not len(y["val"]) or not len(y["test"])
        if too_few or len(np.unique(y["train"])) < 2:
            logger.warning("%s skipped: too few labelled rows or a single class", task)
            continue
        features = {s: x[s][at[s]] for s in SPLITS}
        probabilities = []
        for i, configuration in enumerate(spec["configurations"]):
            probabilities.append(fit(spec, configuration, features, y))
            logger.info("%s %s: configuration %d of %d", model, task, i + 1, len(spec["configurations"]))
        write_cell(cells[task], rows["test"].iloc[at["test"]], y["test"], np.stack(probabilities))


def run_sequence(model: str, spec: dict, config: dict, rows: dict, names: list[str], cells: dict[str, Path]) -> None:
    tabular_dir = Path(config["output_dir"]) / "tabular"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = {s: rows[s][names].to_numpy(dtype=np.float32) for s in SPLITS}
    used = np.flatnonzero(np.isfinite(labels["train"]).any(axis=1))
    counts = [(np.isfinite(column).sum(), np.nansum(column)) for column in labels["train"][used].T]
    heads = [j for j, (n, positives) in enumerate(counts) if n >= sequence_models.MIN_ROWS and 0 < positives < n]
    logger.info("%s: %d heads", model, len(heads))

    data, scalers = {}, {}
    for split in ("train", "val"):
        table = HourlyTable(tabular_dir, split)
        sequence = np.concatenate([sequences(table.windows(chunk, CONTEXT)) for chunk in chunks(rows[split])])
        statics = table.statics(rows[split])
        if split == "train":
            scalers = {"sequence": Standardizer.fit(sequence), "statics": Standardizer.fit(statics)}
        data[split] = {"sequence": scalers["sequence"].transform(sequence, scale=True),
                       "statics": scalers["statics"].transform(statics, scale=True), "labels": labels[split][:, heads]}
        del sequence
    if len(used) < len(rows["train"]):
        data["train"] = {name: value[used] for name, value in data["train"].items()}

    networks = []
    for i, configuration in enumerate(spec["configurations"]):
        network = sequence_models.fit(model, configuration, spec, data["train"], data["val"], spec["seed"], device)
        networks.append(network)
        logger.info("%s: configuration %d of %d", model, i + 1, len(spec["configurations"]))
    del data

    table = HourlyTable(tabular_dir, "test")
    scores = np.empty((len(networks), len(rows["test"]), len(heads)), dtype=np.float32)
    start = 0
    for chunk in chunks(rows["test"], 4 * spec["batch_size"]):
        sequence = scalers["sequence"].transform(sequences(table.windows(chunk, CONTEXT)), scale=True)
        statics = scalers["statics"].transform(table.statics(chunk), scale=True)
        for i, network in enumerate(networks):
            scores[i, start : start + len(chunk)] = sequence_models.logits(
                network, sequence, statics, spec["batch_size"], device)
        start += len(chunk)
    for k, j in enumerate(heads):
        if names[j] in cells:
            at = np.flatnonzero(np.isfinite(labels["test"][:, j]))
            probabilities = expit(scores[:, at, k].astype(np.float64))
            write_cell(cells[names[j]], rows["test"].iloc[at], labels["test"][at, j], probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream.yaml")
    parser.add_argument("--model", required=True, choices=["ridge", "lightgbm", *SEQUENCE_MODELS])
    parser.add_argument("--task_set", required=True, choices=list(TASK_SETS))
    parser.add_argument("--task", help="a single task of the set (ridge and lightgbm)")
    args = parser.parse_args()
    if args.task and args.model in SEQUENCE_MODELS:
        parser.error("--task applies to ridge and lightgbm")
    if args.task and args.task not in TASK_SETS[args.task_set]:
        parser.error(f"{args.task} is not a task of {args.task_set}")

    setup_logging()
    config = load_config(args.config)
    names = TASK_SETS[args.task_set]
    tasks = [name for name in names if name in BY_NAME and args.task in (None, name)]
    directory = Path(config["output_dir"]) / "predictions"
    cells = {name: ensure_dir(directory / name) / f"{args.model}.parquet" for name in tasks}
    cells = {name: path for name, path in cells.items() if not path.exists()}
    if not cells:
        return

    spec = dict(config["baselines"]["sequence" if args.model in SEQUENCE_MODELS else args.model])
    spec["seed"], spec["n_jobs"] = config["baselines"]["seed"], config["n_workers"]
    spec["configurations"] = configurations(spec, spec["seed"])
    rows = evaluation_rows(config, names, hourly=BY_NAME[tasks[0]].hourly)
    if args.model in SEQUENCE_MODELS:
        run_sequence(args.model, spec, config, rows, names, cells)
    else:
        run_tabular(args.model, spec, config, rows, list(cells), cells)


if __name__ == "__main__":
    main()
