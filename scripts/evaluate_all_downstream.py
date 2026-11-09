"""Evaluate all released downstream tasks on patient-disjoint sampled features.

History PCA is reused from the admission-only mortality experiment; future
PCA is fitted only on sampled training windows, separately by horizon.
Future features assume no interventions after the observed context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from clin_jepa.downstream.tasks import TASKS, CONTEXT, feature_horizon
from scripts.patient_bootstrap_metrics import patient_draw_weights, weighted_roc_ap
from scripts.run_mortality_probe import fit_classifier

SPLITS = ("train", "val", "test")
METHODS = ("history_32", "history_64", "history_32_plus_future_32")
HORIZONS = (8, 24, 48)
SEED = 42


def transform_history(history_root, projection):
    result = {}
    for split in SPLITS:
        arr = np.load(history_root / split / "history.npy", mmap_mode="r")
        if arr.shape[1:] != (7, 4096):
            raise ValueError(f"Bad history shape for {split}")
        out = np.empty((len(arr), 64), dtype=np.float32)
        for start in range(0, len(arr), 128):
            block = np.asarray(arr[start:start + 128], dtype=np.float32).reshape(-1, 7 * 4096)
            block = projection["scaler"].transform(block)
            block = projection["pca"].transform(block)
            out[start:start + len(block)] = projection["post_scaler"].transform(block)
        if not np.isfinite(out).all():
            raise ValueError(f"Nonfinite projected history: {split}")
        result[split] = out
        print(f"history {split}: {len(out)} projected windows", flush=True)
    return result


def transform_future(future_root, horizon_index):
    arrays = {s: np.load(future_root / s / "future.npy", mmap_mode="r") for s in SPLITS}
    if any(v.shape[1:] != (3, 3, 4096) for v in arrays.values()):
        raise ValueError("Invalid future feature shape")
    scaler = StandardScaler()
    train = np.asarray(arrays["train"][:, horizon_index], dtype=np.float32).reshape(-1, 3 * 4096)
    train = scaler.fit_transform(train)
    pca = PCA(n_components=32, svd_solver="randomized", iterated_power=2, random_state=SEED)
    pca.fit(train)
    projected = {"train": pca.transform(train).astype(np.float32)}
    del train
    for split in ("val", "test"):
        arr = arrays[split]
        out = np.empty((len(arr), 32), dtype=np.float32)
        for start in range(0, len(arr), 128):
            block = np.asarray(arr[start:start + 128, horizon_index], dtype=np.float32).reshape(-1, 3 * 4096)
            out[start:start + len(block)] = pca.transform(scaler.transform(block))
        projected[split] = out
    post = StandardScaler()
    projected["train"] = post.fit_transform(projected["train"]).astype(np.float32)
    for split in ("val", "test"):
        projected[split] = post.transform(projected[split]).astype(np.float32)
    if any(not np.isfinite(x).all() for x in projected.values()):
        raise ValueError("Nonfinite projected future")
    return projected, {"scaler": scaler, "pca": pca, "post_scaler": post}


def task_rows(task, indexes, label_hourly, label_stay, splits):
    tables = {}
    for split in SPLITS:
        index = indexes[split]
        if not task.hourly:
            index = index[index.start_step == 0]
        index = index.merge(splits, on="stay_id", how="left", validate="many_to_one")
        if index.subject_id.isna().any() or (index["split"] != split).any():
            raise ValueError(f"Patient split mismatch in {task.name}/{split}")
        if task.hourly:
            index["hour"] = index.start_step + CONTEXT - 1
            joined = index.merge(label_hourly[["stay_id", "hour", task.name]],
                                 on=["stay_id", "hour"], how="left", validate="one_to_one")
        else:
            joined = index.merge(label_stay[["stay_id", task.name]],
                                 on="stay_id", how="left", validate="one_to_one")
        joined = joined[joined[task.name].notna()].reset_index(drop=True)
        if not set(joined[task.name].unique()).issubset({0., 1.}):
            raise ValueError(f"Nonbinary labels in {task.name}/{split}")
        tables[split] = joined
    return tables


def counts(table, name):
    labels = table[name].to_numpy(dtype=np.int8)
    positive = labels == 1
    return {"rows": len(labels), "patients": int(table.subject_id.nunique()),
            "positive_rows": int(positive.sum()), "negative_rows": int((~positive).sum()),
            "positive_patients": int(table.subject_id[positive].nunique()),
            "negative_patients": int(table.subject_id[~positive].nunique())}


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return [float(x) for x in np.quantile(values, [.025, .975])] if len(values) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--future", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--history-projector", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=500)
    args = p.parse_args()
    if args.output.exists() or args.bootstrap < 1:
        raise ValueError("Output must be new and bootstrap draws must be positive")
    args.output.mkdir(parents=True)
    indexes = {s: pd.read_parquet(args.history / s / "index.parquet") for s in SPLITS}
    for split, index in indexes.items():
        if index.row.tolist() != list(range(len(index))):
            raise ValueError(f"History row order mismatch: {split}")
        future = np.load(args.future / split / "future.npy", mmap_mode="r")
        if len(future) != len(index):
            raise ValueError(f"Future feature count mismatch: {split}")
    splits = pd.read_csv(args.splits, usecols=["stay_id", "subject_id", "split"])
    if splits.stay_id.duplicated().any() or (splits.groupby("subject_id")["split"].nunique() > 1).any():
        raise ValueError("Patient split is not disjoint")
    label_hourly = pd.read_parquet(args.labels / "hourly.parquet")
    label_stay = pd.read_parquet(args.labels / "stay.parquet")

    old = joblib.load(args.history_projector)
    history_projection = old["history_projection"]
    histories = transform_history(args.history, history_projection)
    futures, future_models = {}, {}
    for i, horizon in enumerate(HORIZONS):
        print(f"Fitting future PCA for horizon {horizon} on training windows", flush=True)
        futures[horizon], future_models[horizon] = transform_future(args.future, i)
    joblib.dump({"history_projection": history_projection, "future_projections": future_models},
                args.output / "projections.joblib")

    results, rows_csv = [], []
    predictions_dir = args.output / "predictions"
    models_dir = args.output / "models"
    predictions_dir.mkdir()
    models_dir.mkdir()
    for task in TASKS:
        tables = task_rows(task, indexes, label_hourly, label_stay, splits)
        sample_counts = {s: counts(tables[s], task.name) for s in SPLITS}
        result = {"task": task.name, "title": task.title, "benchmark": task.benchmark,
                  "horizon": feature_horizon(task), "hourly": task.hourly,
                  "counts": sample_counts, "status": "ok", "warnings": []}
        bad = [s for s in SPLITS if min(sample_counts[s]["positive_rows"], sample_counts[s]["negative_rows"]) == 0]
        if bad:
            result["status"] = "skipped_single_class"
            result["warnings"].append("Single class or no labels in: " + ", ".join(bad))
            results.append(result)
            print(f"{task.name}: skipped, no positive/negative labels in {bad}", flush=True)
            continue
        for split in ("val", "test"):
            if min(sample_counts[split]["positive_patients"], sample_counts[split]["negative_patients"]) < 5:
                result["warnings"].append(f"Fewer than 5 patients in one class in {split}")
        labels = {s: tables[s][task.name].to_numpy(dtype=np.int8) for s in SPLITS}
        h = feature_horizon(task)
        feature_sets = {
            "history_32": {s: histories[s][tables[s].row.to_numpy(dtype=np.int64), :32] for s in SPLITS},
            "history_64": {s: histories[s][tables[s].row.to_numpy(dtype=np.int64)] for s in SPLITS},
            "history_32_plus_future_32": {
                s: np.concatenate([histories[s][tables[s].row.to_numpy(dtype=np.int64), :32],
                                   futures[h][s][tables[s].row.to_numpy(dtype=np.int64)]], axis=1)
                for s in SPLITS},
        }
        predictions, models, metric_results = {}, {}, {}
        weights = patient_draw_weights(tables["test"].subject_id.to_numpy(), args.bootstrap, SEED)
        for method in METHODS:
            model, c, val_auc, grid = fit_classifier(feature_sets[method], labels)
            prob = model.predict_proba(feature_sets[method]["test"])[:, 1]
            auc = float(roc_auc_score(labels["test"], prob))
            ap = float(average_precision_score(labels["test"], prob))
            boot_auc, boot_ap = weighted_roc_ap(labels["test"], prob, weights)
            predictions[method], models[method] = prob, model
            metric_results[method] = {"chosen_C": c, "val_auroc": val_auc, "validation_grid": grid,
                                      "test_auroc": auc, "test_auprc": ap,
                                      "test_auroc_ci95": interval(boot_auc), "test_auprc_ci95": interval(boot_ap),
                                      "valid_bootstrap_draws": int(np.isfinite(boot_auc).sum())}
        result["models"] = metric_results
        pairs = {}
        for baseline in ("history_32", "history_64"):
            x_auc, x_ap = weighted_roc_ap(labels["test"], predictions["history_32_plus_future_32"], weights)
            b_auc, b_ap = weighted_roc_ap(labels["test"], predictions[baseline], weights)
            pairs[baseline] = {
                "auroc_difference": metric_results["history_32_plus_future_32"]["test_auroc"] - metric_results[baseline]["test_auroc"],
                "auroc_difference_ci95": interval(x_auc - b_auc),
                "auprc_difference": metric_results["history_32_plus_future_32"]["test_auprc"] - metric_results[baseline]["test_auprc"],
                "auprc_difference_ci95": interval(x_ap - b_ap),
            }
        result["future_minus"] = pairs
        results.append(result)
        output = tables["test"][["stay_id", "subject_id"]].copy()
        if task.hourly:
            output["target_hour"] = tables["test"]["hour"]
        output["label"] = labels["test"]
        for method, probability in predictions.items():
            output[method] = probability
            row = {"benchmark": task.benchmark, "task": task.name, "horizon": h, "method": method,
                   "test_rows": sample_counts["test"]["rows"], "test_patients": sample_counts["test"]["patients"],
                   "test_positive_rows": sample_counts["test"]["positive_rows"],
                   "test_positive_patients": sample_counts["test"]["positive_patients"],
                   "test_auroc": metric_results[method]["test_auroc"], "test_auprc": metric_results[method]["test_auprc"],
                   "val_auroc": metric_results[method]["val_auroc"],
                   "warning": "; ".join(result["warnings"])}
            rows_csv.append(row)
        output.to_parquet(predictions_dir / f"{task.name}.parquet", index=False)
        joblib.dump(models, models_dir / f"{task.name}.joblib")
        print(f"{task.benchmark} {task.name}: history32 {metric_results['history_32']['test_auroc']:.3f}, "
              f"future {metric_results['history_32_plus_future_32']['test_auroc']:.3f}; "
              f"test positive patients {sample_counts['test']['positive_patients']}", flush=True)
    summary = {"protocol": {"seed": SEED, "bootstrap_draws": args.bootstrap, "bootstrap_unit": "patient",
                             "history_PCA": "reused training-admission projection from mortality probe",
                             "future_PCA": "fit on train windows separately at each horizon",
                             "future_policy": "Observed context then synthetic no-interventions, not medical advice",
                             "validation_selection": "AUROC, per task and method",
                             "test_metric": "window-weighted AUROC and average precision",
                             "caveat": "Small sample, multiple outcomes, no clinical claims"},
               "tasks": results}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rows_csv).to_csv(args.output / "results.csv", index=False)
    print(f"Complete: {sum(r['status'] == 'ok' for r in results)} of {len(TASKS)} tasks evaluated", flush=True)


if __name__ == "__main__":
    main()
