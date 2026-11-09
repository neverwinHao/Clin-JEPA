"""Compare history, predicted continuation, and copy-forward on sampled clinical tasks.

Uses the previous sampled experiment's train-fitted history/prediction PCA, fixed
no-intervention rollouts, patient splits and labels. Copy-forward uses only the
last observed state, summarized identically to predicted future (mean/std/last).
All outputs, including patient-level predictions, must remain on the data server.
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

from clin_jepa.downstream.tasks import TASKS, feature_horizon
from scripts.evaluate_all_downstream import SPLITS, counts, interval, task_rows, transform_history
from scripts.patient_bootstrap_metrics import patient_draw_weights, weighted_roc_ap
from scripts.run_mortality_probe import SEED, fit_classifier

METHODS = ("history_64", "history_64_plus_prediction_32", "history_64_plus_copy_32")
PAIRS = (("prediction_vs_history", METHODS[1], METHODS[0]),
         ("prediction_vs_copy", METHODS[1], METHODS[2]),
         ("copy_vs_history", METHODS[2], METHODS[0]))
HORIZONS = (8, 24, 48)


def copy_forward_summary(history):
    """Mean, standard deviation, and last of a repeated context-final state."""
    h = np.asarray(history)
    if h.ndim != 3 or h.shape[1] != 7:
        raise ValueError("Expected (windows, 7 history blocks, embedding dimensions)")
    last = h[:, 2, :].astype(np.float32)
    return np.stack((last, np.zeros_like(last), last), axis=1)


def apply_projector(matrix, projection):
    x = np.asarray(matrix, dtype=np.float32).reshape(len(matrix), -1)
    x = projection["scaler"].transform(x)
    x = projection["pca"].transform(x)
    return projection["post_scaler"].transform(x).astype(np.float32)


def project_prediction(future_root, projections):
    """Apply exactly the existing train-fitted PCA for each forecast horizon."""
    output = {}
    for i, horizon in enumerate(HORIZONS):
        proj = projections[horizon]
        output[horizon] = {}
        for split in SPLITS:
            arr = np.load(future_root / split / "future.npy", mmap_mode="r")
            if arr.shape[1:] != (3, 3, 4096):
                raise ValueError(f"Invalid cached future shape: {split}")
            result = np.empty((len(arr), 32), dtype=np.float32)
            for start in range(0, len(arr), 128):
                result[start:start+128] = apply_projector(arr[start:start+128, i], proj)
            if not np.isfinite(result).all():
                raise ValueError(f"Nonfinite prediction features: {split}, {horizon}")
            output[horizon][split] = result
        print(f"Prediction features projected: horizon {horizon}h", flush=True)
    return output


def project_copy(history_root):
    """Fit an independent, train-only 32D PCA for copy-forward future features."""
    arr = {s: np.load(history_root / s / "history.npy", mmap_mode="r") for s in SPLITS}
    if any(v.shape[1:] != (7, 4096) for v in arr.values()):
        raise ValueError("Invalid cached history shape")
    scaler = StandardScaler()
    train = copy_forward_summary(arr["train"]).reshape(len(arr["train"]), -1)
    train = scaler.fit_transform(train)
    pca = PCA(n_components=32, svd_solver="randomized", iterated_power=2, random_state=SEED)
    pca.fit(train)
    projected = {"train": pca.transform(train).astype(np.float32)}
    del train
    for split in ("val", "test"):
        out = np.empty((len(arr[split]), 32), dtype=np.float32)
        for start in range(0, len(out), 128):
            block = copy_forward_summary(arr[split][start:start+128]).reshape(-1, 3*4096)
            out[start:start+128] = pca.transform(scaler.transform(block))
        projected[split] = out
    post = StandardScaler()
    projected["train"] = post.fit_transform(projected["train"]).astype(np.float32)
    for split in ("val", "test"):
        projected[split] = post.transform(projected[split]).astype(np.float32)
    if any(not np.isfinite(x).all() for x in projected.values()):
        raise ValueError("Nonfinite copy-forward features")
    print("Copy-forward PCA fitted on train split only", flush=True)
    return projected, {"scaler": scaler, "pca": pca, "post_scaler": post}


def evaluate_task(task, indexes, label_hourly, label_stay, splits, histories, futures, copies, draws):
    tables = task_rows(task, indexes, label_hourly, label_stay, splits)
    sample_counts = {s: counts(tables[s], task.name) for s in SPLITS}
    result = {"task": task.name, "title": task.title, "benchmark": task.benchmark,
              "horizon": feature_horizon(task), "hourly": task.hourly,
              "counts": sample_counts, "status": "ok", "warnings": []}
    bad = [s for s in SPLITS if min(sample_counts[s]["positive_rows"],sample_counts[s]["negative_rows"]) == 0]
    if bad:
        result.update(status="skipped_single_class", warnings=["Single class in " + ", ".join(bad)])
        return result, None, None
    for split in ("val", "test"):
        if min(sample_counts[split]["positive_patients"], sample_counts[split]["negative_patients"]) < 5:
            result["warnings"].append(f"Fewer than 5 patients in one class in {split}")
    labels = {s: tables[s][task.name].to_numpy(dtype=np.int8) for s in SPLITS}
    h = feature_horizon(task)
    idx = {s: tables[s].row.to_numpy(dtype=np.int64) for s in SPLITS}
    features = {
        METHODS[0]: {s: histories[s][idx[s]] for s in SPLITS},
        METHODS[1]: {s: np.concatenate([histories[s][idx[s]], futures[h][s][idx[s]]], axis=1) for s in SPLITS},
        METHODS[2]: {s: np.concatenate([histories[s][idx[s]], copies[s][idx[s]]], axis=1) for s in SPLITS},
    }
    weights = patient_draw_weights(tables["test"].subject_id.to_numpy(), draws, SEED)
    metrics, predictions, models, boot = {}, {}, {}, {}
    for method in METHODS:
        model, c, val_auc, grid = fit_classifier(features[method], labels)
        prob = model.predict_proba(features[method]["test"])[:, 1]
        boot_auc, boot_ap = weighted_roc_ap(labels["test"], prob, weights)
        metrics[method] = {"chosen_C": c, "val_auroc": val_auc, "validation_grid": grid,
                           "test_auroc": float(roc_auc_score(labels["test"], prob)),
                           "test_auprc": float(average_precision_score(labels["test"], prob)),
                           "test_auroc_ci95": interval(boot_auc), "test_auprc_ci95": interval(boot_ap),
                           "valid_bootstrap_draws": int(np.isfinite(boot_auc).sum())}
        predictions[method], models[method] = prob, model
        boot[method] = {"auroc": boot_auc, "auprc": boot_ap}
    comparisons = {}
    for name, a, b in PAIRS:
        comparisons[name] = {}
        for metric in ("auroc","auprc"):
            comparisons[name][metric] = {
                "difference": metrics[a]["test_"+metric] - metrics[b]["test_"+metric],
                "ci95": interval(boot[a][metric] - boot[b][metric]),
                "valid_bootstrap_draws": int(np.isfinite(boot[a][metric] - boot[b][metric]).sum()),
            }
    result["models"], result["comparisons"] = metrics, comparisons
    output = tables["test"][["stay_id","subject_id"]].copy()
    if task.hourly:
        output["target_hour"] = tables["test"]["hour"]
    output["label"] = labels["test"]
    for method, prob in predictions.items():
        output[method] = prob
    return result, output, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--task", action="append", default=[], help="Run selected task names; default: all")
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("Bootstrap draws must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    selected = [task for task in TASKS if not args.task or task.name in args.task]
    if args.task and set(args.task) != {task.name for task in selected}:
        raise ValueError("Unknown task requested")
    exp = args.experiment
    clinical = exp / "clinical_data"
    history_root = clinical / "downstream/features/qwen3_base"
    future_root = clinical / "all_tasks/future"
    labels_root = clinical / "downstream/labels"
    projector_path = clinical / "all_tasks/evaluation/projections.joblib"
    indexes = {s: pd.read_parquet(history_root / s / "index.parquet") for s in SPLITS}
    for split,index in indexes.items():
        if index.row.tolist() != list(range(len(index))):
            raise ValueError(f"Noncontiguous history row indices: {split}")
        future = np.load(future_root / split / "future.npy", mmap_mode="r")
        if len(future) != len(index):
            raise ValueError(f"Future row count mismatch: {split}")
    splits = pd.read_csv(exp / "processed/splits/splits.csv", usecols=["stay_id","subject_id","split"])
    if splits.stay_id.duplicated().any() or (splits.groupby("subject_id")["split"].nunique() > 1).any():
        raise ValueError("Patient split is not disjoint")
    label_hourly = pd.read_parquet(labels_root / "hourly.parquet")
    label_stay = pd.read_parquet(labels_root / "stay.parquet")

    args.output.mkdir(parents=True)
    (args.output / "predictions").mkdir()
    (args.output / "models").mkdir()
    projector = joblib.load(projector_path)
    histories = transform_history(history_root, projector["history_projection"])
    futures = project_prediction(future_root, projector["future_projections"])
    copies, copy_projector = project_copy(history_root)
    joblib.dump({"copy_projection": copy_projector, "source_projection": str(projector_path)},
                args.output / "projection.joblib")

    results = []
    for task in selected:
        result, predictions, models = evaluate_task(task, indexes, label_hourly, label_stay,
                                                     splits, histories, futures, copies, args.bootstrap)
        results.append(result)
        if result["status"] == "ok":
            predictions.to_parquet(args.output / "predictions" / f"{task.name}.parquet", index=False)
            joblib.dump(models, args.output / "models" / f"{task.name}.joblib")
            vals = [result["models"][m]["test_auroc"] for m in METHODS]
            print(f"{task.benchmark} {task.name}: history {vals[0]:.3f}, "
                  f"prediction {vals[1]:.3f}, copy {vals[2]:.3f}; "
                  f"test positive patients {result['counts']['test']['positive_patients']}", flush=True)
        else:
            print(f"{task.name}: {result['status']}", flush=True)
    summary = {"protocol": {
        "seed": SEED, "bootstrap_draws": args.bootstrap, "bootstrap_unit": "patient",
        "history_projection": "existing 64D training-admission history PCA",
        "prediction_projection": "existing 32D training-window PCA per horizon",
        "copy_projection": "separate 32D PCA fitted on training-window copy-forward summaries",
        "copy_definition": "repeat last observed state; future summary [last, zero, last]",
        "future_policy": "Observed context then synthetic no-interventions; not a causal counterfactual",
        "classifier": "L2 logistic regression with balanced classes; per-task C selected by validation AUROC",
        "test_metric": "window-weighted AUROC and average precision",
        "caveat": "small sample and multiple outcomes; no clinical claims"},
        "tasks": results}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Complete: {sum(r['status']=='ok' for r in results)} of {len(selected)} tasks evaluated",flush=True)


if __name__ == "__main__":
    main()
