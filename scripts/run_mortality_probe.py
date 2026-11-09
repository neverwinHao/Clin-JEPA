"""Exploratory patient-disjoint hospital-mortality probe on sampled features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
C_GRID = (0.01, 0.1, 1.0, 10.0)
METHODS = ("history_32", "history_64", "history_32_plus_future_32")


def load_split(path):
    data = np.load(path, allow_pickle=False)
    history, future = data["history"], data["future"]
    labels, subjects, stays = data["labels"], data["subject_ids"], data["stay_ids"]
    if history.ndim != 3 or history.shape[1:] != (7, 4096):
        raise ValueError(f"Bad history shape: {path}")
    if future.shape != (len(history), 3, 4096) or any(len(x) != len(history) for x in (labels, subjects, stays)):
        raise ValueError(f"Bad future or label alignment: {path}")
    if not np.isfinite(history).all() or not np.isfinite(future).all():
        raise ValueError(f"Nonfinite inputs: {path}")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Both mortality classes required: {path}")
    if len(np.unique(stays)) != len(stays):
        raise ValueError(f"Duplicate ICU stays: {path}")
    return {"history": history.reshape(len(history), -1).astype(np.float32),
            "future": future.reshape(len(future), -1).astype(np.float32),
            "labels": labels.astype(np.int8), "subjects": subjects, "stays": stays}


def fit_projector(data, kind, components):
    scaler = StandardScaler()
    train = scaler.fit_transform(data["train"][kind])
    pca = PCA(n_components=components, svd_solver="randomized", iterated_power=2, random_state=SEED)
    pca.fit(train)
    values = {"train": pca.transform(train).astype(np.float32)}
    del train
    for split in ("val", "test"):
        values[split] = pca.transform(scaler.transform(data[split][kind])).astype(np.float32)
    post = StandardScaler()
    values["train"] = post.fit_transform(values["train"]).astype(np.float32)
    for split in ("val", "test"):
        values[split] = post.transform(values[split]).astype(np.float32)
    return values, {"scaler": scaler, "pca": pca, "post_scaler": post}


def fit_classifier(features, labels):
    results = []
    best_model, best_c, best_auc = None, None, -np.inf
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEED)
        model.fit(features["train"], labels["train"])
        score = model.predict_proba(features["val"])[:, 1]
        auc = float(roc_auc_score(labels["val"], score))
        results.append({"C": c, "val_auroc": auc})
        if auc > best_auc:
            best_model, best_c, best_auc = model, c, auc
    return best_model, best_c, best_auc, results


def bootstrap_test(labels, subjects, scores, draws, seed):
    patient_ids = np.unique(subjects)
    groups = [np.flatnonzero(subjects == subject) for subject in patient_ids]
    rng = np.random.default_rng(seed)
    intervals = {name: {"auroc": [], "auprc": []} for name in scores}
    differences = {"auroc": [], "auprc": []}
    skipped = 0
    for _ in range(draws):
        selection = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in selection])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            skipped += 1
            continue
        sampled = {}
        for name, p in scores.items():
            auc = float(roc_auc_score(y, p[idx]))
            ap = float(average_precision_score(y, p[idx]))
            intervals[name]["auroc"].append(auc)
            intervals[name]["auprc"].append(ap)
            sampled[name] = (auc, ap)
        a = sampled["history_32_plus_future_32"]
        b = sampled["history_64"]
        differences["auroc"].append(a[0] - b[0])
        differences["auprc"].append(a[1] - b[1])
    def ci(values):
        return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None
    return ({name: {metric: ci(vals) for metric, vals in metrics.items()} for name, metrics in intervals.items()},
            {metric: ci(vals) for metric, vals in differences.items()}, skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap draws must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    data = {s: load_split(args.data / f"{s}.npz") for s in ("train", "val", "test")}
    sets = {s: set(data[s]["subjects"].tolist()) for s in data}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Patients cross train/validation/test")
    labels = {s: data[s]["labels"] for s in data}
    counts = {s: {"stays": len(labels[s]), "patients": len(sets[s]), "positive_stays": int(labels[s].sum())} for s in data}
    print("Cohort:", counts, flush=True)

    print("Fitting history PCA on training data", flush=True)
    hist, hist_projection = fit_projector(data, "history", 64)
    print("Fitting hypothetical-future PCA on training data", flush=True)
    fut, future_projection = fit_projector(data, "future", 32)
    features = {
        "history_32": {s: hist[s][:, :32] for s in data},
        "history_64": {s: hist[s] for s in data},
        "history_32_plus_future_32": {s: np.concatenate([hist[s][:, :32], fut[s]], axis=1) for s in data},
    }
    models, results, predictions = {}, {}, {}
    for method in METHODS:
        print(f"Fitting {method}; selecting C on validation AUROC", flush=True)
        model, c, val_auc, grid = fit_classifier(features[method], labels)
        prob = model.predict_proba(features[method]["test"])[:, 1]
        models[method] = model
        predictions[method] = prob
        results[method] = {"chosen_C": c, "val_auroc": val_auc, "validation_grid": grid,
                           "test_auroc": float(roc_auc_score(labels["test"], prob)),
                           "test_auprc": float(average_precision_score(labels["test"], prob))}
    print("Computing paired patient-clustered bootstrap", flush=True)
    intervals, diff_intervals, skipped = bootstrap_test(
        labels["test"], data["test"]["subjects"], predictions, args.bootstrap, SEED)
    for method in METHODS:
        results[method]["test_auroc_ci95"] = intervals[method]["auroc"]
        results[method]["test_auprc_ci95"] = intervals[method]["auprc"]
    delta = {metric: results["history_32_plus_future_32"]["test_" + metric] -
                      results["history_64"]["test_" + metric] for metric in ("auroc", "auprc")}
    summary = {
        "task": "hospital mortality, admission window with >=24 hourly steps",
        "protocol": {"history_pca_dims": 64, "future_pca_dims": 32, "C_grid": list(C_GRID),
                     "selection": "validation AUROC", "standardization_and_pca_fit": "train only",
                     "seed": SEED, "bootstrap_draws": args.bootstrap, "bootstrap_skipped": skipped,
                     "bootstrap_unit": "patient", "test_unit": "ICU stay"},
        "caution": "Future features assume no interventions after observed hour 23; exploratory and not a clinical recommendation.",
        "counts": counts, "test_positive_prevalence": float(labels["test"].mean()),
        "results": results, "future_vs_history64_delta": delta,
        "future_vs_history64_delta_ci95": diff_intervals,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    frame = pd.DataFrame({"stay_id": data["test"]["stays"], "subject_id": data["test"]["subjects"],
                          "label": labels["test"], **predictions})
    frame.to_parquet(args.output / "test_predictions.parquet", index=False)
    joblib.dump({"history_projection": hist_projection, "future_projection": future_projection,
                 "classifiers": models, "protocol": summary["protocol"]}, args.output / "probe.joblib")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
