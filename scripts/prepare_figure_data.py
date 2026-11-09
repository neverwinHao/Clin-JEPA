"""Create aggregate-only, reproducible figure data from a sampled experiment."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.patient_bootstrap_metrics import patient_draw_weights

EPOCH = re.compile(r"epoch (\d+)/(\d+) tf ([\d.eE+-]+) rollout ([\d.eE+-]+) val ([\d.eE+-]+)")
STEP = re.compile(r"step (\d+)/(\d+) tf [\d.eE+-]+ rollout [\d.eE+-]+ lr ([\d.eE+-]+)")
METHODS = ("history_32", "history_64", "history_32_plus_future_32")


def parse_training(exp):
    meta = json.loads((exp / "output/predictor_qwen3_base/training_meta.json").read_text())
    epochs, steps = [], []
    for line in (exp / "logs/train_predictor.log").read_text().splitlines():
        if m := EPOCH.search(line):
            epochs.append({"epoch": int(m[1]), "train_tf": float(m[3]),
                           "train_rollout_2step_sum": float(m[4]), "validation_tf": float(m[5])})
        if m := STEP.search(line):
            steps.append({"step": int(m[1]), "learning_rate": float(m[3])})
    if len(epochs) != meta["total_epochs"] or [x["epoch"] for x in epochs] != list(range(1, len(epochs) + 1)):
        raise ValueError("Training log and metadata do not agree")
    if abs(min(x["validation_tf"] for x in epochs) - meta["best_val_loss"]) > .01:
        raise ValueError("Validation losses do not match checkpoint metadata")
    config = meta["config"]["training"]
    return {"epochs": epochs, "logged_learning_rates": steps, "total_steps": meta["total_steps"],
            "learning_rate": float(config["lr"]), "warmup_ratio": config["warmup_ratio"],
            "batch_size": config["batch_size"], "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "parameters": meta["n_params"], "elapsed_seconds": meta["elapsed_seconds"], "seed": meta["seed"]}


def forecast_data(exp, draws):
    folder = exp / "diagnostics/rollout/qwen3_base/test"
    index = pd.read_parquet(folder / "index.parquet")
    splits = pd.read_csv(exp / "processed/splits/splits.csv", usecols=["stay_id", "subject_id"])
    subjects = index.merge(splits, on="stay_id", how="left", validate="many_to_one")["subject_id"]
    if subjects.isna().any():
        raise ValueError("Unmatched test stay")
    weights = patient_draw_weights(subjects.to_numpy(), draws, 42).astype(np.float64)
    e = np.load(folder / "errors.npz")
    other = np.load(exp / "diagnostics/prepared/baseline_errors.npz")
    matrices = {"predictor": e["model_l1"], "copy_forward": e["copy_forward_l1"],
                "context_mean": other["context_l1"], "train_mean": other["train_mean_l1"]}
    if any(v.shape != (len(index), 48) for v in matrices.values()):
        raise ValueError("Rollout errors do not match the test index")
    mask = np.isfinite(matrices["predictor"])
    if any(not np.array_equal(np.isfinite(v), mask) for v in matrices.values()):
        raise ValueError("Rollout baselines do not share valid windows")
    denominator = weights @ mask.astype(np.float64)
    count = mask.sum(axis=0)
    if (count == 0).any():
        raise ValueError("A horizon has no eligible windows")
    point, boot = {}, {}
    for name, matrix in matrices.items():
        clean = np.nan_to_num(matrix.astype(np.float64), nan=0.0)
        point[name] = clean.sum(axis=0) / count
        boot[name] = np.divide(weights @ clean, denominator,
                               out=np.full_like(denominator, np.nan), where=denominator > 0)
    point["mase"] = point["predictor"] / point["copy_forward"]
    point["rollout_drift"] = point["predictor"] / point["predictor"][0]
    point["state_displacement"] = point["copy_forward"] / point["copy_forward"][0]
    point["predictor_degradation"] = point["mase"] / point["mase"][0]
    boot["mase"] = boot["predictor"] / boot["copy_forward"]
    boot["rollout_drift"] = boot["predictor"] / boot["predictor"][:, :1]
    boot["state_displacement"] = boot["copy_forward"] / boot["copy_forward"][:, :1]
    boot["predictor_degradation"] = boot["mase"] / boot["mase"][:, :1]
    boot["context_improvement"] = 1 - boot["predictor"] / boot["context_mean"]
    point["context_improvement"] = 1 - point["predictor"] / point["context_mean"]
    records = []
    for h in range(48):
        subject_count = int(subjects[mask[:, h]].nunique())
        row = {"horizon": h + 1, "windows": int(count[h]), "patients": subject_count}
        for name in point:
            valid = boot[name][:, h]
            valid = valid[np.isfinite(valid)]
            if not len(valid):
                raise ValueError(f"No valid bootstrap draws for {name} at {h + 1}")
            lo, hi = np.quantile(valid, [.025, .975])
            row[name] = float(point[name][h])
            row[name + "_ci"] = [float(lo), float(hi)]
        records.append(row)
    old = pd.read_csv(exp / "diagnostics/prepared/rollout_curves_bootstrap.csv").set_index("horizon")
    for h in (1, 8, 24, 48):
        if abs(records[h - 1]["predictor"] - float(old.loc[h, "predictor_l1"])) > .01:
            raise ValueError("Forecast summary disagrees with saved diagnostics")
    return {"context": 24, "bootstrap_draws": draws, "bootstrap_unit": "patient",
            "policy": "recorded future actions (retrospective)", "horizons": records}


def downstream_data(exp):
    data = json.loads((exp / "clinical_data/all_tasks/evaluation/results.json").read_text())
    groups = []
    for group in ("E1", "E2", "E3"):
        tasks = [x for x in data["tasks"] if x["benchmark"] == group and x["status"] == "ok"]
        if not tasks:
            raise ValueError(f"No results for {group}")
        scores = {}
        for method in METHODS:
            scores[method] = {"auroc": float(np.mean([x["models"][method]["test_auroc"] for x in tasks])),
                              "auprc": float(np.mean([x["models"][method]["test_auprc"] for x in tasks]))}
        groups.append({"group": group, "tasks": len(tasks), "scores": scores})
    return {"groups": groups, "policy": "synthetic no-interventions after hour 23",
            "caveat": "task-unweighted exploratory mean; small and imbalanced samples"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.bootstrap < 1:
        raise ValueError("Bootstrap draws must be positive")
    exp = args.experiment
    sample = json.loads((exp / "processed/sampling_manifest.json").read_text())
    mortality = json.loads((exp / "clinical_data/mortality_probe/summary.json").read_text())
    rank = json.loads((exp / "diagnostics/prepared/rank_diagnostics.json").read_text())
    paired = json.loads((exp / "clinical_data/mortality_probe/paired_comparisons.json").read_text())
    data = {"sampling": sample, "training": parse_training(exp),
            "forecast": forecast_data(exp, args.bootstrap), "rank": rank,
            "mortality": {"counts": mortality["counts"], "test_positive_prevalence": mortality["test_positive_prevalence"],
                          "results": mortality["results"], "paired_comparisons": paired["comparisons"]},
            "downstream": downstream_data(exp)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared aggregate figure data: {args.output}", flush=True)


if __name__ == "__main__":
    main()
