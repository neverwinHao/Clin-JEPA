"""Prepare patient-level rollout baselines and sample-relative rank diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def spectra(x, device):
    # SVD on the centered rows: same singular spectrum as the covariance used
    # by the paper, but economical when the sample has fewer rows than features.
    x = x.to(device=device, dtype=torch.float32)
    x = x - x.mean(dim=0)
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    energy = singular.square()
    if float(energy.sum()) == 0.0:
        return {"rankme": 0.0, "top10_share": 1.0, "mean_std": 0.0}, vh[:64]
    weights = torch.zeros(x.shape[1], dtype=singular.dtype, device=device)
    weights[:len(singular)] = singular / singular.sum()
    weights = weights + 1e-7
    weights = weights / weights.sum()
    entropy = -(weights * weights.log()).sum()
    return {"rankme": float(entropy.exp()), "top10_share": float(energy[:10].sum() / energy.sum()),
            "mean_std": float(x.std(dim=0).mean())}, vh[:64]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = args.data / "embeddings/qwen3_base"
    # This reference is fitted only on training states.
    sum_states = torch.zeros(4096, dtype=torch.float64)
    n_states = 0
    for path in sorted((base / "train").glob("embeddings_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        z = shard["z_state_embeddings"]
        for start in range(0, len(z), 1024):
            sum_states += z[start:start + 1024].double().sum(dim=0)
        n_states += len(z)
    if n_states == 0:
        raise ValueError("No training states")
    train_mean = (sum_states / n_states).float()
    index = pd.read_parquet(args.rollout / "index.parquet")
    pred = np.load(args.rollout / "predicted.npy", mmap_mode="r")
    errors = np.load(args.rollout / "errors.npz")
    if len(index) != len(pred):
        raise ValueError("Rollout index and predictions differ")
    n, steps = pred.shape[:2]
    context_errors = np.full((n, steps), np.nan, dtype=np.float32)
    global_errors = np.full((n, steps), np.nan, dtype=np.float32)
    targets = {h: [] for h in (1, 8, 24, 48)}
    forecasts = {h: [] for h in targets}
    for shard_name, group in index.groupby("shard", sort=False):
        shard = torch.load(base / "test" / shard_name, map_location="cpu", weights_only=False, mmap=True)
        z = shard["z_state_embeddings"]
        for row in group.itertuples(index=False):
            window = shard["windows"][int(row.window_idx)]
            indices = window["emb_indices_state"].long()
            n_inside = min(int(window["length"]) - 24, steps)
            if n_inside <= 0:
                continue
            context_mean = z[indices[:24]].float().mean(dim=0)
            truth = z[indices[24:24 + n_inside]].float()
            context_errors[row.row, :n_inside] = (truth - context_mean).abs().sum(dim=-1).numpy()
            global_errors[row.row, :n_inside] = (truth - train_mean).abs().sum(dim=-1).numpy()
            for h in targets:
                if n_inside >= h:
                    targets[h].append(truth[h - 1])
                    forecasts[h].append(torch.from_numpy(np.array(pred[row.row, h - 1], copy=True)))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in range(1, steps + 1):
        m = errors["model_l1"][:, h - 1]
        c = errors["copy_forward_l1"][:, h - 1]
        k = context_errors[:, h - 1]
        g = global_errors[:, h - 1]
        valid = np.isfinite(m) & np.isfinite(c) & np.isfinite(k) & np.isfinite(g)
        rows.append({"horizon": h, "windows": int(valid.sum()), "model_l1": float(m[valid].mean()),
                     "copy_l1": float(c[valid].mean()), "context_mean_l1": float(k[valid].mean()),
                     "train_mean_l1": float(g[valid].mean())})
    pd.DataFrame(rows).to_csv(args.output / "rollout_curves.csv", index=False)
    np.savez_compressed(args.output / "baseline_errors.npz", context_l1=context_errors, train_mean_l1=global_errors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = {}
    for h in targets:
        if len(targets[h]) < 2:
            continue
        truth = torch.stack(targets[h])
        prediction = torch.stack(forecasts[h])
        true_stats, true_basis = spectra(truth, device)
        pred_stats, pred_basis = spectra(prediction, device)
        k = min(len(true_basis), len(pred_basis), 64)
        overlap = float(((true_basis[:k] @ pred_basis[:k].T).square().sum() / k))
        rank[str(h)] = {"windows": len(truth), "true": true_stats, "predicted": pred_stats,
                        "top64_subspace_overlap": overlap}
        print(f"h={h}: true RankMe {true_stats['rankme']:.1f}, predicted {pred_stats['rankme']:.1f}", flush=True)
    (args.output / "rank_diagnostics.json").write_text(json.dumps(rank, indent=2) + "\n")
    print(f"Prepared {len(rows)} horizon rows using {n_states} training states", flush=True)


if __name__ == "__main__":
    main()
