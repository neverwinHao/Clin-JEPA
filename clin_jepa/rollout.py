"""Autoregressive rollout of a predictor from the first hours of every trajectory window."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.model import ACTransformerPredictor
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

STATISTICS = ("mean", "std", "last")


def load_predictor(path: str | Path, architecture: dict, device: torch.device) -> ACTransformerPredictor:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ACTransformerPredictor(**architecture)
    model.load_state_dict(state.get("model_state_dict", state))
    return model.to(device).eval().requires_grad_(False)


def build_index(shards: list[Path], context: int) -> pd.DataFrame:
    """One row per window with at least ``context`` hours, in shard order."""
    rows = []
    for path in shards:
        windows = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["windows"]
        rows += [
            (int(w["stay_id"]), int(w["start_step"]), int(w["length"]), path.name, i)
            for i, w in enumerate(windows) if int(w["length"]) >= context
        ]
    index = pd.DataFrame(rows, columns=["stay_id", "start_step", "length", "shard", "window_idx"])
    index.insert(0, "row", np.arange(len(index)))
    return index


def gather_batch(shard: dict, window_idx: np.ndarray, n_hours: int, device: torch.device) -> tuple:
    """States ``(B, n_hours, D)``, actions ``(B, n_hours - 1, D)``, demographics ``(B, D)``; zero past the stay end."""
    windows = [shard["windows"][int(i)] for i in window_idx]
    rows = torch.zeros(2, len(windows), n_hours, dtype=torch.long)
    recorded = torch.zeros(len(windows), n_hours, dtype=torch.bool)
    for b, w in enumerate(windows):
        n = min(int(w["length"]), n_hours)
        rows[0, b, :n] = torch.as_tensor(w["emb_indices_state"][:n], dtype=torch.long)
        rows[1, b, :n] = torch.as_tensor(w["emb_indices_action"][:n], dtype=torch.long)
        recorded[b, :n] = True
    states = shard["z_state_embeddings"][rows[0].reshape(-1)].reshape(*recorded.shape, -1)
    actions = shard["z_action_embeddings"][rows[1].reshape(-1)].reshape(*recorded.shape, -1)
    states[~recorded] = 0
    actions[~recorded] = 0
    static = shard["z_statics"][torch.tensor([int(w["static_idx"]) for w in windows])]
    return states.to(device), actions[:, :-1].to(device), static.to(device)


@torch.no_grad()
def rollout_batch(
    model: ACTransformerPredictor, states: torch.Tensor, actions: torch.Tensor, static: torch.Tensor,
    n_inside: torch.Tensor, context: int, steps: int, horizons: list[int],
) -> dict[str, torch.Tensor]:
    """Roll every window forward under its recorded actions.

    ``n_inside`` is the number of steps that fall inside each window's stay. Returns the pooled block
    ``(B, horizons, statistics, D)``, in which the slots past the stay end are the zero vector, the three
    per-step errors ``(B, steps)`` (NaN past the stay end) and the predicted states ``(B, steps, D)``.
    """
    device = states.device
    B, _, D = states.shape
    k = context - 1
    n_inside = n_inside.to(device).float()
    n_run = int(min(steps, max(1, int(n_inside.max().item()))))

    z_hist = states[:, :k].float()
    a_hist = actions[:, :k].clone()
    z_current = states[:, k].float()
    trend = z_current - states[:, k - 1].float()
    z_context = z_current

    count = torch.zeros(B, device=device)
    mean = torch.zeros(B, D, device=device)
    m2 = torch.zeros(B, D, device=device)
    snapshots = {}
    nan = torch.full((B,), float("nan"), device=device)
    errors = {name: torch.full((B, steps), float("nan"), device=device)
              for name in ("model_l1", "copy_forward_l1", "linear_extrap_l1")}
    predicted = torch.zeros(B, steps, D, dtype=torch.float16, device=device)

    for step in range(n_run):
        a_next = actions[:, k + step]
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z_pred = model.forward_rollout_step(z_hist, static.float(), a_hist, a_next, z_current)
        z_pred = z_pred.float()

        inside = step < n_inside
        weight = inside.float().unsqueeze(1)
        count = count + inside.float()
        delta = z_pred - mean
        mean = mean + weight * delta / count.clamp(min=1.0).unsqueeze(1)
        m2 = m2 + weight * delta * (z_pred - mean)
        if step + 1 in horizons:
            snapshots[step + 1] = (count.clone(), mean.clone(), m2.clone(), z_pred.clone())

        z_true = states[:, k + step + 1].float()
        for name, estimate in (("model_l1", z_pred), ("copy_forward_l1", z_context),
                               ("linear_extrap_l1", z_context + (step + 1) * trend)):
            errors[name][:, step] = torch.where(inside, (estimate - z_true).abs().sum(-1), nan)
        predicted[:, step] = torch.where(inside.unsqueeze(1), z_pred, torch.zeros_like(z_pred)).half()

        z_hist = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        a_hist = torch.cat([a_hist, a_next.unsqueeze(1)], dim=1)
        z_current = z_pred

    blocks = []
    for h in horizons:
        c, mu, s2, z_h = snapshots.get(h, (count, mean, m2, z_current))
        c = c.unsqueeze(1)
        reached = (c > 0).float()
        block_mean = mu * c / float(h)
        block_m2 = s2 + c * (mu - block_mean) ** 2 + (float(h) - c) * block_mean ** 2
        block_std = torch.sqrt((block_m2 / float(h - 1)).clamp(min=0.0))
        block_last = torch.where(c >= h, z_h, torch.zeros_like(z_h))
        blocks.append(torch.stack([block_mean * reached, block_std * reached, block_last], dim=1))
    return {"pooled": torch.stack(blocks, dim=1), "predicted": predicted, **errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rollout.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    entry = config["models"][args.model]
    context, steps, horizons = config["context"], config["steps"], config["horizons"]
    if max(horizons) > steps:
        raise ValueError("horizons must not exceed steps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_predictor(entry["predictor"], config["predictor"], device)
    cache_dir = Path(config["embedding_dir"]) / entry["embeddings"] / args.split
    index = build_index(sorted(cache_dir.glob("embeddings_*.pt")), context)
    output_dir = ensure_dir(Path(config["output_dir"]) / args.model / args.split)
    index.to_parquet(output_dir / "index.parquet", index=False)

    n, D = len(index), config["predictor"]["state_dim"]
    pooled = np.lib.format.open_memmap(
        output_dir / "pooled.npy", mode="w+", dtype=np.float16, shape=(n, len(horizons), len(STATISTICS), D),
    )
    predicted = np.lib.format.open_memmap(
        output_dir / "predicted.npy", mode="w+", dtype=np.float16, shape=(n, steps, D),
    ) if args.save_predictions else None
    errors = {name: np.full((n, steps), np.nan, dtype=np.float32)
              for name in ("model_l1", "copy_forward_l1", "linear_extrap_l1")}

    for shard_name, group in index.groupby("shard", sort=False):
        shard = torch.load(cache_dir / shard_name, map_location="cpu", weights_only=False, mmap=True)
        for start in range(0, len(group), config["batch_size"]):
            batch = group.iloc[start : start + config["batch_size"]]
            rows = batch["row"].to_numpy()
            states, actions, static = gather_batch(shard, batch["window_idx"].to_numpy(), context + steps, device)
            n_inside = torch.as_tensor(np.clip(batch["length"].to_numpy() - context, 0, steps))
            out = rollout_batch(model, states, actions, static, n_inside, context, steps, horizons)
            pooled[rows] = out["pooled"].half().cpu().numpy()
            for name in errors:
                errors[name][rows] = out[name].cpu().numpy()
            if predicted is not None:
                predicted[rows] = out["predicted"].cpu().numpy()
        logger.info("%s done (%d of %d windows)", shard_name, int(group["row"].max()) + 1, n)

    pooled.flush()
    if predicted is not None:
        predicted.flush()
    np.savez_compressed(output_dir / "errors.npz", **errors)


if __name__ == "__main__":
    main()
