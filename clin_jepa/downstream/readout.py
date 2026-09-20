"""Read-out of a benchmark's tasks: a small MLP on the frozen context block, alone or with the pooled rollout."""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from clin_jepa.downstream.rows import with_labels
from clin_jepa.downstream.tasks import BENCHMARKS, CONTEXT, benchmark_tasks, feature_horizon
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def build_head(spec: dict, in_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in spec["widths"]:
        layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(spec["dropout"])]
        in_dim = width
    return nn.Sequential(*layers, nn.Linear(in_dim, 1))


def standardization(x: torch.Tensor, chunk: int = 8192, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and 1 / (std + eps) over the rows of ``x``; a constant dimension keeps scale 1."""
    total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    for start in range(0, len(x), chunk):
        total += x[start : start + chunk].float().sum(dim=0, dtype=torch.float64)
    mean = (total / len(x)).float()
    squares = torch.zeros_like(total)
    for start in range(0, len(x), chunk):
        squares += ((x[start : start + chunk].float() - mean).double() ** 2).sum(dim=0)
    std = (squares / max(len(x) - 1, 1)).sqrt()
    scale = torch.where(std < eps, torch.ones_like(std), std + eps)
    return mean, 1.0 / scale.float()


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, mean: torch.Tensor, inv_scale: torch.Tensor, size: int) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), size):
        batch = (x[start : start + size].float() - mean) * inv_scale
        out.append(torch.sigmoid(model(batch).squeeze(-1).float()).cpu())
    return torch.cat(out).numpy()


def train_probe(
    spec: dict, seed: int, x: dict[str, torch.Tensor], y: dict[str, np.ndarray],
) -> tuple[float, np.ndarray]:
    """Fit on train and keep the epoch with the best validation AUROC; returns it and the test probabilities."""
    device = x["train"].device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dim = x["train"].shape[1]
    if spec["standardize"]:
        mean, inv_scale = standardization(x["train"])
    else:
        mean, inv_scale = torch.zeros(dim, device=device), torch.ones(dim, device=device)

    model = build_head(spec, dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    target = torch.as_tensor(y["train"], dtype=torch.float32, device=device)
    n_positive = float(target.sum())
    pos_weight = torch.tensor([(len(target) - n_positive) / max(n_positive, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n, batch_size = len(target), spec["batch_size"]
    scheduler = None
    if spec["warmup_fraction"]:
        total = max((n + batch_size - 1) // batch_size * spec["max_epochs"], 1)
        warmup = max(int(round(total * spec["warmup_fraction"])), 1)

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1.0 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_epoch, best_state, stale = -np.inf, -1, None, 0
    for epoch in range(spec["max_epochs"]):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            batch = (x["train"][rows].float() - mean) * inv_scale
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch).squeeze(-1), target[rows])
            if spec.get("l2"):
                loss = loss + spec["l2"] * sum((w * w).sum() for name, w in model.named_parameters()
                                               if name.endswith("weight"))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        scores = predict(model, x["val"], mean, inv_scale, spec["eval_batch_size"])
        auroc = roc_auc_score(y["val"].astype(np.float64), scores.astype(np.float64))
        if auroc > best:
            best, best_epoch, stale = auroc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= spec["patience"]:
                break
    model.load_state_dict(best_state)
    logger.info("validation AUROC %.4f at epoch %d of %d", best, best_epoch + 1, epoch + 1)
    return best, predict(model, x["test"], mean, inv_scale, spec["eval_batch_size"])


def load_features(config: dict, encoder: str, split: str, variant: str, stay_level: bool) -> tuple[pd.DataFrame, dict]:
    """Rows of a split and their feature blocks: the context block and, for ``full``, the pooled rollout per horizon."""
    feature_dir = Path(config["output_dir"]) / "features" / encoder / split
    rows = pd.read_parquet(feature_dir / "index.parquet")[["stay_id", "start_step"]]
    rows["target_hour"] = rows["start_step"] + CONTEXT - 1
    keep = np.flatnonzero((rows["start_step"] == 0).to_numpy()) if stay_level else np.arange(len(rows))
    history = np.load(feature_dir / "history.npy", mmap_mode="r")
    blocks = {"history": torch.from_numpy(np.ascontiguousarray(history[keep]).reshape(len(keep), -1))}
    if variant == "full":
        rollout_dir = Path(config["rollout_dir"]) / encoder / split
        index = pd.read_parquet(rollout_dir / "index.parquet")
        if not np.array_equal(index[["stay_id", "start_step"]].to_numpy(), rows[["stay_id", "start_step"]].to_numpy()):
            raise ValueError(f"{rollout_dir} and {feature_dir} index different windows")
        pooled = np.load(rollout_dir / "pooled.npy", mmap_mode="r")
        if pooled.shape[1] != len(config["rollout_horizons"]):
            raise ValueError(f"{rollout_dir} does not hold horizons {config['rollout_horizons']}")
        for i, horizon in enumerate(config["rollout_horizons"]):
            blocks[horizon] = torch.from_numpy(np.ascontiguousarray(pooled[keep, i]).reshape(len(keep), -1))
    return rows.iloc[keep].reset_index(drop=True), blocks


def to_device(blocks: dict, names: list, at: np.ndarray, device: torch.device) -> torch.Tensor:
    at = torch.as_tensor(at)
    x = torch.empty((len(at), sum(blocks[name].shape[1] for name in names)), dtype=torch.float16, device=device)
    column = 0
    for name in names:
        width = blocks[name].shape[1]
        x[:, column : column + width] = blocks[name][at]
        column += width
    if not torch.isfinite(x).all():
        raise ValueError("non-finite feature values")
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream.yaml")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--variant", required=True, choices=["hist", "full"])
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    spec, output_dir = config["readout"][args.benchmark], Path(config["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = {(r["encoder"], r["task"], r["variant"]): r["seed"] for r in config["readout"]["seeds"]}

    for stay_level in (False, True):
        tasks = [task for task in benchmark_tasks(args.benchmark) if task.hourly != stay_level]
        if not tasks:
            continue
        rows, blocks = {}, {}
        for split in SPLITS:
            rows[split], blocks[split] = load_features(config, args.encoder, split, args.variant, stay_level)
            rows[split] = with_labels(rows[split], output_dir / "labels", [t.name for t in tasks], not stay_level)
        for task in tasks:
            path = output_dir / "predictions" / task.name / f"{args.encoder}_{args.variant}.parquet"
            if path.exists():
                continue
            labelled = {s: np.flatnonzero(rows[s][task.name].notna().to_numpy()) for s in SPLITS}
            y = {s: rows[s][task.name].to_numpy(dtype=np.float32)[labelled[s]] for s in SPLITS}
            if y["train"].sum() < 1 or y["train"].all() or len(np.unique(y["val"])) < 2:
                logger.warning("%s skipped: training or validation labels hold a single class", task.name)
                continue
            names = ["history"] + ([feature_horizon(task)] if args.variant == "full" else [])
            x = {s: to_device(blocks[s], names, labelled[s], device) for s in SPLITS}
            seed = seeds.get((args.encoder, task.name, args.variant), config["readout"]["seed"])
            _, probabilities = train_probe(spec, seed, x, y)
            del x
            cell = rows["test"].iloc[labelled["test"]][["stay_id", "target_hour"]]
            cell.assign(label=y["test"], p0=probabilities).to_parquet(ensure_dir(path.parent) / path.name, index=False)
            if len(np.unique(y["test"])) == 2:
                logger.info("%s: test AUROC %.4f", task.name, roc_auc_score(y["test"], probabilities))


if __name__ == "__main__":
    main()
