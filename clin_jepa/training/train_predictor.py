"""Train the latent trajectory predictor on the cached embeddings of a frozen encoder."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.predictor_dataset import PredictorDataset, predictor_collate_fn
from clin_jepa.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def teacher_forcing_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    l1 = (preds - targets).abs().sum(dim=-1)
    return (l1 * mask.float()).sum() / mask.float().sum().clamp(min=1)


def rollout_loss(
    model: ACTransformerPredictor, z_states: torch.Tensor, z_static: torch.Tensor,
    z_actions: torch.Tensor, lengths: torch.Tensor, horizon: int,
) -> torch.Tensor:
    """Self-fed l1 over ``horizon`` steps from a random start."""
    device = z_states.device
    min_len = int(lengths.min().item())
    if min_len < horizon + 2:
        return torch.tensor(0.0, device=device)
    k = random.randint(1, min_len - horizon - 1)

    z_hist = z_states[:, :k, :].float()
    za_hist = z_actions[:, :k, :].float()
    z_current = z_states[:, k, :].float()
    total = torch.tensor(0.0, device=device)
    for step in range(horizon):
        z_pred = model.forward_rollout_step(
            z_history=z_hist, z_static=z_static, za_history=za_hist,
            za_next=z_actions[:, k + step, :].float(), z_last=z_current,
        )
        total = total + (z_pred - z_states[:, k + step + 1, :].float()).abs().sum(dim=-1).mean()
        z_hist = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        za_hist = torch.cat([za_hist, z_actions[:, k + step : k + step + 1, :].float()], dim=1)
        z_current = z_pred
    return total


@torch.no_grad()
def evaluate(model, val_loader, device: torch.device, max_batches: int = 100) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        if n >= max_batches:
            break
        z_states = batch["z_states"].to(device)
        preds = model(z_states, batch["z_static"].to(device), batch["z_actions"].to(device),
                      batch["lengths"].to(device))
        total += teacher_forcing_loss(preds, z_states[:, 1:, :].float(),
                                      batch["padding_mask"].to(device)[:, 1:]).item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(config: dict, seed: int, save_dir: Path, max_windows: int | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ACTransformerPredictor(**config["predictor"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    embedding_dir = config["data"]["embedding_dir"]
    train_dataset = PredictorDataset(embedding_dir, "train", max_windows=max_windows)
    val_dataset = PredictorDataset(embedding_dir, "val", max_windows=min(max_windows or 10000, 10000))

    cfg = config["training"]
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=predictor_collate_fn, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=predictor_collate_fn, num_workers=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    noise_std = 0.0
    if cfg["embedding_noise_std"] > 0:
        noise_std = cfg["embedding_noise_std"] * train_dataset[0]["z_states"].float().std().item()

    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("%.1fM parameters, %d epochs x %d steps, seed %d", n_params / 1e6,
                cfg["epochs"], len(train_loader), seed)

    best_val, patience, global_step, t0 = float("inf"), 0, 0, time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        tf_sum = roll_sum = 0.0
        steps = 0
        for batch in train_loader:
            z_states = batch["z_states"].to(device)
            z_actions = batch["z_actions"].to(device)
            z_static = batch["z_static"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["padding_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16"):
                preds = model(z_states, z_static, z_actions, lengths, noise_std=noise_std)
                tf = teacher_forcing_loss(preds, z_states[:, 1:, :].float(), mask[:, 1:])
                roll = rollout_loss(model, z_states, z_static, z_actions, lengths, cfg["rollout_horizon"])
                loss = tf + roll
            loss.backward()
            if cfg["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["gradient_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tf_sum += tf.item()
            roll_sum += roll.item()
            steps += 1
            global_step += 1
            if global_step % config["logging"]["log_every"] == 0:
                logger.info("step %d/%d tf %.4f rollout %.4f lr %.2e", global_step, total_steps,
                            tf_sum / steps, roll_sum / steps, scheduler.get_last_lr()[0])

        val_loss = evaluate(model, val_loader, device)
        logger.info("epoch %d/%d tf %.4f rollout %.4f val %.4f (best %.4f)", epoch + 1, cfg["epochs"],
                    tf_sum / max(steps, 1), roll_sum / max(steps, 1), val_loss, best_val)
        state = {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss,
                 "seed": seed, "config": config}
        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(state, save_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break

    with open(save_dir / "training_meta.json", "w") as f:
        json.dump({"total_steps": global_step, "total_epochs": epoch + 1, "best_val_loss": best_val,
                   "seed": seed, "n_params": n_params, "elapsed_seconds": time.time() - t0,
                   "config": config}, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_windows", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    save_dir = Path(config["save"]["dir"])
    seeds = config["training"].get("seeds")
    if seeds:
        for seed in seeds:
            train(config, seed, save_dir / f"seed_{seed}", args.max_windows)
    else:
        train(config, config["training"]["seed"], save_dir, args.max_windows)


if __name__ == "__main__":
    main()
