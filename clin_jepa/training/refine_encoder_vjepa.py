"""V-JEPA 2-AC style encoder refinement: predict the embeddings of masked hours from the visible ones."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from clin_jepa.model.jepa_predictor import JEPAPredictor
from clin_jepa.training.encoder_ops import (
    any_rank,
    atomic_torch_save,
    collect_lora_adapter_state,
    create_target_encoder_fp32,
    ema_update,
    encode_texts_batched,
    load_encoder,
    manual_allreduce_grads,
    online_target_pairs,
    restore_lora_adapter_state,
    restore_rng_state,
    setup_distributed,
    snapshot_rng_state,
)
from clin_jepa.training.trajectory_dataset import TrajectoryWindowDataset, trajectory_collate_fn
from clin_jepa.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

RESUME_SCHEMA = "clin_jepa_vjepa_refine"
_STOP = False


def _on_signal(signum: int, frame) -> None:
    global _STOP
    _STOP = True


def sample_mask(
    length: int, mask_ratio: float, min_visible: int, rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Visible and masked hours of a window; state and action of an hour are masked together."""
    n_mask = max(min(int(round(mask_ratio * length)), length - min_visible), 0)
    if n_mask == 0:
        return list(range(length)), []
    masked = {int(x) for x in rng.choice(length, size=n_mask, replace=False)}
    return [i for i in range(length) if i not in masked], sorted(masked)


def masked_prediction_loss(
    online, jepa: nn.Module, tokenizer, window: dict, device: torch.device, max_seq_len: int,
    visible: list[int], masked: list[int], train_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """l1 between predicted and target embeddings of the masked hours; also returns the online state embeddings."""
    L = window["length"]
    texts = [window["demographics_text"]] + window["state_texts"] + window["action_texts"]

    online.set_adapter("online")
    z_online = encode_texts_batched(online, tokenizer, texts, device=device, max_seq_len=max_seq_len,
                                    train_mode=train_mode, output_dtype=torch.bfloat16)
    online.set_adapter("target")
    with torch.no_grad():
        z_target = encode_texts_batched(online, tokenizer, texts, device=device, max_seq_len=max_seq_len,
                                        train_mode=False, output_dtype=torch.float32)
    online.set_adapter("online")

    memory, memory_pos = [z_online[0]], [0]
    for t in visible:
        memory += [z_online[1 + t], z_online[1 + L + t]]
        memory_pos += [1 + 2 * t, 2 + 2 * t]
    targets, query_pos = [], []
    for t in masked:
        targets += [z_target[1 + t], z_target[1 + L + t]]
        query_pos += [1 + 2 * t, 2 + 2 * t]

    predicted = jepa(
        torch.stack(memory).unsqueeze(0),
        torch.tensor(memory_pos, device=device, dtype=torch.long).unsqueeze(0),
        torch.tensor(query_pos, device=device, dtype=torch.long).unsqueeze(0),
    )
    targets = torch.stack(targets).unsqueeze(0)
    loss = (predicted.float() - targets.detach().float()).abs().sum(dim=-1).mean()
    return loss, z_online[1 : 1 + L]


@torch.no_grad()
def evaluate(
    online, jepa: nn.Module, tokenizer, val_loader: DataLoader, device: torch.device, max_seq_len: int,
    max_windows: int, mask_ratio: float, min_visible: int, seed: int,
) -> dict:
    jepa.eval()
    rng = np.random.default_rng(seed)
    total, n_windows, z_std = 0.0, 0, []
    for collated in val_loader:
        for window in collated["batch"]:
            if n_windows >= max_windows:
                break
            visible, masked = sample_mask(window["length"], mask_ratio, min_visible, rng)
            if not masked:
                continue
            loss, z_states = masked_prediction_loss(online, jepa, tokenizer, window, device, max_seq_len,
                                                    visible, masked, train_mode=False)
            z_std.append(z_states.float().std(dim=0).mean().detach())
            total += loss.item()
            n_windows += 1
        if n_windows >= max_windows:
            break
    jepa.train()
    return {
        "val_loss": total / max(n_windows, 1),
        "z_std": torch.stack(z_std).mean().item() if z_std else 0.0,
    }


def save_checkpoint(online, jepa: nn.Module, tokenizer, save_dir: Path, tag: str, meta: dict) -> None:
    out = save_dir / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    online.set_adapter("online")
    online.save_pretrained(out, selected_adapters=["online"])
    tokenizer.save_pretrained(out / "online")
    torch.save(jepa.state_dict(), out / "jepa_predictor.pt")
    with open(out / "meta.json", "w") as f:
        json.dump({"tag": tag, **meta}, f, indent=2)


def train(config: dict, args: argparse.Namespace) -> None:
    rank, world_size, distributed = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    if is_main:
        setup_logging()

    cfg = config["training"]
    seed = cfg["seed"]
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    online, tokenizer = load_encoder(config, device)
    online = create_target_encoder_fp32(online)
    ema_pairs = online_target_pairs(online)

    jepa = JEPAPredictor(**config["jepa_predictor"]).to(device)
    if distributed:
        with torch.no_grad():
            for p in jepa.parameters():
                dist.broadcast(p.data, src=0)

    data_cfg, log_cfg = config["data"], config["logging"]
    train_ds = TrajectoryWindowDataset(data_cfg["trajectory_dir"], "train", max_windows=args.max_windows)
    val_ds = TrajectoryWindowDataset(data_cfg["trajectory_dir"], "val", max_windows=log_cfg["val_windows"])

    per_gpu_batch = cfg["per_gpu_batch_size"]
    grad_accum, remainder = divmod(cfg["effective_batch_size"], per_gpu_batch * world_size)
    if remainder or grad_accum < 1:
        raise ValueError("effective_batch_size must be a multiple of per_gpu_batch_size x number of GPUs")

    sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        if distributed else None
    )
    train_loader = DataLoader(
        train_ds, batch_size=per_gpu_batch, sampler=sampler, shuffle=sampler is None,
        collate_fn=trajectory_collate_fn, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=per_gpu_batch, shuffle=False, collate_fn=trajectory_collate_fn, num_workers=2,
    )

    encoder_params = [online_p for online_p, _ in ema_pairs]
    predictor_params = list(jepa.parameters())
    wd = cfg["weight_decay"]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["lr_encoder"], "weight_decay": wd},
        {"params": predictor_params, "lr": cfg["lr_predictor"], "weight_decay": wd},
    ])
    total_steps = (len(train_loader) * cfg["epochs"]) // grad_accum
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = Path(config["save"]["dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    guard = cfg["collapse_guard"]
    early = cfg["early_stop"]
    max_seq_len = data_cfg["max_seq_len"]

    state = {"global_step": 0, "micro_step": 0, "epoch": 0, "micro_step_in_epoch": 0,
             "best_val_loss": float("inf"), "patience": 0}

    resume_path = save_dir / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("schema_version") != RESUME_SCHEMA:
            raise RuntimeError(f"{resume_path} was not written by this trainer")
        restore_lora_adapter_state(online, saved["lora_adapter_state"])
        jepa.load_state_dict(saved["jepa_predictor_state"])
        jepa.to(device)
        optimizer.load_state_dict(saved["optimizer_state"])
        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
        scheduler.load_state_dict(saved["scheduler_state"])
        restore_rng_state(saved["rng_state"])
        state = saved["state"]
        if distributed:
            dist.barrier()

    def save_resume() -> None:
        atomic_torch_save({
            "schema_version": RESUME_SCHEMA,
            "lora_adapter_state": collect_lora_adapter_state(online),
            "jepa_predictor_state": jepa.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "state": state,
            "rng_state": snapshot_rng_state(),
            "config": config,
        }, save_dir / "resume.pt")

    def checkpoint(tag: str, metrics: dict) -> None:
        save_checkpoint(online, jepa, tokenizer, save_dir, tag, {
            "step": state["global_step"], **metrics, "elapsed_seconds": time.time() - t0,
        })

    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if is_main:
        logger.info("%d steps, %d GPUs x %d x accumulation %d", total_steps, world_size, per_gpu_batch, grad_accum)
    online.train()
    jepa.train()
    t0 = time.time()
    rng = np.random.default_rng(seed + rank)
    loss_sum, n_windows = 0.0, 0
    finished = False
    start_epoch, skip = state["epoch"], state["micro_step_in_epoch"]

    for epoch in range(start_epoch, cfg["epochs"]):
        state["epoch"] = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_idx, collated in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < skip:
                for window in collated["batch"]:
                    sample_mask(window["length"], cfg["mask_ratio"], cfg["min_visible"], rng)
                continue
            losses = []
            for window in collated["batch"]:
                visible, masked = sample_mask(window["length"], cfg["mask_ratio"], cfg["min_visible"], rng)
                if masked:
                    losses.append(masked_prediction_loss(online, jepa, tokenizer, window, device, max_seq_len,
                                                         visible, masked, train_mode=True)[0])
            if not losses:
                continue
            micro_loss = torch.stack(losses).mean() / grad_accum
            micro_loss.backward()
            loss_sum += micro_loss.item() * grad_accum * len(losses)
            n_windows += len(losses)
            state["micro_step"] += 1
            state["micro_step_in_epoch"] = batch_idx + 1
            if state["micro_step"] % grad_accum:
                continue

            if distributed:
                manual_allreduce_grads(encoder_params)
                manual_allreduce_grads(predictor_params)
            torch.nn.utils.clip_grad_norm_(encoder_params, cfg["gradient_clip_encoder"])
            torch.nn.utils.clip_grad_norm_(predictor_params, cfg["gradient_clip_predictor"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            state["global_step"] += 1
            step = state["global_step"]

            tau = cfg["tau_init"] + (cfg["tau_final"] - cfg["tau_init"]) * (step / max(total_steps, 1))
            ema_update(ema_pairs, tau)

            if step % 50 == 0:
                if distributed:
                    dist.barrier()
                if is_main:
                    save_resume()
                if distributed:
                    dist.barrier()

            if is_main and step % log_cfg["log_every"] == 0:
                logger.info("step %d/%d loss %.4f tau %.4f lr %.2e/%.2e %.1f min", step, total_steps,
                            loss_sum / max(n_windows, 1), tau, *scheduler.get_last_lr(), (time.time() - t0) / 60)

            if step % log_cfg["eval_every"] == 0:
                if distributed:
                    dist.barrier()
                if is_main:
                    metrics = evaluate(online, jepa, tokenizer, val_loader, device, max_seq_len,
                                       log_cfg["eval_max_windows"], cfg["mask_ratio"], cfg["min_visible"], seed)
                    logger.info("eval step %d val_loss %.4f z_std %.4f (best %.4f)", step,
                                metrics["val_loss"], metrics["z_std"], state["best_val_loss"])
                    if metrics["z_std"] < guard:
                        logger.info("z_std %.4f below %.2f: stopping", metrics["z_std"], guard)
                        finished = True
                    checkpoint("latest", metrics)
                    if metrics["val_loss"] < state["best_val_loss"] * (1.0 - early["min_delta"]):
                        state["best_val_loss"], state["patience"] = metrics["val_loss"], 0
                        checkpoint("best", metrics)
                    else:
                        state["patience"] += 1
                        if step >= early["min_steps"] and state["patience"] >= early["patience"]:
                            finished = True
                    save_resume()
                    online.train()
                    jepa.train()
                if distributed:
                    flag = torch.tensor([int(finished)], device=device, dtype=torch.long)
                    dist.broadcast(flag, src=0)
                    finished = bool(flag.item())
                    dist.barrier()

            if any_rank(_STOP, device):
                if distributed:
                    dist.barrier()
                if is_main:
                    save_resume()
                if distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                sys.exit(0)
            if finished:
                break
        if finished:
            break

    if is_main:
        checkpoint("final", {"best_val_loss": state["best_val_loss"]})
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump({"total_steps": state["global_step"], "best_val_loss": state["best_val_loss"],
                       "elapsed_seconds": time.time() - t0, "config": config}, f, indent=2, default=str)
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_windows", type=int, default=None)
    args = parser.parse_args()

    train(load_config(args.config), args)


if __name__ == "__main__":
    main()
