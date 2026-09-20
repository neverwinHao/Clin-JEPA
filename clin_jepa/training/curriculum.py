"""Co-training curriculum for the encoder LoRA and the latent trajectory predictor.

Phases (optimizer steps each, zero = skipped): warmup -> co_training -> alignment -> sync -> finalization.
"""

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

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.encoder_ops import (
    any_rank,
    atomic_torch_save,
    collect_lora_adapter_state,
    create_target_encoder_fp32,
    ema_update,
    encode_texts_batched,
    hard_sync,
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

RESUME_SCHEMA = "clin_jepa_curriculum"
GUARD_MIN_STEP = 100
PHASES = {
    "warmup": {"train_encoder": False, "ema_target": False},
    "co_training": {"train_encoder": True, "ema_target": True},
    "alignment": {"train_encoder": False, "ema_target": True},
    "finalization": {"train_encoder": False, "ema_target": False},
}

_STOP = False


def _on_signal(signum: int, frame) -> None:
    global _STOP
    _STOP = True


def encode_windows(
    model, tokenizer, windows: list[dict], device: torch.device, max_seq_len: int,
    train_mode: bool, output_dtype: torch.dtype,
) -> dict:
    """Encode the demographics, state and action texts of each window into padded (B, T, 4096) tensors."""
    lengths = torch.tensor([w["length"] for w in windows], dtype=torch.long)
    B, T = len(windows), int(lengths.max().item())
    z_states = torch.zeros(B, T, 4096, dtype=output_dtype, device=device)
    z_actions = torch.zeros(B, T, 4096, dtype=output_dtype, device=device)
    z_static = torch.zeros(B, 4096, dtype=output_dtype, device=device)
    padding_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for i, w in enumerate(windows):
        L = w["length"]
        z = encode_texts_batched(
            model, tokenizer, [w["demographics_text"]] + w["state_texts"] + w["action_texts"],
            device=device, max_seq_len=max_seq_len, train_mode=train_mode, output_dtype=output_dtype,
        )
        z_static[i] = z[0]
        z_states[i, :L] = z[1 : 1 + L]
        z_actions[i, :L] = z[1 + L : 1 + 2 * L]
        padding_mask[i, :L] = True

    return {
        "z_states": z_states, "z_actions": z_actions, "z_static": z_static,
        "lengths": lengths.to(device), "padding_mask": padding_mask,
    }


def teacher_forcing_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    l1 = (preds - targets).abs().sum(dim=-1)
    return (l1 * mask.float()).sum() / mask.float().sum().clamp(min=1)


def rollout_loss(
    predictor, z_states: torch.Tensor, z_static: torch.Tensor, z_actions: torch.Tensor,
    lengths: torch.Tensor, horizon: int, targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """l1 over ``horizon`` steps from a random start; native when ``targets`` is None, else teacher-forced."""
    device = z_states.device
    min_len = int(lengths.min().item())
    if min_len < horizon + 2:
        return torch.tensor(0.0, device=device)
    k = random.randint(1, min_len - horizon - 1)

    total = torch.tensor(0.0, device=device)
    if targets is not None:
        for step in range(horizon):
            z_seq = z_states[:, : k + step + 1, :]
            za_seq = z_actions[:, : k + step + 1, :]
            seq_lengths = torch.full((z_seq.shape[0],), z_seq.shape[1], device=device, dtype=torch.long)
            z_pred = predictor(z_seq, z_static, za_seq, seq_lengths, return_last_only=True)
            total = total + (z_pred - targets[:, k + step + 1, :]).abs().sum(dim=-1).mean()
        return total

    z_hist, za_hist, z_current = z_states[:, :k, :], z_actions[:, :k, :], z_states[:, k, :]
    for step in range(horizon):
        za_next = z_actions[:, k + step, :]
        z_seq = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        za_seq = torch.cat([za_hist, za_next.unsqueeze(1)], dim=1)
        seq_lengths = torch.full((z_seq.shape[0],), z_seq.shape[1], device=device, dtype=torch.long)
        z_pred = predictor(z_seq, z_static, za_seq, seq_lengths, return_last_only=True)
        total = total + (z_pred - z_states[:, k + step + 1, :]).abs().sum(dim=-1).mean()
        z_hist = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        za_hist = torch.cat([za_hist, za_next.unsqueeze(1)], dim=1)
        z_current = z_pred
    return total


def phase_loss(
    phase: str, online, predictor: nn.Module, tokenizer, windows: list[dict],
    device: torch.device, max_seq_len: int, horizon: int,
) -> torch.Tensor:
    spec = PHASES[phase]
    online.set_adapter("online")
    enc = encode_windows(online, tokenizer, windows, device, max_seq_len,
                         train_mode=spec["train_encoder"], output_dtype=torch.bfloat16)
    targets = None
    if spec["ema_target"]:
        online.set_adapter("target")
        with torch.no_grad():
            targets = encode_windows(online, tokenizer, windows, device, max_seq_len,
                                     train_mode=False, output_dtype=torch.float32)["z_states"].detach()
        online.set_adapter("online")

    preds = predictor(enc["z_states"], enc["z_static"], enc["z_actions"], enc["lengths"])
    tf_targets = (targets if targets is not None else enc["z_states"])[:, 1:, :]
    tf = teacher_forcing_loss(preds, tf_targets, enc["padding_mask"][:, 1:])
    roll = rollout_loss(predictor, enc["z_states"], enc["z_static"], enc["z_actions"],
                        enc["lengths"], horizon, targets=targets)
    return tf + roll


def set_encoder_trainable(online: nn.Module, trainable: bool) -> None:
    for name, p in online.named_parameters():
        if "lora_" in name and ".online." in name:
            p.requires_grad = trainable


@torch.no_grad()
def evaluate(
    online, predictor: nn.Module, tokenizer, val_loader: DataLoader, device: torch.device,
    max_seq_len: int, max_windows: int, ema_target: bool,
) -> dict:
    """Validation teacher-forcing loss and z_std, the mean per-feature std of the state embeddings over time."""
    predictor.eval()
    n_windows, total, z_std = 0, 0.0, []
    for collated in val_loader:
        if n_windows >= max_windows:
            break
        windows = collated["batch"]
        if not windows:
            continue
        online.set_adapter("online")
        enc = encode_windows(online, tokenizer, windows, device, max_seq_len,
                             train_mode=False, output_dtype=torch.bfloat16)
        target_states = enc["z_states"]
        if ema_target:
            online.set_adapter("target")
            target_states = encode_windows(online, tokenizer, windows, device, max_seq_len,
                                           train_mode=False, output_dtype=torch.float32)["z_states"]
            online.set_adapter("online")
        preds = predictor(enc["z_states"], enc["z_static"], enc["z_actions"], enc["lengths"])
        total += teacher_forcing_loss(preds, target_states[:, 1:, :].float(), enc["padding_mask"][:, 1:]).item()
        n_windows += len(windows)
        z_std.append(enc["z_states"].float().std(dim=1).mean().detach())
    predictor.train()
    return {
        "val_loss": total / max(1, len(z_std)),
        "z_std": torch.stack(z_std).mean().item() if z_std else 0.0,
    }


def save_checkpoint(online, predictor: nn.Module, tokenizer, save_dir: Path, tag: str, meta: dict) -> None:
    out = save_dir / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    online.set_adapter("online")
    online.save_pretrained(out, selected_adapters=["online"])
    tokenizer.save_pretrained(out / "online")
    torch.save(predictor.state_dict(), out / "predictor.pt")
    with open(out / "meta.json", "w") as f:
        json.dump({"tag": tag, **meta}, f, indent=2)


def phase_at(step: int, schedule: list[tuple[str, int]]) -> str:
    end = 0
    for name, steps in schedule:
        end += steps
        if step < end:
            return name
    return schedule[-1][0]


def train(config: dict, args: argparse.Namespace) -> None:
    rank, world_size, distributed = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    if is_main:
        setup_logging()

    train_cfg = config["training"]
    seed = train_cfg["seed"]
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    online, tokenizer = load_encoder(config, device)
    online = create_target_encoder_fp32(online)
    ema_pairs = online_target_pairs(online)

    predictor = ACTransformerPredictor(**config["predictor"]).to(device)
    if distributed:
        with torch.no_grad():
            for p in predictor.parameters():
                dist.broadcast(p.data, src=0)

    data_cfg, log_cfg = config["data"], config["logging"]
    train_ds = TrajectoryWindowDataset(data_cfg["trajectory_dir"], "train", max_windows=args.max_windows)
    val_ds = TrajectoryWindowDataset(data_cfg["trajectory_dir"], "val", max_windows=log_cfg["val_windows"])

    per_gpu_batch = train_cfg["per_gpu_batch_size"]
    grad_accum, remainder = divmod(train_cfg["effective_batch_size"], per_gpu_batch * world_size)
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

    if set(train_cfg["phases"]) != set(PHASES):
        raise ValueError(f"training.phases must have the keys {list(PHASES)}")
    schedule = [(name, int(train_cfg["phases"][name])) for name in PHASES]
    total_steps = sum(steps for _, steps in schedule)
    max_steps = train_cfg.get("max_steps")
    guard = train_cfg.get("collapse_guard")

    encoder_params = [online_p for online_p, _ in ema_pairs]
    predictor_params = list(predictor.parameters())
    wd = train_cfg["weight_decay"]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": train_cfg["lr_encoder"], "weight_decay": wd},
        {"params": predictor_params, "lr": train_cfg["lr_predictor"], "weight_decay": wd},
    ])
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = Path(config["save"]["dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    max_seq_len = data_cfg["max_seq_len"]
    horizon = train_cfg["rollout_horizon"]
    tau = train_cfg["ema_tau"]

    state = {"global_step": 0, "micro_step": 0, "epoch": 0, "micro_step_in_epoch": 0,
             "phase": phase_at(0, schedule), "last_eval": None}

    resume_path = save_dir / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("schema_version") != RESUME_SCHEMA:
            raise RuntimeError(f"{resume_path} was not written by this trainer")
        restore_lora_adapter_state(online, saved["lora_adapter_state"])
        predictor.load_state_dict(saved["predictor_state"])
        predictor.to(device)
        optimizer.load_state_dict(saved["optimizer_state"])
        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
        scheduler.load_state_dict(saved["scheduler_state"])
        restore_rng_state(saved["rng_state"])
        state = saved["state"]
        if is_main:
            logger.info("resumed from %s at step %d (%s)", resume_path, state["global_step"], state["phase"])
        if distributed:
            dist.barrier()
    set_encoder_trainable(online, PHASES[state["phase"]]["train_encoder"])

    def save_resume() -> None:
        atomic_torch_save({
            "schema_version": RESUME_SCHEMA,
            "lora_adapter_state": collect_lora_adapter_state(online),
            "predictor_state": predictor.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "state": state,
            "rng_state": snapshot_rng_state(),
            "config": config,
        }, save_dir / "resume.pt")

    def checkpoint(tag: str) -> None:
        save_checkpoint(online, predictor, tokenizer, save_dir, tag, {
            "step": state["global_step"], "phase": state["phase"],
            "last_eval": state["last_eval"], "elapsed_seconds": time.time() - t0,
        })

    def broadcast_adapters() -> None:
        if distributed:
            dist.barrier()
            with torch.no_grad():
                for online_p, target_p in ema_pairs:
                    dist.broadcast(online_p.data, src=0)
                    dist.broadcast(target_p.data, src=0)
            dist.barrier()

    def stop(reason: str) -> None:
        if distributed:
            dist.barrier()
        if is_main:
            save_resume()
            logger.info("%s at step %d (%s); resume state saved", reason, state["global_step"], state["phase"])
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if is_main:
        logger.info("schedule %s, %d steps, %d GPUs x %d x accumulation %d",
                    schedule, total_steps, world_size, per_gpu_batch, grad_accum)
    online.train()
    predictor.train()
    t0 = time.time()

    if sampler is not None:
        sampler.set_epoch(state["epoch"])
    train_iter = iter(train_loader)
    try:
        for _ in range(state["micro_step_in_epoch"]):
            next(train_iter)
    except StopIteration:
        state["epoch"] += 1
        state["micro_step_in_epoch"] = 0
        if sampler is not None:
            sampler.set_epoch(state["epoch"])
        train_iter = iter(train_loader)

    while state["global_step"] < total_steps:
        try:
            collated = next(train_iter)
            state["micro_step_in_epoch"] += 1
        except StopIteration:
            state["epoch"] += 1
            if sampler is not None:
                sampler.set_epoch(state["epoch"])
            train_iter = iter(train_loader)
            collated = next(train_iter)
            state["micro_step_in_epoch"] = 1
        windows = collated["batch"]
        if not windows:
            continue

        phase = state["phase"]
        loss = phase_loss(phase, online, predictor, tokenizer, windows, device, max_seq_len, horizon)
        (loss / grad_accum).backward()
        state["micro_step"] += 1
        if state["micro_step"] % grad_accum:
            continue

        train_encoder = PHASES[phase]["train_encoder"]
        if distributed:
            if train_encoder:
                manual_allreduce_grads(encoder_params)
            manual_allreduce_grads(predictor_params)
        if train_encoder:
            torch.nn.utils.clip_grad_norm_(encoder_params, train_cfg["gradient_clip_encoder"])
        torch.nn.utils.clip_grad_norm_(predictor_params, train_cfg["gradient_clip_predictor"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        state["global_step"] += 1
        step = state["global_step"]

        if PHASES[phase]["ema_target"]:
            ema_update(ema_pairs, tau)

        new_phase = phase_at(step, schedule) if step < total_steps else phase
        if new_phase != phase:
            if is_main:
                checkpoint(f"after_{phase}")
            if new_phase == "finalization" and train_cfg["sync"]:
                hard_sync(ema_pairs)
                state["phase"] = "sync"
                if is_main:
                    checkpoint("after_sync")
            set_encoder_trainable(online, PHASES[new_phase]["train_encoder"])
            broadcast_adapters()
            state["phase"] = new_phase
            if is_main:
                logger.info("step %d: %s -> %s", step, phase, new_phase)

        if is_main and step % log_cfg["log_every"] == 0:
            logger.info("step %d/%d [%s] loss %.4f lr %.1e/%.1e %.1f min", step, total_steps, state["phase"],
                        loss.item(), *scheduler.get_last_lr(), (time.time() - t0) / 60)

        if step % 50 == 0:
            if distributed:
                dist.barrier()
            if is_main:
                save_resume()
            if distributed:
                dist.barrier()

        if step % log_cfg["eval_every"] == 0:
            if distributed:
                dist.barrier()
            if is_main:
                metrics = evaluate(online, predictor, tokenizer, val_loader, device, max_seq_len,
                                   log_cfg["eval_max_windows"], PHASES[state["phase"]]["ema_target"])
                state["last_eval"] = {"step": step, **metrics}
                logger.info("eval step %d [%s] val_loss %.4f z_std %.4f",
                            step, state["phase"], metrics["val_loss"], metrics["z_std"])
                checkpoint("latest")
                if guard is not None and metrics["z_std"] < guard and step > GUARD_MIN_STEP:
                    global _STOP
                    _STOP = True
                save_resume()
                online.train()
                predictor.train()
            if distributed:
                dist.barrier()

        if max_steps is not None and step >= max_steps:
            stop("max_steps reached")
        if any_rank(_STOP, device):
            stop("stopping")

    if is_main:
        checkpoint("final")
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump({"total_steps": state["global_step"], "schedule": schedule,
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
