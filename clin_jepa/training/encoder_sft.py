"""Encoder SFT: next-token fine-tuning of a Qwen3-8B LoRA on the per-hour state and action texts."""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

apply_liger_kernel_to_qwen3()

from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening  # noqa: E402

from clin_jepa.training.encoder_ops import setup_distributed  # noqa: E402
from clin_jepa.utils import load_config, set_seed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


class PerHourSFTDataset(Dataset):
    """Every hour's state text and action text as independent samples."""

    def __init__(
        self, shard_dir: str | Path, tokenizer, max_seq_len: int,
        source: str = "both", max_samples: int | None = None,
    ) -> None:
        files = sorted(Path(shard_dir).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"no .pt shards in {shard_dir}")
        self.texts: list[str] = []
        for path in files:
            per_hour = torch.load(path, weights_only=False, map_location="cpu")["per_hour"]
            if source == "both":
                for state, action in zip(per_hour["state_texts"], per_hour["action_texts"]):
                    self.texts += [state, action]
            else:
                self.texts.extend(per_hour[f"{source}_texts"])
        if max_samples:
            self.texts = self.texts[:max_samples]
        self.tokenizer, self.max_seq_len = tokenizer, max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        ids = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_seq_len - 1,
                             add_special_tokens=True)["input_ids"]
        ids.append(self.tokenizer.eos_token_id)
        return {"input_ids": ids, "labels": ids.copy()}


def create_model(config: dict, device: torch.device) -> tuple:
    model_cfg, offline = config["model"], config.get("hf", {}).get("offline", False)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg["attn_implementation"], local_files_only=offline,
    )
    for param in model.parameters():
        param.requires_grad = False
    lora = model_cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], target_modules=lora["target_modules"],
        lora_dropout=lora["dropout"], bias=lora["bias"], task_type=TaskType.CAUSAL_LM,
    ))
    if model_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], local_files_only=offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(device), tokenizer


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        if n >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        total += model(**batch).loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(config: dict, args: argparse.Namespace) -> None:
    rank, world_size, distributed = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    if is_main:
        setup_logging()
    set_seed(config["training"]["seed"] + rank)

    model, tokenizer = create_model(config, device)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    raw_model = model.module if distributed else model
    trainable = [p for p in raw_model.parameters() if p.requires_grad]

    data_cfg, cfg, log_cfg = config["data"], config["training"], config["logging"]
    trajectory_dir, max_seq_len = Path(data_cfg["trajectory_dir"]), data_cfg["max_seq_len"]
    train_ds = PerHourSFTDataset(trajectory_dir / "train", tokenizer, max_seq_len, "both", args.max_samples)
    n_val = min(args.max_samples or 5000, 5000)
    val_sets = [PerHourSFTDataset(trajectory_dir / "val", tokenizer, max_seq_len, kind, n_val)
                for kind in ("state", "action")]

    per_gpu_batch = cfg["per_gpu_batch_size"]
    grad_accum, remainder = divmod(cfg["effective_batch_size"], per_gpu_batch * world_size)
    if remainder or grad_accum < 1:
        raise ValueError("effective_batch_size must be a multiple of per_gpu_batch_size x number of GPUs")

    collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=per_gpu_batch, sampler=sampler, shuffle=sampler is None,
                              collate_fn=collator, num_workers=4, pin_memory=True, drop_last=True)
    val_loaders = [DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator, num_workers=2) for ds in val_sets]

    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
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

    def save(tag: str) -> None:
        raw_model.save_pretrained(save_dir / tag)
        tokenizer.save_pretrained(save_dir / tag)

    if is_main:
        logger.info("%d steps, %d GPUs x %d x accumulation %d", total_steps, world_size, per_gpu_batch, grad_accum)
    best_val, step, micro_step, t0 = float("inf"), 0, 0, time.time()
    loss_sum, n_tokens = 0.0, 0
    model.train()
    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / grad_accum).backward()
            tokens = (batch["labels"] != -100).sum().item()
            loss_sum += loss.item() * tokens
            n_tokens += tokens
            micro_step += 1
            if micro_step % grad_accum:
                continue

            if cfg["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg["gradient_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if is_main and step % log_cfg["log_every"] == 0:
                logger.info("step %d/%d loss %.4f lr %.2e %.1f min", step, total_steps,
                            loss_sum / max(n_tokens, 1), scheduler.get_last_lr()[0], (time.time() - t0) / 60)
            if is_main and step % log_cfg["eval_every"] == 0:
                val_state, val_action = (evaluate(model, vl, device, log_cfg["eval_max_batches"]) for vl in val_loaders)
                val_loss = (val_state + val_action) / 2
                logger.info("eval step %d val_loss %.4f (state %.4f, action %.4f)",
                            step, val_loss, val_state, val_action)
                save("latest")
                if val_loss < best_val:
                    best_val = val_loss
                    save("best")

    if is_main:
        save("final")
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump({"total_steps": step, "best_val_loss": best_val, "elapsed_seconds": time.time() - t0,
                       "config": config}, f, indent=2, default=str)
    if distributed:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    train(load_config(args.config), args)


if __name__ == "__main__":
    main()
