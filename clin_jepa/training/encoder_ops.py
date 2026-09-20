"""Encoder forward pass, the online/target LoRA adapter pair with its fp32 EMA, and resume helpers."""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from liger_kernel.transformers import apply_liger_kernel_to_qwen3

apply_liger_kernel_to_qwen3()

from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithFlattening,
)

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, bool]:
    """(rank, world_size, distributed) — initialises NCCL when launched by torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        return rank, dist.get_world_size(), True
    return 0, 1, False


def any_rank(flag: bool, device: torch.device) -> bool:
    """True on every rank if ``flag`` is set on any rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return flag
    t = torch.tensor([int(flag)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


_LORA_NAME_PAT = re.compile(r"(.*\.lora_[AB])\.(\w+)\.weight$")


def _split_lora_name(full_name: str) -> tuple[str, str] | None:
    """Parse a PEFT LoRA param name into (prefix, adapter)."""
    m = _LORA_NAME_PAT.match(full_name)
    if m is None:
        return None
    return (m.group(1), m.group(2))


def online_target_pairs(
    model: nn.Module,
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    """Return list of (online_param, target_param) pairs for all LoRA layers."""
    prefix_to_adapters: dict[str, dict[str, nn.Parameter]] = {}
    for name, p in model.named_parameters():
        parsed = _split_lora_name(name)
        if parsed is None:
            continue
        prefix, adapter = parsed
        prefix_to_adapters.setdefault(prefix, {})[adapter] = p

    pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for prefix, d in prefix_to_adapters.items():
        if "online" not in d or "target" not in d:
            raise RuntimeError(f"expected 'online' and 'target' LoRA adapters at {prefix!r}, found {list(d.keys())}")
        pairs.append((d["online"], d["target"]))

    if not pairs:
        raise RuntimeError("no LoRA online/target parameter pairs found")
    return pairs


def _rename_peft_adapter(model: nn.Module, old: str, new: str) -> None:
    """Rename a PEFT adapter from `old` to `new` in-place."""
    adapter_dict_attrs = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
        "lora_dropout",
        "scaling",
        "r",
        "lora_alpha",
        "lora_bias",
        "use_rslora",
        "use_dora",
    )

    for module in model.modules():
        for attr in adapter_dict_attrs:
            if not hasattr(module, attr):
                continue
            d = getattr(module, attr)
            if isinstance(d, (nn.ModuleDict, nn.ParameterDict)) and old in d:
                d[new] = d.pop(old)
            elif isinstance(d, dict) and old in d:
                d[new] = d.pop(old)

    if hasattr(model, "peft_config") and old in model.peft_config:
        model.peft_config[new] = model.peft_config.pop(old)

    for active_attr in ("active_adapter", "_active_adapter"):
        if hasattr(model, active_attr):
            cur = getattr(model, active_attr)
            if isinstance(cur, str) and cur == old:
                setattr(model, active_attr, new)
            elif isinstance(cur, list):
                setattr(model, active_attr, [new if a == old else a for a in cur])

    if hasattr(model, "active_adapters"):
        try:
            cur = model.active_adapters
            if isinstance(cur, list):
                model.active_adapters = [new if a == old else a for a in cur]
        except Exception:
            pass

    for module in model.modules():
        if module is model:
            continue
        for active_attr in ("active_adapter", "_active_adapter"):
            if hasattr(module, active_attr):
                cur = getattr(module, active_attr)
                if isinstance(cur, str) and cur == old:
                    try:
                        setattr(module, active_attr, new)
                    except Exception:
                        pass
                elif isinstance(cur, list):
                    try:
                        setattr(module, active_attr, [new if a == old else a for a in cur])
                    except Exception:
                        pass


def load_encoder(
    config: dict,
    device: torch.device,
) -> tuple[PeftModel, AutoTokenizer]:
    """Base model with the LoRA adapter at ``model.checkpoint_dir`` loaded as the trainable 'online' adapter."""
    model_cfg = config["model"]
    hf_cfg = config.get("hf", {})
    if model_cfg["attn_implementation"] != "flash_attention_2":
        raise ValueError("the packed encoder forward needs attn_implementation: flash_attention_2")

    base_name = model_cfg["base_model"]
    ckpt_abs = Path(model_cfg["checkpoint_dir"])
    if not ckpt_abs.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found at {ckpt_abs}")

    logger.info("Loading base model: %s", base_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg["attn_implementation"],
        local_files_only=hf_cfg.get("offline", False),
    )

    for p in base.parameters():
        p.requires_grad = False

    logger.info("Loading LoRA adapter from: %s", ckpt_abs)
    online = PeftModel.from_pretrained(
        base,
        str(ckpt_abs),
        is_trainable=True,
    )
    online = online.to(device)

    _rename_peft_adapter(online, old="default", new="online")
    online.set_adapter("online")
    if not any(".lora_A.online." in name for name, _ in online.named_parameters()):
        raise RuntimeError("the LoRA adapter was not registered as 'online'")

    if model_cfg["gradient_checkpointing"]:
        online.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt_abs),
        local_files_only=hf_cfg.get("offline", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_lora_params = sum(
        p.numel() for n, p in online.named_parameters()
        if "lora_" in n and p.requires_grad
    )
    logger.info(
        "Loaded online encoder: %s + LoRA (%.1fM trainable)",
        base_name, n_lora_params / 1e6,
    )

    return online, tokenizer


def create_target_encoder_fp32(online: PeftModel) -> PeftModel:
    """Add a 'target' adapter as fp32 EMA copy of 'online'."""
    online_config = online.peft_config["online"]
    online.add_adapter("target", online_config)

    online.set_adapter("online")

    pairs = online_target_pairs(online)
    with torch.no_grad():
        for online_p, target_p in pairs:
            target_p.data = online_p.detach().float().clone()
            target_p.requires_grad = False

    online.set_adapter("online")

    n_target_params = sum(p.numel() for _, p in pairs)
    logger.info(
        "Created target adapter as fp32 EMA copy of online (%.1fM params, frozen)",
        n_target_params / 1e6,
    )

    return online


@torch.no_grad()
def ema_update(pairs: list[tuple[nn.Parameter, nn.Parameter]], tau: float) -> None:
    """target <- tau * target + (1 - tau) * online."""
    for online_p, target_p in pairs:
        online_fp32 = online_p.detach().float()
        target_p.data.mul_(tau).add_(online_fp32, alpha=1.0 - tau)


@torch.no_grad()
def manual_allreduce_grads(params: list[nn.Parameter]) -> None:
    """Manually all-reduce gradients across DDP ranks via dist.all_reduce."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)


@torch.no_grad()
def hard_sync(pairs: list[tuple[nn.Parameter, nn.Parameter]]) -> None:
    """target <- online."""
    for online_p, target_p in pairs:
        target_p.data.copy_(online_p.detach().float())


_COLLATOR: DataCollatorWithFlattening | None = None


def _get_collator() -> DataCollatorWithFlattening:
    global _COLLATOR
    if _COLLATOR is None:
        _COLLATOR = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
    return _COLLATOR


def encode_texts_batched(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_seq_len: int = 4096,
    train_mode: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Encode a list of texts into (N, 4096) last-token hidden state embeddings."""
    collator = _get_collator()

    token_ids_list: list[list[int]] = []
    for text in texts:
        ids = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )["input_ids"]
        token_ids_list.append(ids)

    chunks: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, ids in enumerate(token_ids_list):
        n_tok = len(ids)
        if current and current_tokens + n_tok > max_seq_len:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += n_tok
    if current:
        chunks.append(current)

    all_embeddings: list[torch.Tensor | None] = [None] * len(texts)

    for chunk_indices in chunks:
        features = [
            {"input_ids": token_ids_list[i], "labels": token_ids_list[i]}
            for i in chunk_indices
        ]
        batch = collator(features)
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        forward_kwargs = dict(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            cu_seq_lens_q=batch["cu_seq_lens_q"],
            cu_seq_lens_k=batch["cu_seq_lens_k"],
            max_length_q=batch["max_length_q"],
            max_length_k=batch["max_length_k"],
            output_hidden_states=True,
            use_cache=False,
        )

        if train_mode:
            outputs = model(**forward_kwargs)
        else:
            with torch.no_grad():
                outputs = model(**forward_kwargs)

        cu = batch["cu_seq_lens_q"]
        last_positions = (cu[1:] - 1).long()
        last_hidden = outputs.hidden_states[-1]
        chunk_embs = last_hidden[0, last_positions, :]

        chunk_embs = chunk_embs.to(output_dtype)
        for out_idx, text_idx in enumerate(chunk_indices):
            all_embeddings[text_idx] = chunk_embs[out_idx]

        del outputs, batch, last_hidden, chunk_embs

    assert all(e is not None for e in all_embeddings), "encode_texts_batched dropped some texts"

    return torch.stack(all_embeddings, dim=0)


@torch.no_grad()
def collect_lora_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot all LoRA adapter weights (online + target) for resume."""
    state: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if _split_lora_name(name) is None:
            continue
        state[name] = p.detach().cpu().clone()
    return state


@torch.no_grad()
def restore_lora_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore LoRA adapter weights from a snapshot produced by collect_lora_adapter_state."""
    name_to_param = {n: p for n, p in model.named_parameters() if _split_lora_name(n)}
    missing = [n for n in name_to_param if n not in state]
    extra = [n for n in state if n not in name_to_param]
    if missing:
        raise RuntimeError(
            f"restore_lora_adapter_state: {len(missing)} live params missing from snapshot. "
            f"First few: {missing[:5]}"
        )
    if extra:
        logger.warning(
            "restore_lora_adapter_state: %d snapshot entries have no live param "
            "(ignored). First few: %s",
            len(extra), extra[:5],
        )
    for name, p in name_to_param.items():
        p.data.copy_(state[name].to(p.device))


def snapshot_rng_state() -> dict:
    """torch CPU and CUDA, numpy and python RNG states."""
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict) -> None:
    """Restore the states captured by ``snapshot_rng_state``."""
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        saved_cuda_states = state["torch_cuda"]
        n_available = torch.cuda.device_count()
        if len(saved_cuda_states) > n_available:
            logger.warning(
                "restore_rng_state: saved %d CUDA RNG states but %d devices available; clipping to %d.",
                len(saved_cuda_states), n_available, n_available,
            )
            saved_cuda_states = saved_cuda_states[:n_available]
        torch.cuda.set_rng_state_all(saved_cuda_states)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def atomic_torch_save(state: dict, path: Path) -> None:
    """torch.save to a temporary file, then rename it onto ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
