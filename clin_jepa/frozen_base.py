"""Frozen Qwen base encoding without an SFT adapter or Flash Attention dependency."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def load_frozen_base(config: dict, device: torch.device):
    model_config = config["model"]
    base = model_config["base_model"]
    offline = config.get("hf", {}).get("offline", False)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        local_files_only=offline,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


@torch.inference_mode()
def encode_texts_batched(model, tokenizer, texts, device, max_seq_len=4096,
                         output_dtype=torch.float16, batch_size=32,
                         max_batch_tokens=4096, sort_window=1024):
    """Encode last tokens, grouping similar lengths and restoring the original order."""
    if batch_size < 1 or max_batch_tokens < 1 or sort_window < 1:
        raise ValueError("batch_size, max_batch_tokens and sort_window must be positive")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have a pad token")
    if not texts:
        return torch.empty((0, model.config.hidden_size), dtype=output_dtype)
    blocks = []
    for start in range(0, len(texts), sort_window):
        block = texts[start : start + sort_window]
        encoded = tokenizer(block, add_special_tokens=True, truncation=True,
                            max_length=max_seq_len, padding=False, return_attention_mask=False)
        ids = encoded["input_ids"]
        if any(len(row) == 0 for row in ids):
            raise ValueError("an input text tokenized to an empty sequence")
        order = sorted(range(len(ids)), key=lambda index: len(ids[index]))
        restored = [None] * len(ids)
        groups, group = [], []
        for index in order:
            length = len(ids[index])
            if group and (len(group) >= batch_size or length * (len(group) + 1) > max_batch_tokens):
                groups.append(group)
                group = []
            group.append(index)
        if group:
            groups.append(group)
        for group in groups:
            width = max(len(ids[index]) for index in group)
            input_ids = torch.full((len(group), width), tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros((len(group), width), dtype=torch.long)
            lengths = []
            for row, index in enumerate(group):
                tokens = ids[index]
                lengths.append(len(tokens))
                input_ids[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                attention[row, :len(tokens)] = 1
            batch = {"input_ids": input_ids.to(device), "attention_mask": attention.to(device)}
            states = model(**batch, use_cache=False).last_hidden_state
            rows = torch.arange(len(group), device=device)
            last = torch.tensor(lengths, device=device) - 1
            values = states[rows, last].to(output_dtype).cpu()
            for index, value in zip(group, values):
                restored[index] = value
        blocks.append(torch.stack(restored))
        completed = min(start + sort_window, len(texts))
        if completed == len(texts) or completed // 8192 > start // 8192:
            logger.info("Encoded %d / %d texts", completed, len(texts))
    return torch.cat(blocks, dim=0)
