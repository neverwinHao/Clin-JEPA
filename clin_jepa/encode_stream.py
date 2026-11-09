"""Continuously encode complete trajectory shards on one GPU with work stealing."""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import time
from functools import partial
from pathlib import Path

import torch

from clin_jepa.encode import encode_shard
from clin_jepa.frozen_base import encode_texts_batched, load_frozen_base
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def available_shards(trajectory_dir: Path):
    for split in ("train", "val", "test"):
        for path in sorted((trajectory_dir / split).glob("trajectories_*.pt")):
            yield split, path


def output_path(embedding_dir: Path, model_name: str, split: str, path: Path) -> Path:
    return embedding_dir / model_name / split / path.name.replace("trajectories_", "embeddings_")


def process_available(config: dict, model, tokenizer, device: torch.device, model_name: str) -> int:
    data = config["data"]
    trajectory_dir = Path(data["trajectory_dir"])
    embedding_dir = Path(data["embedding_dir"])
    encode_fn = partial(encode_texts_batched, batch_size=data.get("batch_size", 1),
                        max_batch_tokens=data.get("max_batch_tokens", 4096))
    completed = 0
    for split, path in available_shards(trajectory_dir):
        output = output_path(embedding_dir, model_name, split, path)
        if output.exists():
            continue
        ensure_dir(output.parent)
        lock_path = output.with_suffix(output.suffix + ".lock")
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if output.exists():
                continue
            logger.info("Encoding %s/%s", split, path.name)
            shard = torch.load(path, map_location="cpu", weights_only=False)
            result = encode_shard(shard, model, tokenizer, device, data["max_seq_len"], encode_fn)
            temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
            try:
                torch.save(result, temporary)
                temporary.replace(output)
            finally:
                if temporary.exists():
                    temporary.unlink()
            logger.info("Completed %s/%s (%d windows)", split, path.name, len(shard["windows"]))
            completed += 1
            del shard, result
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/encode_frozen_base.yaml")
    parser.add_argument("--model", default="qwen3_base")
    parser.add_argument("--complete-marker", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if not config["model"].get("frozen_base") or args.model not in config["encoders"]:
        parser.error("stream encoding requires a configured frozen base model")
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    device = torch.device("cuda")
    model, tokenizer = load_frozen_base(config, device)
    trajectory_dir = Path(config["data"]["trajectory_dir"])
    embedding_dir = Path(config["data"]["embedding_dir"])
    while True:
        completed = process_available(config, model, tokenizer, device, args.model)
        if args.complete_marker.exists():
            paths = list(available_shards(trajectory_dir))
            if not paths:
                raise RuntimeError("completion marker exists but no trajectory shards were found")
            if all(output_path(embedding_dir, args.model, split, path).exists() for split, path in paths):
                logger.info("All %d trajectory shards encoded", len(paths))
                return
        if completed == 0:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
