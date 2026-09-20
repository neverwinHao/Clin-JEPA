"""Context block of every window: mean, standard deviation and last of the first 24 state and action embeddings,
and the demographics embedding."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from clin_jepa.downstream.tasks import CONTEXT
from clin_jepa.rollout import build_index
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

BLOCKS = ("state_mean", "state_std", "state_last", "action_mean", "action_std", "action_last", "demographics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    cache_dir = Path(config["embedding_dir"]) / args.encoder / args.split
    shards = sorted(cache_dir.glob("embeddings_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no embedding shards in {cache_dir}")
    index = build_index(shards, CONTEXT)
    output_dir = ensure_dir(Path(config["output_dir"]) / "features" / args.encoder / args.split)
    index.to_parquet(output_dir / "index.parquet", index=False)
    store = np.lib.format.open_memmap(output_dir / "history.npy", mode="w+", dtype=np.float16,
                                      shape=(len(index), len(BLOCKS), config["embedding_dim"]))

    for shard_name, group in index.groupby("shard", sort=False):
        shard = torch.load(cache_dir / shard_name, map_location="cpu", weights_only=False, mmap=True)
        windows = [shard["windows"][int(i)] for i in group["window_idx"]]
        rows = group["row"].to_numpy()
        for start in range(0, len(windows), 2048):
            chunk = windows[start : start + 2048]
            blocks = []
            for kind in ("state", "action"):
                at = torch.stack([torch.as_tensor(w[f"emb_indices_{kind}"][:CONTEXT], dtype=torch.long) for w in chunk])
                z = shard[f"z_{kind}_embeddings"][at.reshape(-1)].reshape(len(chunk), CONTEXT, -1).float()
                blocks += [z.mean(dim=1), z.std(dim=1), z[:, -1]]
            blocks.append(shard["z_statics"][torch.tensor([int(w["static_idx"]) for w in chunk])].float())
            store[rows[start : start + 2048]] = torch.stack(blocks, dim=1).half().numpy()
        logger.info("%s done", shard_name)
    store.flush()


if __name__ == "__main__":
    main()
