"""One split of an embedding cache as a single state table."""

from pathlib import Path

import torch


def load_states(cache_dir: Path) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, int]]:
    """State table ``(N, D)`` fp16, each window's rows into it (shard order), and the first window of each shard."""
    tables, windows, first_window, offset = [], [], {}, 0
    for path in sorted(cache_dir.glob("embeddings_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        first_window[path.name] = len(windows)
        windows += [w["emb_indices_state"].long() + offset for w in shard["windows"]]
        tables.append(shard["z_state_embeddings"])
        offset += len(tables[-1])
    return torch.cat(tables), windows, first_window
