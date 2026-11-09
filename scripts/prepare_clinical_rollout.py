"""Prepare history-only hypothetical rollouts for a 24h mortality probe.

Only observed state/actions through hour 23 are read. From hour 24 onward the
synthetic action text says "No active interventions". This is an artificial
stress test, not a medical treatment recommendation or causal estimate.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.frozen_base import encode_texts_batched, load_frozen_base
from clin_jepa.rollout import load_predictor
from clin_jepa.utils import load_config


@torch.inference_mode()
def future_features(model, states, actions, static, synthetic, device):
    states = states.to(device).float()
    actions = actions.to(device).float()
    static = static.to(device).float()
    synthetic = synthetic.to(device).float()
    z_hist, a_hist = states[:, :23], actions[:, :23]
    current = states[:, 23]
    generated = []
    for step in range(24):
        # The action at hour 23 is observed. Later actions are a fixed,
        # synthetic no-intervention assumption, not the recorded future.
        next_action = actions[:, 23] if step == 0 else synthetic[step - 1].expand(len(states), -1)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            next_state = model.forward_rollout_step(z_hist, static, a_hist, next_action, current)
        next_state = next_state.float()
        generated.append(next_state)
        z_hist = torch.cat((z_hist, current.unsqueeze(1)), dim=1)
        a_hist = torch.cat((a_hist, next_action.unsqueeze(1)), dim=1)
        current = next_state
    generated = torch.stack(generated, dim=1)
    return torch.stack((generated.mean(dim=1), generated.std(dim=1), generated[:, -1]), dim=1).half().cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder-config", default="configs/encode_frozen_base.yaml")
    p.add_argument("--predictor-config", default="configs/rollout_frozen_base.yaml")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / split / "future.npy").exists() for split in ("train", "val", "test")):
        raise FileExistsError("Refusing to overwrite existing future features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.encoder_config)
    encoder, tokenizer = load_frozen_base(config, device)
    texts = [f"t={h}h | No active interventions." for h in range(24, 47)]
    action_vectors = encode_texts_batched(
        encoder, tokenizer, texts, device, max_seq_len=config["data"]["max_seq_len"],
        output_dtype=torch.float16, batch_size=config["data"]["batch_size"],
        max_batch_tokens=config["data"]["max_batch_tokens"])
    if action_vectors.shape != (23, 4096) or not torch.isfinite(action_vectors).all():
        raise ValueError("Invalid synthetic action embeddings")
    torch.save({"hours": list(range(24, 47)), "embeddings": action_vectors,
                "description": "synthetic no-intervention actions, hours 24 through 46"},
               args.output / "synthetic_actions.pt")
    del encoder, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    architecture = load_config(args.predictor_config)["predictor"]
    model = load_predictor(args.checkpoint, architecture, device)
    for split in ("train", "val", "test"):
        original_index = pd.read_parquet(args.history / split / "index.parquet")
        selected = original_index[original_index["start_step"] == 0].copy().reset_index(drop=True)
        if selected["stay_id"].duplicated().any() or (selected["length"] < 24).any():
            raise ValueError(f"Invalid admission windows in {split}")
        selected.insert(0, "feature_row", np.arange(len(selected)))
        folder = args.output / split
        folder.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(folder / "index.parquet", index=False)
        dest = folder / "future.npy"
        tmp = folder / "future.npy.tmp"
        store = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(selected), 3, 4096))
        for shard_name, group in selected.groupby("shard", sort=False):
            shard = torch.load(args.data / "embeddings/qwen3_base" / split / shard_name,
                               map_location="cpu", weights_only=False, mmap=True)
            for start in range(0, len(group), args.batch_size):
                batch = group.iloc[start:start + args.batch_size]
                windows = [shard["windows"][int(i)] for i in batch["window_idx"]]
                state_at = torch.stack([w["emb_indices_state"][:24].long() for w in windows])
                action_at = torch.stack([w["emb_indices_action"][:24].long() for w in windows])
                states = shard["z_state_embeddings"][state_at]
                actions = shard["z_action_embeddings"][action_at]
                static = shard["z_statics"][torch.tensor([int(w["static_idx"]) for w in windows])]
                blocks = future_features(model, states, actions, static, action_vectors, device)
                if not torch.isfinite(blocks).all():
                    raise ValueError(f"Nonfinite future features in {split}/{shard_name}")
                store[batch["feature_row"].to_numpy()] = blocks.numpy()
            print(f"{split}: {shard_name} ({len(group)} windows)", flush=True)
        store.flush()
        del store
        tmp.replace(dest)
        print(f"{split}: prepared {len(selected)} admission windows", flush=True)
    (args.output / "metadata.json").write_text(json.dumps({
        "context_hours": 24, "forecast_hours": 24,
        "action_policy": "Observed hour 23; synthetic no-active-interventions hours 24-46",
        "warning": "Hypothetical and out of distribution; not a treatment recommendation.",
        "encoder": config["model"]["base_model"], "predictor_checkpoint": str(args.checkpoint),
        "future_features": ["mean", "std", "last"],
        "history_reading": "Only state/action indices 0 through 23; no future observations/actions."
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
