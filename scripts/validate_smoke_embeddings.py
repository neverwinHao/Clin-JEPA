"""Validate smoke-run embedding shapes, finiteness, and trajectory indices."""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--dim", type=int, default=4096)
    args = p.parse_args()
    for split in ("train", "val"):
        trajectory = torch.load(args.data / "trajectories" / split / "trajectories_smoke.pt", map_location="cpu", weights_only=False)
        path = args.data / "embeddings" / "qwen3_base" / split / "embeddings_smoke.pt"
        cache = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "z_state_embeddings": len(trajectory["per_hour"]["state_texts"]),
            "z_action_embeddings": len(set(trajectory["per_hour"]["action_texts"])),
            "z_statics": len(trajectory["per_stay"]["stay_ids"]),
        }
        for name, count in expected.items():
            values = cache[name]
            if values.shape != (count, args.dim) or values.dtype != torch.float16:
                raise ValueError(f"{split} {name}: unexpected shape or dtype: {values.shape} {values.dtype}")
            if not torch.isfinite(values).all():
                raise ValueError(f"{split} {name}: non-finite values")
        if len(cache["windows"]) != len(trajectory["windows"]):
            raise ValueError(f"{split}: window count mismatch")
        for window, source in zip(cache["windows"], trajectory["windows"]):
            if window["length"] != source["length"] or not torch.equal(window["emb_indices_state"].long(), torch.from_numpy(source["hour_indices"]).long()):
                raise ValueError(f"{split}: window mismatch")
            if window["emb_indices_action"].min() < 0 or window["emb_indices_action"].max() >= expected["z_action_embeddings"]:
                raise ValueError(f"{split}: action index out of bounds")
            if not 0 <= window["static_idx"] < expected["z_statics"]:
                raise ValueError(f"{split}: static index out of bounds")
        print(f"{split}: finite FP16 embeddings {expected}; {len(cache['windows'])} windows and indices OK")


if __name__ == "__main__":
    main()
