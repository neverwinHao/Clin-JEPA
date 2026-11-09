"""Verify every sampled window and its hour texts match its source shard."""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--sample", type=Path, required=True)
    args = p.parse_args()
    for split in ("train", "val", "test"):
        count = 0
        for path in sorted((args.sample / "trajectories" / split).glob("trajectories_*.pt")):
            sample = torch.load(path, map_location="cpu", weights_only=False)
            original = torch.load(args.source / "trajectories" / split / path.name, map_location="cpu", weights_only=False)
            orig_stays = {int(s): i for i, s in enumerate(original["per_stay"]["stay_ids"])}
            orig_windows = {(int(w["stay_index"]), int(w["start_step"])): w for w in original["windows"]}
            for w in sample["windows"]:
                si = int(w["stay_index"])
                original_si = orig_stays[int(sample["per_stay"]["stay_ids"][si])]
                ref = orig_windows[(original_si, int(w["start_step"]))]
                if w["length"] != ref["length"]:
                    raise ValueError(f"Window length mismatch in {path.name}")
                for field in ("stay_ids", "hours", "state_texts", "action_texts", "sofa"):
                    a = [sample["per_hour"][field][i] for i in w["hour_indices"]]
                    b = [original["per_hour"][field][i] for i in ref["hour_indices"]]
                    if field == "sofa":
                        equal = np.array_equal(a, b, equal_nan=True)
                    else:
                        equal = a == b
                    if not equal:
                        raise ValueError(f"{field} mismatch in {path.name}")
                for field in ("stay_ids", "ages", "genders", "races"):
                    if sample["per_stay"][field][si] != original["per_stay"][field][original_si]:
                        raise ValueError(f"{field} mismatch in {path.name}")
                count += 1
        print(f"{split}: {count} windows match source", flush=True)


if __name__ == "__main__":
    main()
