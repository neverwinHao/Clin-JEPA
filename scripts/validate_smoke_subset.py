"""Check a small trajectory subset retains its original windows and hourly data."""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--subset", type=Path, required=True)
    args = p.parse_args()
    ids = {}
    for split in ("train", "val"):
        source_files = sorted((args.source / split).glob("trajectories_*.pt"))
        if not source_files:
            raise FileNotFoundError(split)
        original = torch.load(source_files[0], map_location="cpu", weights_only=False)
        subset = torch.load(args.subset / split / "trajectories_smoke.pt", map_location="cpu", weights_only=False)
        original_indices = {int(stay): i for i, stay in enumerate(original["per_stay"]["stay_ids"])}
        subset_indices = [original_indices[int(stay)] for stay in subset["per_stay"]["stay_ids"]]
        ref = {(int(w["stay_index"]), int(w["start_step"])): w for w in original["windows"]}
        for window in subset["windows"]:
            j = int(window["stay_index"])
            original_window = ref[(subset_indices[j], int(window["start_step"]))]
            assert window["length"] == original_window["length"] == len(window["hour_indices"])
            for k, h in zip(window["hour_indices"], original_window["hour_indices"]):
                for field in ("stay_ids", "hours", "state_texts", "action_texts"):
                    assert subset["per_hour"][field][k] == original["per_hour"][field][h], field
        ids[split] = set(map(int, subset["per_stay"]["stay_ids"]))
        print(f"{split}: {len(ids[split])} stays, {len(subset['windows'])} windows; source match OK")
    assert ids["train"].isdisjoint(ids["val"])
    print("Train/validation ICU-stay IDs disjoint; subject-level split checked in full dataset validation.")


if __name__ == "__main__":
    main()
