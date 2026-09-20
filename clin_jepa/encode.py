"""Embedding cache: every state, action and demographics text of the trajectory shards under one encoder."""

import argparse
import logging
from pathlib import Path

import torch

from clin_jepa.training.encoder_ops import atomic_torch_save, encode_texts_batched, load_encoder
from clin_jepa.training.trajectory_dataset import demographics_text
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def encode_shard(shard: dict, model, tokenizer, device: torch.device, max_seq_len: int) -> dict:
    """State rows per hour, action rows per distinct text, one row per stay; windows index into them."""
    per_hour, per_stay = shard["per_hour"], shard["per_stay"]

    def encode(texts: list[str]) -> torch.Tensor:
        return encode_texts_batched(
            model, tokenizer, texts, device=device, max_seq_len=max_seq_len, output_dtype=torch.float16,
        ).cpu()

    action_row: dict[str, int] = {}
    action_of_hour = [action_row.setdefault(text, len(action_row)) for text in per_hour["action_texts"]]
    demographics = [
        demographics_text(age=int(round(float(age))), gender=gender, race=race)
        for age, gender, race in zip(per_stay["ages"], per_stay["genders"], per_stay["races"])
    ]

    z_state = encode(per_hour["state_texts"])
    z_action = encode(list(action_row))
    z_static = encode(demographics)

    action_of_hour = torch.tensor(action_of_hour, dtype=torch.int32)
    windows = []
    for w in shard["windows"]:
        hours = torch.from_numpy(w["hour_indices"]).long()
        windows.append({
            "stay_id": int(per_stay["stay_ids"][w["stay_index"]]),
            "start_step": int(w["start_step"]),
            "length": int(w["length"]),
            "emb_indices_state": hours.int(),
            "emb_indices_action": action_of_hour[hours],
            "static_idx": int(w["stay_index"]),
        })
    return {
        "z_state_embeddings": z_state,
        "z_action_embeddings": z_action,
        "z_statics": z_static,
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/encode.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    config["model"]["checkpoint_dir"] = config["encoders"][args.model]
    device = torch.device("cuda")

    model, tokenizer = load_encoder(config, device)
    model.eval()
    model.requires_grad_(False)

    shards = sorted((Path(config["data"]["trajectory_dir"]) / args.split).glob("trajectories_*.pt"))
    output_dir = ensure_dir(Path(config["data"]["embedding_dir"]) / args.model / args.split)
    for path in shards[args.task_id :: args.num_tasks]:
        output = output_dir / path.name.replace("trajectories_", "embeddings_")
        if output.exists():
            continue
        shard = torch.load(path, map_location="cpu", weights_only=False)
        atomic_torch_save(
            encode_shard(shard, model, tokenizer, device, config["data"]["max_seq_len"]), output,
        )
        logger.info("%s: %d windows", output.name, len(shard["windows"]))


if __name__ == "__main__":
    main()
