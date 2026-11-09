"""Reload a smoke checkpoint, confirm training changed it, and reevaluate it."""
import argparse
import math
from pathlib import Path

import torch

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.predictor_dataset import PredictorDataset, predictor_collate_fn
from clin_jepa.training.train_predictor import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--embedding-dir", type=Path, required=True)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    torch.manual_seed(ckpt["seed"])
    model = ACTransformerPredictor(**cfg["predictor"])
    changed = 0
    for name, initial in model.state_dict().items():
        trained = ckpt["model_state_dict"][name]
        if not torch.isfinite(trained).all():
            raise ValueError(f"non-finite checkpoint values: {name}")
        changed += torch.count_nonzero(initial != trained).item()
    if changed == 0:
        raise ValueError("Predictor parameters did not change from initialization")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = PredictorDataset(args.embedding_dir, "val", max_windows=10000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["training"]["eval_batch_size"],
                                         collate_fn=predictor_collate_fn, shuffle=False, num_workers=0)
    score = evaluate(model, loader, device)
    saved_score = float(ckpt["val_loss"])
    if not math.isfinite(score) or abs(score - saved_score) > max(0.01, abs(saved_score) * 1e-4):
        raise ValueError(f"reloaded validation loss mismatch: {score} versus {saved_score}")
    print(f"Reloaded checkpoint epoch {ckpt['epoch']}; {changed} values changed from seeded initialization")
    print(f"Validation loss: saved={saved_score:.4f}, reloaded={score:.4f}; finite and consistent")


if __name__ == "__main__":
    main()
