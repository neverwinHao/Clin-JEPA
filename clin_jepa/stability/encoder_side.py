"""Encoder-side diagnostics of a test-split embedding cache: scale, z_std, rank, mean-state errors."""

import argparse
import json
import logging
from pathlib import Path

import torch

from clin_jepa.stability.cache import load_states
from clin_jepa.stability.spectrum import CHUNK, Spectrum, covariance_spectrum, effective_rank, rankme, top_share
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def rank_summary(spectrum: Spectrum) -> dict:
    return {"rankme": rankme(spectrum), "effective_rank": effective_rank(spectrum), "top10_share": top_share(spectrum)}


def mean_state_errors(
    z: torch.Tensor, windows: list[torch.Tensor], global_mean: torch.Tensor, context: int, horizon: int,
) -> dict:
    """Mean l1 error at ``horizon`` of three predictors that use no dynamics, over the windows that reach it."""
    reach = [w for w in windows if len(w) > context - 1 + horizon]
    z_true = z[torch.stack([w[context - 1 + horizon] for w in reach])].float()
    context_mean = torch.stack([z[w[:context]].float().mean(dim=0) for w in reach])
    references = {"global": global_mean.float(), "horizon": z_true.mean(dim=0), "context": context_mean}
    return {name: float((z_true - ref).abs().sum(dim=-1).mean()) for name, ref in references.items()}


def analyse(cache_dir: Path, config: dict, device: torch.device) -> dict:
    z, windows, _ = load_states(cache_dir)
    total = torch.zeros(z.shape[1], dtype=torch.float64, device=device)
    l1_norm = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, len(z), CHUNK):
        block = z[start : start + CHUNK].to(device=device, dtype=torch.float64)
        total += block.sum(dim=0)
        l1_norm += block.abs().sum()

    sample = [w for w in windows[: config["dynamic_windows"]] if len(w) >= 2]
    z_std = [float(z[w].to(device=device, dtype=torch.float32).std(dim=0).mean()) for w in sample]
    differences = torch.cat([z[w][1:].float() - z[w][:-1].float() for w in sample])
    return {
        "n_states": len(z),
        "n_windows": len(windows),
        "mean_l1_norm": float(l1_norm / len(z)),
        "z_std": sum(z_std) / len(z_std),
        "static": rank_summary(covariance_spectrum(z, device)),
        "dynamic": rank_summary(covariance_spectrum(differences, device)),
        "mean_state_l1": mean_state_errors(z, windows, (total / len(z)).cpu(), config["context"], config["horizon"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stability.yaml")
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = analyse(Path(config["embedding_dir"]) / args.encoder / "test", config, device)
    output = ensure_dir(Path(config["output_dir"]) / "encoder_side") / f"{args.encoder}.json"
    output.write_text(json.dumps(result, indent=2))
    logger.info("%s: static RankMe %.0f, dynamic RankMe %.0f, z_std %.3f", args.encoder,
                result["static"]["rankme"], result["dynamic"]["rankme"], result["z_std"])


if __name__ == "__main__":
    main()
