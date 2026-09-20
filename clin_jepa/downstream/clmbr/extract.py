"""CLMBR-T-base representation of every evaluation row: the last token at or before the row's cutoff."""

import argparse
import logging
import pickle
from pathlib import Path

import femr.models.processor
import femr.models.tokenizer
import femr.models.transformer
import numpy as np
import pandas as pd
import torch

from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def token_representations(processor, model, patient: dict, max_tokens: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Representation and time (seconds) of the patient's last ``max_tokens`` tokens, in time order."""
    batch = processor.collate([processor.convert_patient(patient, tensor_type="pt")])
    n_tokens = batch["batch"]["transformer"]["tokens"].shape[-1]
    if n_tokens == 0:
        return None
    if n_tokens > max_tokens:
        batch = processor.collate([processor.convert_patient(
            patient, offset=n_tokens - max_tokens, max_length=max_tokens, tensor_type="pt")])
    result = model(**batch)[1]
    vectors, seconds = result["representations"].float().numpy(), result["timestamps"].numpy()
    order = np.argsort(seconds, kind="stable")
    return vectors[order], seconds[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/clmbr.yaml")
    parser.add_argument("--rows", required=True, choices=["stay", "hourly"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    torch.set_num_threads(config["n_workers"])
    root = Path(config["output_dir"]) / "clmbr"
    timeline_dir = root / "timelines" / args.rows / args.split
    output_dir = ensure_dir(root / "features" / args.rows / args.split)
    rows = pd.read_parquet(timeline_dir / "rows.parquet").sort_values(["subject_id", "cutoff"], kind="stable")
    cutoffs = {subject: group for subject, group in rows.groupby("subject_id")}

    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(config["model_dir"])
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer)
    model = femr.models.transformer.FEMRModel.from_pretrained(config["model_dir"]).eval()

    for path in sorted(timeline_dir.glob("patients_*.pkl"))[args.task_id :: args.num_tasks]:
        output = output_dir / f"{path.stem}.parquet"
        if output.exists():
            continue
        keys, vectors = [], []
        for patient in pickle.loads(path.read_bytes()):
            tokens = token_representations(processor, model, patient, config["max_tokens"])
            if tokens is None:
                continue
            group = cutoffs[patient["patient_id"]]
            seconds = group["cutoff"].map(lambda t: int(t.timestamp())).to_numpy()
            at = np.searchsorted(tokens[1], seconds, side="right") - 1
            keys.append(group[["subject_id", "stay_id", "target_hour"]][at >= 0])
            vectors.append(tokens[0][at[at >= 0]])
        features = pd.DataFrame(np.concatenate(vectors), columns=[f"f{i}" for i in range(vectors[0].shape[1])])
        pd.concat([pd.concat(keys, ignore_index=True), features], axis=1).to_parquet(output, index=False)
        logger.info("%s: %d rows", output.name, len(features))


if __name__ == "__main__":
    main()
