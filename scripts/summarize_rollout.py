"""Summarize saved rollout errors without exposing patient-level results."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--errors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = np.load(args.errors)
    summary = {}
    for horizon in (1, 8, 24, 48):
        model = data["model_l1"][:, horizon - 1]
        baseline = data["copy_forward_l1"][:, horizon - 1]
        linear = data["linear_extrap_l1"][:, horizon - 1]
        mask = np.isfinite(model) & np.isfinite(baseline) & np.isfinite(linear)
        n = int(mask.sum())
        if n:
            mean_model = float(model[mask].mean())
            mean_base = float(baseline[mask].mean())
            summary[str(horizon)] = {"windows": n, "predictor_l1": mean_model,
                                     "copy_forward_l1": mean_base,
                                     "linear_extrap_l1": float(linear[mask].mean()),
                                     "mase_vs_copy_forward": mean_model / mean_base if mean_base else None}
        else:
            summary[str(horizon)] = {"windows": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
