"""Patient-clustered paired comparisons preserve matching predictions."""
import numpy as np
import pandas as pd
from scripts.compare_mortality_predictions import compare


def test_identical_predictions_have_zero_paired_difference():
    score = np.array([.1, .2, .8, .9, .3, .7])
    frame = pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4, 5],
        "label": [0, 0, 1, 1, 0, 1],
        "history_32": score,
        "history_64": score,
        "history_32_plus_future_32": score,
    })
    result = compare(frame, draws=100, seed=42)
    for baseline in ("history_32", "history_64"):
        for metric in ("auroc", "auprc"):
            assert result["comparisons"][baseline][metric]["difference"] == 0
            assert result["comparisons"][baseline][metric]["ci95"] == [0, 0]
