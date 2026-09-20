import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from clin_jepa.downstream.readout import build_head, standardization, train_probe

SPEC = {"widths": [32], "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256, "max_epochs": 15,
        "patience": 5, "standardize": True, "warmup_fraction": 0.1, "eval_batch_size": 1024}


def problem():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 24)).astype(np.float32) * rng.uniform(0.1, 20, size=24).astype(np.float32)
    y = ((x / x.std(axis=0)) @ rng.normal(size=24) + rng.normal(scale=0.5, size=3000) > 1.0).astype(np.float32)
    parts = {"train": slice(0, 1800), "val": slice(1800, 2400), "test": slice(2400, 3000)}
    return {s: torch.from_numpy(x[i]).half() for s, i in parts.items()}, {s: y[i] for s, i in parts.items()}


def test_head_shapes():
    assert [m.out_features for m in build_head(dict(SPEC, widths=[]), 7) if hasattr(m, "out_features")] == [1]
    widths = [m.out_features for m in build_head(dict(SPEC, widths=[16, 4]), 7) if hasattr(m, "out_features")]
    assert widths == [16, 4, 1]


def test_standardization_keeps_constant_dimensions():
    x = torch.tensor([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    mean, inv_scale = standardization(x)
    np.testing.assert_allclose(mean, [3.0, 5.0])
    np.testing.assert_allclose(inv_scale, [1 / (2.0 + 1e-6), 1.0], rtol=1e-6)


def test_probe_separates_and_is_deterministic():
    x, y = problem()
    best, first = train_probe(SPEC, 42, x, y)
    _, second = train_probe(SPEC, 42, x, y)
    np.testing.assert_array_equal(first, second)
    assert best > 0.85 and roc_auc_score(y["test"], first) > 0.85
    assert not np.array_equal(first, train_probe(SPEC, 43, x, y)[1])
