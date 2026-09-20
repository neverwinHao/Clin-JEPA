import numpy as np
import torch

from clin_jepa.downstream.sequence_models import build, fit, logits, mean_auroc

SPEC = {"max_epochs": 3, "patience": 2, "min_improvement": 1e-6, "batch_size": 64, "grad_clip": 1.0,
        "final_lr_ratio": 0.01, "head_hidden": 16, "tcn_base_layers": 2, "tcn_kernel": 3}
CONFIG = {"hidden_size": 8, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1}


def split(n, seed):
    rng = np.random.default_rng(seed)
    sequence, statics = rng.normal(size=(n, 24, 6)).astype(np.float32), rng.normal(size=(n, 3)).astype(np.float32)
    signal = sequence[:, -4:, 0].mean(axis=1)
    labels = np.stack([signal > 0, signal < -0.3], axis=1).astype(np.float32)
    labels[rng.random(labels.shape) < 0.3] = np.nan
    return {"sequence": sequence, "statics": statics, "labels": labels}


def test_tcn_output_does_not_depend_on_later_inputs():
    torch.manual_seed(0)
    net = build("tcn", 6, 3, 2, CONFIG, SPEC).eval()
    x = torch.randn(2, 24, 6)
    changed = x.clone()
    changed[:, 12:] += 1.0
    early = net.encoder.blocks(x.transpose(1, 2))[:, :, :12]
    torch.testing.assert_close(early, net.encoder.blocks(changed.transpose(1, 2))[:, :, :12])


def test_fit_handles_missing_labels_for_both_models():
    train, val = split(300, 0), split(120, 1)
    for model in ("lstm", "tcn"):
        net = fit(model, CONFIG, SPEC, train, val, 42, torch.device("cpu"))
        out = logits(net, val["sequence"], val["statics"], 64, torch.device("cpu"))
        assert out.shape == (120, 2) and np.isfinite(out).all()


def test_mean_auroc_skips_heads_without_enough_labelled_rows():
    labels = np.full((100, 2), np.nan, dtype=np.float32)
    labels[:, 0] = np.arange(100) % 2
    labels[:10, 1] = np.arange(10) % 2
    scores = np.stack([labels[:, 0], np.zeros(100, dtype=np.float32)], axis=1)
    assert mean_auroc(labels, scores) == 1.0
    assert np.isnan(mean_auroc(labels[:, 1:], scores[:, 1:]))
