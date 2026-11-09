"""The small-sample SVD matches the paper's covariance-spectrum metrics."""
import torch
from clin_jepa.stability.spectrum import covariance_spectrum, rankme, top_share
from scripts.prepare_predictor_diagnostics import spectra


def test_small_svd_matches_author_spectrum():
    torch.manual_seed(19)
    x = torch.randn(60, 128)
    fast, _ = spectra(x, 'cpu')
    reference = covariance_spectrum(x, 'cpu')
    assert abs(fast['rankme'] - rankme(reference)) < 0.01
    assert abs(fast['top10_share'] - top_share(reference)) < 1e-5
