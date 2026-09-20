import numpy as np
import torch

from clin_jepa.stability.drift import decompose, horizon_means
from clin_jepa.stability.spectrum import covariance_spectrum, effective_rank, rankme, subspace_overlap, top_share


def test_drift_is_degradation_times_displacement():
    rng = np.random.default_rng(0)
    l1, cf = rng.uniform(1, 5, size=(7, 48)), rng.uniform(1, 5, size=(7, 48))
    parts = decompose(l1, cf)
    np.testing.assert_allclose(parts["drift"], parts["degradation"] * parts["displacement"], rtol=1e-12)
    np.testing.assert_allclose(parts["drift"][:, 0], 1.0)


def test_horizon_means_ignore_steps_past_the_stay_end():
    values = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan]])
    np.testing.assert_array_equal(horizon_means(values[:, :2]), [2.0, 2.0])


def test_rank_measures_on_known_spectra():
    torch.manual_seed(0)
    isotropic = covariance_spectrum(torch.randn(20000, 16), "cpu")
    assert 15.5 < rankme(isotropic) <= 16.0 and 15.5 < effective_rank(isotropic) <= 16.0
    one_direction = covariance_spectrum(torch.randn(5000, 1) * torch.ones(1, 16), "cpu")
    assert rankme(one_direction) < 1.1 and top_share(one_direction, 1) > 0.999


def test_subspace_overlap_bounds():
    basis = torch.linalg.qr(torch.randn(12, 12))[0]
    assert abs(subspace_overlap(basis[:, :4], basis[:, :4]) - 1.0) < 1e-6
    assert subspace_overlap(basis[:, :4], basis[:, 4:8]) < 1e-6
