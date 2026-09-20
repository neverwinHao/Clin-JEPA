import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from clin_jepa.downstream.bootstrap import bootstrap, stay_draws


def data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    stays = rng.integers(0, 120, size=n) * 11 + 5
    labels = (rng.random(n) < 0.2).astype(np.float32)
    scores = np.round(rng.random((2, n)) * 0.6 + 0.3 * labels, 2).astype(np.float32)
    return labels, scores, stays


def test_draws_depend_on_the_stays_and_the_seed_only():
    labels, _, stays = data()
    unique, counts = stay_draws(stays, 20, 42)
    shuffled = np.random.default_rng(1).permutation(stays)
    np.testing.assert_array_equal(counts, stay_draws(shuffled, 20, 42)[1])
    assert (counts.sum(axis=1) == len(unique)).all()
    assert not np.array_equal(counts, stay_draws(stays, 20, 43)[1])


def test_bootstrap_matches_weighted_scikit_learn_metrics():
    labels, scores, stays = data()
    out = bootstrap(labels, scores, stays, 15, 42)
    unique, counts = stay_draws(stays, 15, 42)
    weights = counts[:, np.searchsorted(unique, stays)].astype(float)
    for b in range(15):
        auroc = np.mean([roc_auc_score(labels, s, sample_weight=weights[b]) for s in scores])
        auprc = np.mean([average_precision_score(labels, s, sample_weight=weights[b]) for s in scores])
        np.testing.assert_allclose([out["auroc"][b], out["auprc"][b]], [auroc, auprc], rtol=1e-10)
    assert out["n"] == len(labels) and out["n_positive"] == int(labels.sum())


def test_a_draw_without_positives_is_undefined():
    labels = np.array([1, 0, 0, 0], dtype=np.float32)
    out = bootstrap(labels, np.array([[0.9, 0.1, 0.2, 0.3]], dtype=np.float32), np.array([1, 2, 3, 4]), 200, 0)
    assert np.isnan(out["auroc"]).any() and np.isfinite(out["auroc"]).any()
    assert set(np.unique(out["auroc"][np.isfinite(out["auroc"])])) == {1.0}
