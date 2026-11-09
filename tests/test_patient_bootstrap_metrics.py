"""Weighted metrics match sklearn with repeated patients and score ties."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts.patient_bootstrap_metrics import patient_draw_weights, weighted_roc_ap


def test_weighted_metrics_match_sklearn_with_ties():
    labels = np.array([0, 1, 0, 1, 0, 1, 0])
    scores = np.array([.1, .5, .5, .8, .8, .3, .2])
    subjects = np.array([1, 1, 2, 3, 4, 5, 6])
    weights = patient_draw_weights(subjects, 100, 42)
    auc, ap = weighted_roc_ap(labels, scores, weights)
    for i, w in enumerate(weights):
        if w[labels == 0].sum() and w[labels == 1].sum():
            assert abs(auc[i] - roc_auc_score(labels, scores, sample_weight=w)) < 1e-12
            assert abs(ap[i] - average_precision_score(labels, scores, sample_weight=w)) < 1e-12
        else:
            assert np.isnan(auc[i]) and np.isnan(ap[i])
