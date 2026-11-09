"""Vectorized AUROC/AP for patient-clustered bootstrap, including score ties."""
import numpy as np


def patient_draw_weights(subjects, draws, seed):
    patients, inverse = np.unique(subjects, return_inverse=True)
    if draws < 1 or len(patients) < 2:
        raise ValueError('Need positive draws and at least two patients')
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(patients), np.full(len(patients), 1.0 / len(patients)), size=draws)
    return counts[:, inverse]


def weighted_roc_ap(y, score, weights):
    y = np.asarray(y)
    score = np.asarray(score)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if len(y) != len(score) or weights.shape[1] != len(y) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError('Invalid binary labels, scores or weights')
    if not np.isfinite(score).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Scores and weights must be finite and weights nonnegative')
    order = np.argsort(-score, kind='stable')
    score = score[order]
    labels = y[order]
    w = weights[:, order]
    starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
    pos = np.add.reduceat(w * labels[None, :], starts, axis=1)
    neg = np.add.reduceat(w * (1 - labels)[None, :], starts, axis=1)
    total_pos, total_neg = pos.sum(axis=1), neg.sum(axis=1)
    cum_pos, cum_neg = pos.cumsum(axis=1), neg.cumsum(axis=1)
    valid = (total_pos > 0) & (total_neg > 0)
    denominator = total_pos * total_neg
    numerator = (pos * (total_neg[:, None] - cum_neg + .5 * neg)).sum(axis=1)
    auc = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=valid)
    total_seen = cum_pos + cum_neg
    precision = np.divide(cum_pos, total_seen, out=np.zeros_like(cum_pos), where=total_seen > 0)
    ap_num = (pos * precision).sum(axis=1)
    ap = np.divide(ap_num, total_pos, out=np.full(len(weights), np.nan), where=valid)
    return auc, ap
