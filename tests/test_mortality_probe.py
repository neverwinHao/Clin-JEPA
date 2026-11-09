"""Check split isolation and paired patient-bootstrap behavior."""
import numpy as np
from scripts.run_mortality_probe import bootstrap_test, fit_classifier, fit_projector


def test_projector_is_fitted_on_train_only():
    rng = np.random.default_rng(42)
    d1 = {s: {'history': rng.normal(size=(n, 8)).astype('float32')} for s, n in [('train', 20), ('val', 5), ('test', 6)]}
    d2 = {s: {'history': v['history'].copy()} for s, v in d1.items()}
    d2['val']['history'] += 1000
    d2['test']['history'] -= 1000
    x1, m1 = fit_projector(d1, 'history', 2)
    x2, m2 = fit_projector(d2, 'history', 2)
    assert np.allclose(x1['train'], x2['train'])
    assert np.allclose(m1['pca'].components_, m2['pca'].components_)
    assert np.allclose(m1['scaler'].mean_, m2['scaler'].mean_)


def test_classifier_selection_does_not_read_test_features():
    x = np.arange(20, dtype='float32').reshape(10, 2)
    train = np.r_[np.zeros(5, dtype='int8'), np.ones(5, dtype='int8')]
    val_x = np.array([[0, 0], [1, 1], [4, 4], [5, 5]], dtype='float32')
    labels = {'train': train, 'val': np.array([0, 0, 1, 1]), 'test': np.array([0, 1])}
    f1 = {'train': x, 'val': val_x, 'test': np.zeros((2, 2))}
    f2 = dict(f1, test=np.ones((2, 2)) * 1000)
    m1, c1, auc1, _ = fit_classifier(f1, labels)
    m2, c2, auc2, _ = fit_classifier(f2, labels)
    assert c1 == c2 and auc1 == auc2
    assert np.allclose(m1.coef_, m2.coef_)


def test_bootstrap_delta_zero_for_identical_predictions():
    y = np.array([0, 0, 1, 1, 0, 1], dtype='int8')
    subjects = np.array([1, 1, 2, 3, 4, 5])
    scores = np.array([.1, .2, .8, .9, .3, .7])
    methods = {'history_32': scores, 'history_64': scores, 'history_32_plus_future_32': scores}
    intervals, delta, skipped = bootstrap_test(y, subjects, methods, 100, 42)
    assert delta['auroc'] == [0., 0.]
    assert delta['auprc'] == [0., 0.]
    assert intervals['history_64']['auroc'] == [1., 1.]
    assert skipped >= 0
