import numpy as np

from clin_jepa.geometry.cohorts import assign
from clin_jepa.geometry.metrics import balanced_accuracy, cohens_d, leave_one_out_means

MAIN = {"deteriorating_delta": 3, "stable_std": 1.0, "stable_range": 2}


def test_cohort_assignment_tests_deterioration_first():
    rows = [{"delta": 3, "std": 0.5, "range": 1}, {"delta": 0, "std": 0.9, "range": 2},
            {"delta": 2, "std": 0.9, "range": 3}, {"delta": -4, "std": 2.0, "range": 5}]
    np.testing.assert_array_equal(assign(rows, MAIN), [1, 0, -1, -1])
    np.testing.assert_array_equal(assign(rows, dict(MAIN, stable_range=None)), [1, 0, 0, -1])


def test_cohens_d_and_balanced_accuracy():
    x, y = np.array([2.0, 4.0, 6.0]), np.array([1.0, 3.0, 5.0])
    np.testing.assert_allclose(cohens_d(x, y), 1.0 / 2.0)
    labels = np.array([1, 1, 1, 0])
    assert balanced_accuracy(np.array([True, True, False, True]), labels) == (2 / 3 + 1.0) / 2


def test_leave_one_out_means_exclude_the_stay_itself():
    v = np.array([[0.0], [2.0], [10.0], [20.0]], dtype=np.float32)
    y = np.array([1, 1, 0, 0])
    deteriorating, stable = leave_one_out_means(v, y)
    np.testing.assert_array_equal(deteriorating[:, 0], [2.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(stable[:, 0], [15.0, 15.0, 20.0, 10.0])
