import numpy as np

from clin_jepa.downstream.labels import any_ahead, extreme_ahead, extubation_failure, next_true, onset_label
from clin_jepa.downstream.organ_failure import early_event_label, fill, interpolate

NAN = np.nan


def test_onset_label_looks_ahead_and_stops_at_the_onset():
    label = onset_label(5.0, 8, horizon=3)
    np.testing.assert_array_equal(label, [0, 0, 1, 1, 1, NAN, NAN, NAN])
    np.testing.assert_array_equal(onset_label(float("inf"), 4, horizon=3), [0, 0, 0, 0])


def test_any_ahead_excludes_the_current_hour():
    flag = np.array([True, False, False, False, True, False])
    np.testing.assert_array_equal(any_ahead(flag, 2, 6), [False, False, True, True, False, False])


def test_extreme_ahead_ignores_missing_values():
    values = np.array([1.0, NAN, 4.0, 2.0, NAN])
    np.testing.assert_array_equal(extreme_ahead(values, 2, 5, largest=True), [4.0, 4.0, 2.0, NAN, NAN])
    np.testing.assert_array_equal(extreme_ahead(values, 2, 5, largest=False), [4.0, 2.0, 2.0, NAN, NAN])


def test_next_true():
    np.testing.assert_array_equal(next_true(np.array([False, True, False, False])), [1, 1, 4, 4, 4])


def test_early_event_label():
    state = np.array([0, 0, NAN, 1, 1, 0, NAN, NAN], dtype=np.float32)
    np.testing.assert_array_equal(early_event_label(state, 2), [0, 1, 1, NAN, NAN, NAN, NAN, NAN])


def test_fill_and_interpolate_respect_their_limits():
    values = np.array([NAN, 2.0, NAN, NAN, NAN, 6.0, NAN], dtype=np.float32)
    np.testing.assert_array_equal(fill(values, 1), [2.0, 2.0, 2.0, NAN, 6.0, 6.0, 6.0])
    np.testing.assert_array_equal(interpolate(values, 4, values.copy()), [NAN, 2.0, 3.0, 4.0, 5.0, 6.0, NAN])
    np.testing.assert_array_equal(interpolate(values, 3, values.copy()), values)


def test_extubation_failure():
    episodes = (np.array([2.0, 60.0]), np.array([40.0, 70.0]))
    assert extubation_failure(episodes, 100.0) == (1.0, 40.0)
    assert extubation_failure((np.array([2.0]), np.array([40.0])), 100.0) == (0.0, 40.0)
    assert np.isnan(extubation_failure((np.array([2.0]), np.array([20.0])), 100.0)[0])
    assert np.isnan(extubation_failure(None, 100.0)[0])
