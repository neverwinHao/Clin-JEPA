import numpy as np

from clin_jepa.downstream.windows import Standardizer, sequences, summaries

NAN = np.nan


def test_summaries_of_one_window():
    column = np.array([NAN, 2.0, NAN, 6.0, 4.0, NAN], dtype=np.float32)
    empty = np.full(6, NAN, dtype=np.float32)
    out = summaries(np.stack([column, empty], axis=1)[None])
    np.testing.assert_allclose(out[0, :6], [4.0, 4.0, 2.0, 6.0, np.std([2.0, 6.0, 4.0]), 2.0], rtol=1e-6)
    assert np.isnan(out[0, 6:12]).all()
    np.testing.assert_array_equal(out[0, 12:], [3.0, 0.0])


def test_sequences_append_the_observed_mask():
    window = np.array([[[1.0, NAN], [NAN, 3.0]]], dtype=np.float32)
    out = sequences(window)
    assert out.shape == (1, 2, 4)
    np.testing.assert_array_equal(out[0, :, 2:], [[1.0, 0.0], [0.0, 1.0]])


def test_standardizer_imputes_with_the_training_median():
    train = np.array([[1.0, NAN, 5.0], [3.0, NAN, 5.0], [8.0, NAN, 5.0]], dtype=np.float32)
    scaler = Standardizer.fit(train)
    raw = scaler.transform(np.array([[NAN, 7.0, NAN]], dtype=np.float32), scale=False)
    np.testing.assert_array_equal(raw, [[3.0, 7.0, 5.0]])
    scaled = scaler.transform(train, scale=True)
    np.testing.assert_allclose(scaled[:, 0].mean(), 0.0, atol=1e-6)
    np.testing.assert_array_equal(scaled[:, 1:], np.zeros((3, 2), dtype=np.float32))
