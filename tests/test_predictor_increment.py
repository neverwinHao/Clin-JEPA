import numpy as np
import pytest

from scripts.evaluate_predictor_increment import copy_forward_summary, project_copy


def test_copy_forward_uses_only_last_observed_state():
    history = np.arange(2 * 7 * 4, dtype=np.float32).reshape(2, 7, 4)
    result = copy_forward_summary(history)
    np.testing.assert_array_equal(result[:, 0], history[:, 2])
    np.testing.assert_array_equal(result[:, 1], np.zeros((2, 4)))
    np.testing.assert_array_equal(result[:, 2], history[:, 2])
    changed = history.copy()
    changed[:, [0, 1, 3, 4, 5, 6]] += 1000
    np.testing.assert_array_equal(copy_forward_summary(changed), result)
    with pytest.raises(ValueError):
        copy_forward_summary(np.zeros((3, 6, 4)))


def test_copy_projection_fitted_only_on_train(tmp_path):
    rng = np.random.default_rng(42)
    for split, n in (("train", 40), ("val", 5), ("test", 5)):
        folder = tmp_path / split
        folder.mkdir()
        history = np.zeros((n, 7, 4096), dtype=np.float32)
        history[:, 2] = rng.normal(size=(n, 4096))
        np.save(folder / "history.npy", history)
    first, first_model = project_copy(tmp_path)
    test_path = tmp_path / "test" / "history.npy"
    test = np.load(test_path)
    test[:, 2] += 5
    np.save(test_path, test)
    second, second_model = project_copy(tmp_path)
    np.testing.assert_allclose(first_model["scaler"].mean_, second_model["scaler"].mean_)
    np.testing.assert_allclose(first_model["pca"].components_, second_model["pca"].components_)
    np.testing.assert_allclose(first["train"], second["train"])
    assert not np.allclose(first["test"], second["test"])
