import numpy as np
import pytest
from pypfopt.efficient_frontier import EfficientFrontier
from tests.utilities_for_tests import get_data, setup_efficient_frontier


def test_custom_upper_bound():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0, 0.10)
    )
    ef.max_sharpe()
    ef.portfolio_performance()
    assert ef.weights.max() <= 0.1
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_lower_bound():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0.02, 1)
    )
    ef.max_sharpe()
    assert ef.weights.min() >= 0.02
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_bounds():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0.03, 0.13)
    )
    ef.max_sharpe()
    assert ef.weights.min() >= 0.03
    assert ef.weights.max() <= 0.13
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_bounds_errors():
    with pytest.raises(ValueError):
        EfficientFrontier(
            *setup_efficient_frontier(data_only=True), weight_bounds=(0.06, 1)
        )
    assert EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0, 1)
    )

    with pytest.raises(ValueError):
        EfficientFrontier(
            *setup_efficient_frontier(data_only=True), weight_bounds=(0.06, 1, 3)
        )


def test_clean_weights():
    ef = setup_efficient_frontier()
    ef.max_sharpe()
    number_tiny_weights = sum(ef.weights < 1e-4)
    cleaned = ef.clean_weights(cutoff=1e-4, rounding=5)
    cleaned_weights = cleaned.values()
    clean_number_tiny_weights = sum(i < 1e-4 for i in cleaned_weights)
    assert clean_number_tiny_weights == number_tiny_weights
    #  Check rounding
    cleaned_weights_str_length = [len(str(i)) for i in cleaned_weights]
    assert all([length == 7 or length ==
                3 for length in cleaned_weights_str_length])


def test_clean_weights_short():
    ef = setup_efficient_frontier()
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.max_sharpe()
    # In practice we would never use such a high cutoff
    number_tiny_weights = sum(np.abs(ef.weights) < 0.05)
    cleaned = ef.clean_weights(cutoff=0.05)
    cleaned_weights = cleaned.values()
    clean_number_tiny_weights = sum(abs(i) < 0.05 for i in cleaned_weights)
    assert clean_number_tiny_weights == number_tiny_weights


def test_clean_weights_error():
    ef = setup_efficient_frontier()
    ef.max_sharpe()
    with pytest.raises(ValueError):
        ef.clean_weights(rounding=1.3)
    with pytest.raises(ValueError):
        ef.clean_weights(rounding=0)
    assert ef.clean_weights(rounding=3)


def test_efficient_frontier_init_errors():
    df = get_data()
    mean_returns = df.pct_change().dropna(how="all").mean()
    with pytest.raises(TypeError):
        EfficientFrontier("test", "string")

    with pytest.raises(TypeError):
        EfficientFrontier(mean_returns, mean_returns)
