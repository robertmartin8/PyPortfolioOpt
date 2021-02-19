import json
import os
import numpy as np
import pandas as pd
import pytest
import cvxpy as cp
from pypfopt import EfficientFrontier
from pypfopt import exceptions
from pypfopt.base_optimizer import portfolio_performance, BaseOptimizer
from tests.utilities_for_tests import get_data, setup_efficient_frontier


def test_base_optimizer():
    """ Cover code not covered elsewhere."""
    # Test tickers not provided
    bo = BaseOptimizer(2)
    assert bo.tickers == [0, 1]
    w = {0: 0.4, 1: 0.6}
    bo.set_weights(w)
    assert dict(bo.clean_weights()) == w


def test_custom_bounds():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0.02, 0.13)
    )
    ef.min_volatility()
    np.testing.assert_allclose(ef._lower_bounds, np.array([0.02] * ef.n_assets))
    np.testing.assert_allclose(ef._upper_bounds, np.array([0.13] * ef.n_assets))

    assert ef.weights.min() >= 0.02
    assert ef.weights.max() <= 0.13
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_bounds_different_values():
    bounds = [(0.01, 0.13), (0.02, 0.11)] * 10
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=bounds
    )
    ef.min_volatility()
    assert (0.01 <= ef.weights[::2]).all() and (ef.weights[::2] <= 0.13).all()
    assert (0.02 <= ef.weights[1::2]).all() and (ef.weights[1::2] <= 0.11).all()
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    bounds = ((0.01, 0.13), (0.02, 0.11)) * 10
    assert EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=bounds
    )


def test_weight_bounds_minus_one_to_one():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    assert ef.max_sharpe()
    assert ef.min_volatility()


def test_none_bounds():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, 0.3)
    )
    ef.min_volatility()
    w1 = ef.weights

    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 0.3)
    )
    ef.min_volatility()
    w2 = ef.weights
    np.testing.assert_array_almost_equal(w1, w2)


def test_bound_input_types():
    bounds = [0.01, 0.13]
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=bounds
    )
    assert ef
    np.testing.assert_allclose(ef._lower_bounds, np.array([0.01] * ef.n_assets))
    np.testing.assert_allclose(ef._upper_bounds, np.array([0.13] * ef.n_assets))

    lb = np.array([0.01, 0.02] * 10)
    ub = np.array([0.07, 0.2] * 10)
    assert EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(lb, ub)
    )
    bounds = ((0.01, 0.13), (0.02, 0.11)) * 10
    assert EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=bounds
    )


def test_bound_failure():
    # Ensure optimization fails when lower bound is too high or upper bound is too low
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0.06, 0.13)
    )
    with pytest.raises(exceptions.OptimizationError):
        ef.min_volatility()

    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0, 0.04)
    )
    with pytest.raises(exceptions.OptimizationError):
        ef.min_volatility()


def test_bounds_errors():
    assert EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0, 1)
    )

    with pytest.raises(TypeError):
        EfficientFrontier(
            *setup_efficient_frontier(data_only=True), weight_bounds=(0.06, 1, 3)
        )

    with pytest.raises(TypeError):
        # Not enough bounds
        bounds = [(0.01, 0.13), (0.02, 0.11)] * 5
        EfficientFrontier(
            *setup_efficient_frontier(data_only=True), weight_bounds=bounds
        )


def test_clean_weights():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    number_tiny_weights = sum(ef.weights < 1e-4)
    cleaned = ef.clean_weights(cutoff=1e-4, rounding=5)
    cleaned_weights = cleaned.values()
    clean_number_tiny_weights = sum(i < 1e-4 for i in cleaned_weights)
    assert clean_number_tiny_weights == number_tiny_weights
    #  Check rounding
    cleaned_weights_str_length = [len(str(i)) for i in cleaned_weights]
    assert all([length == 7 or length == 3 for length in cleaned_weights_str_length])


def test_clean_weights_short():
    ef = setup_efficient_frontier()
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.min_volatility()
    # In practice we would never use such a high cutoff
    number_tiny_weights = sum(np.abs(ef.weights) < 0.05)
    cleaned = ef.clean_weights(cutoff=0.05)
    cleaned_weights = cleaned.values()
    clean_number_tiny_weights = sum(abs(i) < 0.05 for i in cleaned_weights)
    assert clean_number_tiny_weights == number_tiny_weights


def test_clean_weights_error():
    ef = setup_efficient_frontier()
    with pytest.raises(AttributeError):
        ef.clean_weights()
    ef.min_volatility()
    with pytest.raises(ValueError):
        ef.clean_weights(rounding=1.3)
    with pytest.raises(ValueError):
        ef.clean_weights(rounding=0)
    assert ef.clean_weights(rounding=3)


def test_clean_weights_no_rounding():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    # ensure the call does not fail
    # in previous commits, this call would raise a ValueError
    cleaned = ef.clean_weights(rounding=None, cutoff=0)
    assert cleaned
    np.testing.assert_array_almost_equal(
        np.sort(ef.weights), np.sort(list(cleaned.values()))
    )


def test_efficient_frontier_init_errors():
    df = get_data()
    mean_returns = df.pct_change().dropna(how="all").mean()
    with pytest.raises(TypeError):
        EfficientFrontier("test", "string")

    with pytest.raises(TypeError):
        EfficientFrontier(mean_returns, mean_returns)


def test_set_weights():
    ef = setup_efficient_frontier()
    w1 = ef.min_volatility()
    test_weights = ef.weights
    ef.min_volatility()
    ef.set_weights(w1)
    np.testing.assert_array_almost_equal(test_weights, ef.weights)


def test_save_weights_to_file():
    ef = setup_efficient_frontier()
    ef.min_volatility()

    ef.save_weights_to_file("tests/test.txt")
    with open("tests/test.txt", "r") as f:
        file = f.read()
    parsed = json.loads(file.replace("'", '"'))
    assert ef.clean_weights() == parsed

    ef.save_weights_to_file("tests/test.json")
    with open("tests/test.json", "r") as f:
        parsed = json.load(f)
    assert ef.clean_weights() == parsed

    ef.save_weights_to_file("tests/test.csv")
    with open("tests/test.csv", "r") as f:
        df = pd.read_csv(
            f,
            header=None,
            names=["ticker", "weight"],
            index_col=0,
            float_precision="high",
        )
    parsed = df["weight"].to_dict()
    assert ef.clean_weights() == parsed

    os.remove("tests/test.txt")
    os.remove("tests/test.json")
    os.remove("tests/test.csv")

    with pytest.raises(NotImplementedError):
        ef.save_weights_to_file("tests/test.xml")


def test_portfolio_performance():
    """
    Cover logic in base_optimizer.portfolio_performance not covered elsewhere.
    """
    ef = setup_efficient_frontier()
    ef.min_volatility()
    expected = ef.portfolio_performance()

    # Cover verbose logic
    assert (
        portfolio_performance(ef.weights, ef.expected_returns, ef.cov_matrix, True)
        == expected
    )
    # including when used without expected returns too.
    assert portfolio_performance(ef.weights, None, ef.cov_matrix, True) == (
        None,
        expected[1],
        None,
    )
    # Internal ticker creations when weights param is a dict and ...
    w_dict = dict(zip(ef.tickers, ef.weights))
    # ... expected_returns is a Series
    er = pd.Series(ef.expected_returns, index=ef.tickers)
    assert portfolio_performance(w_dict, er, ef.cov_matrix) == expected
    # ... cov_matrix is a DataFrame
    cov = pd.DataFrame(data=ef.cov_matrix, index=ef.tickers, columns=ef.tickers)
    assert portfolio_performance(w_dict, ef.expected_returns, cov) == expected

    # Will only support 'tickers' as dict keys that are ints starting from zero.
    w_dict = dict(zip(range(len(ef.weights)), ef.weights))
    assert portfolio_performance(w_dict, ef.expected_returns, ef.cov_matrix) == expected

    # Weights must not sum to zero.
    w_dict = dict(zip(range(len(ef.weights)), np.zeros(len(ef.weights))))
    with pytest.raises(ValueError):
        portfolio_performance(w_dict, ef.expected_returns, ef.cov_matrix)


def test_add_constraint_exception():
    ef = setup_efficient_frontier()
    # Must be callable.
    with pytest.raises(TypeError):
        ef.add_constraint(42)


def test_problem_access():
    ef = setup_efficient_frontier()
    ef.max_sharpe()
    assert isinstance(ef._opt, cp.Problem)
