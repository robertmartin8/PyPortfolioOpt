import numpy as np
from tests.utilities_for_tests import setup_efficient_frontier
from pypfopt import objective_functions


def test_custom_objective_equal_weights():
    ef = setup_efficient_frontier()

    def new_objective(weights):
        return (weights ** 2).sum()

    ef.custom_objective(new_objective)
    np.testing.assert_allclose(ef.weights, np.array([1 / 20] * 20))


def test_custom_objective_min_var():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    built_in = ef.weights

    # With custom objective
    ef = setup_efficient_frontier()
    ef.custom_objective(objective_functions.volatility, ef.cov_matrix, 0)
    custom = ef.weights
    np.testing.assert_allclose(built_in, custom, atol=1e-7)


def test_custom_objective_sharpe_L2():
    ef = setup_efficient_frontier()
    ef.gamma = 2
    ef.max_sharpe()
    built_in = ef.weights

    # With custom objective
    ef = setup_efficient_frontier()
    ef.custom_objective(objective_functions.negative_sharpe,
                        ef.expected_returns, ef.cov_matrix, 2)
    custom = ef.weights
    np.testing.assert_allclose(built_in, custom, atol=1e-7)


def test_custom_logarithmic_barrier():
    # 60 Years of Portfolio Optimisation, Kolm et al (2014)
    ef = setup_efficient_frontier()

    def logarithmic_barrier(weights, cov_matrix, k=0.1):
        log_sum = np.sum(np.log(weights))
        portfolio_volatility = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_volatility - k * log_sum

    w = ef.custom_objective(logarithmic_barrier, ef.cov_matrix, 0.1)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_deviation_risk_parity():
    # 60 Years of Portfolio Optimisation, Kolm et al (2014)
    ef = setup_efficient_frontier()

    def deviation_risk_parity(w, cov_matrix):
        diff = w * np.dot(cov_matrix, w) - \
            (w * np.dot(cov_matrix, w)).reshape(-1, 1)
        return (diff ** 2).sum().sum()

    w = ef.custom_objective(deviation_risk_parity, ef.cov_matrix)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_utility_objective():
    ef = setup_efficient_frontier()

    def utility_obj(weights, mu, cov_matrix, k=1):
        return -weights.dot(mu) + k * np.dot(weights.T, np.dot(cov_matrix, weights))

    w = ef.custom_objective(utility_obj, ef.expected_returns, ef.cov_matrix, 1)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    vol1 = ef.portfolio_performance()[1]

    # If we increase k, volatility should decrease
    ef.custom_objective(utility_obj, ef.expected_returns, ef.cov_matrix, 2)
    vol2 = ef.portfolio_performance()[1]
    assert vol2 < vol1
