import numpy as np
import cvxpy as cp
import pytest
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
from pypfopt import exceptions
from tests.utilities_for_tests import setup_efficient_frontier


def test_custom_convex_equal_weights():
    ef = setup_efficient_frontier()

    def new_objective(w):
        return cp.sum(w ** 2)

    ef.convex_objective(new_objective)
    np.testing.assert_allclose(ef.weights, np.array([1 / 20] * 20))


def test_custom_convex_min_var():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    built_in = ef.weights

    # With custom objective
    ef = setup_efficient_frontier()
    ef.convex_objective(
        objective_functions.portfolio_variance, cov_matrix=ef.cov_matrix
    )
    custom = ef.weights
    np.testing.assert_allclose(built_in, custom, atol=1e-7)


def test_custom_convex_objective_market_neutral_efficient_risk():
    target_risk = 0.19
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.efficient_risk(target_risk, market_neutral=True)
    built_in = ef.weights

    # Recreate the market-neutral efficient_risk optimiser using this API
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.add_constraint(lambda x: cp.sum(x) == 0)
    ef.add_constraint(lambda x: cp.quad_form(x, ef.cov_matrix) <= target_risk ** 2)
    ef.convex_objective(lambda x: -x @ ef.expected_returns, weights_sum_to_one=False)
    custom = ef.weights
    np.testing.assert_allclose(built_in, custom, atol=1e-7)


def test_convex_sharpe_raises_error():
    # With custom objective
    with pytest.raises(exceptions.OptimizationError):
        ef = setup_efficient_frontier()
        ef.convex_objective(
            objective_functions.sharpe_ratio,
            expected_returns=ef.expected_returns,
            cov_matrix=ef.cov_matrix,
        )


def test_custom_convex_logarithmic_barrier():
    # 60 Years of Portfolio Optimisation, Kolm et al (2014)
    ef = setup_efficient_frontier()

    def logarithmic_barrier(w, cov_matrix, k=0.1):
        log_sum = cp.sum(cp.log(w))
        var = cp.quad_form(w, cov_matrix)
        return var - k * log_sum

    w = ef.convex_objective(logarithmic_barrier, cov_matrix=ef.cov_matrix)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.23978400459553223, 0.21100848889958182, 1.041588448605623),
    )


def test_custom_convex_deviation_risk_parity_error():
    # 60 Years of Portfolio Optimisation, Kolm et al (2014)
    ef = setup_efficient_frontier()

    def deviation_risk_parity(w, cov_matrix):
        n = cov_matrix.shape[0]
        rp = (w * (cov_matrix @ w)) / cp.quad_form(w, cov_matrix)
        return cp.sum_squares(rp - 1 / n)

    with pytest.raises(exceptions.OptimizationError):
        ef.convex_objective(deviation_risk_parity, cov_matrix=ef.cov_matrix)


def test_custom_nonconvex_min_var():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    original_vol = ef.portfolio_performance()[1]

    # With custom objective
    ef = setup_efficient_frontier()
    ef.nonconvex_objective(
        objective_functions.portfolio_variance, objective_args=ef.cov_matrix
    )
    custom_vol = ef.portfolio_performance()[1]
    # Scipy should be close but not as good for this simple objective
    np.testing.assert_almost_equal(custom_vol, original_vol, decimal=5)
    assert original_vol < custom_vol


def test_custom_nonconvex_logarithmic_barrier():
    # 60 Years of Portfolio Optimisation, Kolm et al (2014)
    ef = setup_efficient_frontier()

    def logarithmic_barrier(weights, cov_matrix, k=0.1):
        log_sum = np.sum(np.log(weights))
        portfolio_volatility = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_volatility - k * log_sum

    w = ef.nonconvex_objective(logarithmic_barrier, objective_args=(ef.cov_matrix, 0.2))
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_nonconvex_deviation_risk_parity_1():
    # 60 Years of Portfolio Optimisation, Kolm et al (2014) - first definition
    ef = setup_efficient_frontier()

    def deviation_risk_parity(w, cov_matrix):
        diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
        return (diff ** 2).sum().sum()

    w = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_nonconvex_deviation_risk_parity_2():
    # 60 Years of Portfolio Optimisation, Kolm et al (2014) - second definition
    ef = setup_efficient_frontier()

    def deviation_risk_parity(w, cov_matrix):
        n = cov_matrix.shape[0]
        rp = (w * (cov_matrix @ w)) / cp.quad_form(w, cov_matrix)
        return cp.sum_squares(rp - 1 / n).value

    w = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_nonconvex_utility_objective():
    ef = setup_efficient_frontier()

    def utility_obj(weights, mu, cov_matrix, k=1):
        return -weights.dot(mu) + k * np.dot(weights.T, np.dot(cov_matrix, weights))

    w = ef.nonconvex_objective(
        utility_obj, objective_args=(ef.expected_returns, ef.cov_matrix, 1)
    )
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    vol1 = ef.portfolio_performance()[1]

    # If we increase k, volatility should decrease
    w = ef.nonconvex_objective(
        utility_obj, objective_args=(ef.expected_returns, ef.cov_matrix, 3)
    )
    vol2 = ef.portfolio_performance()[1]
    assert vol2 < vol1


def test_custom_nonconvex_objective_market_neutral_efficient_risk():
    # Recreate the market-neutral efficient_risk optimiser using this API
    target_risk = 0.19
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )

    weight_constr = {"type": "eq", "fun": lambda w: np.sum(w)}
    risk_constr = {
        "type": "eq",
        "fun": lambda w: target_risk ** 2 - np.dot(w.T, np.dot(ef.cov_matrix, w)),
    }
    constraints = [weight_constr, risk_constr]

    ef.nonconvex_objective(
        lambda w, mu: -w.T.dot(mu),
        objective_args=(ef.expected_returns),
        weights_sum_to_one=False,
        constraints=constraints,
    )
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2309497754562942, target_risk, 1.1102600451243954),
        atol=1e-6,
    )
