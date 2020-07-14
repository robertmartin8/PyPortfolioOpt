import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import pytest
import scipy.optimize as sco

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import objective_functions
from pypfopt import exceptions
from tests.utilities_for_tests import get_data, setup_efficient_frontier


def test_data_source():
    df = get_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 20
    assert len(df) == 7126
    assert df.index.is_all_dates


def test_returns_dataframe():
    df = get_data()
    returns_df = df.pct_change().dropna(how="all")
    assert isinstance(returns_df, pd.DataFrame)
    assert returns_df.shape[1] == 20
    assert len(returns_df) == 7125
    assert returns_df.index.is_all_dates
    assert not ((returns_df > 1) & returns_df.notnull()).any().any()


def test_efficient_frontier_inheritance():
    ef = setup_efficient_frontier()
    assert ef.clean_weights
    assert ef.n_assets
    assert ef.tickers
    assert isinstance(ef._constraints, list)
    assert isinstance(ef._lower_bounds, np.ndarray)
    assert isinstance(ef._upper_bounds, np.ndarray)


def test_portfolio_performance():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.portfolio_performance()
    ef.min_volatility()
    perf = ef.portfolio_performance()
    assert isinstance(perf, tuple)
    assert len(perf) == 3
    assert isinstance(perf[0], float)


def test_min_volatility():
    ef = setup_efficient_frontier()
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.17931232481259154, 0.15915084514118694, 1.00101463282373),
    )


def test_min_volatility_different_solver():
    ef = setup_efficient_frontier()
    ef.solver = "ECOS"
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    test_performance = (0.179312, 0.159151, 1.001015)
    np.testing.assert_allclose(ef.portfolio_performance(), test_performance, atol=1e-5)

    ef = setup_efficient_frontier()
    ef.solver = "OSQP"
    w = ef.min_volatility()
    np.testing.assert_allclose(ef.portfolio_performance(), test_performance, atol=1e-5)

    ef = setup_efficient_frontier()
    ef.solver = "SCS"
    w = ef.min_volatility()
    np.testing.assert_allclose(ef.portfolio_performance(), test_performance, atol=1e-3)


def test_min_volatility_no_rets():
    # Should work with no rets, see issue #82
    df = get_data()
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(None, S)
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_almost_equal(ef.portfolio_performance()[1], 0.15915084514118694)


def test_min_volatility_tx_costs():
    # Baseline
    ef = setup_efficient_frontier()
    ef.min_volatility()
    w1 = ef.weights

    # Pretend we were initally equal weight
    ef = setup_efficient_frontier()
    prev_w = np.array([1 / ef.n_assets] * ef.n_assets)
    ef.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    ef.min_volatility()
    w2 = ef.weights

    # TX cost should  pull closer to prev portfolio
    assert np.abs(prev_w - w2).sum() < np.abs(prev_w - w1).sum()


def test_min_volatility_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.1721356467349655, 0.1555915367269669, 0.9777887019776287),
    )

    # Shorting should reduce volatility
    volatility = ef.portfolio_performance()[1]
    ef_long_only = setup_efficient_frontier()
    ef_long_only.min_volatility()
    long_only_volatility = ef_long_only.portfolio_performance()[1]
    assert volatility < long_only_volatility


def test_min_volatility_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    weights = ef.min_volatility()
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])

    ef2 = setup_efficient_frontier()
    ef2.min_volatility()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.23129890623344232, 0.1955254118258614, 1.080672349748733),
    )


def test_min_volatility_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for _ in range(10):
        ef.add_objective(objective_functions.L2_reg, gamma=0.05)
        ef.min_volatility()
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_min_volatility_L2_reg_limit_case():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1e10)
    ef.min_volatility()
    equal_weights = np.array([1 / ef.n_assets] * ef.n_assets)
    np.testing.assert_array_almost_equal(ef.weights, equal_weights)


def test_min_volatility_L2_reg_increases_vol():
    # L2 reg should reduce the number of small weights
    # but increase in-sample volatility.
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.min_volatility()
    vol_no_reg = ef_no_reg.portfolio_performance()[1]
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    ef.min_volatility()
    vol = ef.portfolio_performance()[1]
    assert vol > vol_no_reg


def test_min_volatility_tx_costs_L2_reg():
    ef = setup_efficient_frontier()
    prev_w = np.array([1 / ef.n_assets] * ef.n_assets)
    ef.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    ef.add_objective(objective_functions.L2_reg)
    ef.min_volatility()

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2316565265271545, 0.1959773703677164, 1.0800049318450338),
    )


def test_min_volatility_cvxpy_vs_scipy():
    # cvxpy
    ef = setup_efficient_frontier()
    ef.min_volatility()
    w1 = ef.weights

    # scipy
    args = (ef.cov_matrix,)
    initial_guess = np.array([1 / ef.n_assets] * ef.n_assets)
    result = sco.minimize(
        objective_functions.portfolio_variance,
        x0=initial_guess,
        args=args,
        method="SLSQP",
        bounds=[(0, 1)] * 20,
        constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - 1}],
    )
    w2 = result["x"]

    cvxpy_var = objective_functions.portfolio_variance(w1, ef.cov_matrix)
    scipy_var = objective_functions.portfolio_variance(w2, ef.cov_matrix)
    assert cvxpy_var <= scipy_var


def test_min_volatility_sector_constraints():
    sector_mapper = {
        "T": "auto",
        "UAA": "airline",
        "SHLD": "retail",
        "XOM": "energy",
        "RRC": "energy",
        "BBY": "retail",
        "MA": "fig",
        "PFE": "pharma",
        "JPM": "fig",
        "SBUX": "retail",
        "GOOG": "tech",
        "AAPL": "tech",
        "FB": "tech",
        "AMZN": "tech",
        "BABA": "tech",
        "GE": "utility",
        "AMD": "tech",
        "WMT": "retail",
        "BAC": "fig",
        "GM": "auto",
    }

    sector_upper = {
        "tech": 0.2,
        "utility": 0.1,
        "retail": 0.2,
        "fig": 0.4,
        "airline": 0.05,
        "energy": 0.2,
    }
    sector_lower = {"utility": 0.01, "fig": 0.02, "airline": 0.01}

    # ef = setup_efficient_frontier()
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    weights = ef.min_volatility()

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_min_volatility_vs_max_sharpe():
    # Test based on issue #75
    expected_returns_daily = pd.Series(
        [0.043622, 0.120588, 0.072331, 0.056586], index=["AGG", "SPY", "GLD", "HYG"]
    )
    covariance_matrix = pd.DataFrame(
        [
            [0.000859, -0.000941, 0.001494, -0.000062],
            [-0.000941, 0.022400, -0.002184, 0.005747],
            [0.001494, -0.002184, 0.011518, -0.000129],
            [-0.000062, 0.005747, -0.000129, 0.002287],
        ],
        index=["AGG", "SPY", "GLD", "HYG"],
        columns=["AGG", "SPY", "GLD", "HYG"],
    )

    ef = EfficientFrontier(expected_returns_daily, covariance_matrix)
    ef.min_volatility()
    vol_min_vol = ef.portfolio_performance(risk_free_rate=0.00)[1]

    ef = EfficientFrontier(expected_returns_daily, covariance_matrix)
    ef.max_sharpe(risk_free_rate=0.00)
    vol_max_sharpe = ef.portfolio_performance(risk_free_rate=0.00)[1]

    assert vol_min_vol < vol_max_sharpe


def test_min_volatility_nonconvex_objective():
    ef = setup_efficient_frontier()
    ef.add_objective(lambda x: cp.sum((x + 1) / (x + 2) ** 2))
    with pytest.raises(exceptions.OptimizationError):
        ef.min_volatility()


def test_min_volatility_nonlinear_constraint():
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda x: (x + 1) / (x + 2) ** 2 <= 0.5)
    with pytest.raises(exceptions.OptimizationError):
        ef.min_volatility()


def test_max_sharpe_long_only():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.33035037367760506, 0.21671276571944567, 1.4320816434015786),
    )


def test_max_sharpe_long_weight_bounds():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0.03, 0.13)
    )
    ef.max_sharpe()
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert ef.weights.min() >= 0.03
    assert ef.weights.max() <= 0.13

    bounds = [(0.01, 0.13), (0.02, 0.11)] * 10
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=bounds
    )
    ef.max_sharpe()
    assert (0.01 <= ef.weights[::2]).all() and (ef.weights[::2] <= 0.13).all()
    assert (0.02 <= ef.weights[1::2]).all() and (ef.weights[1::2] <= 0.11).all()


def test_max_sharpe_explicit_bound():
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: w[0] >= 0.2)
    ef.add_constraint(lambda w: w[2] == 0.15)
    ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

    ef.max_sharpe()
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert ef.weights[0] >= 0.2 - 1e-5
    np.testing.assert_almost_equal(ef.weights[2], 0.15)
    assert ef.weights[3] + ef.weights[4] <= 0.10 + 1e-5


def test_max_sharpe_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.4072439477276246, 0.24823487545231313, 1.5599900981762558),
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.max_sharpe()
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_max_sharpe_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=5)

    with warnings.catch_warnings(record=True) as w:
        weights = ef.max_sharpe()
        assert len(w) == 1

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2936875354933478, 0.22783545277575057, 1.2012508683744123),
    )

    ef2 = setup_efficient_frontier()
    ef2.max_sharpe()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_max_sharpe_L2_reg_many_values():
    warnings.filterwarnings("ignore")

    ef = setup_efficient_frontier()
    ef.max_sharpe()
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for i in range(1, 20, 2):
        ef = setup_efficient_frontier()
        ef.add_objective(objective_functions.L2_reg, gamma=0.05 * i)
        ef.max_sharpe()
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_max_sharpe_L2_reg_different_gamma():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    ef.max_sharpe()

    ef2 = setup_efficient_frontier()
    ef2.add_objective(objective_functions.L2_reg, gamma=0.01)
    ef2.max_sharpe()

    # Higher gamma should pull close to equal weight
    equal_weight = np.array([1 / ef.n_assets] * ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_max_sharpe_L2_reg_reduces_sharpe():
    # L2 reg should reduce the number of small weights at the cost of Sharpe
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.max_sharpe()
    sharpe_no_reg = ef_no_reg.portfolio_performance()[2]
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    ef.max_sharpe()
    sharpe = ef.portfolio_performance()[2]
    assert sharpe < sharpe_no_reg


def test_max_sharpe_L2_reg_with_shorts():
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.max_sharpe()
    initial_number = sum(ef_no_reg.weights > 0.01)

    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    ef.add_objective(objective_functions.L2_reg)
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3076093180094401, 0.22415982749409985, 1.2830546901496447),
    )
    new_number = sum(ef.weights > 0.01)
    assert new_number >= initial_number


def test_max_sharpe_risk_free_rate():
    ef = setup_efficient_frontier()
    ef.max_sharpe()
    _, _, initial_sharpe = ef.portfolio_performance()
    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0.10)
    _, _, new_sharpe = ef.portfolio_performance(risk_free_rate=0.10)
    assert new_sharpe <= initial_sharpe

    ef = setup_efficient_frontier()
    ef.max_sharpe(risk_free_rate=0)
    _, _, new_sharpe = ef.portfolio_performance(risk_free_rate=0)
    assert new_sharpe >= initial_sharpe


def test_min_vol_pair_constraint():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    old_sum = ef.weights[:2].sum()
    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: (w[1] + w[0] <= old_sum / 2))
    ef.min_volatility()
    new_sum = ef.weights[:2].sum()
    assert new_sum <= old_sum / 2 + 1e-4


def test_max_sharpe_pair_constraint():
    ef = setup_efficient_frontier()
    ef.max_sharpe()
    old_sum = ef.weights[:2].sum()

    ef = setup_efficient_frontier()
    ef.add_constraint(lambda w: (w[1] + w[0] <= old_sum / 2))
    ef.max_sharpe()
    new_sum = ef.weights[:2].sum()
    assert new_sum <= old_sum / 2 + 1e-4


def test_max_sharpe_sector_constraints_manual():
    sector_mapper = {
        "GOOG": "tech",
        "AAPL": "tech",
        "FB": "tech",
        "AMZN": "tech",
        "BABA": "tech",
        "GE": "utility",
        "AMD": "tech",
        "WMT": "retail",
        "BAC": "fig",
        "GM": "auto",
        "T": "auto",
        "UAA": "airline",
        "SHLD": "retail",
        "XOM": "energy",
        "RRC": "energy",
        "BBY": "retail",
        "MA": "fig",
        "PFE": "pharma",
        "JPM": "fig",
        "SBUX": "retail",
    }

    sector_upper = {
        "tech": 0.2,
        "utility": 0.1,
        "retail": 0.2,
        "fig": 0.4,
        "airline": 0.05,
        "energy": 0.2,
    }
    sector_lower = {"utility": 0.01, "fig": 0.02, "airline": 0.01}

    ef = setup_efficient_frontier()
    for sector in sector_upper:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
    for sector in sector_lower:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    weights = ef.max_sharpe()

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_max_sharpe_sector_constraints_auto():
    sector_mapper = {
        "GOOG": "tech",
        "AAPL": "tech",
        "FB": "tech",
        "AMZN": "tech",
        "BABA": "tech",
        "GE": "utility",
        "AMD": "tech",
        "WMT": "retail",
        "BAC": "fig",
        "GM": "auto",
        "T": "auto",
        "UAA": "airline",
        "SHLD": "retail",
        "XOM": "energy",
        "RRC": "energy",
        "BBY": "retail",
        "MA": "fig",
        "PFE": "pharma",
        "JPM": "fig",
        "SBUX": "retail",
    }

    sector_upper = {
        "tech": 0.2,
        "utility": 0.1,
        "retail": 0.2,
        "fig": 0.4,
        "airline": 0.05,
        "energy": 0.2,
    }
    sector_lower = {"utility": 0.01, "fig": 0.02, "airline": 0.01}

    ef = setup_efficient_frontier()
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = ef.max_sharpe()

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_efficient_risk_sector_constraints_manual():
    sector_mapper = {
        "GOOG": "tech",
        "AAPL": "tech",
        "FB": "tech",
        "AMZN": "tech",
        "BABA": "tech",
        "GE": "utility",
        "AMD": "tech",
        "WMT": "retail",
        "BAC": "fig",
        "GM": "auto",
        "T": "auto",
        "UAA": "airline",
        "SHLD": "retail",
        "XOM": "energy",
        "RRC": "energy",
        "BBY": "retail",
        "MA": "fig",
        "PFE": "pharma",
        "JPM": "fig",
        "SBUX": "retail",
    }

    sector_upper = {
        "tech": 0.2,
        "utility": 0.1,
        "retail": 0.2,
        "fig": 0.4,
        "airline": 0.05,
        "energy": 0.2,
    }
    sector_lower = {"utility": 0.01, "fig": 0.02, "airline": 0.01}

    ef = setup_efficient_frontier()

    for sector in sector_upper:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
    for sector in sector_lower:
        is_sector = [sector_mapper[t] == sector for t in ef.tickers]
        ef.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    weights = ef.efficient_risk(0.19)

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_efficient_risk_sector_constraints_auto():
    sector_mapper = {
        "GOOG": "tech",
        "AAPL": "tech",
        "FB": "tech",
        "AMZN": "tech",
        "BABA": "tech",
        "GE": "utility",
        "AMD": "tech",
        "WMT": "retail",
        "BAC": "fig",
        "GM": "auto",
        "T": "auto",
        "UAA": "airline",
        "SHLD": "retail",
        "XOM": "energy",
        "RRC": "energy",
        "BBY": "retail",
        "MA": "fig",
        "PFE": "pharma",
        "JPM": "fig",
        "SBUX": "retail",
    }

    sector_upper = {
        "tech": 0.2,
        "utility": 0.1,
        "retail": 0.2,
        "fig": 0.4,
        "airline": 0.05,
        "energy": 0.2,
    }
    sector_lower = {"utility": 0.01, "fig": 0.02, "airline": 0.01}
    ef = setup_efficient_frontier()
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = ef.efficient_risk(0.19)
    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_max_quadratic_utility():
    ef = setup_efficient_frontier()
    w = ef.max_quadratic_utility(risk_aversion=2)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.40064324249527605, 0.2917825266124642, 1.3045443362029479),
    )

    ret1, var1, _ = ef.portfolio_performance()
    # increasing risk_aversion should lower both vol and return
    ef.max_quadratic_utility(10)
    ret2, var2, _ = ef.portfolio_performance()
    assert ret2 < ret1 and var2 < var1


def test_max_quadratic_utility_with_shorts():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.max_quadratic_utility()
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (1.3318330413711252, 1.0198436183533854, 1.2863080356272452),
    )


def test_max_quadratic_utility_market_neutral():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.max_quadratic_utility(market_neutral=True)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (1.13434841843883, 0.9896404148973286, 1.1260134506071473),
    )


def test_max_quadratic_utility_limit():
    # in limit of large risk_aversion, this should approach min variance.
    ef = setup_efficient_frontier()
    ef.max_quadratic_utility(risk_aversion=1e10)

    ef2 = setup_efficient_frontier()
    ef2.min_volatility()
    np.testing.assert_array_almost_equal(ef.weights, ef2.weights)


def test_max_quadratic_utility_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=5)
    weights = ef.max_quadratic_utility()

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2602803268728476, 0.21603540587515674, 1.112226608872166),
    )

    ef2 = setup_efficient_frontier()
    ef2.max_quadratic_utility()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_max_quadratic_utility_error():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.max_quadratic_utility(0)
    with pytest.raises(ValueError):
        ef.max_quadratic_utility(-1)


def test_efficient_risk():
    ef = setup_efficient_frontier()
    w = ef.efficient_risk(0.19)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.28577452556155075, 0.19, 1.3988132892376837),
        atol=1e-6,
    )


def test_efficient_risk_error():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    min_possible_vol = ef.portfolio_performance()[1]

    ef = setup_efficient_frontier()
    assert ef.efficient_risk(min_possible_vol + 0.01)

    ef = setup_efficient_frontier()
    with pytest.raises(exceptions.OptimizationError):
        # This volatility is too low
        ef.efficient_risk(min_possible_vol - 0.01)


def test_efficient_risk_many_values():
    for target_risk in np.array([0.16, 0.21, 0.30]):
        ef = setup_efficient_frontier()
        ef.efficient_risk(target_risk)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        volatility = ef.portfolio_performance()[1]
        assert abs(target_risk - volatility) < 1e-5


def test_efficient_risk_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    w = ef.efficient_risk(0.19)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.30468522897430295, 0.19, 1.4983424153337392),
        atol=1e-6,
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_risk(0.19)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_efficient_risk_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=5)
    weights = ef.efficient_risk(0.19)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.24087463760460398, 0.19, 1.162498090632486),
        atol=1e-6,
    )

    ef2 = setup_efficient_frontier()
    ef2.efficient_risk(0.19)

    # L2_reg should pull close to equal weight
    equal_weight = np.full((ef.n_assets,), 1 / ef.n_assets)
    assert (
        np.abs(equal_weight - ef.weights).sum()
        < np.abs(equal_weight - ef2.weights).sum()
    )


def test_efficient_risk_market_neutral():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    w = ef.efficient_risk(0.21, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2552600197428133, 0.21, 1.1202858085349783),
        atol=1e-6,
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_risk(0.21)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]
    assert long_only_sharpe > sharpe


def test_efficient_risk_market_neutral_L2_reg():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.add_objective(objective_functions.L2_reg)

    w = ef.efficient_risk(0.19, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.10755645826336145, 0.11079556786108302, 0.7902523535340413),
        atol=1e-6,
    )


def test_efficient_risk_market_neutral_warning():
    ef = setup_efficient_frontier()
    with warnings.catch_warnings(record=True) as w:
        ef.efficient_risk(0.19, market_neutral=True)
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert (
            str(w[0].message)
            == "Market neutrality requires shorting - bounds have been amended"
        )


def test_efficient_return():
    ef = setup_efficient_frontier()
    w = ef.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.25, 0.1738852429895079, 1.3227114391408021),
        atol=1e-6,
    )


def test_efficient_return_error():
    ef = setup_efficient_frontier()
    max_ret = ef.expected_returns.max()

    with pytest.raises(ValueError):
        ef.efficient_return(-0.1)
    with pytest.raises(ValueError):
        # This return is too high
        ef.efficient_return(max_ret + 0.01)


def test_efficient_return_many_values():
    ef = setup_efficient_frontier()
    for target_return in np.arange(0.25, 0.20, 0.28):
        ef.efficient_return(target_return)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        assert all([i >= 0 for i in ef.weights])
        mean_return = ef.portfolio_performance()[0]
        assert abs(target_return - mean_return) < 0.05


def test_efficient_return_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.16826225873038014, 1.3669137793315087)
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_efficient_return_L2_reg():
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    w = ef.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.20033592447690426, 1.1480716731187948)
    )


def test_efficient_return_market_neutral():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    w = ef.efficient_return(0.25, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_almost_equal(
        ef.portfolio_performance(), (0.25, 0.20567263154580923, 1.1182819914898223)
    )
    sharpe = ef.portfolio_performance()[2]
    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]
    assert long_only_sharpe > sharpe


def test_efficient_return_market_neutral_warning():
    # This fails
    ef = setup_efficient_frontier()
    with warnings.catch_warnings(record=True) as w:
        ef.efficient_return(0.25, market_neutral=True)
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert (
            str(w[0].message)
            == "Market neutrality requires shorting - bounds have been amended"
        )


def test_max_sharpe_semicovariance():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2972184894480104, 0.06443145011260347, 4.302533762060766),
    )


def test_max_sharpe_short_semicovariance():
    df = get_data()
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3564305116656491, 0.07201282488003401, 4.671813836300796),
    )


def test_min_volatilty_shrunk_L2_reg():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg)

    ef.cov_matrix = risk_models.CovarianceShrinkage(df).ledoit_wolf(
        shrinkage_target="constant_correlation"
    )
    w = ef.min_volatility()

    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.23127396601517256, 0.19563960638632416, 1.0799140824173181),
    )


def test_efficient_return_shrunk():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.CovarianceShrinkage(df).ledoit_wolf(
        shrinkage_target="single_factor"
    )
    w = ef.efficient_return(0.22)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.22, 0.0849639369932322, 2.353939884117318)
    )


def test_max_sharpe_exp_cov():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3678817256187322, 0.1753405505478982, 1.9840346373481956),
    )


def test_min_volatility_exp_cov_L2_reg():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.add_objective(objective_functions.L2_reg)
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2434082300792007, 0.17835412793427002, 1.2526103694192867),
    )


def test_efficient_risk_exp_cov_market_neutral():
    df = get_data()
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.efficient_risk(0.19, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3908928033782067, 0.18999999995323363, 1.9520673866815672),
        atol=1e-6,
    )
