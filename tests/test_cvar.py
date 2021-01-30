import numpy as np
import pytest
from pypfopt import (
    expected_returns,
    EfficientCVaR,
    objective_functions,
)
from tests.utilities_for_tests import setup_efficient_cvar, get_data
from pypfopt.exceptions import OptimizationError

def test_cvar_example():
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df).dropna()

    cv = EfficientCVaR(mu, historical_rets)
    w = cv.efficient_return(0.2)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])
    print(cv.portfolio_performance())
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.20, 0.032664),
        rtol=1e-4,
        atol=1e-4,
    )


def test_cvar_errors():
    df = get_data()
    mu = expected_returns.mean_historical_return(df)
    historical_rets = expected_returns.returns_from_prices(df)

    with pytest.warns(UserWarning):
        EfficientCVaR(mu, historical_rets)

    historical_rets = historical_rets.dropna(axis=0, how="any")
    assert EfficientCVaR(mu, historical_rets)

    historical_rets = historical_rets.iloc[:, :-1]
    with pytest.raises(ValueError):
        EfficientCVaR(mu, historical_rets)


def test_cvar_example_weekly():
    df = get_data()
    df = df.resample("W").first()
    mu = expected_returns.mean_historical_return(df, frequency=52)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    cv = EfficientCVaR(mu, historical_rets, frequency=52)
    cv.efficient_return(0.2)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.2000000562544616, 0.068891),
        rtol=1e-4,
        atol=1e-4,
    )


def test_cvar_example_monthly():
    df = get_data()
    df = df.resample("M").first()
    mu = expected_returns.mean_historical_return(df, frequency=12)
    historical_rets = expected_returns.returns_from_prices(df).dropna()
    cv = EfficientCVaR(mu, historical_rets, frequency=12)

    cv.efficient_return(0.3)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.3, 0.032726),
        rtol=1e-4,
        atol=1e-4,
    )


def test_min_cvar():
    cv = setup_efficient_cvar()
    w = cv.min_cvar()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.12629, 0.032176),
        rtol=1e-4,
        atol=1e-4,
    )


def test_min_cvar_extra_constraints():
    cv = setup_efficient_cvar()
    w = cv.min_cvar()
    assert w["GOOG"] < 0.02 and w["AAPL"] > 0.02

    cv = setup_efficient_cvar()
    cv.add_constraint(lambda x: x[0] >= 0.03)
    cv.add_constraint(lambda x: x[1] <= 0.03)
    w = cv.min_cvar()
    assert w["GOOG"] >= 0.025 and w["AAPL"] <= 0.035


def test_min_cvar_different_solver():
    cv = setup_efficient_cvar(solver="ECOS")
    w = cv.min_cvar()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    test_performance = (0.12629, 0.032176)
    np.testing.assert_allclose(
        cv.portfolio_performance(), test_performance, rtol=1e-2, atol=1e-2
    )

    cv = setup_efficient_cvar(solver="OSQP")
    w = cv.min_cvar()
    np.testing.assert_allclose(
        cv.portfolio_performance(), test_performance, rtol=1e-2, atol=1e-2
    )


def test_min_cvar_tx_costs():
    # Baseline
    cv = setup_efficient_cvar()
    cv.min_cvar()
    w1 = cv.weights

    # Pretend we were initally equal weight
    cv = setup_efficient_cvar()
    prev_w = np.array([1 / cv.n_assets] * cv.n_assets)
    cv.add_objective(objective_functions.transaction_cost, w_prev=prev_w)
    cv.min_cvar()
    w2 = cv.weights

    # TX cost should  pull closer to prev portfolio
    assert np.abs(prev_w - w2).sum() < np.abs(prev_w - w1).sum()


def test_min_cvar_L2_reg():
    cv = setup_efficient_cvar()
    cv.add_objective(objective_functions.L2_reg, gamma=.01)
    weights = cv.min_cvar()
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= 0 for i in weights.values()])

    cv2 = setup_efficient_cvar()
    cv2.min_cvar()

    # L2_reg should pull close to equal weight
    equal_weight = np.full((cv.n_assets,), 1 / cv.n_assets)
    assert (
            np.abs(equal_weight - cv.weights).sum()
            < np.abs(equal_weight - cv2.weights).sum()
    )

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.163864, 0.032973),
        rtol=1e-4,
        atol=1e-4,
    )


def test_min_cvar_sector_constraints():
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

    cv = setup_efficient_cvar()
    cv.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    weights = cv.min_cvar()

    for sector in list(set().union(sector_upper, sector_lower)):
        sector_sum = 0
        for t, v in weights.items():
            if sector_mapper[t] == sector:
                sector_sum += v
        assert sector_sum <= sector_upper.get(sector, 1) + 1e-5
        assert sector_sum >= sector_lower.get(sector, 0) - 1e-5


def test_efficient_risk():
    cv = setup_efficient_cvar()
    w = cv.efficient_risk(0.2)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.462833, 0.2),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_risk_low_risk():
    cv = setup_efficient_cvar()
    cv.min_cvar()
    min_value = cv.portfolio_performance()[1]

    # Should fail below
    with pytest.raises(OptimizationError):
        cv = setup_efficient_cvar()
        cv.efficient_risk(min_value - 0.01)

    cv = setup_efficient_cvar()
    cv.efficient_risk(min_value + 0.01)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.359426, min_value + 0.01),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_risk_L2_reg():
    cv = setup_efficient_cvar()
    cv.add_objective(objective_functions.L2_reg, gamma=1)
    weights = cv.efficient_risk(0.19)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    np.testing.assert_array_less(np.zeros(len(weights)), cv.weights + 1e-4)
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.313392, 0.189838),
        rtol=1e-4,
        atol=1e-4,
    )

    ef2 = setup_efficient_cvar()
    cv.add_objective(objective_functions.L2_reg, gamma=1)
    ef2.efficient_risk(0.19)

    # L2_reg should pull close to equal weight
    equal_weight = np.full((cv.n_assets,), 1 / cv.n_assets)
    assert (
            np.abs(equal_weight - cv.weights).sum()
            < np.abs(equal_weight - ef2.weights).sum()
    )


def test_efficient_return():
    cv = setup_efficient_cvar()
    w = cv.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.25, 0.034611),
        rtol=1e-4,
        atol=1e-4,
    )


def test_efficient_return_L2_reg():
    cv = setup_efficient_cvar()
    cv.add_objective(objective_functions.L2_reg, gamma=1)
    w = cv.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cv.tickers)
    np.testing.assert_almost_equal(cv.weights.sum(), 1)
    print(w)
    assert all([i >= -1e-5 for i in w.values()])
    np.testing.assert_allclose(
        cv.portfolio_performance(),
        (0.25, 0.041786),
        rtol=1e-4,
        atol=1e-4,
    )
