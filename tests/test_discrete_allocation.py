import numpy as np
import pandas as pd
import pytest
from pypfopt.discrete_allocation import get_latest_prices, DiscreteAllocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from tests.utilities_for_tests import get_data


def test_get_latest_prices():
    df = get_data()
    latest_prices = get_latest_prices(df)
    assert len(latest_prices) == 20
    assert list(latest_prices.index) == list(df.columns)
    assert latest_prices.name == pd.Timestamp(2018, 4, 11)


def test_get_latest_prices_error():
    df = get_data()
    with pytest.raises(TypeError):
        get_latest_prices(df.values)


def test_remove_zero_positions():
    raw = {"MA": 14, "FB": 12, "XOM": 0, "PFE": 51, "BABA": 5, "GOOG": 0}

    da = DiscreteAllocation({}, pd.Series())
    assert da._remove_zero_positions(raw) == {"MA": 14, "FB": 12, "PFE": 51, "BABA": 5}


def test_greedy_portfolio_allocation():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    allocation, leftover = da.greedy_portfolio()

    assert allocation == {
        "MA": 14,
        "FB": 12,
        "PFE": 51,
        "BABA": 5,
        "AAPL": 5,
        "BBY": 9,
        "SBUX": 6,
        "GOOG": 1,
        "AMD": 1,
    }

    total = 0
    for ticker, num in allocation.items():
        total += num * latest_prices[ticker]
    np.testing.assert_almost_equal(total + leftover, 10000, decimal=4)


def test_greedy_allocation_rmse_error():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    da.greedy_portfolio()
    np.testing.assert_almost_equal(da._allocation_rmse_error(), 0.0257368)


def test_greedy_portfolio_allocation_short():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    allocation, leftover = da.greedy_portfolio()

    assert da.allocation == {
        "MA": 15,
        "PFE": 45,
        "FB": 8,
        "BABA": 5,
        "AAPL": 4,
        "BBY": 8,
        "AMZN": 1,
        "SBUX": 9,
        "WMT": 2,
        "XOM": 2,
        "BAC": -32,
        "GM": -16,
        "GE": -43,
        "SHLD": -110,
        "AMD": -34,
        "JPM": -1,
        "T": -1,
        "UAA": -1,
    }
    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]
    np.testing.assert_almost_equal(
        long_total + short_total + leftover, 13000, decimal=4
    )


def test_greedy_allocation_rmse_error_short():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    da.greedy_portfolio()
    np.testing.assert_almost_equal(da._allocation_rmse_error(), 0.03306318)


def test_greedy_portfolio_allocation_short_different_params():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(
        w, latest_prices, total_portfolio_value=50000, short_ratio=0.5
    )
    allocation, leftover = da.greedy_portfolio()

    assert da.allocation == {
        "MA": 77,
        "PFE": 225,
        "FB": 41,
        "BABA": 25,
        "AAPL": 23,
        "BBY": 44,
        "AMZN": 2,
        "SBUX": 45,
        "GOOG": 3,
        "WMT": 11,
        "XOM": 11,
        "BAC": -271,
        "GM": -133,
        "GE": -355,
        "SHLD": -923,
        "AMD": -284,
        "JPM": -6,
        "T": -13,
        "UAA": -7,
        "RRC": -2,
    }
    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]
    np.testing.assert_almost_equal(long_total + short_total + leftover, 75000)


def test_lp_portfolio_allocation():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    allocation, leftover = da.lp_portfolio()

    assert da.allocation == {
        "AAPL": 5.0,
        "FB": 11.0,
        "BABA": 5.0,
        "AMZN": 1.0,
        "BBY": 7.0,
        "MA": 14.0,
        "PFE": 50.0,
        "SBUX": 5.0,
    }
    total = 0
    for ticker, num in allocation.items():
        total += num * latest_prices[ticker]
    np.testing.assert_almost_equal(total + leftover, 10000, decimal=4)


def test_lp_allocation_rmse_error():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    da.lp_portfolio()
    np.testing.assert_almost_equal(da._allocation_rmse_error(verbose=False), 0.0170634)


def test_lp_portfolio_allocation_short():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    allocation, leftover = da.lp_portfolio()

    assert da.allocation == {
        "GOOG": 1.0,
        "AAPL": 5.0,
        "FB": 8.0,
        "BABA": 5.0,
        "WMT": 2.0,
        "XOM": 2.0,
        "BBY": 9.0,
        "MA": 16.0,
        "PFE": 46.0,
        "SBUX": 9.0,
        "GE": -43.0,
        "AMD": -34.0,
        "BAC": -32.0,
        "GM": -16.0,
        "T": -1.0,
        "UAA": -1.0,
        "SHLD": -110.0,
        "JPM": -1.0,
    }
    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]
    np.testing.assert_almost_equal(
        long_total + short_total + leftover, 13000, decimal=5
    )


def test_lp_allocation_rmse_error_short():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    da.lp_portfolio()
    np.testing.assert_almost_equal(da._allocation_rmse_error(), 0.02699558)


def test_lp_portfolio_allocation_different_params():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(
        w, latest_prices, total_portfolio_value=80000, short_ratio=0.4
    )
    allocation, leftover = da.lp_portfolio()

    assert da.allocation == {
        "GOOG": 1.0,
        "AAPL": 43.0,
        "FB": 95.0,
        "BABA": 44.0,
        "AMZN": 4.0,
        "AMD": 1.0,
        "SHLD": 3.0,
        "BBY": 69.0,
        "MA": 114.0,
        "PFE": 412.0,
        "SBUX": 51.0,
    }
    total = 0
    for ticker, num in allocation.items():
        total += num * latest_prices[ticker]
    np.testing.assert_almost_equal(total + leftover, 80000, decimal=4)


def test_rmse_decreases_with_value():
    # As total_portfolio_value increases, rmse should decrease.
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()
    latest_prices = get_latest_prices(df)

    da1 = DiscreteAllocation(w, latest_prices, total_portfolio_value=10000)
    da1.greedy_portfolio()
    rmse1 = da1._allocation_rmse_error(verbose=False)
    da2 = DiscreteAllocation(w, latest_prices, total_portfolio_value=100000)
    da2.greedy_portfolio()
    rmse2 = da2._allocation_rmse_error(verbose=False)
    assert rmse2 < rmse1

    da3 = DiscreteAllocation(w, latest_prices, total_portfolio_value=10000)
    da3.lp_portfolio()
    rmse3 = da3._allocation_rmse_error(verbose=False)
    da4 = DiscreteAllocation(w, latest_prices, total_portfolio_value=30000)
    da4.lp_portfolio()
    rmse4 = da4._allocation_rmse_error(verbose=False)
    assert rmse4 < rmse3


def test_allocation_errors():
    df = get_data()
    mu = mean_historical_return(df)
    S = sample_cov(df)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()
    latest_prices = get_latest_prices(df)

    with pytest.raises(TypeError):
        DiscreteAllocation(ef.weights, latest_prices)
    with pytest.raises(TypeError):
        DiscreteAllocation(w, latest_prices.values.tolist())
    with pytest.raises(ValueError):
        DiscreteAllocation(w, latest_prices, total_portfolio_value=0)
    with pytest.raises(ValueError):
        DiscreteAllocation(w, latest_prices, short_ratio=-0.4)
