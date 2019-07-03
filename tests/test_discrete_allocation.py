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


def test_greedy_portfolio_allocation():
    df = get_data()
    e_ret = mean_historical_return(df)
    cov = sample_cov(df)
    ef = EfficientFrontier(e_ret, cov)
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    allocation, leftover = da.greedy_portfolio()

    assert da.allocation == {
        "MA": 14,
        "FB": 12,
        "PFE": 51,
        "BABA": 5,
        "AAPL": 5,
        "AMZN": 0,
        "BBY": 9,
        "SBUX": 6,
        "GOOG": 1,
    }
    total = 0
    for ticker, num in allocation.items():
        total += num * latest_prices[ticker]
    np.testing.assert_almost_equal(total + leftover, 10000)


def test_greedy_portfolio_allocation_short():
    df = get_data()
    e_ret = mean_historical_return(df)
    cov = sample_cov(df)
    ef = EfficientFrontier(e_ret, cov, weight_bounds=(-1, 1))
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
        "GOOG": 0,
        "WMT": 2,
        "XOM": 2,
        "BAC": -33,
        "GM": -16,
        "GE": -43,
        "SHLD": -114,
        "AMD": -35,
        "JPM": -1,
    }
    long_total = 0
    short_total = 0
    for ticker, num in allocation.items():
        if num > 0:
            long_total += num * latest_prices[ticker]
        else:
            short_total -= num * latest_prices[ticker]
    np.testing.assert_almost_equal(long_total + short_total + leftover, 13000)


def test_lp_portfolio_allocation_short():
    pass


def test_allocation_rmse_error():
    df = get_data()
    e_ret = mean_historical_return(df)
    cov = sample_cov(df)
    ef = EfficientFrontier(e_ret, cov, weight_bounds=(-1, 1))
    w = ef.max_sharpe()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(w, latest_prices)
    assert da._allocation_rmse_error()


def test_allocation_errors():
    df = get_data()
    e_ret = mean_historical_return(df)
    cov = sample_cov(df)
    ef = EfficientFrontier(e_ret, cov)
    w = ef.max_sharpe()
    latest_prices = get_latest_prices(df)

    with pytest.raises(TypeError):
        DiscreteAllocation(ef.weights, latest_prices)
    with pytest.raises(TypeError):
        DiscreteAllocation(w, latest_prices.values.tolist())
    with pytest.raises(ValueError):
        DiscreteAllocation(w, latest_prices, min_allocation=0.5)
    with pytest.raises(ValueError):
        DiscreteAllocation(w, latest_prices, total_portfolio_value=0)
    with pytest.raises(ValueError):
        DiscreteAllocation(w, latest_prices, short_ratio=-0.4)
