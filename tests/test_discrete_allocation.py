import numpy as np
import pandas as pd
import pytest
from pypfopt import discrete_allocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from tests.utilities_for_tests import get_data


def test_get_latest_prices():
    df = get_data()
    latest_prices = discrete_allocation.get_latest_prices(df)
    assert len(latest_prices) == 20
    assert list(latest_prices.index) == list(df.columns)
    assert latest_prices.name == pd.Timestamp(2018, 4, 11)


def test_get_latest_prices_error():
    df = get_data()
    with pytest.raises(TypeError):
        discrete_allocation.get_latest_prices(df.values)


def test_portfolio_allocation():
    df = get_data()
    e_ret = mean_historical_return(df)
    cov = sample_cov(df)
    ef = EfficientFrontier(e_ret, cov)
    w = ef.max_sharpe()

    latest_prices = discrete_allocation.get_latest_prices(df)
    allocation, leftover = discrete_allocation.portfolio(w, latest_prices)
    assert allocation == {
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


def test_portfolio_allocation_errors():
    df = get_data()
    e_ret = mean_historical_return(df)
    cov = sample_cov(df)
    ef = EfficientFrontier(e_ret, cov)
    w = ef.max_sharpe()
    latest_prices = discrete_allocation.get_latest_prices(df)

    with pytest.raises(TypeError):
        discrete_allocation.portfolio(ef.weights, latest_prices)

    with pytest.raises(TypeError):
        discrete_allocation.portfolio(w, latest_prices.values.tolist())

    with pytest.raises(ValueError):
        discrete_allocation.portfolio(w, latest_prices, min_allocation=0.5)

    with pytest.raises(ValueError):
        discrete_allocation.portfolio(w, latest_prices, total_portfolio_value=0)
