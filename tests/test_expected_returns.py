import warnings
import pandas as pd
import numpy as np
from pypfopt import expected_returns
from tests.utilities_for_tests import get_data


def test_returns_dataframe():
    df = get_data()
    returns_df = expected_returns.returns_from_prices(df)
    assert isinstance(returns_df, pd.DataFrame)
    assert returns_df.shape[1] == 20
    assert len(returns_df) == 7125
    assert returns_df.index.is_all_dates
    assert not ((returns_df > 1) & returns_df.notnull()).any().any()


def test_prices_from_returns():
    df = get_data()
    returns_df = df.pct_change()  # keep NaN row

    # convert pseudo-price to price
    pseudo_prices = expected_returns.prices_from_returns(returns_df)
    initial_prices = df.bfill().iloc[0]
    test_prices = pseudo_prices * initial_prices

    # check equality, robust to floating point issues
    assert ((test_prices[1:] - df[1:]).fillna(0) < 1e-10).all().all()


def test_mean_historical_returns_dummy():
    data = pd.DataFrame(
        [
            [4.0, 2.0, 0.6, -12],
            [4.2, 2.1, 0.59, -13.2],
            [3.9, 2.0, 0.58, -11.3],
            [4.3, 2.1, 0.62, -11.7],
            [4.1, 2.2, 0.63, -10.1],
        ]
    )
    mean = expected_returns.mean_historical_return(data, frequency=1)
    test_answer = pd.Series([0.00865598, 0.025, 0.01286968, -0.03632333])
    pd.testing.assert_series_equal(mean, test_answer)


def test_mean_historical_returns():
    df = get_data()
    mean = expected_returns.mean_historical_return(df)
    assert isinstance(mean, pd.Series)
    assert list(mean.index) == list(df.columns)
    assert mean.notnull().all()
    assert mean.dtype == "float64"
    correct_mean = np.array(
        [
            0.26770284,
            0.3637864,
            0.31709032,
            0.22616723,
            0.49982007,
            0.16888704,
            0.22754479,
            0.14783539,
            0.19001915,
            0.08150653,
            0.12826351,
            0.25797816,
            0.07580128,
            0.16087243,
            0.20510267,
            0.3511536,
            0.38808003,
            0.24635612,
            0.21798433,
            0.28474973,
        ]
    )
    np.testing.assert_array_almost_equal(mean.values, correct_mean)


def test_mean_historical_returns_type_warning():
    df = get_data()
    mean = expected_returns.mean_historical_return(df)

    with warnings.catch_warnings(record=True) as w:
        mean_from_array = expected_returns.mean_historical_return(np.array(df))
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert str(w[0].message) == "prices are not in a dataframe"

    np.testing.assert_array_almost_equal(mean.values, mean_from_array.values, decimal=6)


def test_mean_historical_returns_frequency():
    df = get_data()
    mean = expected_returns.mean_historical_return(df)
    mean2 = expected_returns.mean_historical_return(df, frequency=52)
    np.testing.assert_array_almost_equal(mean / 252, mean2 / 52)


def test_ema_historical_return():
    df = get_data()
    mean = expected_returns.ema_historical_return(df)
    assert isinstance(mean, pd.Series)
    assert list(mean.index) == list(df.columns)
    assert mean.notnull().all()
    assert mean.dtype == "float64"


def test_ema_historical_return_frequency():
    df = get_data()
    mean = expected_returns.ema_historical_return(df)
    mean2 = expected_returns.ema_historical_return(df, frequency=52)
    np.testing.assert_array_almost_equal(mean / 252, mean2 / 52)


def test_ema_historical_return_limit():
    df = get_data()
    sma = expected_returns.mean_historical_return(df)
    ema = expected_returns.ema_historical_return(df, span=1e10)
    np.testing.assert_array_almost_equal(ema.values, sma.values)
