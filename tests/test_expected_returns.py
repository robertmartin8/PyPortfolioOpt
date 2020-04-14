import warnings
import pandas as pd
import numpy as np
import pytest
from pypfopt import expected_returns
from tests.utilities_for_tests import get_data, get_benchmark_data


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


def test_returns_from_prices():
    df = get_data()
    returns_df = expected_returns.returns_from_prices(df)
    pd.testing.assert_series_equal(returns_df.iloc[-1], df.pct_change().iloc[-1])


def test_log_returns_from_prices():
    df = get_data()
    old_nan = df.isnull().sum(axis=1).sum()
    log_rets = expected_returns.log_returns_from_prices(df)
    new_nan = log_rets.isnull().sum(axis=1).sum()
    assert new_nan == old_nan
    np.testing.assert_almost_equal(log_rets.iloc[-1, -1], 0.0001682740081102576)


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
    mean = expected_returns.mean_historical_return(data, compounding=True, frequency=1)
    pd.testing.assert_series_equal(mean, test_answer)


def test_mean_historical_returns_compounding():
    df = get_data()
    mean = expected_returns.mean_historical_return(df)
    mean2 = expected_returns.mean_historical_return(df, compounding=True)
    assert (mean2 >= mean).all()


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

    mean3 = expected_returns.ema_historical_return(df, compounding=True)
    assert (abs(mean3) > mean).all()


def test_ema_historical_return_limit():
    df = get_data()
    sma = expected_returns.mean_historical_return(df)
    ema = expected_returns.ema_historical_return(df, span=1e10)
    np.testing.assert_array_almost_equal(ema.values, sma.values)


def test_james_stein():
    df = get_data()
    js = expected_returns.james_stein_shrinkage(df)
    correct_mean = np.array(
        [
            0.25870218,
            0.32318595,
            0.29184719,
            0.23082673,
            0.41448111,
            0.19238474,
            0.23175124,
            0.17825652,
            0.20656697,
            0.13374178,
            0.16512141,
            0.25217574,
            0.12991287,
            0.18700597,
            0.21668984,
            0.3147078,
            0.33948993,
            0.24437593,
            0.225335,
            0.27014272,
        ]
    )
    np.testing.assert_array_almost_equal(js.values, correct_mean)

    # Test shrinkage
    y = expected_returns.returns_from_prices(df).mean(axis=0) * 252
    nu = y.mean()
    assert (((js <= nu) & (js >= y)) | ((js >= nu) & (js <= y))).all()


def test_capm_no_benchmark():
    df = get_data()
    mu = expected_returns.capm_return(df)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(df.columns)
    assert mu.notnull().all()
    assert mu.dtype == "float64"
    correct_mu = np.array(
        [
            0.21803135,
            0.27902605,
            0.14475533,
            0.14668971,
            0.40944875,
            0.22361704,
            0.39057166,
            0.164807,
            0.31280876,
            0.17018046,
            0.15044284,
            0.34609161,
            0.3233097,
            0.1479624,
            0.26403991,
            0.31124465,
            0.27312086,
            0.16703193,
            0.30396023,
            0.25182927,
        ]
    )
    np.testing.assert_array_almost_equal(mu.values, correct_mu)


def test_capm_with_benchmark():
    df = get_data()
    mkt_df = get_benchmark_data()
    mu = expected_returns.capm_return(df, market_prices=mkt_df, compounding=True)

    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(df.columns)
    assert mu.notnull().all()
    assert mu.dtype == "float64"
    correct_mu = np.array(
        [
            0.10903299,
            0.11891232,
            0.0659977,
            0.07369941,
            0.15948144,
            0.12308759,
            0.15907944,
            0.08680978,
            0.15778843,
            0.0903294,
            0.09043133,
            0.14716681,
            0.12510181,
            0.0927869,
            0.10990104,
            0.12317033,
            0.13596521,
            0.09344662,
            0.15457909,
            0.11430041,
        ]
    )
    np.testing.assert_array_almost_equal(mu.values, correct_mu)


def test_risk_matrix_and_returns_data():
    # Test the switcher method for simple calls
    df = get_data()

    for method in {
        "mean_historical_return",
        "ema_historical_return",
        "james_stein_shrinkage",
        "capm_return",
    }:
        mu = expected_returns.return_model(df, method=method)

        assert isinstance(mu, pd.Series)
        assert list(mu.index) == list(df.columns)
        assert mu.notnull().all()
        assert mu.dtype == "float64"

        mu2 = expected_returns.return_model(
            expected_returns.returns_from_prices(df), method=method, returns_data=True
        )
        pd.testing.assert_series_equal(mu, mu2)


def test_return_model_additional_kwargs():
    df = get_data()
    mkt_prices = get_benchmark_data()

    mu1 = expected_returns.return_model(
        df, method="capm_return", market_prices=mkt_prices, risk_free_rate=0.03
    )
    mu2 = expected_returns.capm_return(
        df, market_prices=mkt_prices, risk_free_rate=0.03
    )
    pd.testing.assert_series_equal(mu1, mu2)


def test_return_model_not_implemented():
    df = get_data()
    with pytest.raises(NotImplementedError):
        expected_returns.return_model(df, method="fancy_new!")
