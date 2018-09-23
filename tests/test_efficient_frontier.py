import warnings
import numpy as np
import pandas as pd
import pytest
from pypfopt.efficient_frontier import EfficientFrontier
from tests.utilities_for_tests import get_data, setup_efficient_frontier
from pypfopt import risk_models


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


def test_portfolio_performance():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.portfolio_performance()
    ef.max_sharpe()
    assert ef.portfolio_performance()


def test_efficient_frontier_inheritance():
    ef = setup_efficient_frontier()
    assert ef.clean_weights
    assert isinstance(ef.initial_guess, np.ndarray)
    assert isinstance(ef.constraints, list)


def test_max_sharpe_long_only():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3303554237026972, 0.21671629636481254, 1.4288438866031374),
    )


def test_max_sharpe_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.40723757138191374, 0.24823079451957306, 1.5524922427959371),
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.max_sharpe()
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_max_sharpe_L2_reg():
    ef = setup_efficient_frontier()
    ef.gamma = 1
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3062919882686126, 0.20291367026287507, 1.4087639167552641),
    )


def test_max_sharpe_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.max_sharpe()
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.gamma = a
        ef.max_sharpe()
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_max_sharpe_L2_reg_limit_case():
    ef = setup_efficient_frontier()
    ef.gamma = 1e10
    ef.max_sharpe()
    equal_weights = np.array([1 / ef.n_assets] * ef.n_assets)
    np.testing.assert_array_almost_equal(ef.weights, equal_weights)


def test_max_sharpe_L2_reg_reduces_sharpe():
    # L2 reg should reduce the number of small weights at the cost of Sharpe
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.max_sharpe()
    sharpe_no_reg = ef_no_reg.portfolio_performance()[2]
    ef = setup_efficient_frontier()
    ef.gamma = 1
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
    ef.gamma = 1
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3236047844566581, 0.20241509723550233, 1.4969817524033966),
    )
    new_number = sum(ef.weights > 0.01)
    assert new_number >= initial_number


def test_max_sharpe_risk_free_rate():
    ef = setup_efficient_frontier()
    ef.max_sharpe()
    _, _, initial_sharpe = ef.portfolio_performance()
    ef.max_sharpe(risk_free_rate=0.10)
    _, _, new_sharpe = ef.portfolio_performance(risk_free_rate=0.10)
    assert new_sharpe <= initial_sharpe

    ef.max_sharpe(risk_free_rate=0)
    _, _, new_sharpe = ef.portfolio_performance(risk_free_rate=0)
    assert new_sharpe >= initial_sharpe


def test_max_sharpe_input_errors():
    with pytest.raises(ValueError):
        ef = EfficientFrontier(
            *setup_efficient_frontier(data_only=True), gamma="2"
        )

    with warnings.catch_warnings(record=True) as w:
        ef = EfficientFrontier(
            *setup_efficient_frontier(data_only=True), gamma=-1)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert (
            str(w[0].message)
            == "in most cases, gamma should be positive"
        )

    with pytest.raises(ValueError):
        ef.max_sharpe(risk_free_rate="0.2")


def test_min_volatility():
    ef = setup_efficient_frontier()
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.17915572327783236, 0.1591542642140098, 0.9971057459792518),
    )


def test_min_volatility_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.1719799158957379, 0.15559547854162945, 0.9734986722620801),
    )

    # Shorting should reduce volatility
    volatility = ef.portfolio_performance()[1]
    ef_long_only = setup_efficient_frontier()
    ef_long_only.min_volatility()
    long_only_volatility = ef_long_only.portfolio_performance()[1]
    assert volatility < long_only_volatility


def test_min_volatility_L2_reg():
    ef = setup_efficient_frontier()
    ef.gamma = 1
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2313619320427517, 0.195525914008473, 1.0799317402364261),
    )


def test_min_volatility_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.gamma = a
        ef.min_volatility()
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_efficient_risk():
    ef = setup_efficient_frontier()
    w = ef.efficient_risk(0.19)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.2857747021121558, 0.19, 1.396492876), atol=1e-6
    )


def test_efficient_risk_many_values():
    ef = setup_efficient_frontier()
    for target_risk in np.arange(0.16, 0.21, 0.01):
        ef.efficient_risk(target_risk)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        volatility = ef.portfolio_performance()[1]
        assert abs(target_risk - volatility) < 0.05


def test_efficient_risk_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.efficient_risk(0.19)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.30468522897560224, 0.19, 1.4947624032507056),
        atol=1e6,
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_efficient_risk_L2_reg():
    ef = setup_efficient_frontier()
    ef.gamma = 1
    w = ef.efficient_risk(0.19)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2843888327412046, 0.19, 1.3895318474675356),
        atol=1e-6,
    )


def test_efficient_risk_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.efficient_risk(0.19)
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.gamma = a
        ef.efficient_risk(0.19)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_efficient_risk_market_neutral():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    w = ef.efficient_risk(0.19, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_almost_equal(
        ef.portfolio_performance(),
        (0.2309497469661495, 0.19000021138101422, 1.1021245569881066)
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]
    assert long_only_sharpe > sharpe


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
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.17388778912324757, 1.3204920206007777), atol=1e-6
    )


def test_efficient_return_many_values():
    ef = setup_efficient_frontier()
    for target_return in np.arange(0.19, 0.30, 0.01):
        ef.efficient_return(target_return)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        mean_return = ef.portfolio_performance()[0]
        assert abs(target_return - mean_return) < 0.05


def test_efficient_return_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.168264744226909, 1.3640929002973508)
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_efficient_return_L2_reg():
    ef = setup_efficient_frontier()
    ef.gamma = 1
    w = ef.efficient_return(0.25)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.20032972838376054, 1.1470454626523598)
    )


def test_efficient_return_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.efficient_return(0.25)
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.gamma = a
        ef.efficient_return(0.25)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_efficient_return_market_neutral():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    w = ef.efficient_return(0.25, market_neutral=True)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_almost_equal(
        ef.portfolio_performance(),
        (0.25, 0.20567621957041887, 1.1087335497769277)
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]
    assert long_only_sharpe > sharpe


def test_efficient_return_market_neutral_warning():
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
    # f
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2972237362989219, 0.064432672830601, 4.297294313174586)
    )


def test_min_volatilty_semicovariance_L2_reg():
    # f
    df = get_data()

    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.20661406122127524, 0.055515981410304394, 3.3567606718215663)
    )


def test_efficient_return_semicovariance():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.efficient_return(0.12)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.12000000000871075, 0.06948386214063361, 1.4319423610177537)
    )
