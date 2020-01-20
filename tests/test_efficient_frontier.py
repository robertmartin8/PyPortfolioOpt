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
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3303554227420522, 0.21671629569400466, 1.4320816150358278),
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
        (0.4072375737868628, 0.24823079606119094, 1.5599900573634125)
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.max_sharpe()
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_weight_bounds_minus_one_to_one():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(-1, 1)
    )
    assert ef.max_sharpe()
    assert ef.min_volatility()
    assert ef.efficient_return(0.05)
    assert ef.efficient_risk(0.20)


def test_max_sharpe_L2_reg():
    ef = setup_efficient_frontier()
    ef.gamma = 1
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3062919877378972, 0.20291366982652356, 1.4109053765705188),
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
        (0.32360478341793864, 0.20241509658051923, 1.499911758296975),
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


def test_max_unconstrained_utility():
    ef = setup_efficient_frontier()
    w = ef.max_unconstrained_utility(2)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (1.3507326549906276, 0.8218067458322021, 1.6192768698230409)
    )

    ret1, var1, _ = ef.portfolio_performance()
    # increasing risk_aversion should lower both vol and return
    ef.max_unconstrained_utility(10)
    ret2, var2, _ = ef.portfolio_performance()
    assert ret2 < ret1 and var2 < var1


def test_max_unconstrained_utility_error():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.max_unconstrained_utility(0)
    with pytest.raises(ValueError):
        ef.max_unconstrained_utility(-1)


def test_min_volatility():
    ef = setup_efficient_frontier()
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.1791557243114251, 0.15915426422116669, 1.0000091740567905),
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
        (0.1719799152621441, 0.1555954785460613, 0.9767630568850568),
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
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.23136193240984504, 0.1955259140191799, 1.0809919159314694),
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
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.2857747021087114, 0.19, 1.3988133092245933), atol=1e-6
    )


def test_efficient_risk_error():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    min_possible_vol = ef.portfolio_performance()[1]
    with pytest.raises(ValueError):
        # This volatility is too low
        ef.efficient_risk(min_possible_vol - 0.01)


def test_efficient_risk_many_values():
    ef = setup_efficient_frontier()
    for target_risk in np.arange(0.16, 0.21, 0.30):
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
        (0.30468522897430295, 0.19, 1.4983424153337392),
        atol=1e-6,
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
    assert all([i >= 0 for i in w.values()])

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.28437776398043807, 0.19, 1.3914587310224322),
        atol=1e-6,
    )


def test_efficient_risk_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.efficient_risk(0.19)
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.gamma = a
        ef.efficient_risk(0.2)
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
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2309497469633197, 0.19, 1.1102605909328953),
        atol=1e-6
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
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.1738877891235972, 1.3226920714748545), atol=1e-6
    )


def test_efficient_return_error():
    ef = setup_efficient_frontier()
    max_ret = ef.expected_returns.max()
    with pytest.raises(ValueError):
        # This volatility is too low
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
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.1682647442258144, 1.3668935881968987)
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
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.20032972845476912, 1.1481071819692497)
    )


def test_efficient_return_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.efficient_return(0.25)
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.gamma = a
        ef.efficient_return(0.20)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        assert all([i >= 0 for i in ef.weights])
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
        (0.25, 0.20567621957479246, 1.1182624830289896)
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
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2972237371625498, 0.06443267303123411, 4.302533545801584)
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
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3564654865246848, 0.07202031837368413, 4.671813373260894)
    )


def test_min_volatilty_semicovariance_L2_reg():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.gamma = 1
    ef.cov_matrix = risk_models.semicovariance(df, benchmark=0)
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.23803779483710888, 0.0962263031034166, 2.265885603053655)
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
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.11999999997948813, 0.06948386215256849, 1.4391830977949114)
    )


def test_max_sharpe_exp_cov():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.3678835305574766, 0.17534146043561463, 1.9840346355802103)
    )


def test_min_volatility_exp_cov_L2_reg():
    df = get_data()
    ef = setup_efficient_frontier()
    ef.gamma = 1
    ef.cov_matrix = risk_models.exp_cov(df)
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(ef.tickers)
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    assert all([i >= 0 for i in w.values()])
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.24340406492258035, 0.17835396894670616, 1.2525881326999546)
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
    assert set(w.keys()) == set(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.39089308906686077, 0.19, 1.9520670176494717),
        atol=1e-6
    )
