import warnings
import numpy as np
import pandas as pd
import pytest

from pypfopt.efficient_frontier import EfficientFrontier
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


def test_portfolio_performance():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.portfolio_performance()
    ef.max_sharpe()
    assert ef.portfolio_performance()


def test_efficient_frontier_init_errors():
    df = get_data()
    mean_returns = df.pct_change().dropna(how="all").mean()
    with pytest.raises(TypeError):
        EfficientFrontier("test", "string")

    with pytest.raises(TypeError):
        EfficientFrontier(mean_returns, mean_returns)


def test_max_sharpe_long_only():
    ef = setup_efficient_frontier()
    w = ef.max_sharpe()
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
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
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
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
    w = ef.max_sharpe(gamma=1)
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
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
        ef.max_sharpe(gamma=a)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_max_sharpe_L2_reg_limit_case():
    ef = setup_efficient_frontier()
    ef.max_sharpe(gamma=1e10)
    equal_weights = np.array([1 / ef.n_assets] * ef.n_assets)
    np.testing.assert_array_almost_equal(ef.weights, equal_weights)


def test_max_sharpe_L2_reg_reduces_sharpe():
    # L2 reg should reduce the number of small weights at the cost of Sharpe
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.max_sharpe()
    sharpe_no_reg = ef_no_reg.portfolio_performance()[2]
    ef = setup_efficient_frontier()
    ef.max_sharpe(gamma=1)
    sharpe = ef.portfolio_performance()[2]

    assert sharpe < sharpe_no_reg


def test_max_sharpe_L2_reg_with_shorts():
    ef_no_reg = setup_efficient_frontier()
    ef_no_reg.max_sharpe()
    initial_number = sum(ef_no_reg.weights > 0.01)

    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.max_sharpe(gamma=1)
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
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
    _, _, new_sharpe = ef.portfolio_performance()
    assert new_sharpe <= initial_sharpe

    ef.max_sharpe(risk_free_rate=0)
    _, _, new_sharpe = ef.portfolio_performance()
    assert new_sharpe >= initial_sharpe


def test_max_sharpe_input_errors():
    ef = setup_efficient_frontier()
    with pytest.raises(ValueError):
        ef.max_sharpe(gamma="2")

    with warnings.catch_warnings(record=True) as w:
        ef.max_sharpe(gamma=-1)
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
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.1793245141665063, 0.15915107045094778, 0.9981835740658117),
    )


def test_min_volatility_short():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(None, None)
    )
    w = ef.min_volatility()
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.17225673749865328, 0.15559209747801794, 0.9752992044136976),
    )

    # Shorting should reduce volatility
    volatility = ef.portfolio_performance()[1]
    ef_long_only = setup_efficient_frontier()
    ef_long_only.min_volatility()
    long_only_volatility = ef_long_only.portfolio_performance()[1]
    assert volatility < long_only_volatility


def test_min_volatility_L2_reg():
    ef = setup_efficient_frontier()
    w = ef.min_volatility(gamma=1)
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(),
        (0.2211888419683154, 0.18050174016287326, 1.1133499289183508),
    )


def test_min_volatility_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.min_volatility()
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.min_volatility(gamma=a)
        np.testing.assert_almost_equal(ef.weights.sum(), 1)
        new_number = sum(ef.weights > 0.01)
        # Higher gamma should reduce the number of small weights
        assert new_number >= initial_number
        initial_number = new_number


def test_efficient_risk():
    ef = setup_efficient_frontier()
    w = ef.efficient_risk(0.19)
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.285775, 0.19, 1.396493), atol=1e-6
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
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
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
    w = ef.efficient_risk(0.19, gamma=1)
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
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
        ef.efficient_risk(0.19, gamma=a)
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
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
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
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.173885, 1.320507), atol=1e-6
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
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)
    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.16826260520748268, 1.3641098601259731)
    )
    sharpe = ef.portfolio_performance()[2]

    ef_long_only = setup_efficient_frontier()
    ef_long_only.efficient_return(0.25)
    long_only_sharpe = ef_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_efficient_return_L2_reg():
    ef = setup_efficient_frontier()
    w = ef.efficient_return(0.25, gamma=1)
    assert isinstance(w, dict)
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 1)

    np.testing.assert_allclose(
        ef.portfolio_performance(), (0.25, 0.18813935436629708, 1.221273523695721)
    )


def test_efficient_return_L2_reg_many_values():
    ef = setup_efficient_frontier()
    ef.efficient_return(0.25)
    # Count the number of weights more 1%
    initial_number = sum(ef.weights > 0.01)
    for a in np.arange(0.5, 5, 0.5):
        ef.efficient_return(0.25, gamma=a)
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
    assert list(w.keys()) == ef.tickers
    assert list(w.keys()) == list(ef.expected_returns.index)
    np.testing.assert_almost_equal(ef.weights.sum(), 0)
    assert (ef.weights < 1).all() and (ef.weights > -1).all()
    np.testing.assert_almost_equal(
        ef.portfolio_performance(),
        (0.24999999999755498, 0.20567338787141307, 1.1087493060316183),
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


def test_custom_upper_bound():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0, 0.10)
    )
    ef.max_sharpe()
    ef.portfolio_performance()
    assert ef.weights.max() <= 0.1
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_lower_bound():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0.02, 1)
    )
    ef.max_sharpe()
    assert ef.weights.min() >= 0.02
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_custom_bounds():
    ef = EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0.03, 0.13)
    )
    ef.max_sharpe()
    assert ef.weights.min() >= 0.03
    assert ef.weights.max() <= 0.13
    np.testing.assert_almost_equal(ef.weights.sum(), 1)


def test_bounds_errors():
    with pytest.raises(ValueError):
        EfficientFrontier(
            *setup_efficient_frontier(data_only=True), weight_bounds=(0.06, 1)
        )
    assert EfficientFrontier(
        *setup_efficient_frontier(data_only=True), weight_bounds=(0, 1)
    )

    with pytest.raises(ValueError):
        EfficientFrontier(
            *setup_efficient_frontier(data_only=True), weight_bounds=(0.06, 1, 3)
        )
