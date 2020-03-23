import numpy as np
import pytest
from tests.utilities_for_tests import get_data, setup_cla
from pypfopt import risk_models
from pypfopt.cla import CLA


def test_portfolio_performance():
    cla = setup_cla()
    with pytest.raises(ValueError):
        cla.portfolio_performance()
    cla.max_sharpe()
    assert cla.portfolio_performance()


def test_cla_inheritance():
    cla = setup_cla()
    assert cla.clean_weights
    assert cla.set_weights


def test_cla_max_sharpe_long_only():
    cla = setup_cla()
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)

    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.3253436663900292, 0.21333530089904357, 1.4312852355106793),
    )


def test_cla_max_sharpe_short():
    cla = CLA(*setup_cla(data_only=True), weight_bounds=(-1, 1))
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.3799273115521356, 0.23115368271125736, 1.5570909679242886),
    )
    sharpe = cla.portfolio_performance()[2]

    cla_long_only = setup_cla()
    cla_long_only.max_sharpe()
    long_only_sharpe = cla_long_only.portfolio_performance()[2]

    assert sharpe > long_only_sharpe


def test_cla_custom_bounds():
    bounds = [(0.01, 0.13), (0.02, 0.11)] * 10
    cla = CLA(*setup_cla(data_only=True), weight_bounds=bounds)
    df = get_data()
    cla.cov_matrix = risk_models.exp_cov(df).values
    w = cla.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    assert (0.01 <= cla.weights[::2]).all() and (cla.weights[::2] <= 0.13).all()
    assert (0.02 <= cla.weights[1::2]).all() and (cla.weights[1::2] <= 0.11).all()


def test_cla_min_volatility():
    cla = setup_cla()
    w = cla.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.1793123248125915, 0.15915084514118688, 1.00101463282373),
    )


def test_cla_error():
    cla = setup_cla()
    w = cla.min_volatility()
    with pytest.raises(NotImplementedError):
        cla.set_weights(w)


def test_cla_two_assets():
    mu = np.array([[0.02569294], [0.16203987]])
    cov = np.array([[0.0012765, -0.00212724], [-0.00212724, 0.01616983]])
    assert CLA(mu, cov)


def test_cla_max_sharpe_semicovariance():
    df = get_data()
    cla = setup_cla()
    cla.cov_matrix = risk_models.semicovariance(df, benchmark=0).values
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.2936179968144084, 0.06362345488289835, 4.300583759841616),
    )


def test_cla_max_sharpe_exp_cov():
    df = get_data()
    cla = setup_cla()
    cla.cov_matrix = risk_models.exp_cov(df).values
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.3619453128519127, 0.1724297730592084, 1.9830990135009723),
    )


def test_cla_min_volatility_exp_cov_short():
    cla = CLA(*setup_cla(data_only=True), weight_bounds=(-1, 1))
    df = get_data()
    cla.cov_matrix = risk_models.exp_cov(df).values
    w = cla.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.2634735528776959, 0.13259590618253303, 1.8362071642131053),
    )


def test_cla_efficient_frontier():
    cla = setup_cla()

    cla.efficient_frontier()

    mu, sigma, weights = cla.efficient_frontier()
    assert len(mu) == len(sigma) and len(sigma) == len(weights)
    # higher return = higher risk
    assert sigma[-1] < sigma[0] and mu[-1] < mu[0]
    assert weights[0].shape == (20, 1)
