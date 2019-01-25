import warnings
import numpy as np
import pandas as pd
import pytest
from pypfopt.cla import CLA
from tests.utilities_for_tests import get_data, setup_cla
from pypfopt import risk_models


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


def test_max_sharpe_long_only():
    cla = setup_cla()
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)

    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.3253436657555845, 0.2133353004830236, 1.4281588303044812),
    )


def test_min_volatility():
    cla = setup_cla()
    w = cla.min_volatility()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.1793123248125915, 0.15915084514118688, 0.9981061956268638),
    )


def test_max_sharpe_semicovariance():
    # f
    df = get_data()
    cla = setup_cla()
    cla.covar = risk_models.semicovariance(df, benchmark=0)
    w = cla.max_sharpe()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(cla.tickers)
    np.testing.assert_almost_equal(cla.weights.sum(), 1)
    np.testing.assert_allclose(
        cla.portfolio_performance(),
        (0.3253436657555845, 0.2133353004830236, 1.4281588303044812),
    )
