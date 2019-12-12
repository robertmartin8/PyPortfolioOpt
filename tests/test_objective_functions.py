import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt import objective_functions
from pypfopt.risk_models import sample_cov
from tests.utilities_for_tests import get_data


def test_negative_mean_return_dummy():
    w = np.array([0.3, 0.1, 0.2, 0.25, 0.15])
    e_rets = pd.Series([0.19, 0.08, 0.09, 0.23, 0.17])

    negative_mu = objective_functions.negative_mean_return(w, e_rets)
    assert isinstance(negative_mu, float)
    assert negative_mu < 0
    np.testing.assert_almost_equal(negative_mu, -w.dot(e_rets))
    np.testing.assert_almost_equal(negative_mu, -(w * e_rets).sum())


def test_negative_mean_return_real():
    df = get_data()
    e_rets = mean_historical_return(df)
    w = np.array([1 / len(e_rets)] * len(e_rets))
    negative_mu = objective_functions.negative_mean_return(w, e_rets)
    assert isinstance(negative_mu, float)
    assert negative_mu < 0
    assert negative_mu == -w.dot(e_rets)
    assert negative_mu == -(w * e_rets).sum()
    np.testing.assert_almost_equal(-e_rets.sum() / len(e_rets), negative_mu)


def test_negative_sharpe():
    df = get_data()
    e_rets = mean_historical_return(df)
    S = sample_cov(df)
    w = np.array([1 / len(e_rets)] * len(e_rets))

    sharpe = objective_functions.negative_sharpe(w, e_rets, S)
    assert isinstance(sharpe, float)
    assert sharpe < 0

    sigma = np.sqrt(np.dot(w, np.dot(S, w.T)))
    negative_mu = objective_functions.negative_mean_return(w, e_rets)
    np.testing.assert_almost_equal(sharpe * sigma - 0.02, negative_mu)

    # Risk free rate increasing should lead to negative Sharpe increasing.
    assert sharpe < objective_functions.negative_sharpe(
        w, e_rets, S, risk_free_rate=0.1
    )


def test_negative_quadratic_utility():
    df = get_data()
    e_rets = mean_historical_return(df)
    S = sample_cov(df)
    w = np.array([1 / len(e_rets)] * len(e_rets))
    utility = objective_functions.negative_quadratic_utility(
        w, e_rets, S, risk_aversion=3
    )
    assert isinstance(utility, float)
    assert utility < 0

    mu = -objective_functions.negative_mean_return(w, e_rets)
    variance = np.dot(w, np.dot(S, w.T))
    np.testing.assert_almost_equal(-utility + 3 / 2 * variance, mu)


def test_volatility_dummy():
    w = np.array([0.4, 0.4, 0.2])
    data = np.diag([0.5, 0.8, 0.9])
    test_var = objective_functions.volatility(w, data)
    np.testing.assert_almost_equal(test_var, 0.244)


def test_volatility():
    df = get_data()
    S = sample_cov(df)
    w = np.array([1 / df.shape[1]] * df.shape[1])
    var = objective_functions.volatility(w, S)
    np.testing.assert_almost_equal(var, 0.04498224489292057)


def test_cvar():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    w = np.array([1 / df.shape[1]] * df.shape[1])
    cvar0 = objective_functions.negative_cvar(w, returns, s=5000, random_state=0)
    assert cvar0 > 0
    cvar1 = objective_functions.negative_cvar(
        w, returns, s=5000, beta=0.98, random_state=0
    )
    assert cvar1 > 0

    # Nondeterministic
    cvar2 = objective_functions.negative_cvar(w, returns, s=5000, random_state=1)
    assert not cvar0 == cvar2
