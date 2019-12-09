import numpy as np
import pandas as pd
import pytest

from pypfopt.black_litterman import BlackLittermanModel, black_litterman_cov
from pypfopt import risk_models, expected_returns
from tests.utilities_for_tests import get_data


def test_input_errors():
    df = get_data()
    S = risk_models.sample_cov(df)
    # Insufficient args
    with pytest.raises(TypeError):
        BlackLittermanModel(S)


def test_dataframe_input():
    pass


def test_parse_views():
    df = get_data()
    S = risk_models.sample_cov(df)
    viewdict = {"AAPL": 0.20, "GOOG": -0.30, "XOM": 0.40, "fail": 0.1}

    with pytest.raises(ValueError):
        bl = BlackLittermanModel(S, absolute_views=viewdict)
    del viewdict["fail"]
    bl = BlackLittermanModel(S, absolute_views=viewdict)

    # Check the picking matrix is correct
    test_P = np.copy(bl.P)
    test_P[0, 1] -= 1
    test_P[1, 0] -= 1
    test_P[2, 13] -= 1
    np.testing.assert_array_equal(test_P, np.zeros((len(bl.Q), bl.n_assets)))

    # Check views vector is correct
    np.testing.assert_array_equal(
        bl.Q, np.array(list(viewdict.values())).reshape(-1, 1)
    )


def test_check_attribute_dimensions():
    # change an attribute (or pass wrong) and make sure it raises error
    pass


def test_default_omega():
    pass


def test_bl_returns_no_prior():
    df = get_data()
    S = risk_models.sample_cov(df)
    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
    bl = BlackLittermanModel(S, absolute_views=viewdict)
    rets = bl.bl_returns()

    # Make sure it gives the same answer as explicit inverse
    test_rets = np.linalg.inv(
        np.linalg.inv(bl.tau * bl.cov_matrix) + bl.P.T @ np.linalg.inv(bl.omega) @ bl.P
    ) @ (bl.P.T @ np.linalg.inv(bl.omega) @ bl.Q)
    np.testing.assert_array_almost_equal(rets.values.reshape(-1, 1), test_rets)


def test_bl_returns_all_views():
    # TODO: fix this test.
    df = get_data()
    prior = expected_returns.ema_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    Q = pd.Series(0.1, index=S.columns)

    # TODO: error is somewhere here. For some reason, P and Q are changing shape
    # during the init. Very bizarre, but shouldn't be hard to fix.
    bl = BlackLittermanModel(S, pi=prior, Q=Q)
    posterior_rets = bl.bl_returns()
    assert isinstance(posterior_rets, pd.Series)
    assert list(posterior_rets.index) == list(df.columns)
    assert posterior_rets.notnull().all()
    assert posterior_rets.dtype == "float64"


def test_black_litterman_cov_default():
    df = get_data()
    Sigma = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    S = black_litterman_cov(Sigma)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
