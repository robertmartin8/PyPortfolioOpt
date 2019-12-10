import numpy as np
import pandas as pd
import pytest

from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import risk_models, expected_returns
from tests.utilities_for_tests import get_data


def test_input_errors():
    df = get_data()
    S = risk_models.sample_cov(df)
    views = pd.Series(0.1, index=S.columns)

    # Insufficient args
    with pytest.raises(TypeError):
        BlackLittermanModel(S)

    assert BlackLittermanModel(S, Q=views)

    with pytest.raises(TypeError):
        BlackLittermanModel(S)

    with pytest.raises(ValueError):
        BlackLittermanModel(S, Q=views, tau=-0.1)

    # P and Q don't match dimensions
    P = np.eye(len(S))[:, :-1]
    with pytest.raises(ValueError):
        # This doesn't raise the error from the expected place!
        # Because default_omega uses matrix mult on P
        BlackLittermanModel(S, Q=views, P=P)
    with pytest.raises(ValueError):
        BlackLittermanModel(S, Q=views, P=P, omega=np.eye(len(views)))

    # pi and S don't match dimensions
    with pytest.raises(ValueError):
        BlackLittermanModel(S, Q=views, pi=df.pct_change().mean()[:-1])


def test_parse_views():
    df = get_data()
    S = risk_models.sample_cov(df)

    viewlist = ["AAPL", 0.20, "GOOG", -0.30, "XOM", 0.40]  # incorrect type
    viewdict = {"AAPL": 0.20, "GOOG": -0.30, "XOM": 0.40, "fail": 0.1}

    with pytest.raises(TypeError):
        bl = BlackLittermanModel(S, absolute_views=viewlist)
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


def test_dataframe_input():
    df = get_data()
    S = risk_models.sample_cov(df)

    view_df = pd.DataFrame(pd.Series(0.1, index=S.columns))
    bl = BlackLittermanModel(S, Q=view_df)
    np.testing.assert_array_equal(bl.P, np.eye(len(view_df)))

    # views on the first 10 assets
    view_df = pd.DataFrame(pd.Series(0.1, index=S.columns)[:10])
    picking = np.eye(len(S))[:10, :]
    assert BlackLittermanModel(S, Q=view_df, P=picking)

    prior_df = df.pct_change().mean()
    assert BlackLittermanModel(S, pi=prior_df, Q=view_df, P=picking)
    omega_df = S.iloc[:10, :10]
    assert BlackLittermanModel(S, pi=prior_df, Q=view_df, P=picking, omega=omega_df)


def test_default_omega():
    df = get_data()
    S = risk_models.sample_cov(df)
    views = pd.Series(0.1, index=S.columns)
    bl = BlackLittermanModel(S, Q=views)

    # Check square and diagonal
    assert bl.omega.shape == (len(S), len(S))
    np.testing.assert_array_equal(bl.omega, np.diag(np.diagonal(bl.omega)))

    # In this case, we should have omega = tau * diag(S)
    np.testing.assert_array_almost_equal(np.diagonal(bl.omega), bl.tau * np.diagonal(S))


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
    df = get_data()
    prior = expected_returns.ema_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    views = pd.Series(0.1, index=S.columns)

    bl = BlackLittermanModel(S, pi=prior, Q=views)
    posterior_rets = bl.bl_returns()
    assert isinstance(posterior_rets, pd.Series)
    assert list(posterior_rets.index) == list(df.columns)
    assert posterior_rets.notnull().all()
    assert posterior_rets.dtype == "float64"
    np.testing.assert_array_almost_equal(
        posterior_rets,
        np.array(
            [
                0.11774473,
                0.1709139,
                0.12180833,
                0.21202423,
                0.28120945,
                -0.2787358,
                0.17274774,
                0.12714698,
                0.25492005,
                0.11229777,
                0.07182723,
                -0.01521839,
                -0.21235465,
                0.06399515,
                -0.11738365,
                0.28865661,
                0.23828607,
                0.12038049,
                0.2331218,
                0.10485376,
            ]
        ),
    )


def test_black_litterman_cov_default():
    df = get_data()
    cov_matrix = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
    S = bl.bl_cov()
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()


def test_black_litterman_market_prior():
    # proper test of BL, making sure weights deviate expectedly.
    pass
