import numpy as np
import pandas as pd
import pytest

from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import risk_models, expected_returns
from tests.utilities_for_tests import get_data, get_market_caps, resource


def test_input_errors():
    df = get_data()
    S = risk_models.sample_cov(df)
    views = pd.Series(0.1, index=S.columns)

    # Insufficient args
    with pytest.raises(TypeError):
        BlackLittermanModel(S)

    assert BlackLittermanModel(S, Q=views)

    with pytest.raises(ValueError):
        BlackLittermanModel(S, Q=views, tau=-0.1)

    # P and Q don't match dimensions
    P = np.eye(len(S))[:, :-1]
    with pytest.raises(AssertionError):
        # This doesn't raise the error from the expected place!
        # Because default_omega uses matrix mult on P
        BlackLittermanModel(S, Q=views, P=P)

    with pytest.raises(AssertionError):
        BlackLittermanModel(S, Q=views, P=P, omega=np.eye(len(views)))

    # pi and S don't match dimensions
    with pytest.raises(AssertionError):
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
    np.testing.assert_array_equal(bl.Q, np.array([0.20, -0.30, 0.40]).reshape(-1, 1))


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


def test_bl_equal_prior():
    df = get_data()
    S = risk_models.sample_cov(df)

    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
    bl = BlackLittermanModel(S, absolute_views=viewdict, pi="equal")
    np.testing.assert_array_almost_equal(bl.pi, np.ones((20, 1)) * 0.05)

    bl.bl_weights()
    np.testing.assert_allclose(
        bl.portfolio_performance(),
        (0.1877432247395778, 0.3246889329226965, 0.5166274785827545),
    )


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


def test_bl_relative_views():
    df = get_data()
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()

    # 1. SBUX will drop by 20%
    # 2. GOOG outperforms FB by 10%
    # 3. BAC and JPM will outperform T and GE by 15%
    views = np.array([-0.20, 0.10, 0.15]).reshape(-1, 1)
    picking = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
        ]
    )

    bl = BlackLittermanModel(S, Q=views, P=picking)
    rets = bl.bl_returns()
    assert rets["SBUX"] < 0
    assert rets["GOOG"] > rets["FB"]
    assert (rets["BAC"] > rets["T"]) and (rets["JPM"] > rets["GE"])


def test_bl_cov_default():
    df = get_data()
    cov_matrix = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
    S = bl.bl_cov()
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()


def test_market_risk_aversion():
    prices = pd.read_csv(
        resource("spy_prices.csv"), parse_dates=True, index_col=0, squeeze=True
    )
    delta = black_litterman.market_implied_risk_aversion(prices)
    assert np.round(delta, 5) == 2.68549

    # check it works for df
    prices = pd.read_csv(resource("spy_prices.csv"), parse_dates=True, index_col=0)
    delta = black_litterman.market_implied_risk_aversion(prices)
    assert np.round(delta.iloc[0], 5) == 2.68549


def test_bl_weights():
    df = get_data()
    S = risk_models.sample_cov(df)

    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
    bl = BlackLittermanModel(S, absolute_views=viewdict)

    prices = pd.read_csv(
        resource("spy_prices.csv"), parse_dates=True, index_col=0, squeeze=True
    )

    delta = black_litterman.market_implied_risk_aversion(prices)
    bl.bl_weights(delta)
    w = bl.clean_weights()
    assert abs(sum(w.values()) - 1) < 1e-5

    # check weights are allocated in same direction as views
    # (in absence of priors)
    assert all(viewdict[t] * w[t] >= 0 for t in viewdict)

    # numerical check
    test_weights = {
        "GOOG": 0.0,
        "AAPL": 1.40675,
        "FB": 0.0,
        "BABA": 0.0,
        "AMZN": 0.0,
        "GE": 0.0,
        "AMD": 0.0,
        "WMT": 0.0,
        "BAC": 0.02651,
        "GM": 0.0,
        "T": 2.81117,
        "UAA": 0.0,
        "SHLD": 0.0,
        "XOM": 0.0,
        "RRC": 0.0,
        "BBY": -1.44667,
        "MA": 0.0,
        "PFE": 0.0,
        "JPM": 0.0,
        "SBUX": -1.79776,
    }
    assert w == test_weights

    bl = BlackLittermanModel(S, absolute_views=viewdict)
    bl.optimize(delta)
    w2 = bl.clean_weights()
    assert w2 == w

    bl = BlackLittermanModel(S, absolute_views=pd.Series(viewdict))
    bl.optimize(delta)
    w2 = bl.clean_weights()
    assert w2 == w


def test_market_implied_prior():
    df = get_data()
    S = risk_models.sample_cov(df)

    prices = pd.read_csv(
        resource("spy_prices.csv"), parse_dates=True, index_col=0, squeeze=True
    )
    delta = black_litterman.market_implied_risk_aversion(prices)

    mcaps = get_market_caps()
    pi = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    assert isinstance(pi, pd.Series)
    assert list(pi.index) == list(df.columns)
    assert pi.notnull().all()
    assert pi.dtype == "float64"
    np.testing.assert_array_almost_equal(
        pi.values,
        np.array(
            [
                0.14933293,
                0.2168623,
                0.11219185,
                0.10362374,
                0.28416295,
                0.12196098,
                0.19036819,
                0.08860159,
                0.17724273,
                0.08779627,
                0.0791797,
                0.16460474,
                0.12854665,
                0.08657863,
                0.11230036,
                0.13875465,
                0.15017163,
                0.09066484,
                0.1696369,
                0.13270213,
            ]
        ),
    )

    mcaps = pd.Series(mcaps)
    pi2 = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    pd.testing.assert_series_equal(pi, pi2, check_exact=False)

    # Test alternate syntax
    bl = BlackLittermanModel(
        S,
        pi="market",
        market_caps=mcaps,
        absolute_views={"AAPL": 0.1},
        risk_aversion=delta,
    )
    pi = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=0)
    np.testing.assert_array_almost_equal(bl.pi, pi.values.reshape(-1, 1))


def test_bl_market_prior():
    df = get_data()
    S = risk_models.sample_cov(df)

    prices = pd.read_csv(
        resource("spy_prices.csv"), parse_dates=True, index_col=0, squeeze=True
    )

    delta = black_litterman.market_implied_risk_aversion(prices)

    mcaps = get_market_caps()
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    viewdict = {"GOOG": 0.40, "AAPL": -0.30, "FB": 0.30, "BABA": 0}
    bl = BlackLittermanModel(S, pi=prior, absolute_views=viewdict)
    rets = bl.bl_returns()

    # compare posterior with prior
    for v in viewdict:
        assert (prior[v] <= rets[v] <= viewdict[v]) or (
            viewdict[v] <= rets[v] <= prior[v]
        )

    with pytest.raises(ValueError):
        bl.portfolio_performance()

    bl.bl_weights(delta)
    np.testing.assert_allclose(
        bl.portfolio_performance(),
        (0.2580693114409672, 0.265445955488424, 0.8968654692926723),
    )
    # Check that bl.cov() has been called and used
    assert bl.posterior_cov is not None


def test_bl_market_automatic():
    df = get_data()
    S = risk_models.sample_cov(df)

    mcaps = get_market_caps()
    viewdict = {"GOOG": 0.40, "AAPL": -0.30, "FB": 0.30, "BABA": 0}
    bl = BlackLittermanModel(S, pi="market", absolute_views=viewdict, market_caps=mcaps)
    rets = bl.bl_returns()

    # Compare with explicit
    prior = black_litterman.market_implied_prior_returns(mcaps, 1, S, 0)
    bl2 = BlackLittermanModel(S, pi=prior, absolute_views=viewdict)
    rets2 = bl2.bl_returns()
    pd.testing.assert_series_equal(rets, rets2)


def test_bl_tau():
    df = get_data()
    S = risk_models.sample_cov(df)

    prices = pd.read_csv(
        resource("spy_prices.csv"), parse_dates=True, index_col=0, squeeze=True
    )

    delta = black_litterman.market_implied_risk_aversion(prices)

    mcaps = get_market_caps()
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    viewdict = {"GOOG": 0.40, "AAPL": -0.30, "FB": 0.30, "BABA": 0}

    # Need to change omega for this test to work
    omega = np.diag([0.01, 0.01, 0.01, 0.01])

    bl0 = BlackLittermanModel(
        S, pi=prior, absolute_views=viewdict, tau=1e-10, omega=omega
    )
    bl1 = BlackLittermanModel(
        S, pi=prior, absolute_views=viewdict, tau=0.01, omega=omega
    )
    bl2 = BlackLittermanModel(
        S, pi=prior, absolute_views=viewdict, tau=0.1, omega=omega
    )

    # For tiny tau, posterior should roughly equal prior
    np.testing.assert_allclose(bl0.bl_returns(), bl0.pi.flatten(), rtol=1e-5)

    # For bigger tau, GOOG should be given more weight
    assert bl1.bl_returns()["GOOG"] > bl0.bl_returns()["GOOG"]
    assert bl2.bl_returns()["GOOG"] > bl1.bl_returns()["GOOG"]


def test_bl_no_uncertainty():
    df = get_data()
    S = risk_models.sample_cov(df)
    omega = np.diag([0, 0, 0, 0])

    viewdict = {"GOOG": 0.40, "AAPL": -0.30, "FB": 0.30, "BABA": 0}
    bl = BlackLittermanModel(S, absolute_views=viewdict, omega=omega)
    rets = bl.bl_returns()

    # For 100% confidence, posterior return should equal view return.
    for k, v in viewdict.items():
        assert np.abs(rets[k] - v) < 1e-5

    # If only one view has 100% confidencee, only that asset will have post = prior.
    omega = np.diag([0, 0.2, 0.2, 0.2])
    bl = BlackLittermanModel(S, absolute_views=viewdict, omega=omega)
    rets = bl.bl_returns()
    assert np.abs(bl.bl_returns()["GOOG"] - viewdict["GOOG"]) < 1e-5
    assert np.abs(rets["AAPL"] - viewdict["AAPL"]) > 0.01


def test_idzorek_confidences_error():
    # if no confidences have been passed
    S = pd.DataFrame(np.diag(np.ones((5,))), index=range(5), columns=range(5))
    # Constant view of 0.3 return
    views = {k: 0.3 for k in range(5)}
    # Prior
    pi = pd.Series(0.1, index=range(5))

    with pytest.raises(ValueError):
        BlackLittermanModel(S, pi=pi, absolute_views=views, omega="idzorek")

    with pytest.raises(ValueError):
        # Wrong number of views
        BlackLittermanModel(
            S, pi=pi, absolute_views=views, omega="idzorek", view_confidences=[0.2] * 4
        )

    with pytest.raises(ValueError):
        #  Conf greater than 1
        BlackLittermanModel(
            S, pi=pi, absolute_views=views, omega="idzorek", view_confidences=[1.1] * 5
        )

    with pytest.raises(ValueError):
        #  Conf less than zero
        BlackLittermanModel(
            S, pi=pi, absolute_views=views, omega="idzorek", view_confidences=[-0.1] * 5
        )


def test_idzorek_basic():
    # Identity covariance
    S = pd.DataFrame(np.diag(np.ones((5,))), index=range(5), columns=range(5))
    # Constant view of 0.3 return
    views = {k: 0.3 for k in range(5)}
    # Prior
    pi = pd.Series(0.1, index=range(5))

    # Perfect confidence - should equal views
    bl = BlackLittermanModel(
        S, pi=pi, absolute_views=views, omega=np.diag(np.zeros(5))  # perfect confidence
    )
    pd.testing.assert_series_equal(bl.bl_returns(), pd.Series([0.3] * 5))

    # No confidence - should equal priors
    bl = BlackLittermanModel(S, pi=pi, absolute_views=views, omega=S * 1e6)
    pd.testing.assert_series_equal(bl.bl_returns(), pi)

    # Idzorek 100% confidence
    bl = BlackLittermanModel(
        S, pi=pi, absolute_views=views, omega="idzorek", view_confidences=[1] * 5
    )
    np.testing.assert_array_almost_equal(bl.omega, np.zeros((5, 5)))
    pd.testing.assert_series_equal(bl.bl_returns(), pd.Series(0.3, index=range(5)))

    # Idzorek 0% confidence
    bl = BlackLittermanModel(
        S, pi=pi, absolute_views=views, omega="idzorek", view_confidences=[0] * 5
    )
    np.testing.assert_array_almost_equal(bl.omega, np.diag([1e6] * 5))
    pd.testing.assert_series_equal(bl.bl_returns(), pi)

    # Idzorek confidence range
    for i, conf in enumerate(np.arange(0, 1.2, 0.2)):
        bl = BlackLittermanModel(
            S, pi=pi, absolute_views=views, omega="idzorek", view_confidences=[conf] * 5
        )
        # Linear spacing
        np.testing.assert_almost_equal(bl.bl_returns()[0], 0.1 + i * 0.2 / 5)


def test_idzorek_input_formats():
    # Identity covariance
    S = pd.DataFrame(np.diag(np.ones((5,))), index=range(5), columns=range(5))
    # Constant view of 0.3 return
    views = {k: 0.3 for k in range(5)}
    # Prior
    pi = pd.Series(0.1, index=range(5))

    test_result = pd.Series(0.2, index=range(5))

    bl = BlackLittermanModel(
        S, pi=pi, absolute_views=views, omega="idzorek", view_confidences=[0.5] * 5
    )
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)

    bl = BlackLittermanModel(
        S,
        pi=pi,
        absolute_views=views,
        omega="idzorek",
        view_confidences=(0.5, 0.5, 0.5, 0.5, 0.5),
    )
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)

    bl = BlackLittermanModel(
        S,
        pi=pi,
        absolute_views=views,
        omega="idzorek",
        view_confidences=np.array([0.5] * 5),
    )
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)

    bl = BlackLittermanModel(
        S,
        pi=pi,
        absolute_views=views,
        omega="idzorek",
        view_confidences=np.array([0.5] * 5).reshape(-1, 1),
    )
    pd.testing.assert_series_equal(bl.bl_returns(), test_result)


def test_idzorek_with_priors():
    df = get_data()
    S = risk_models.sample_cov(df)

    mcaps = get_market_caps()

    viewdict = {"GOOG": 0.40, "AAPL": -0.30, "FB": 0.30, "BABA": 0}
    bl = BlackLittermanModel(
        S,
        pi="market",
        market_caps=mcaps,
        absolute_views=viewdict,
        omega="idzorek",
        view_confidences=[1, 1, 0.25, 0.25],
    )
    rets = bl.bl_returns()
    assert bl.omega[0, 0] == 0
    np.testing.assert_almost_equal(rets["AAPL"], -0.3)

    with pytest.raises(ValueError):
        bl.portfolio_performance()

    bl.bl_weights()
    np.testing.assert_allclose(
        bl.portfolio_performance(),
        (0.943431295405105, 0.5361412623208567, 1.722365653051476),
    )
    # Check that bl.cov() has been called and used
    assert bl.posterior_cov is not None
