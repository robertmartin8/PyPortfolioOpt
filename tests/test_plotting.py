import matplotlib
from tests.utilities_for_tests import get_data
from pypfopt import plotting, risk_models, expected_returns
from pypfopt import HRPOpt, CLA


def test_correlation_plot():
    df = get_data()

    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    ax = plotting.plot_covariance(S, showfig=False)
    assert len(ax.findobj()) == 256
    ax = plotting.plot_covariance(S, show_tickers=False, showfig=False)
    assert len(ax.findobj()) == 136


def test_dendrogram_plot():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    hrp.optimize()

    ax = plotting.plot_dendrogram(hrp, showfig=False)
    assert len(ax.findobj()) == 185
    assert type(ax.findobj()[0]) == matplotlib.collections.LineCollection

    ax = plotting.plot_dendrogram(hrp, show_tickers=False, showfig=False)
    assert len(ax.findobj()) == 65
    assert type(ax.findobj()[0]) == matplotlib.collections.LineCollection


def test_ef_plot():
    df = get_data()
    rets = expected_returns.mean_historical_return(df)
    S = risk_models.exp_cov(df)
    cla = CLA(rets, S)

    ax = plotting.plot_efficient_frontier(cla, showfig=False)
    assert len(ax.findobj()) == 137
    ax = plotting.plot_efficient_frontier(cla, show_assets=False, showfig=False)
    assert len(ax.findobj()) == 149


def test_weight_plot():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    w = hrp.optimize()

    ax = plotting.plot_weights(w, showfig=False)
    assert len(ax.findobj()) == 197
