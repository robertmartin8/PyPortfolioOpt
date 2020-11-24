import numpy as np

from pypfopt import risk_models, EfficientFrontier
from pypfopt.efficient_semivariance import EfficientSemivariance
from tests.utilities_for_tests import setup_efficient_semivariance


def test_efficient_return():
    es = setup_efficient_semivariance()
    w = es.efficient_return(0.25/252)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(es.tickers)
    np.testing.assert_almost_equal(es.weights.sum(), 1)
    assert all([i >= -1e-5 for i in w.values()])

    np.testing.assert_allclose(
        es.portfolio_performance(),
        (0.2500001777470832, 0.10035962239578998, 2.291760094911752),
    )


def test_efficient_semivariance_vs_heuristic():
    benchmark = 0
    es = setup_efficient_semivariance()
    es.efficient_return(0.20/252)
    mu_es, semi_deviation, _ = es.portfolio_performance()

    mean_return, historic_returns = setup_efficient_semivariance(data_only=True)

    pairwise_semivariance = risk_models.semicovariance(historic_returns, returns_data=True, benchmark=0, frequency=1)
    ef = EfficientFrontier(mean_return, pairwise_semivariance)
    ef.efficient_return(0.20/252)
    mu_ef, _, _ = ef.portfolio_performance()
    mu_ef *= 252
    portfolio_returns = historic_returns @ ef.weights
    drops = np.fmin(portfolio_returns - benchmark, 0)
    T = historic_returns.shape[0]
    semivariance = np.sum(np.square(drops)) / T * 252
    semi_deviation_ef = np.sqrt(semivariance)

    assert semi_deviation < semi_deviation_ef
    assert mu_es / semi_deviation > mu_ef / semi_deviation_ef


def test_efficient_semivariance_vs_heuristic_weekly():
    benchmark = 0

    _, historic_returns = setup_efficient_semivariance(data_only=True)
    monthly_returns = historic_returns.resample('W').sum()
    mean_monthly_returns = monthly_returns.mean(axis=0)

    es = EfficientSemivariance(mean_monthly_returns, monthly_returns, frequency=52)
    es.efficient_return(0.20/52)
    mu_es, semi_deviation, _ = es.portfolio_performance()

    pairwise_semivariance = risk_models.semicovariance(monthly_returns, returns_data=True, benchmark=0, frequency=1)
    ef = EfficientFrontier(mean_monthly_returns, pairwise_semivariance)
    ef.efficient_return(0.20/52)
    mu_ef, _, _ = ef.portfolio_performance()
    mu_ef *= 52
    portfolio_returns = historic_returns @ ef.weights
    drops = np.fmin(portfolio_returns - benchmark, 0)
    T = monthly_returns.shape[0]
    semivariance = np.sum(np.square(drops)) / T * 52
    semi_deviation_ef = np.sqrt(semivariance)

    assert semi_deviation < semi_deviation_ef
    assert mu_es / semi_deviation > mu_ef / semi_deviation_ef





