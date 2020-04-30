import numpy as np
import pandas as pd

import pytest
from pypfopt import HRPOpt, CovarianceShrinkage
from tests.utilities_for_tests import get_data, resource


def test_hrp_portfolio():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    w = hrp.optimize()

    # uncomment this line if you want generating a new file
    # pd.Series(w).to_csv(resource("weights_hrp.csv"))

    x = pd.read_csv(resource("weights_hrp.csv"), squeeze=True, index_col=0)
    pd.testing.assert_series_equal(
        x, pd.Series(w), check_names=False, check_less_precise=True
    )

    assert isinstance(w, dict)
    assert set(w.keys()) == set(df.columns)
    np.testing.assert_almost_equal(sum(w.values()), 1)
    assert all([i >= 0 for i in w.values()])


def test_portfolio_performance():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    with pytest.raises(ValueError):
        hrp.portfolio_performance()
    hrp.optimize()
    np.testing.assert_allclose(
        hrp.portfolio_performance(),
        (0.21353402380950973, 0.17844159743748936, 1.084579081272277),
    )


def test_pass_cov_matrix():
    df = get_data()
    S = CovarianceShrinkage(df).ledoit_wolf()
    hrp = HRPOpt(cov_matrix=S)
    hrp.optimize()
    perf = hrp.portfolio_performance()
    assert perf[0] is None and perf[2] is None
    np.testing.assert_almost_equal(perf[1], 0.10002783894982334)


def test_cluster_var():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    cov = returns.cov()
    tickers = ["SHLD", "AMD", "BBY", "RRC", "FB", "WMT", "T", "BABA", "PFE", "UAA"]
    var = HRPOpt._get_cluster_var(cov, tickers)
    np.testing.assert_almost_equal(var, 0.00012842967106653283)


def test_quasi_dag():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    hrp.optimize()
    clusters = hrp.clusters
    assert HRPOpt._get_quasi_diag(clusters)[:5] == [12, 6, 15, 14, 2]
