import numpy as np
from pypfopt.hierarchical_risk_parity import hrp_portfolio
from tests.utilities_for_tests import get_data


def test_hrp_portfolio():
    df = get_data()
    cov = df.pct_change().cov()
    corr = df.pct_change().corr()
    w = hrp_portfolio(cov, corr)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(df.columns)
    np.testing.assert_almost_equal(sum(w.values()), 1)
