import numpy as np
import pytest
from pypfopt import HRPOpt
from tests.utilities_for_tests import get_data


def test_hrp_portfolio():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    w = hrp.optimize()
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
    assert hrp.portfolio_performance()


def test_cluster_var():
    #  TODO
    pass


def test_quasi_dag():
    # TODO
    pass


def test_raw_allocation():
    # TODO
    pass
