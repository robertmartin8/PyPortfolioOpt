import numpy as np
import pytest
from pypfopt.value_at_risk import CVAROpt
from tests.utilities_for_tests import get_data


def test_init_cvar():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    vr = CVAROpt(returns)
    assert list(vr.tickers) == list(df.columns)

    # Inheritance
    assert vr.bounds == ((0, 1),) * len(df.columns)
    assert vr.clean_weights
    assert isinstance(vr.initial_guess, np.ndarray)
    assert isinstance(vr.constraints, list)


def test_init_cvar_errors():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    with pytest.raises(ValueError):
        vr = CVAROpt(returns, weight_bounds=(0.5, 1))
    with pytest.raises(AttributeError):
        vr = CVAROpt(returns)
        vr.clean_weights()
    with pytest.raises(TypeError):
        vr = CVAROpt(returns.values)
    returns_list = df.values.tolist()
    with pytest.raises(TypeError):
        vr = CVAROpt(returns_list)


def test_cvar_weights():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    vr = CVAROpt(returns)
    w = vr.min_cvar(s=100, random_state=0)
    assert isinstance(w, dict)
    assert set(w.keys()) == set(df.columns)
    assert set(w.keys()) == set(vr.tickers)
    np.testing.assert_almost_equal(vr.weights.sum(), 1)


def test_cvar_bounds():
    # TODO
    pass


def test_cvar_beta():
    # TODO
    pass
