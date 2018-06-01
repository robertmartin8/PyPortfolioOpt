import pandas as pd
import numpy as np
from pypfopt import risk_models
from tests.utilities_for_tests import get_data
import warnings
import pytest


def test_sample_cov_dummy():
    data = pd.DataFrame(
        [
            [4.0, 2.0, 0.6],
            [4.2, 2.1, 0.59],
            [3.9, 2.0, 0.58],
            [4.3, 2.1, 0.62],
            [4.1, 2.2, 0.63],
        ]
    )
    test_answer = pd.DataFrame(
        [
            [0.006661687937656102, 0.00264970955585574, 0.0020849735375206195],
            [0.00264970955585574, 0.0023450491307634215, 0.00096770864287974],
            [0.0020849735375206195, 0.00096770864287974, 0.0016396416271856837],
        ]
    )
    S = risk_models.sample_cov(data) / 252
    pd.testing.assert_frame_equal(S, test_answer)


def test_sample_cov_real():
    df = get_data()
    S = risk_models.sample_cov(df)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()


def test_sample_cov_type_warning():
    df = get_data()
    cov_from_df = risk_models.sample_cov(df)

    returns_as_array = np.array(df)
    with warnings.catch_warnings(record=True) as w:
        cov_from_array = risk_models.sample_cov(returns_as_array)

        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert str(w[0].message) == "prices are not in a dataframe"

    np.testing.assert_array_almost_equal(
        cov_from_df.values, cov_from_array.values, decimal=6
    )


@pytest.mark.xfail()
def test_constant_correlation_shrinkage_target():
    df = get_data()
    lw = risk_models.LedoitWolfShrinkage(df)
    F = lw.constant_correlation_shrinkage_target()
    assert F.shape == (20, 20)
    np.testing.assert_array_almost_equal(df.std().values, F.diagonal())
    test_array = np.array(
        [
            [0.01923241, 0.00756082, 0.00681764],
            [0.00756082, 0.02869772, 0.00832801],
            [0.00681764, 0.00832801, 0.02333343],
        ]
    )
    np.testing.assert_array_almost_equal(F[0:3, 0:3], test_array)
