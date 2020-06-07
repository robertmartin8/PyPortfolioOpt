import warnings
import pandas as pd
import numpy as np
import pytest
from pypfopt import risk_models, expected_returns
from tests.utilities_for_tests import get_data


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


def test_sample_cov_real_data():
    df = get_data()
    S = risk_models.sample_cov(df)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)


def test_sample_cov_type_warning():
    df = get_data()
    cov_from_df = risk_models.sample_cov(df)

    returns_as_array = np.array(df)
    with warnings.catch_warnings(record=True) as w:
        cov_from_array = risk_models.sample_cov(returns_as_array)

        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert str(w[0].message) == "data is not in a dataframe"

    np.testing.assert_array_almost_equal(
        cov_from_df.values, cov_from_array.values, decimal=6
    )


def test_sample_cov_npd():
    S = np.array([[0.03818144, 0.04182824], [0.04182824, 0.04149209]])
    assert not risk_models._is_positive_semidefinite(S)

    for method in {"spectral", "diag"}:
        with warnings.catch_warnings(record=True) as w:
            S2 = risk_models.fix_nonpositive_semidefinite(S, fix_method=method)
            assert risk_models._is_positive_semidefinite(S2)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert (
                str(w[0].message)
                == "The covariance matrix is non positive semidefinite. Amending eigenvalues."
            )


def test_fix_npd_different_method():
    df = get_data()
    S = risk_models.sample_cov(df)
    assert risk_models._is_positive_semidefinite(S)
    S = risk_models.sample_cov(df, fix_method="diag")
    assert risk_models._is_positive_semidefinite(S)


def test_sample_cov_frequency():
    df = get_data()
    S = risk_models.sample_cov(df)
    S2 = risk_models.sample_cov(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)


def test_semicovariance():
    df = get_data()
    S = risk_models.semicovariance(df)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)
    S2 = risk_models.semicovariance(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)


def test_semicovariance_benchmark():
    df = get_data()
    # When the benchmark is very negative, the cov matrix should be zeroes
    S_negative_benchmark = risk_models.semicovariance(df, benchmark=-0.5)
    np.testing.assert_allclose(S_negative_benchmark, 0, atol=1e-4)

    # Increasing the benchmark should increase covariances on average
    S = risk_models.semicovariance(df, benchmark=0)
    S2 = risk_models.semicovariance(df, benchmark=1)
    assert S2.sum().sum() > S.sum().sum()


def test_exp_cov_matrix():
    df = get_data()
    S = risk_models.exp_cov(df)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)
    S2 = risk_models.exp_cov(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)


def test_exp_cov_limits():
    df = get_data()
    sample_cov = risk_models.sample_cov(df)
    S = risk_models.exp_cov(df)
    assert not np.allclose(sample_cov, S)

    # As span gets larger, it should tend towards sample covariance
    S2 = risk_models.exp_cov(df, span=1e20)
    assert np.abs(S2 - sample_cov).max().max() < 1e-3


def test_min_cov_det():
    df = get_data()
    S = risk_models.min_cov_determinant(df, random_state=8)
    assert S.shape == (20, 20)
    assert S.index.equals(df.columns)
    assert S.index.equals(S.columns)
    assert S.notnull().all().all()
    # assert risk_models._is_positive_semidefinite(S)


def test_cov_to_corr():
    df = get_data()
    rets = risk_models.returns_from_prices(df).dropna()
    test_corr = risk_models.cov_to_corr(rets.cov())
    pd.testing.assert_frame_equal(test_corr, rets.corr())

    with warnings.catch_warnings(record=True) as w:
        test_corr_numpy = risk_models.cov_to_corr(rets.cov().values)
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert str(w[0].message) == "cov_matrix is not a dataframe"
        np.testing.assert_array_almost_equal(test_corr_numpy, rets.corr().values)


def test_corr_to_cov():
    df = get_data()
    rets = risk_models.returns_from_prices(df).dropna()
    test_corr = risk_models.cov_to_corr(rets.cov())
    new_cov = risk_models.corr_to_cov(test_corr, rets.std())
    pd.testing.assert_frame_equal(new_cov, rets.cov())


def test_covariance_shrinkage_init():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    assert cs.S.shape == (20, 20)
    assert not (np.isnan(cs.S)).any()


def test_shrunk_covariance():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.shrunk_covariance(0.2)
    assert cs.delta == 0.2
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)


def test_shrunk_covariance_extreme_delta():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    # if delta = 0, no shrinkage occurs
    shrunk_cov = cs.shrunk_covariance(0)
    np.testing.assert_array_almost_equal(shrunk_cov.values, risk_models.sample_cov(df))
    # if delta = 1, sample cov does not contribute to shrunk cov
    shrunk_cov = cs.shrunk_covariance(1)
    N = df.shape[1]
    F = np.identity(N) * np.trace(cs.S) / N
    np.testing.assert_array_almost_equal(shrunk_cov.values, F * 252)


def test_shrunk_covariance_frequency():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df, frequency=52)
    # if delta = 0, no shrinkage occurs
    shrunk_cov = cs.shrunk_covariance(0)

    S = risk_models.sample_cov(df, frequency=52)
    np.testing.assert_array_almost_equal(shrunk_cov.values, S)


def test_ledoit_wolf_default():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.ledoit_wolf()
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)


def test_ledoit_wolf_single_index():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.ledoit_wolf(shrinkage_target="single_factor")
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)


def test_ledoit_wolf_constant_correlation():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.ledoit_wolf(shrinkage_target="constant_correlation")
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)


def test_ledoit_wolf_raises_not_implemented():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    with pytest.raises(NotImplementedError):
        cs.ledoit_wolf(shrinkage_target="I have not been implemented!")


def test_oracle_approximating():
    df = get_data()
    cs = risk_models.CovarianceShrinkage(df)
    shrunk_cov = cs.oracle_approximating()
    assert 0 < cs.delta < 1
    assert shrunk_cov.shape == (20, 20)
    assert list(shrunk_cov.index) == list(df.columns)
    assert list(shrunk_cov.columns) == list(df.columns)
    assert not shrunk_cov.isnull().any().any()
    assert risk_models._is_positive_semidefinite(shrunk_cov)


def test_risk_matrix_and_returns_data():
    # Test the switcher method for simple calls
    df = get_data()

    for method in {
        "sample_cov",
        "semicovariance",
        "exp_cov",
        "ledoit_wolf",
        "ledoit_wolf_constant_variance",
        "ledoit_wolf_single_factor",
        "ledoit_wolf_constant_correlation",
        "oracle_approximating",
    }:

        S = risk_models.risk_matrix(df, method=method)
        assert S.shape == (20, 20)
        assert S.notnull().all().all()
        assert risk_models._is_positive_semidefinite(S)

        S2 = risk_models.risk_matrix(
            expected_returns.returns_from_prices(df), returns_data=True, method=method
        )
        pd.testing.assert_frame_equal(S, S2)


def test_risk_matrix_additional_kwargs():
    df = get_data()
    S = risk_models.sample_cov(df)
    S2 = risk_models.risk_matrix(df, frequency=2)
    pd.testing.assert_frame_equal(S / 126, S2)

    S = risk_models.risk_matrix(
        df, method="semicovariance", benchmark=0.0004, frequency=52
    )
    assert S.shape == (20, 20)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)

    S = risk_models.risk_matrix(
        expected_returns.returns_from_prices(df),
        returns_data=True,
        method="exp_cov",
        span=60,
        fix_method="diag",
    )
    assert S.shape == (20, 20)
    assert S.notnull().all().all()
    assert risk_models._is_positive_semidefinite(S)


def test_risk_matrix_not_implemented():
    df = get_data()
    with pytest.raises(NotImplementedError):
        risk_models.risk_matrix(df, method="fancy_new!")
