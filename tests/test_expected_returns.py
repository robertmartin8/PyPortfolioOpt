import pandas as pd
import numpy as np
from pypfopt import expected_returns
from tests.utilities_for_tests import get_data

# TODO test mean historical returns frequency
# TODO test ema


def test_mean_historical_returns_dummy():
    data = pd.DataFrame(
        [
            [4.0, 2.0, 0.6, -12],
            [4.2, 2.1, 0.59, -13.2],
            [3.9, 2.0, 0.58, -11.3],
            [4.3, 2.1, 0.62, -11.7],
            [4.1, 2.2, 0.63, -10.1],
        ]
    )
    mean = expected_returns.mean_historical_return(data, frequency=1)
    test_answer = pd.Series([4.1, 2.08, 0.604, -11.66])
    pd.testing.assert_series_equal(mean, test_answer)


def test_mean_historical_returns():
    df = get_data()
    mean = expected_returns.mean_historical_return(df)
    assert isinstance(mean, pd.Series)
    assert list(mean.index) == list(df.columns)
    assert mean.notnull().all()
    assert mean.dtype == "float64"
    correct_mean = np.array(
        [
            0.26770284,
            0.3637864,
            0.31709032,
            0.22616723,
            0.49982007,
            0.16888704,
            0.22754479,
            0.14783539,
            0.19001915,
            0.08150653,
            0.12826351,
            0.25797816,
            0.07580128,
            0.16087243,
            0.20510267,
            0.3511536,
            0.38808003,
            0.24635612,
            0.21798433,
            0.28474973,
        ]
    )
    np.testing.assert_array_almost_equal(mean.values, correct_mean)


def test_mean_historical_returns_frequency():
    pass
