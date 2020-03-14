"""
The ``base_optimizer`` module houses the parent classes ``BaseOptimizer`` and
``BaseConvexOptimizer``, from which all optimisers will inherit. The later is for
optimisers that use the scipy solver.

Additionally, we define a general utility function ``portfolio_performance`` to
evaluate return and risk for a given set of portfolio weights.
"""

import json
import numpy as np
import pandas as pd
import cvxpy as cp
from . import objective_functions


class BaseOptimizer:

    """
    Instance variables:

    - ``n_assets`` - int
    - ``tickers`` - str list
    - ``weights`` - np.ndarray

    Public methods:

    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, n_assets, tickers=None):
        """
        :param n_assets: number of assets
        :type n_assets: int
        :param tickers: name of assets
        :type tickers: list
        """
        self.n_assets = n_assets
        if tickers is None:
            self.tickers = list(range(n_assets))
        else:
            self.tickers = tickers
        # Outputs
        self.weights = None

    def set_weights(self, weights):
        """
        Utility function to set weights.

        :param weights: {ticker: weight} dictionary
        :type weights: dict
        """
        self.weights = np.array([weights[ticker] for ticker in self.tickers])

    def clean_weights(self, cutoff=1e-4, rounding=5):
        """
        Helper method to clean the raw weights, setting any weights whose absolute
        values are below the cutoff to zero, and rounding the rest.

        :param cutoff: the lower bound, defaults to 1e-4
        :type cutoff: float, optional
        :param rounding: number of decimal places to round the weights, defaults to 5.
                         Set to None if rounding is not desired.
        :type rounding: int, optional
        :return: asset weights
        :rtype: dict
        """
        if self.weights is None:
            raise AttributeError("Weights not yet computed")
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        if rounding is not None:
            if not isinstance(rounding, int) or rounding < 1:
                raise ValueError("rounding must be a positive integer")
            clean_weights = np.round(clean_weights, rounding)
        return dict(zip(self.tickers, clean_weights))

    def save_weights_to_file(self, filename="weights.csv"):
        """
        Utility method to save weights to a text file.

        :param filename: name of file. Should be csv, json, or txt.
        :type filename: str
        """
        clean_weights = self.clean_weights()

        ext = filename.split(".")[1]
        if ext == "csv":
            pd.Series(clean_weights).to_csv(filename, header=False)
        elif ext == "json":
            with open(filename, "w") as fp:
                json.dump(clean_weights, fp)
        else:
            with open(filename, "w") as f:
                f.write(str(clean_weights))


class BaseConvexOptimizer(BaseOptimizer):

    """
    Instance variables:

    - ``n_assets`` - int
    - ``tickers`` - str list
    - ``weights`` - np.ndarray
    - ``bounds`` - float tuple OR (float tuple) list
    - ``constraints`` - dict list

    Public methods:

    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, n_assets, tickers=None, weight_bounds=(0, 1)):
        """
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        """
        super().__init__(n_assets, tickers)

        # Optimisation variables
        self._w = cp.Variable(n_assets)
        self._objective = None
        self._additional_objectives = []
        self._constraints = []
        self._map_bounds_to_constraints(weight_bounds)

    def _map_bounds_to_constraints(self, test_bounds):
        """
        Process input bounds into a form acceptable by cvxpy and add to the constraints list.

        :param test_bounds: minimum and maximum weight of each asset OR single min/max pair
                            if all identical OR pair of arrays corresponding to lower/upper bounds. defaults to (0, 1).
        :type test_bounds: tuple OR list/tuple of tuples OR pair of np arrays
        :raises TypeError: if ``test_bounds`` is not of the right type
        :return: bounds suitable for cvxpy
        :rtype: tuple pair of np.ndarray
        """
        # If it is a collection with the right length, assume they are all bounds.
        if len(test_bounds) == self.n_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=np.float)
            lower = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            upper = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            # Otherwise this must be a pair.
            if len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list)):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) "
                    "OR a collection of bounds for each asset"
                )
            lower, upper = test_bounds

            # Replace None values with the appropriate infinity.
            if np.isscalar(lower) or lower is None:
                lower = -np.inf if lower is None else lower
                upper = np.inf if upper is None else upper
            else:
                lower = np.nan_to_num(lower, nan=-np.inf)
                upper = np.nan_to_num(upper, nan=np.inf)

        self._constraints.append(self._w >= lower)
        self._constraints.append(self._w <= upper)

    @staticmethod
    def _make_scipy_bounds():
        """
        Convert the current cvxpy bounds to scipy bounds
        """
        raise NotImplementedError


def portfolio_performance(
    expected_returns, cov_matrix, weights, verbose=False, risk_free_rate=0.02
):
    """
    After optimising, calculate (and optionally print) the performance of the optimal
    portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

    :param expected_returns: expected returns for each asset. Set to None if
                             optimising for volatility only.
    :type expected_returns: pd.Series, list, np.ndarray
    :param cov_matrix: covariance of returns for each asset
    :type cov_matrix: pd.DataFrame or np.array
    :param weights: weights or assets
    :type weights: list, np.array or dict, optional
    :param verbose: whether performance should be printed, defaults to False
    :type verbose: bool, optional
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
    :type risk_free_rate: float, optional
    :raises ValueError: if weights have not been calcualted yet
    :return: expected return, volatility, Sharpe ratio.
    :rtype: (float, float, float)
    """
    if isinstance(weights, dict):
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(range(len(expected_returns)))
        new_weights = np.zeros(len(tickers))
        for i, k in enumerate(tickers):
            if k in weights:
                new_weights[i] = weights[k]
        if new_weights.sum() == 0:
            raise ValueError("Weights add to zero, or ticker names don't match")
    elif weights is not None:
        new_weights = np.asarray(weights)
    else:
        raise ValueError("Weights is None")

    sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, cov_matrix))
    mu = new_weights.dot(expected_returns)

    sharpe = -objective_functions.negative_sharpe(
        new_weights, expected_returns, cov_matrix, risk_free_rate=risk_free_rate
    )
    if verbose:
        print("Expected annual return: {:.1f}%".format(100 * mu))
        print("Annual volatility: {:.1f}%".format(100 * sigma))
        print("Sharpe Ratio: {:.2f}".format(sharpe))
    return mu, sigma, sharpe
