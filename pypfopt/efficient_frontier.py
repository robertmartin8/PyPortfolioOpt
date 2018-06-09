"""
The ``efficient_frontier`` module houses the EfficientFrontier object, which
generates optimal portfolios for various possible objective functions and parameters.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.optimize as sco
from . import objective_functions


class EfficientFrontier:

    """
    An EfficientFrontier object contains multiple optimisation methods that can be called
    (corresponding to different objective functions) with various parameters.

    Instance variables:

    - Inputs:

        - ``cov_matrix``
        - ``n_assets``
        - ``tickers``
        - ``bounds``

    - Optimisation parameters:

        - ``initial_guess``
        - ``constraints``

    - Output: ``weights``

    Public methods:

    - ``max_sharpe()`` optimises for maximal Sharpe ratio (a.k.a the tangency portfolio)
    - ``min_volatility()`` optimises for minimum volatility
    - ``efficient_risk()`` maximises Sharpe for a given target risk
    - ``efficient_return()`` minimises risk for a given target return
    - ``portfolio_performance()`` calculates the expected return, volatility and Sharpe ratio for
      the optimised portfolio.
    """

    def __init__(self, expected_returns, cov_matrix, weight_bounds=(0, 1), gamma=0):
        """
        :param expected_returns: expected returns for each asset
        :type expected_returns: pd.Series, list, np.ndarray
        :param cov_matrix: covariance of returns for each asset
        :type cov_matrix: pd.DataFrame or np.array
        :param weight_bounds: minimum and maximum weight of an asset, defaults to (0, 1).
                              Must be changed to (-1, 1) for portfolios with shorting.
        :type weight_bounds: tuple, optional
        :param gamma: L2 regularisation parameter, defaults to 0. Increase if you want more
                      non-negligible weights
        :type gamma: float, optional
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        :raises TypeError: if ``cov_matrix`` is not a dataframe or array
        """
        # Inputs
        if not isinstance(expected_returns, (pd.Series, list, np.ndarray)):
            raise TypeError("Expected returns is not a series, list or array")
        if not isinstance(cov_matrix, (pd.DataFrame, np.ndarray)):
            raise TypeError("cov_matrix is not a dataframe or array")

        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.n_assets = len(expected_returns)
        self.tickers = list(expected_returns.index)
        self.bounds = self._make_valid_bounds(weight_bounds)

        if not isinstance(gamma, (int, float)):
            raise ValueError("gamma should be numeric")
        if gamma < 0:
            warnings.warn("in most cases, gamma should be positive", UserWarning)
        self.gamma = gamma

        # Optimisation parameters
        self.initial_guess = np.array([1 / self.n_assets] * self.n_assets)
        self.constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        # Outputs
        self.weights = None

    def _make_valid_bounds(self, test_bounds):
        """
        Private method: process input bounds into a form acceptable by scipy.optimize,
        and check the validity of said bounds.

        :param test_bounds: minimum and maximum weight of an asset
        :type test_bounds: tuple
        :raises ValueError: if ``test_bounds`` is not a tuple of length two.
        :raises ValueError: if the lower bound is too high
        :return: a tuple of bounds, e.g ((0, 1), (0, 1), (0, 1) ...)
        :rtype: tuple of tuples
        """
        if len(test_bounds) != 2 or not isinstance(test_bounds, tuple):
            raise ValueError(
                "test_bounds must be a tuple of (lower bound, upper bound)"
            )
        if test_bounds[0] is not None:
            if test_bounds[0] * self.n_assets > 1:
                raise ValueError("Lower bound is too high")
        return (test_bounds,) * self.n_assets

    def max_sharpe(self, risk_free_rate=0.02):
        """
        Maximise the Sharpe Ratio. The result is also referred to as the tangency portfolio,
        as it is the tangent to the efficient frontier curve that intercepts the risk-free
        rate.

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: dict
        """
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        args = (self.expected_returns, self.cov_matrix, self.gamma, risk_free_rate)
        constraints = self.constraints
        result = sco.minimize(
            objective_functions.negative_sharpe,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
        )
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def min_volatility(self):
        """
        Minimise volatility.

        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the volatility-minimising portfolio
        :rtype: dict
        """
        args = (self.cov_matrix, self.gamma)
        constraints = self.constraints
        result = sco.minimize(
            objective_functions.volatility,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
        )
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def efficient_risk(self, target_risk, risk_free_rate=0.02, market_neutral=False):
        """
        Calculate the Sharpe-maximising portfolio for a given volatility (i.e max return
        for a target risk).

        :param target_risk: the desired volatility of the resulting portfolio.
        :type target_risk: float
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :raises ValueError: if ``target_risk`` is not a positive float
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the efficient risk portfolio
        :rtype: dict
        """
        if not isinstance(target_risk, float) or target_risk < 0:
            raise ValueError("target_risk should be a positive float")
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        self.n_assets = len(self.expected_returns)
        args = (self.expected_returns, self.cov_matrix, self.gamma, risk_free_rate)
        target_constraint = {
            "type": "ineq",
            "fun": lambda w: target_risk
            - objective_functions.volatility(w, self.cov_matrix),
        }
        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            if self.bounds[0][0] is not None and self.bounds[0][0] >= 0:
                warnings.warn(
                    "Market neutrality requires shorting - bounds have been amended",
                    RuntimeWarning,
                )
                self.bounds = self._make_valid_bounds((-1, 1))
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x)},
                target_constraint,
            ]
        else:
            constraints = self.constraints + [target_constraint]

        result = sco.minimize(
            objective_functions.negative_sharpe,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
        )
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def efficient_return(self, target_return, market_neutral=False):
        """
        Calculate the 'Markowitz portfolio', minimising volatility for a given target return.

        :param target_return: the desired return of the resulting portfolio.
        :type target_return: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :type market_neutral: bool, optional
        :raises ValueError: if ``target_return`` is not a positive float
        :return: asset weights for the Markowitz portfolio
        :rtype: dict
        """
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_risk should be a positive float")

        self.n_assets = len(self.expected_returns)
        args = (self.cov_matrix, self.gamma)
        target_constraint = {
            "type": "eq",
            "fun": lambda w: w.dot(self.expected_returns) - target_return,
        }
        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            if self.bounds[0][0] is not None and self.bounds[0][0] >= 0:
                warnings.warn(
                    "Market neutrality requires shorting - bounds have been amended",
                    RuntimeWarning,
                )
                self.bounds = self._make_valid_bounds((-1, 1))
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x)},
                target_constraint,
            ]
        else:
            constraints = self.constraints + [target_constraint]

        result = sco.minimize(
            objective_functions.volatility,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
        )
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

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
        if not isinstance(rounding, int) or rounding < 1:
            raise ValueError("rounding must be a positive integer")
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        if rounding is not None:
            clean_weights = np.round(clean_weights, rounding)
        return dict(zip(self.tickers, clean_weights))

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        if self.weights is None:
            raise ValueError("Weights not calculated yet")
        sigma = objective_functions.volatility(self.weights, self.cov_matrix)
        mu = self.weights.dot(self.expected_returns)

        sharpe = -objective_functions.negative_sharpe(
            self.weights, self.expected_returns, self.cov_matrix, risk_free_rate
        )
        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * sigma))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
        return mu, sigma, sharpe
