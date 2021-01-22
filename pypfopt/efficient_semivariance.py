"""
The ``efficient_semivariance`` module houses the EfficientSemivariance class, which
generates optimal portfolios with respect to the semivariance of the portfolio.
"""

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from . import base_optimizer, objective_functions


class EfficientSemivariance(base_optimizer.BaseConvexOptimizer):

    # Note: expected_returns and historic_returns have to be in the same frequency
    def __init__(
        self,
        expected_returns,
        historic_returns,
        frequency=252,
        benchmark=0,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
    ):
        # Inputs
        self.expected_returns = EfficientSemivariance._validate_expected_returns(
            expected_returns
        )
        self.historic_returns = historic_returns
        self.frequency = frequency
        self.benchmark = benchmark
        self.T = self.historic_returns.shape[0]

        # Labels
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(historic_returns, pd.DataFrame):
            tickers = list(historic_returns.columns)
        else:  # use integer labels
            tickers = list(range(len(expected_returns)))

        if expected_returns is not None:
            if historic_returns.shape[1] != len(expected_returns):
                raise ValueError(
                    "Historic return matrix does not match expected returns"
                )

        super().__init__(
            len(tickers), tickers, weight_bounds, solver=solver, verbose=verbose
        )

    @staticmethod
    def _validate_expected_returns(expected_returns):
        if expected_returns is None:
            warnings.warn(
                "No expected returns provided. You may only use ef.min_volatility()"
            )
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError("expected_returns is not a series, list or array")

    def _market_neutral_bounds_check(self):
        """
        Helper method to make sure bounds are suitable for a market neutral
        optimisation.
        """
        portfolio_possible = np.any(self._lower_bounds < 0)
        if not portfolio_possible:
            warnings.warn(
                "Market neutrality requires shorting - bounds have been amended",
                RuntimeWarning,
            )
            self._map_bounds_to_constraints((-1, 1))
            # Delete original constraints
            del self._constraints[0]
            del self._constraints[0]

    def efficient_return(self, target_return, market_neutral=False):

        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_return should be a positive float")
        if target_return > np.abs(self.expected_returns).max():
            raise ValueError(
                "target_return must be lower than the largest expected return"
            )

        p = cp.Variable(self.T, nonneg=True)
        n = cp.Variable(self.T, nonneg=True)
        self._objective = cp.sum(cp.square(n))

        self._constraints.append(
            cp.sum(self._w @ self.expected_returns) >= target_return
        )
        B = (self.historic_returns.values - self.benchmark) / np.sqrt(self.T)
        self._constraints.append(B @ self._w - p + n == 0)

        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        return self._solve_cvxpy_opt_problem()

    def min_semivariance(self, market_neutral=False):

        p = cp.Variable(self.T, nonneg=True)
        n = cp.Variable(self.T, nonneg=True)
        self._objective = cp.sum(cp.square(n))
        B = (self.historic_returns.values - self.benchmark) / np.sqrt(self.T)
        self._constraints.append(B @ self._w - p + n == 0)

        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_semi_deviation, market_neutral=False):

        self._objective = objective_functions.portfolio_return(
            self._w, self.expected_returns
        )

        p = cp.Variable(self.T, nonneg=True)
        n = cp.Variable(self.T, nonneg=True)

        self._constraints.append(cp.sum(cp.square(n)) <= (target_semi_deviation ** 2))

        B = (self.historic_returns.values - self.benchmark) / np.sqrt(self.T)
        self._constraints.append(B @ self._w - p + n == 0)

        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):

        mu = objective_functions.portfolio_return(
            self.weights, self.expected_returns, negative=False
        )
        mu *= self.frequency

        portfolio_returns = self.historic_returns @ self.weights
        drops = np.fmin(portfolio_returns - self.benchmark, 0)
        semivariance = np.sum(np.square(drops)) / self.T * self.frequency
        semi_deviation = np.sqrt(semivariance)
        sortino_ratio = (mu - risk_free_rate) / semi_deviation

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual semi-deviation: {:.1f}%".format(100 * semi_deviation))
            print("Sortino Ratio: {:.2f}".format(sortino_ratio))

        return mu, semi_deviation, sortino_ratio
