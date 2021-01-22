import math
import numpy as np
import pandas as pd
from . import base_optimizer
from . import expected_returns
from scipy import optimize
import cvxpy as cp
import warnings


class CVar(base_optimizer.BaseConvexOptimizer):
    def __init__(self, returns, expected_returns,
                 cov_matrix, beta=0.95, verbose=False):

        if returns is None:
            raise ValueError("Returns must be provided")

        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.cov_matrix = CVar._validate_cov_matrix(cov_matrix)
        self.expected_returns = CVar._validate_expected_returns(
            expected_returns
        )
        self.beta = beta
        self._alpha = cp.Variable()
        self._u = cp.Variable(len(self.returns))

        tickers = list(returns.columns)
        super().__init__(len(tickers), tickers, verbose=verbose)

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

    @staticmethod
    def _validate_cov_matrix(cov_matrix):
        if cov_matrix is None:
            raise ValueError("cov_matrix must be provided")
        elif isinstance(cov_matrix, pd.DataFrame):
            return cov_matrix.values
        elif isinstance(cov_matrix, np.ndarray):
            return cov_matrix
        else:
            raise TypeError("cov_matrix is not a series, list or array")

    def optimize(self, target_return):
        self._objective = self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u)

        self._constraints = [
            cp.sum(self._w) == 1.,
            self._w >= 0.,
            self._u >= 0.,
            self.returns.values @ self._w + self._alpha + self._u >= 0.
        ]

        ret = self.expected_returns.T @ self._w
        self._constraints.append(ret >= target_return)

        self._solve_cvxpy_opt_problem()
        # Inverse-transform

        self.weights = (self._w.value).round(16) + 0.0
        return self._make_output_weights()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        return base_optimizer.portfolio_performance(
            self.weights,
            self.expected_returns,
            self.cov_matrix,
            verbose,
            risk_free_rate,
        )
