import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
from . import objective_functions, base_optimizer


class EfficientCVaR(base_optimizer.BaseConvexOptimizer):
    def __init__(self, expected_returns, historic_returns,
                 beta=0.95, frequency=21, verbose=False):

        if historic_returns is None:
            raise ValueError("Returns must be provided")

        if not isinstance(historic_returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.frequency = frequency
        self.expected_returns = EfficientCVaR._validate_expected_returns(
            expected_returns
        )
        self.historic_returns = self._validate_historic_returns(historic_returns)

        self.beta = beta
        self._alpha = cp.Variable()
        self._u = cp.Variable(len(self.historic_returns))

        tickers = list(historic_returns.columns)
        super().__init__(len(tickers), tickers, verbose=verbose)

    def _validate_historic_returns(self, historic_returns):
        if not isinstance(historic_returns, (pd.DataFrame, np.ndarray)):
            raise TypeError("historic_returns should be a pd.Dataframe or np.ndarray")

        historic_returns_df = pd.DataFrame(historic_returns)
        if historic_returns_df.isnull().values.any():
            warnings.warn(
                "Removing NaNs from historic returns",
                UserWarning,
            )
            historic_returns_df = historic_returns_df.dropna(axis=0, how="any")

        if self.expected_returns is not None:
            if historic_returns_df.shape[1] != len(self.expected_returns):
                raise ValueError(
                    "historic_returns columns do not match expected_returns. Please check your tickers."
                )

        return historic_returns_df

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


    def min_cvar(self):
        self._objective = self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u)
        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints += [
            cp.sum(self._w) == 1.,
            self._w >= 0.,
            self._u >= 0.,
            self.historic_returns.values @ self._w + self._alpha + self._u >= 0.
        ]

        ret = self.expected_returns.T @ self._w

        self._solve_cvxpy_opt_problem()
        # Inverse-transform

        self.weights = (self._w.value).round(16) + 0.0
        return self._make_output_weights()

    def efficient_return(self, target_return):
        self._objective = self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u)
        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints += [
            cp.sum(self._w) == 1.,
            self._w >= 0.,
            self._u >= 0.,
            self.historic_returns.values @ self._w + self._alpha + self._u >= 0.
        ]

        ret = self.expected_returns.T @ self._w
        self._constraints.append(ret >= target_return)

        self._solve_cvxpy_opt_problem()
        # Inverse-transform

        self.weights = (self._w.value).round(16) + 0.0
        return self._make_output_weights()

    def efficient_risk(self, target_cvar):
        self._objective = objective_functions.portfolio_return(
            self._w, self.expected_returns
        )
        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints += [
            self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u) <= target_cvar,
            cp.sum(self._w) == 1.,
            self._w >= 0.,
            self._u >= 0.,
            self.historic_returns.values @ self._w + self._alpha + self._u >= 0.
        ]

        self._solve_cvxpy_opt_problem()

        self.weights = (self._w.value).round(16) + 0.0
        return self._make_output_weights()

    def portfolio_performance(self, verbose=False):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio, specifically: expected return, semideviation, Sortino ratio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, CVaR.
        :rtype: (float, float)
        """

        mu = objective_functions.portfolio_return(
            self.weights, self.expected_returns, negative=False
        )

        portfolio_returns = self.historic_returns @ self.weights
        cvar = (self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u)).value

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Conditional Value at Risk: {:.2f}%".format(100 * cvar))

        return mu, cvar
