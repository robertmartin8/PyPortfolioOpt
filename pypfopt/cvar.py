import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
from . import objective_functions, efficient_frontier


class EfficientCVaR(efficient_frontier.EfficientFrontier):
    def __init__(self, expected_returns, historic_returns, beta=0.95, weight_bounds=(0, 1),
                 solver='ECOS',
                 verbose=False,
                 solver_options=None):
        super().__init__(
            expected_returns=expected_returns,
            cov_matrix=np.zeros((len(expected_returns),) * 2),  # dummy
            weight_bounds=weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

        if historic_returns is None:
            raise ValueError("Returns must be provided")

        if not isinstance(historic_returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.historic_returns = self._validate_historic_returns(historic_returns)

        self.beta = beta
        self._alpha = cp.Variable()
        self._u = cp.Variable(len(self.historic_returns))

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

    def min_cvar(self, market_neutral=False):
        self._objective = self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u)
        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints += [
            self._u >= 0.,
            self.historic_returns.values @ self._w + self._alpha + self._u >= 0.
        ]

        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        self._solve_cvxpy_opt_problem()
        # Inverse-transform

        self.weights = (self._w.value).round(16) + 0.0
        return self._make_output_weights()

    def efficient_return(self, target_return, market_neutral=False):
        self._objective = self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u)
        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints += [
            self._u >= 0.,
            self.historic_returns.values @ self._w + self._alpha + self._u >= 0.
        ]

        ret = self.expected_returns.T @ self._w
        self._constraints.append(ret >= target_return)

        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        self._solve_cvxpy_opt_problem()
        # Inverse-transform

        self.weights = (self._w.value).round(16) + 0.0
        return self._make_output_weights()

    def efficient_risk(self, target_cvar, market_neutral=False):
        self._objective = objective_functions.portfolio_return(
            self._w, self.expected_returns
        )
        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints += [
            self._alpha + 1. / len(self.tickers) / (1 - self.beta) * cp.sum(self._u) <= target_cvar,
            self._u >= 0.,
            self.historic_returns.values @ self._w + self._alpha + self._u >= 0.
        ]
        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

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
