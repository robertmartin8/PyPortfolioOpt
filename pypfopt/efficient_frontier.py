import numpy as np
import pandas as pd
import scipy.optimize as sco
from . import objective_functions
import warnings


class EfficientFrontier:

    def __init__(self, expected_returns, cov_matrix, weight_bounds=(0, 1)):
        """
        :param expected_returns: expected returns for each asset
        :type expected_returns: pd.Series, list, np.ndarray
        :param cov_matrix: covariance of returns for each asset
        :type cov_matrix: pd.DataFrame or np.array
        :param weight_bounds: minimum and maximum weight of an asset, defaults to (0, 1)
        :param weight_bounds: tuple, optional
        """
        # Inputs
        if not isinstance(expected_returns, (pd.Series, list, np.ndarray)):
            raise TypeError("Expected returns is not a series, list or array")
        self.expected_returns = expected_returns

        if not isinstance(cov_matrix, (pd.DataFrame, np.ndarray)):
            raise TypeError("cov_matrix is not a dataframe or array")
        self.cov_matrix = cov_matrix
        self.n_assets = len(expected_returns)
        self.tickers = list(expected_returns.index)
        # Optimisation parameters
        self.initial_guess = np.array([1 / self.n_assets] * self.n_assets)
        self.constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        self.bounds = self._make_valid_bounds(weight_bounds)
        # Optional
        self.risk_free_rate = 0.02
        # Outputs
        self.weights = None

    def _make_valid_bounds(self, test_bounds):
        if len(test_bounds) != 2:
            raise ValueError("test_bounds must have lower and upper bounds")
        if test_bounds[0] is not None:
            if test_bounds[0] * self.n_assets > 1:
                raise ValueError("Lower bound is too high")
        return (test_bounds,) * self.n_assets

    def max_sharpe(self, alpha=0, risk_free_rate=0.02):
        """
        The 'tangent' portfolio that maximises the Sharpe Ratio. The Sharpe ratio is defined as
        .. math::
        \frac{\mu - R_f}{\sigma}
        :param risk_free_rate: risk free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :return: portfolio weights
        :rtype: dictionary: keys are tickers (string), values are weights (float)
        """
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha should be numeric")
        if alpha < 0:
            warnings.warn("in most cases, alpha should be positive", UserWarning)
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        self.risk_free_rate = risk_free_rate
        args = (self.expected_returns, self.cov_matrix, alpha, risk_free_rate)
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

    def min_volatility(self, alpha=0):
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha should be numeric")
        if alpha < 0:
            warnings.warn("in most cases, alpha should be positive", UserWarning)

        args = (self.cov_matrix, alpha)
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

    def efficient_risk(
        self, target_risk, alpha=0, risk_free_rate=0.02, market_neutral=False
    ):
        """
        Calculates the Sharpe-maximising portfolio for a given target risk
        :param self.expected_returns: array of mean returns for a number of stocks
        :param self.cov_matrix: covariance of these stocks.
        :param target_risk: the target return
        :param risk_free_rate: defaults to zero
        :return: the weights of the portfolio that minimise risk for this target return
        """
        if not isinstance(target_risk, float) or target_risk < 0:
            raise ValueError("target_risk should be a positive float")
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha should be numeric")
        if alpha < 0:
            warnings.warn("in most cases, alpha should be positive", UserWarning)
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        self.n_assets = len(self.expected_returns)
        args = (self.expected_returns, self.cov_matrix, alpha, risk_free_rate)

        target_constraint = {
            "type": "ineq",
            "fun": lambda w: target_risk
            - objective_functions.volatility(w, self.cov_matrix),
        }

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

    def efficient_return(self, target_return, alpha=0, market_neutral=False):
        """
        Calculates the "Markowitz" portfolio, minimising risk for a target return
        :param self.expected_returns: array of mean returns for a number of stocks
        :param self.cov_matrix: covariance of these stocks.
        :param target_risk: the target return
        :param risk_free_rate: defaults to zero
        :return: the weights of the portfolio that minimise risk for this target return
        """
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_risk should be a positive float")
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha should be numeric")
        if alpha < 0:
            warnings.warn("in most cases, alpha should be positive", UserWarning)

        self.n_assets = len(self.expected_returns)
        args = (self.cov_matrix, alpha)
        target_constraint = {
            "type": "eq",
            "fun": lambda w: w.dot(self.expected_returns) - target_return,
        }
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

    def portfolio_performance(self, verbose=False):
        """
        Calculates the performance given the calculated weights of the portfolio
        :return: [description]
        :rtype: [type]
        """
        if self.weights is None:
            raise ValueError("Weights not calculated yet")
        sigma = objective_functions.volatility(self.weights, self.cov_matrix)
        mu = self.weights.dot(self.expected_returns)

        sharpe = -objective_functions.negative_sharpe(
            self.weights, self.expected_returns, self.cov_matrix, self.risk_free_rate
        )
        if verbose:
            print("Expected return:", mu)
            print("Volatility:", sigma)
            print("Sharpe:", sharpe)
        return mu, sigma, sharpe
