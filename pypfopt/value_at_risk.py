"""
The ``value_at_risk`` module allows for optimisation with a (conditional)
value-at-risk (CVaR) objective, which requires Monte Carlo simulation.
"""

import pandas as pd
from .base_optimizer import BaseOptimizer
from . import objective_functions
import noisyopt


class CVAROpt(BaseOptimizer):

    """
    A CVAROpt object (inheriting from BaseOptimizer) provides a method for
    optimising the CVaR (a.k.a expected shortfall) of a portfolio.

    Instance variables:

    - Inputs
        - ``tickers``
        - ``returns``
        - ``bounds``

    - Optimisation parameters:

        - ``s``: the number of Monte Carlo simulations
        - ``beta``: the critical value

    - Output: ``weights``

    Public methods:

    - ``min_cvar()``
    - ``normalize_weights()``
    """

    def __init__(self, returns, weight_bounds=(0, 1)):
        """
        :param returns: asset historical returns
        :type returns: pd.DataFrame
        :param weight_bounds: minimum and maximum weight of an asset, defaults to (0, 1).
                              Must be changed to (-1, 1) for portfolios with shorting.
                              For CVaR opt, this is not a hard boundary.
        :type weight_bounds: tuple, optional
        :raises TypeError: if ``returns`` is not a dataframe
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")
        self.returns = returns
        self.tickers = returns.columns
        super().__init__(returns.shape[1], weight_bounds)  # bounds

    def min_cvar(self, s=10000, beta=0.95, random_state=None):
        """
        Find the portfolio weights that minimises the CVaR, via
        Monte Carlo sampling from the return distribution.

        :param s: number of bootstrap draws, defaults to 10000
        :type s: int, optional
        :param beta: "significance level" (i. 1 - q), defaults to 0.95
        :type beta: float, optional
        :param random_state: seed for random sampling, defaults to None
        :type random_state: int, optional
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: dict
        """
        args = (self.returns, s, beta, random_state)
        result = noisyopt.minimizeSPSA(
            objective_functions.negative_cvar,
            args=args,
            bounds=self.bounds,
            x0=self.initial_guess,
            niter=1000,
            paired=False,
        )
        self.weights = self.normalize_weights(result["x"])
        return dict(zip(self.tickers, self.weights))

    @staticmethod
    def normalize_weights(raw_weights):
        """
        Make all weights sum to 1

        :param raw_weights: input weights which do not sum to 1
        :type raw_weights: np.array, pd.Series
        :return: normalized weights
        :rtype: np.array, pd.Series
        """
        return raw_weights / raw_weights.sum()
