"""
The ``value_at_risk`` module allows for optimisation with a (conditional)
value-at-risk (CVaR) objective, which requires Monte Carlo simulation.
"""

import pandas as pd
from . import base_optimizer
from . import objective_functions

# Extra dependency
try:
    import noisyopt
except (ModuleNotFoundError, ImportError):
    raise ImportError("Please install noisyopt via pip or poetry")


class CVAROpt(base_optimizer.BaseScipyOptimizer):

    """
    A CVAROpt object (inheriting from BaseScipyOptimizer) provides a method for
    optimising the CVaR (a.k.a expected shortfall) of a portfolio.

    Instance variables:

    - Inputs

        - ``tickers`` - str list
        - ``returns`` - pd.DataFrame
        - ``bounds`` - float tuple OR (float tuple) list

    - Optimisation parameters:

        - ``s`` - int (the number of Monte Carlo simulations)
        - ``beta`` - float (the critical value)

    - Output: ``weights`` - np.ndarray

    Public methods:

    - ``min_cvar()``
    - ``normalize_weights()``
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, returns, weight_bounds=(0, 1)):
        """
        :param returns: asset historical returns
        :type returns: pd.DataFrame
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :raises TypeError: if ``returns`` is not a dataframe
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")
        self.returns = returns
        tickers = returns.columns
        super().__init__(len(tickers), tickers, weight_bounds)

    @staticmethod
    def _normalize_weights(raw_weights):
        """
        Utility function to make all weights sum to 1

        :param raw_weights: input weights which do not sum to 1
        :type raw_weights: np.array, pd.Series
        :return: normalized weights
        :rtype: np.array, pd.Series
        """
        return raw_weights / raw_weights.sum()

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
        self.weights = CVAROpt._normalize_weights(result["x"])
        return dict(zip(self.tickers, self.weights))
