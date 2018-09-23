# TODO module docstring

import numpy as np


class BaseOptimizer:
    def __init__(self, n_assets, weight_bounds=(0, 1)):
        """
        :param weight_bounds: minimum and maximum weight of an asset, defaults to (0, 1).
                              Must be changed to (-1, 1) for portfolios with shorting.
        :type weight_bounds: tuple, optional
        """
        self.n_assets = n_assets
        self.bounds = self._make_valid_bounds(weight_bounds)
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
