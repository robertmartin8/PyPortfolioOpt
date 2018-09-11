# import warnings
# import numpy as np
# import pandas as pd
from .base_optimizer import BaseOptimizer
from . import objective_functions
import noisyopt


class CVAROpt(BaseOptimizer):

    def __init__(self, returns, weight_bounds=(0, 1)):
        # TODO documentation and type checks
        self.returns = returns
        self.tickers = returns.columns
        super().__init__(returns.shape[1], weight_bounds)

    def min_cvar(self, s=10000, beta=0.95, random_state=None):
        args = (self.returns, s, beta, random_state)
        result = noisyopt.minimizeSPSA(
            objective_functions.negative_cvar,
            args=args,
            bounds=self.bounds,
            x0=self.initial_guess,
            niter=1000,
            paired=False,
        )
        self.weights = self.post_process_weights(result["x"])
        return dict(zip(self.tickers, self.weights))

    @staticmethod
    def post_process_weights(raw_weights):
        # must manually make weights sum to 1
        return raw_weights / raw_weights.sum()
