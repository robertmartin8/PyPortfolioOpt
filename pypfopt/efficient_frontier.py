"""
The ``efficient_frontier`` module houses the EfficientFrontier class, which
generates optimal portfolios for various possible objective functions and parameters.
"""

import warnings
import numpy as np
import pandas as pd
import cvxpy as cp

from . import objective_functions, base_optimizer


class EfficientFrontier(base_optimizer.BaseConvexOptimizer):

    """
    An EfficientFrontier object (inheriting from BaseConvexOptimizer) contains multiple
    optimisation methods that can be called (corresponding to different objective
    functions) with various parameters. Note: a new EfficientFrontier object should
    be instantiated if you want to make any change to objectives/constraints/bounds/parameters.

    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``bounds`` - float tuple OR (float tuple) list
        - ``cov_matrix`` - np.ndarray
        - ``expected_returns`` - np.ndarray
        - ``solver`` - str

    - Output: ``weights`` - np.ndarray

    Public methods:

    - ``max_sharpe()`` optimises for maximal Sharpe ratio (a.k.a the tangency portfolio)
    - ``min_volatility()`` optimises for minimum volatility
    - ``max_quadratic_utility()`` maximises the quadratic utility, given some risk aversion.
    - ``efficient_risk()`` maximises Sharpe for a given target risk
    - ``efficient_return()`` minimises risk for a given target return

    - ``add_objective()`` adds a (convex) objective to the optimisation problem
    - ``add_constraint()`` adds a (linear) constraint to the optimisation problem
    - ``convex_objective()`` solves for a generic convex objective with linear constraints
    - ``nonconvex_objective()`` solves for a generic nonconvex objective using the scipy backend.
      This is prone to getting stuck in local minima and is generally *not* recommended.

    - ``portfolio_performance()`` calculates the expected return, volatility and Sharpe ratio for
      the optimised portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, expected_returns, cov_matrix, weight_bounds=(0, 1), gamma=0):
        """
        :param expected_returns: expected returns for each asset. Can be None if
                                optimising for volatility only (but not recommended).
        :type expected_returns: pd.Series, list, np.ndarray
        :param cov_matrix: covariance of returns for each asset. This **must** be
                           positive semidefinite, otherwise optimisation will fail.
        :type cov_matrix: pd.DataFrame or np.array
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param gamma: L2 regularisation parameter, defaults to 0. Increase if you want more
                      non-negligible weights
        :type gamma: float, optional
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        :raises TypeError: if ``cov_matrix`` is not a dataframe or array
        """
        # Inputs
        self.cov_matrix = EfficientFrontier._validate_cov_matrix(cov_matrix)
        self.expected_returns = EfficientFrontier._validate_expected_returns(
            expected_returns
        )

        # Labels
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:  # use integer labels
            tickers = list(range(len(expected_returns)))

        if expected_returns is not None:
            if cov_matrix.shape != (len(expected_returns), len(expected_returns)):
                raise ValueError("Covariance matrix does not match expected returns")

        super().__init__(len(tickers), tickers, weight_bounds)

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

    def min_volatility(self):
        """
        Minimise volatility.

        :return: asset weights for the volatility-minimising portfolio
        :rtype: OrderedDict
        """
        self._objective = objective_functions.portfolio_variance(
            self._w, self.cov_matrix
        )
        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints.append(cp.sum(self._w) == 1)

        return self._solve_cvxpy_opt_problem()

    def max_sharpe(self, risk_free_rate=0.02):
        """
        Maximise the Sharpe Ratio. The result is also referred to as the tangency portfolio,
        as it is the portfolio for which the capital market line is tangent to the efficient frontier.

        This is a convex optimisation problem after making a certain variable substitution. See
        `Cornuejols and Tutuncu (2006) <http://web.math.ku.dk/~rolf/CT_FinOpt.pdf>`_ for more.

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: OrderedDict
        """
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        # max_sharpe requires us to make a variable transformation.
        # Here we treat w as the transformed variable.
        self._objective = cp.quad_form(self._w, self.cov_matrix)
        k = cp.Variable()

        # Note: objectives are not scaled by k. Hence there are subtle differences
        # between how these objectives work for max_sharpe vs min_volatility
        if len(self._additional_objectives) > 0:
            warnings.warn(
                "max_sharpe transforms the optimisation problem so additional objectives may not work as expected."
            )
        for obj in self._additional_objectives:
            self._objective += obj

        new_constraints = []
        # Must rebuild the constraints
        for constr in self._constraints:
            if isinstance(constr, cp.constraints.nonpos.Inequality):
                # Either the first or second item is the expression
                if isinstance(
                    constr.args[0], cp.expressions.constants.constant.Constant
                ):
                    new_constraints.append(constr.args[1] >= constr.args[0] * k)
                else:
                    new_constraints.append(constr.args[0] <= constr.args[1] * k)
            elif isinstance(constr, cp.constraints.zero.Equality):
                new_constraints.append(constr.args[0] == constr.args[1] * k)
            else:
                raise TypeError(
                    "Please check that your constraints are in a suitable format"
                )

        # Transformed max_sharpe convex problem:
        self._constraints = [
            (self.expected_returns - risk_free_rate).T @ self._w == 1,
            cp.sum(self._w) == k,
            k >= 0,
        ] + new_constraints

        self._solve_cvxpy_opt_problem()
        # Inverse-transform
        self.weights = (self._w.value / k.value).round(16) + 0.0
        return self._make_output_weights()

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        r"""
        Maximise the given quadratic utility, i.e:

        .. math::

            \max_w w^T \mu - \frac \delta 2 w^T \Sigma w

        :param risk_aversion: risk aversion parameter (must be greater than 0),
                              defaults to 1
        :type risk_aversion: positive float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :return: asset weights for the maximum-utility portfolio
        :rtype: OrderedDict
        """
        if risk_aversion <= 0:
            raise ValueError("risk aversion coefficient must be greater than zero")

        self._objective = objective_functions.quadratic_utility(
            self._w, self.expected_returns, self.cov_matrix, risk_aversion=risk_aversion
        )
        for obj in self._additional_objectives:
            self._objective += obj

        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_volatility, market_neutral=False):
        """
        Maximise return for a target risk. The resulting portfolio will have a volatility
        less than the target (but not guaranteed to be equal).

        :param target_volatility: the desired maximum volatility of the resulting portfolio.
        :type target_volatility: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :raises ValueError: if ``target_volatility`` is not a positive float
        :raises ValueError: if no portfolio can be found with volatility equal to ``target_volatility``
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the efficient risk portfolio
        :rtype: OrderedDict
        """
        if not isinstance(target_volatility, (float, int)) or target_volatility < 0:
            raise ValueError("target_volatility should be a positive float")

        self._objective = objective_functions.portfolio_return(
            self._w, self.expected_returns
        )
        variance = objective_functions.portfolio_variance(self._w, self.cov_matrix)

        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints.append(variance <= target_volatility ** 2)

        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        """
        Calculate the 'Markowitz portfolio', minimising volatility for a given target return.

        :param target_return: the desired return of the resulting portfolio.
        :type target_return: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :type market_neutral: bool, optional
        :raises ValueError: if ``target_return`` is not a positive float
        :raises ValueError: if no portfolio can be found with return equal to ``target_return``
        :return: asset weights for the Markowitz portfolio
        :rtype: OrderedDict
        """
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_return should be a positive float")
        if target_return > self.expected_returns.max():
            raise ValueError(
                "target_return must be lower than the largest expected return"
            )

        self._objective = objective_functions.portfolio_variance(
            self._w, self.cov_matrix
        )
        ret = objective_functions.portfolio_return(
            self._w, self.expected_returns, negative=False
        )

        self.objective = cp.quad_form(self._w, self.cov_matrix)
        ret = self.expected_returns.T @ self._w

        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints.append(ret >= target_return)

        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            self._market_neutral_bounds_check()
            self._constraints.append(cp.sum(self._w) == 0)
        else:
            self._constraints.append(cp.sum(self._w) == 1)

        return self._solve_cvxpy_opt_problem()

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
