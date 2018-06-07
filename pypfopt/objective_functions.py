""" Objective functions for optimisation methods

Implements the actual objective functions called by the EfficientFrontier object's
optimisation methods. These methods are primarily designed for internal use during optimisation
(via scipy.optimize), and require a certain signature (which is why they have not been factored into
a class).

Because scipy.optimize only minimises, any objectives that we want to maximise must be made negative.

Currently implemented:
- negative mean return
- negative Sharpe ratio
- volatility
"""

import numpy as np


def negative_mean_return(weights, expected_returns):
    """
    Calculate the negative mean return of a portfolio

    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param expected_returns: expected return of each asset
    :type expected_returns: pd.Series
    :return: negative mean return
    :rtype: float
    """
    return -weights.dot(expected_returns)


def negative_sharpe(
    weights, expected_returns, cov_matrix, gamma=0, risk_free_rate=0.02
):
    """
    Calculate the negative Sharpe ratio of a portfolio

    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param expected_returns: expected return of each asset
    :type expected_returns: pd.Series
    :param cov_matrix: the covariance matrix of asset returns
    :type cov_matrix: pd.DataFrame
    :param gamma: L2 regularisation parameter, defaults to 0. Increase if you want more
                    non-negligible weights
    :param gamma: float, optional
    :param risk_free_rate: risk free rate of borrowing/lending, defaults to 0.02
    :type risk_free_rate: float, optional
    :return: negative Sharpe ratio
    :rtype: float
    """
    mu = weights.dot(expected_returns)
    sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
    L2_reg = gamma * (weights ** 2).sum()
    return -(mu - risk_free_rate) / sigma + L2_reg


def volatility(weights, cov_matrix, gamma=0):
    """
    Calculate the volatility of a portfolio
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param cov_matrix: the covariance matrix of asset returns
    :type cov_matrix: pd.DataFrame
    :return: portfolio volatility
    :rtype: float
    """
    L2_reg = gamma * (weights ** 2).sum()
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) + L2_reg
