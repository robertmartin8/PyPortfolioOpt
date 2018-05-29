"""
This model implements possible objective functions for efficient optimisation

:return: [description]
:rtype: [type]
"""
import numpy as np


def negative_mean_return(weights, expected_returns):
    """
    Negative mean return of a portfolio
    :param weights: normalised weights
    :type weights: np.array
    :param expected_returns: mean returns of the assets
    :type expected_returns: pd.Series
    :return: negative mean return
    :rtype: float
    """
    return -weights.dot(expected_returns)


def negative_sharpe(
    weights, expected_returns, cov_matrix, alpha=0, risk_free_rate=0.02
):
    """
    Negative Sharpe Ratio of a given portfolio

    :param weights: normalised weights
    :param expected_returns: mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :param risk_free_rate: defaults to zero
    :return: the negative Sharpe ratio
    """
    mu = weights.dot(expected_returns)
    sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
    L2_reg = alpha * (weights ** 2).sum()
    return -(mu - risk_free_rate) / sigma + L2_reg


def volatility(weights, cov_matrix, alpha=0):
    """
    Volatility of a given portfolio
    :param weights: normalised weights
    :param cov_matrix: covariance of these stocks.
    :return:
    """
    L2_reg = alpha * (weights ** 2).sum()
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) + L2_reg
