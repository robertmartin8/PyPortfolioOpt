"""
This module implements possible models for risk of a portfolio
"""
import pandas as pd
import warnings


def sample_cov(daily_returns):
    """
    Calculates the sample covariance matrix of daily returns, then annualises.
    :param daily_returns: Daily returns, each row is a date and each column is a ticker
    :type daily_returns: pd.DataFrame or array-like
    :returns: annualised sample covariance matrix of daily returns
    :rtype: pd.DataFrame
    """
    if not isinstance(daily_returns, pd.DataFrame):
        warnings.warn("daily_returns is not a dataframe", RuntimeWarning)
        daily_returns = pd.DataFrame(daily_returns)
    return daily_returns.cov() * 252
