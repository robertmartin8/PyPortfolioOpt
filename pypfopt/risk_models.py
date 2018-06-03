"""
This module implements possible models for risk of a portfolio
"""
import warnings
import numpy as np
import pandas as pd
from sklearn import covariance


def sample_cov(prices, frequency=252):
    """
    Calculates the sample covariance matrix of daily returns, then annualises.
    If data is not daily, change the frequency.
    :param daily_returns: Daily returns, each row is a date and each column is a ticker
    :type daily_returns: pd.DataFrame or array-like
    :returns: annualised sample covariance matrix of daily returns
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    daily_returns = prices.pct_change().dropna(how="all")

    return daily_returns.cov() * frequency


def min_cov_determinant(prices, frequency=252, random_state=None):
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    X = prices.pct_change().dropna(how="all")
    X = np.nan_to_num(X.values)
    raw_cov_array = covariance.fast_mcd(X, random_state=random_state)[1]
    return pd.DataFrame(raw_cov_array, index=assets, columns=assets) * frequency


class CovarianceShrinkage:
    """
        The regularised covariance is::
        (1 - shrinkage)*cov
                + shrinkage*mu*np.identity(n_features)
    :return: [description]
    :rtype: [type]
    """

    def __init__(self, prices, frequency=252):
        if not isinstance(prices, pd.DataFrame):
            warnings.warn("prices are not in a dataframe", RuntimeWarning)
            prices = pd.DataFrame(prices)
        self.frequency = frequency
        self.X = prices.pct_change().dropna(how="all")
        self.S = self.X.cov().values
        self.delta = None  # shrinkage constant

    def format_and_annualise(self, raw_cov_array):
        assets = self.X.columns
        return (
            pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        )

    def shrunk_covariance(self, delta=0.2):
        self.delta = delta
        N = self.S.shape[1]
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu  # shrinkage target
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self.format_and_annualise(shrunk_cov)

    def ledoit_wolf(self):
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = covariance.ledoit_wolf(X)
        return self.format_and_annualise(shrunk_cov)

    def oracle_approximating(self):
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = covariance.oas(X)
        return self.format_and_annualise(shrunk_cov)
