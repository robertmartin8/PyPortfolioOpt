""" Given historical asset price data, calculate expected returns.

It is assumed that daily returns are provided, though in reality the methods are agnostic
to the time period (just changed the frequency parameter to annualise).

Currently implemented:
- mean historical return
- exponentially weighted mean historical return
"""

import warnings
import pandas as pd


def mean_historical_return(prices, frequency=252):
    """
    Calculate annualised mean daily historical return from input (daily) asset prices.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param frequency: number of time periods in a year, defaults to 252
                      (trading days in a year).
    :param frequency: int, optional
    :return: annualised mean daily return for each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    daily_returns = prices.pct_change().dropna(how="all")
    return daily_returns.mean() * frequency


def ema_historical_return(prices, frequency=252, span=500):
    """
    Annualised exponentially-weighted mean of daily historical return, giving
    higher weight to more recent data.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param frequency: number of time periods in a year, defaults to 252
                      (trading days in a year).
    :param frequency: int, optional
    :param span: the time-span for the EMA, defaults to 500-day EMA.
    :type span: int, optional
    :return: annualised exponentially-weighted mean daily return of each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    daily_returns = prices.pct_change().dropna(how="all")
    return daily_returns.ewm(span=span).mean().iloc[-1] * frequency
