"""
This module implements possible models for the expected return.
It is assumed that daily returns are provided, though in reality the below methods are agnostic
to the time period (just changed the frequency parameter to annualise).
"""
import warnings
import pandas as pd


def mean_historical_return(prices, frequency=252):
    """
    Calculates annualised mean daily historical return from input daily stock prices.
    If data is not daily, change the frequency.
    :param prices: daily adjusted closing prices of the asset. Each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param frequency: number of days (more generally, number of your desired time period)
    in a trading year, defaults to 252 days.
    :param frequency: int, optional
    :return: annualised mean daily return
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
    :param prices: daily adjusted closing prices of the asset. Each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param frequency: number of days (more generally, number of your desired time period)
                      in a trading year, defaults to 252 days.
    :param frequency: int, optional
    :param span: the time period for the EMA, defaults to 500-day EMA.
    :type span: int, optional
    :return: annualised exponentially-weighted mean daily return
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    daily_returns = prices.pct_change().dropna(how="all")
    return daily_returns.ewm(span=span).mean().iloc[-1] * frequency
