"""
This module implements possible models for the expected return.
It is assumed that daily returns are provided, though in reality the below methods are agnostic
to the time period (just changed the frequency parameter to annualise).
"""


def mean_historical_return(daily_returns, frequency=252):
    """
    Annualises mean daily historical return.
    :param daily_returns: Daily returns, each row is a date and each column is a ticker
    :type daily_returns: pd.DataFrame
    :param frequency: number of days (more generally, number of your desired time period)
    in a trading year, defaults to 252 days.
    :param frequency: int, optional
    :return: annualised mean daily return
    :rtype: pd.Series
    """
    return daily_returns.mean() * frequency


def ema_historical_return(daily_returns, frequency=252, span=500):
    """
    Annualised exponentially-weighted mean of daily historical return, giving
    higher weight to more recent data.
    :param daily_returns: Daily returns, each row is a date and each column is a ticker
    :type daily_returns: pd.DataFrame
    :param frequency: number of days (more generally, number of your desired time period)
                      in a trading year, defaults to 252 days.
    :param frequency: int, optional
    :param span: the time period for the EMA, defaults to 500-day EMA.
    :type span: int, optional
    :return: annualised exponentially-weighted mean daily return
    :rtype: pd.Series
    """
    return daily_returns.ewm(span=span).mean().iloc[-1] * frequency
