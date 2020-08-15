"""
The ``expected_returns`` module provides functions for estimating the expected returns of
the assets, which is a required input in mean-variance optimisation.

By convention, the output of these methods is expected *annual* returns. It is assumed that
*daily* prices are provided, though in reality the functions are agnostic
to the time period (just change the ``frequency`` parameter). Asset prices must be given as
a pandas dataframe, as per the format described in the :ref:`user-guide`.

All of the functions process the price data into percentage returns data, before
calculating their respective estimates of expected returns.

Currently implemented:

    - general return model function, allowing you to run any return model from one function.
    - mean historical return
    - exponentially weighted mean historical return
    - James-Stein shrinkage
    - CAPM estimate of returns

Additionally, we provide utility functions to convert from returns to prices and vice-versa.
"""

import warnings
import pandas as pd
import numpy as np


def returns_from_prices(prices):
    """
    Calculate the returns given prices.

    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    return prices.pct_change().dropna(how="all")


def log_returns_from_prices(prices):
    """
    Calculate the log returns given prices.

    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    return np.log(1 + prices.pct_change()).dropna(how="all")


def prices_from_returns(returns):
    """
    Calculate the pseudo-prices given returns. These are not true prices because
    the initial prices are all set to 1, but it behaves as intended when passed
    to any PyPortfolioOpt method.

    :param returns: (daily) percentage returns of the assets
    :type returns: pd.DataFrame
    :return: (daily) pseudo-prices.
    :rtype: pd.DataFrame
    """
    ret = 1 + returns
    ret.iloc[0] = 1  # set first day pseudo-price
    return ret.cumprod()


def return_model(prices, method="mean_historical_return", **kwargs):
    """
    Compute an estimate of future returns, using the return model specified in ``method``.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param method: the return model to use. Should be one of:

        - ``mean_historical_return``
        - ``ema_historical_return``
        - ``james_stein_shrinkage``
        - ``capm_return``

    :type method: str, optional
    :raises NotImplementedError: if the supplied method is not recognised
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if method == "mean_historical_return":
        return mean_historical_return(prices, **kwargs)
    elif method == "ema_historical_return":
        return ema_historical_return(prices, **kwargs)
    elif method == "james_stein_shrinkage":
        return james_stein_shrinkage(prices, **kwargs)
    elif method == "capm_return":
        return capm_return(prices, **kwargs)
    else:
        raise NotImplementedError("Return model {} not implemented".format(method))


def mean_historical_return(
    prices, returns_data=False, compounding=False, frequency=252
):
    """
    Calculate annualised mean (daily) historical return from input (daily) asset prices.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param compounding: whether to properly compound the returns, optional.
    :type compounding: bool, defaults to False
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :return: annualised mean (daily) return for each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices)
    if compounding:
        return (1 + returns.mean()) ** frequency - 1
    else:
        return returns.mean() * frequency


def ema_historical_return(
    prices, returns_data=False, compounding=False, span=500, frequency=252
):
    """
    Calculate the exponentially-weighted mean of (daily) historical returns, giving
    higher weight to more recent data.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param compounding: whether to properly compound the returns, optional.
    :type compounding: bool, defaults to False
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param span: the time-span for the EMA, defaults to 500-day EMA.
    :type span: int, optional
    :return: annualised exponentially-weighted mean (daily) return of each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices)

    if compounding:
        return (1 + returns.ewm(span=span).mean().iloc[-1]) ** frequency - 1
    else:
        return returns.ewm(span=span).mean().iloc[-1] * frequency


def james_stein_shrinkage(prices, returns_data=False, compounding=False, frequency=252):
    r"""
    Compute the James-Stein shrinkage estimator, i.e

    .. math::

        \hat{\mu}_i^{JS} = \hat{\kappa} \bar{\mu} + (1-\hat{\kappa}) \mu_i,

    where :math:`\kappa` is the shrinkage parameter, :math:`\bar{\mu}` is the shrinkage
    target (grand average), and :math:`\mu` is the vector of mean returns.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param compounded: whether to properly compound the returns, optional.
    :type compounding: bool, defaults to False
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :return: James-Stein estimate of annualised return
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices)

    T, n = returns.shape
    mu = returns.mean(axis=0)
    mu_bar = mu.mean()
    sigma_squared = 1 / T * mu_bar * (1 - mu_bar)  # binomial estimate
    kappa = 1 - (n - 3) * sigma_squared / np.sum((mu - mu_bar) ** 2)
    theta_js = (1 - kappa) * mu + kappa * mu_bar

    if compounding:
        return (1 + theta_js) ** frequency - 1
    else:
        return theta_js * frequency


def capm_return(
    prices,
    market_prices=None,
    returns_data=False,
    risk_free_rate=0.02,
    compounding=False,
    frequency=252,
):
    """
    Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM,
    asset returns are equal to market returns plus a :math:`\beta` term encoding
    the relative risk of the asset.

    .. math::

        R_i = R_f + \beta_i (E(R_m) - R_f)


    :param prices: adjusted closing prices of the asset, each row is a date
                    and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param market_prices: adjusted closing prices of the benchmark, defaults to None
    :type market_prices: pd.DataFrame, optional
    :param returns_data: if true, the first arguments are returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                           You should use the appropriate time period, corresponding
                           to the frequency parameter.
    :type risk_free_rate: float, optional
    :param compounding: whether to properly compound the returns, optional.
    :type compounding: bool, defaults to False
    :param frequency: number of time periods in a year, defaults to 252 (the number
                        of trading days in a year)
    :type frequency: int, optional
    :return: annualised return estimate
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
        market_returns = market_prices
    else:
        returns = returns_from_prices(prices)
        if market_prices is not None:
            market_returns = returns_from_prices(market_prices)
        else:
            market_returns = None
    # Use the equally-weighted dataset as a proxy for the market
    if market_returns is None:
        # Append market return to right and compute sample covariance matrix
        returns["mkt"] = returns.mean(axis=1)

    else:
        market_returns.columns = ["mkt"]
        returns = returns.join(market_returns, how="left")

    # Compute covariance matrix for the new dataframe (including markets)
    cov = returns.cov()
    # The far-right column of the cov matrix is covariances to market
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")
    # Find mean market return on a given time period
    if compounding:
        mkt_mean_ret = (1 + returns["mkt"].mean()) ** frequency - 1
    else:
        mkt_mean_ret = returns["mkt"].mean() * frequency

    # CAPM formula
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)
