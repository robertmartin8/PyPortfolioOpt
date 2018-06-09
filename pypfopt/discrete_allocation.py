"""
The ``discrete_allocation`` module contains functions to generate a discrete
allocation from continuous weights.
"""

import numpy as np
import pandas as pd


def get_latest_prices(prices):
    """
    A helper tool which retrieves the most recent asset prices from a dataframe of
    asset prices, required in order to generate a discrete allocation.

    :param prices: historical asset prices
    :type prices: pd.DataFrame
    :raises TypeError: if prices are not in a dataframe
    :return: the most recent price of each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices not in a dataframe")
    return prices.ffill().iloc[-1]


def portfolio(weights, latest_prices, min_allocation=0.01, total_portfolio_value=10000):
    """
    For a long only portfolio, convert the continuous weights to a discrete allocation
    in a greedy iterative approach. This can be thought of as a clever way to round
    the continuous weights to an integer number of shares

    :param weights: continuous weights generated from the ``efficient_frontier`` module
    :type weights: dict
    :param latest_prices: the most recent price for each asset
    :type latest_prices: pd.Series or dict
    :param min_allocation: any weights less than this number are considered negligible,
                           defaults to 0.01
    :type min_allocation: float, optional
    :param total_portfolio_value: the desired total value of the portfolio, defaults to 10000
    :type total_portfolio_value: int/float, optional
    :raises TypeError: if ``weights`` is not a dict
    :raises TypeError: if ``latest_prices`` isn't a series
    :raises ValueError: if ``0 < min_allocation < 0.3``
    :return: the number of shares of each ticker that should be purchased, along with the amount
             of funds leftover.
    :rtype: (dict, float)
    """
    if not isinstance(weights, dict):
        raise TypeError("weights should be a dictionary of {ticker: weight}")
    if not isinstance(latest_prices, (pd.Series, dict)):
        raise TypeError("latest_prices should be a pd.Series")
    if min_allocation > 0.3:
        raise ValueError("min_allocation should be a small float")
    if total_portfolio_value <= 0:
        raise ValueError("total_portfolio_value must be greater than zero")

    # Drop any companies with negligible weights. We use a tuple because order matters.
    nonzero_weights = [(k, v) for k, v in weights.items() if v > min_allocation]
    print(
        "{} out of {} tickers were removed".format(
            len(weights) - len(nonzero_weights), len(weights)
        )
    )
    # Sort in descending order of weight
    nonzero_weights.sort(key=lambda x: x[1], reverse=True)
    available_funds = total_portfolio_value
    shares_bought = []
    buy_prices = []

    # First round
    for ticker, weight in nonzero_weights:
        price = latest_prices[ticker]
        # Attempt to buy the lower integer number of shares
        n_shares = int(weight * total_portfolio_value / price)
        cost = n_shares * price
        if cost > available_funds:
            # Buy as many as possible
            n_shares = available_funds // price
            if n_shares == 0:
                print("Insufficient funds")
        available_funds -= cost
        shares_bought.append(n_shares)
        buy_prices.append(price)

    # Second round
    while available_funds > 0:
        # Calculate the equivalent continuous weights of the shares that
        # have already been bought
        current_weights = np.array(buy_prices) * np.array(shares_bought)
        current_weights /= current_weights.sum()
        ideal_weights = np.array([i[1] for i in nonzero_weights])
        deficit = ideal_weights - current_weights

        # Attempt to buy the asset whose current weights deviate the most
        idx = np.argmax(deficit)
        ticker, weight = nonzero_weights[idx]
        price = latest_prices[ticker]

        # If we can't afford this asset, search for the next highest deficit that we
        # can purchase.
        counter = 0
        while price > available_funds:
            deficit[idx] = 0  # we can no longer purchase the asset at idx
            idx = np.argmax(deficit)  # find the next most deviant asset

            # If either of these conditions is met, we break out of both while loops
            # hence the repeated statement below
            if deficit[idx] < 0 or counter == 10:
                break

            ticker, weight = nonzero_weights[idx]
            price = latest_prices[ticker]
            counter += 1

        if deficit[idx] <= 0 or counter == 10:
            # See https://stackoverflow.com/questions/189645/
            break

        # Buy one share at a time
        shares_bought[idx] += 1
        available_funds -= price

    print("Funds remaining: {:.2f}".format(available_funds))

    num_shares = dict(zip([i[0] for i in nonzero_weights], shares_bought))
    return num_shares, available_funds
