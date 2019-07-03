"""
The ``discrete_allocation`` module contains the ``DiscreteAllocation`` class, which
offers multile methods to generate a discrete portfolio allocation from continuous weights.
"""
import numpy as np
import pandas as pd
import pulp


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


class DiscreteAllocation:
    """
    Generate a discrete portfolio allocation from continuous weights

    Instance variables:

    - Inputs:

        - ``weights``
        - ``latest_prices``
        - ``min_allocation``
        - ``total_portfolio_value``
        - ``short_ratio``

    - Output: ``allocation``

    Public methods:

    - ``greedy_portfolio()``
    - ``lp_portfolio()``
    """

    def __init__(
        self,
        weights,
        latest_prices,
        min_allocation=0.01,
        total_portfolio_value=10000,
        short_ratio=0.30,
    ):
        """
        :param weights: continuous weights generated from the ``efficient_frontier`` module
        :type weights: dict
        :param latest_prices: the most recent price for each asset
        :type latest_prices: pd.Series or dict
        :param min_allocation: any weights less than this number are considered negligible,
                               defaults to 0.01
        :type min_allocation: float, optional
        :param total_portfolio_value: the desired total value of the portfolio, defaults to 10000
        :type total_portfolio_value: int/float, optional
        :param short_ratio: the short ratio, e.g 0.3 corresponds to 130/30
        :type short_ratio: float
        :raises TypeError: if ``weights`` is not a dict
        :raises TypeError: if ``latest_prices`` isn't a series
        :raises ValueError: if not ``0 < min_allocation < 0.3``
        :raises ValueError: if ``short_ratio < 0``
        """
        if not isinstance(weights, dict):
            raise TypeError("weights should be a dictionary of {ticker: weight}")
        if not isinstance(latest_prices, (pd.Series, dict)):
            raise TypeError("latest_prices should be a pd.Series")
        if min_allocation > 0.3:
            raise ValueError("min_allocation should be a small float")
        if total_portfolio_value <= 0:
            raise ValueError("total_portfolio_value must be greater than zero")
        if short_ratio <= 0:
            raise ValueError("short_ratio must be positive")

        # Drop any companies with negligible weights. Use a tuple because order matters.
        self.weights = [
            (k, v) for k, v in weights.items() if np.abs(v) > min_allocation
        ]
        print(
            "{} out of {} tickers were removed".format(
                len(weights) - len(self.weights), len(weights)
            )
        )
        self.latest_prices = latest_prices
        self.min_allocation = min_allocation
        self.total_portfolio_value = total_portfolio_value
        self.short_ratio = short_ratio

    def _allocation_rmse_error(self, verbose=True):
        """
        Utility function to calculate and print RMSE error between discretised
        weights and continuous weights. RMSE was usen instead of MAE because we
        want to penalise large variations.

        :param verbose: print weight discrepancies?
        :type verbose: bool
        :return: rmse error
        :rtype: float
        """
        portfolio_val = 0
        for ticker, num in self.allocation.items():
            portfolio_val += num * self.latest_prices[ticker]

        sse = 0  # sum of square errors
        for ticker, weight in self.weights:
            if ticker in self.allocation:
                allocation_weight = (
                    self.allocation[ticker] * self.latest_prices[ticker] / portfolio_val
                )
            else:
                allocation_weight = 0
            sse += (weight - allocation_weight) ** 2
            if verbose:
                print(
                    "{}: allocated {:.3f}, desired {:.3f}".format(
                        ticker, allocation_weight, weight
                    )
                )
        rmse = np.sqrt(sse / len(self.weights))
        print("Allocation has RMSE: {:.3f}".format(rmse))
        return rmse

    def greedy_portfolio(self, verbose=False):
        """
        Convert continuous weights into a discrete portfolio allocation
        using a greedy iterative approach.

        :param verbose: print error analysis?
        :type verbose: bool
        :return: the number of shares of each ticker that should be purchased,
                 along with the amount of funds leftover.
        :rtype: (dict, float)
        """
        # Sort in descending order of weight
        self.weights.sort(key=lambda x: x[1], reverse=True)

        # If portfolio contains shorts
        if self.weights[-1][1] < 0:
            longs = {t: w for t, w in self.weights if w >= 0}
            shorts = {t: -w for t, w in self.weights if w < 0}

            # Make them sum to one
            long_total_weight = sum(longs.values())
            short_total_weight = sum(shorts.values())
            longs = {t: w / long_total_weight for t, w in longs.items()}
            shorts = {t: w / short_total_weight for t, w in shorts.items()}

            # Construct long-only discrete allocations for each
            short_val = self.total_portfolio_value * self.short_ratio

            print("\nAllocating long sub-portfolio:")
            da1 = DiscreteAllocation(
                longs,
                self.latest_prices[longs.keys()],
                min_allocation=0,
                total_portfolio_value=self.total_portfolio_value,
            )
            long_alloc, long_leftover = da1.greedy_portfolio()

            print("\nAllocating short sub-portfolio:")
            da2 = DiscreteAllocation(
                shorts,
                self.latest_prices[shorts.keys()],
                min_allocation=0,
                total_portfolio_value=short_val,
            )
            short_alloc, short_leftover = da2.greedy_portfolio()
            short_alloc = {t: -w for t, w in short_alloc.items()}

            # Combine and return
            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            return self.allocation, long_leftover + short_leftover

        # Otherwise, portfolio is long only and we proceed with greedy algo
        available_funds = self.total_portfolio_value
        shares_bought = []
        buy_prices = []

        # First round
        for ticker, weight in self.weights:
            price = self.latest_prices[ticker]
            # Attempt to buy the lower integer number of shares
            n_shares = int(weight * self.total_portfolio_value / price)
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
            ideal_weights = np.array([i[1] for i in self.weights])
            deficit = ideal_weights - current_weights

            # Attempt to buy the asset whose current weights deviate the most
            idx = np.argmax(deficit)
            ticker, weight = self.weights[idx]
            price = self.latest_prices[ticker]

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

                ticker, weight = self.weights[idx]
                price = self.latest_prices[ticker]
                counter += 1

            if deficit[idx] <= 0 or counter == 10:
                # Dirty solution to break out of both loops
                break

            # Buy one share at a time
            shares_bought[idx] += 1
            available_funds -= price

        self.allocation = dict(zip([i[0] for i in self.weights], shares_bought))

        if verbose:
            print("Funds remaining: {:.2f}".format(available_funds))
            self._allocation_rmse_error(verbose)
        return self.allocation, available_funds

    def lp_portfolio(self, verbose=False):
        """
        Convert continuous weights into a discrete portfolio allocation
        using integer programming.

        :param verbose: print error analysis?
        :type verbose: bool
        :return: the number of shares of each ticker that should be purchased, along with the amount
                of funds leftover.
        :rtype: (dict, float)
        """

        if any([w < 0 for _, w in self.weights]):
            longs = {t: w for t, w in self.weights if w >= 0}
            shorts = {t: -w for t, w in self.weights if w < 0}

            # Make them sum to one
            long_total_weight = sum(longs.values())
            short_total_weight = sum(shorts.values())
            longs = {t: w / long_total_weight for t, w in longs.items()}
            shorts = {t: w / short_total_weight for t, w in shorts.items()}

            # Construct long-only discrete allocations for each
            short_val = self.total_portfolio_value * self.short_ratio

            print("\nAllocating long sub-portfolio:")
            da1 = DiscreteAllocation(
                longs,
                self.latest_prices[longs.keys()],
                min_allocation=0,
                total_portfolio_value=self.total_portfolio_value,
            )
            long_alloc, long_leftover = da1.lp_portfolio()

            print("\nAllocating short sub-portfolio:")
            da2 = DiscreteAllocation(
                shorts,
                self.latest_prices[shorts.keys()],
                min_allocation=0,
                total_portfolio_value=short_val,
            )
            short_alloc, short_leftover = da2.lp_portfolio()
            short_alloc = {t: -w for t, w in short_alloc.items()}

            # Combine and return
            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            return self.allocation, long_leftover + short_leftover

        opt = pulp.LpProblem("PfAlloc", pulp.LpMinimize)
        vals = {}
        realvals = {}
        etas = {}
        abss = {}
        remaining = pulp.LpVariable("remaining", 0)
        for k, w in self.weights:
            # Each ticker is an optimisation variable
            vals[k] = pulp.LpVariable("x_" + k, 0, cat="Integer")
            # Allocated weights
            realvals[k] = self.latest_prices[k] * vals[k]
            # Deviation between allocated and ideal weights
            etas[k] = w * self.total_portfolio_value - realvals[k]
            abss[k] = pulp.LpVariable("u_" + k, 0)
            # Constraints
            opt += etas[k] <= abss[k]
            opt += -etas[k] <= abss[k]

        # Constraint: fixed total value
        opt += remaining + pulp.lpSum(realvals.values()) == self.total_portfolio_value

        # Objective function: leftover + sum of abs errors
        opt += pulp.lpSum(abss.values()) + remaining
        opt.solve()
        self.allocation = {k: int(val.varValue) for k, val in vals.items()}
        if verbose:
            print("Funds remaining: {:.2f}".format(remaining.varValue))
            self._allocation_rmse_error()
        return self.allocation, remaining.varValue
