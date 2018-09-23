.. _post-processing:

#######################
Post-processing weights
#######################

After optimal weights have been generated, it is often necessary to do some
post-processing before they can be used practically. In particular, you are
likely using portfolio optimisation techniques to generate a
**portfolio allocation** – a list of tickers and corresponding integer quantities
that you could go and purchase at a broker.

However, it is not trivial to convert the continuous weights (output by any
optimisation) into an actionable allocation. In fact, owing to its
similarity to the `Knapsack Problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_,
I would guess that finding the optimal discrete allocation is an NP problem.

For example, let us say that we have $10,000 that we
would like to allocate. If we multiply the weights by this total portfolio
value, the result will be dollar amounts of each asset. So if the optimal weight
for Apple is 0.15, we need $1500 worth of Apple stock. However, Apple shares come
in discrete units ($190 at the time of writing), so we will not be able to buy
exactly $1500 of stock. The best we can do is to buy the number of shares that
gets us closest to the desired dollar value.

PyPortfolioOpt implements a greedy iterative algorithm, which proceeds in two 'rounds'.
In the first round, we buy as many shares as we can for each asset. In the Apple
example, :math:`1500/190 \approx 7.89`, so we buy 7 shares at a cost of $1330. After
iterating through all of the assets, we will have a lot of money left over (since
we always rounded down).

In the second round, we calculate how far the current weights deviate from the
existing weights for each asset. We wanted Apple to form 15% of the portfolio
(with total value $10,000), but we only bought $1330 worth of Apple stock, so
there is a deviation of :math:`0.15 - 0.133`. Some assets will have a higher
deviation from the ideal, so we will purchase shares of these first. We then
repeat the process, always buying shares of the asset whose current weight is
furthest away from the ideal weight. Though this algorithm will not guarantee
the optimal solution, I have found that it allows us to generate discrete
allocations with very little money left over.


.. warning::
    This does not support portfolios with short selling.

.. automodule:: pypfopt.discrete_allocation
    :members:
