Expected Returns
================

Mean-variance optimisation requries knowledge of the mean returns. In practice, this is
of course impossible to know – if we knew the expected return of a stock, life would be
much easier. Thus the best we can do is to come up with estimates; one way is to
extrapolate historical price data. This is where the main flaw in efficient frontier
lies – the optimisation procedure is sound, and provides strong mathematical guarantees,
*given the correct inputs*. This is one of the reasons why I have emphasised modularity:
users should be able to come up with their own superior models and feed them into the
optimiser.

.. automodule:: expected_returns

    .. autofunction:: mean_historical_return

        This is probably the default textbook approach. It is intuitive and easily interpretable,
        however the estimates are unlikely to be accurate. That being said, one of the advantages
        of efficient frontier is that the estimation error is reduced by having multiple assets, so
        perhaps this inaccuracy is not such an issue. In some informal backtests, I've found
        that vanilla efficient frontier portfolios (using mean historical returns and sample covariance)
        actually do have a statistically significant outperformance over the S&P500 (in the order of
        3-5%). At some stage, I may redo these backtests rigorously and add them to the repo
        (see the :ref:`roadmap` page for more).


    .. autofunction:: ema_historical_return

        Using the exponential moving average is a simple improvement over the mean historical
        return; it gives more credence to recent returns and thus aims to increase the relevance
        of the estimates. This is paramaterised by the ``span`` parameter, which gives users
        the ability to decide exactly how much more weight is given to recent data.
        Generally, I would err on the side of a higher span - in the limit, this tends towards
        the mean historical return. However, if you plan on rebalancing much more frequently,
        there is a case to be made for lowering the span in order to capture recent trends.
