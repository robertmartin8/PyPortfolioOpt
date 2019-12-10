.. _expected-returns:

################
Expected Returns
################

Mean-variance optimisation requires knowledge of the expected returns. In practice,
these are rather difficult to know with any certainty. Thus the best we can do is to
come up with estimates, for example by extrapolating historical data, This is where the
main flaw in efficient frontier lies – the optimisation procedure is sound, and provides
strong mathematical guarantees, *given the correct inputs*. This is one of the reasons
why I have emphasised modularity: users should be able to come up with their own
superior models and feed them into the optimiser.

.. caution::

    In my experience, supplying expected returns often does more harm than good. If
    predicting stock returns were as easy as calculating the mean historical return,
    we'd all be rich! For most use-cases, I would suggest that you focus your efforts
    on choosing an appropriate risk model (see :ref:`risk-models`). 

    As of v0.5.0, you can use :ref:`black-litterman` to greatly improve the quality of
    your expected returns estimate.

.. automodule:: pypfopt.expected_returns

    .. autofunction:: mean_historical_return

        This is probably the default textbook approach. It is intuitive and easily interpretable,
        however the estimates are unlikely to be accurate. This is a problem especially in the
        context of a quadratic optimiser, which will maximise the erroneous inputs, In some informal
        backtests, I've found that vanilla efficient frontier portfolios (using mean historical
        returns and sample covariance) actually do have a statistically significant outperformance
        over the S&P500 (in the order of 3-5%), though the same isn't true for cryptoasset portfolios.
        At some stage, I may redo these backtests rigorously and add them to the repo
        (see the :ref:`roadmap` page for more).


    .. autofunction:: ema_historical_return

        The exponential moving average is a simple improvement over the mean historical
        return; it gives more credence to recent returns and thus aims to increase the relevance
        of the estimates. This is parameterised by the ``span`` parameter, which gives users
        the ability to decide exactly how much more weight is given to recent data.
        Generally, I would err on the side of a higher span – in the limit, this tends towards
        the mean historical return. However, if you plan on rebalancing much more frequently,
        there is a case to be made for lowering the span in order to capture recent trends.

    .. autofunction:: returns_from_prices

    .. autofunction:: prices_from_returns
