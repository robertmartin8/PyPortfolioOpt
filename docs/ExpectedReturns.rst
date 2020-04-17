.. _expected-returns:

################
Expected Returns
################

Mean-variance optimisation requires knowledge of the expected returns. In practice,
these are rather difficult to know with any certainty. Thus the best we can do is to
come up with estimates, for example by extrapolating historical data, This is the
main flaw in mean-variance optimisation – the optimisation procedure is sound, and provides
strong mathematical guarantees, *given the correct inputs*. This is one of the reasons
why I have emphasised modularity: users should be able to come up with their own
superior models and feed them into the optimiser.

.. caution::

    In my experience, supplying expected returns often does more harm than good. If
    predicting stock returns were as easy as calculating the mean historical return,
    we'd all be rich! For most use-cases, I would suggest that you focus your efforts
    on choosing an appropriate risk model (see :ref:`risk-models`). 

    As of v0.5.0, you can use :ref:`black-litterman` to greatly improve the quality of
    your estimate of the expected returns.

.. automodule:: pypfopt.expected_returns

    .. note::

        For any of these methods, if you would prefer to pass returns (the default is prices),
        set the boolean flag ``returns_data=True``

    .. autofunction:: mean_historical_return

        This is probably the default textbook approach. It is intuitive and easily interpretable,
        however the estimates are subject to large uncertainty. This is a problem especially in the
        context of a quadratic optimiser, which will maximise the erroneous inputs.


    .. autofunction:: ema_historical_return

        The exponential moving average is a simple improvement over the mean historical
        return; it gives more credence to recent returns and thus aims to increase the relevance
        of the estimates. This is parameterised by the ``span`` parameter, which gives users
        the ability to decide exactly how much more weight is given to recent data.
        Generally, I would err on the side of a higher span – in the limit, this tends towards
        the mean historical return. However, if you plan on rebalancing much more frequently,
        there is a case to be made for lowering the span in order to capture recent trends.

    .. autofunction:: james_stein_shrinkage

        A surprising result in statistics is that the MLE estimator for a 3+ dimensional
        Normal distribution is an *inadmissible* estimator. That is, there exists an estimator
        :math:`\hat{\mu}^{JS}` such that:

        .. math::

            E\{ \lVert \hat{\mu}^{JS} - \mu \rVert^2 \} < E \{ \lVert \bar{\mu} - \mu \rVert^2 \}

        In essence, to reduce account for the fact that our sample may not be representative and
        thus reduce loss, we shrink the sample means to the "grand average" (mean of means). For
        a more detailed explanation, refer to Efron and Hastie (2010) [1]_

    .. autofunction:: capm_return

    .. autofunction:: returns_from_prices

    .. autofunction:: log_returns_from_prices

    .. autofunction:: prices_from_returns


References
==========

.. [1] Efron and Hastie (2010) `Empirical Bayes and the James–Stein Estimator <http://statweb.stanford.edu/~ckirby/brad/LSI/chapter1.pdf>`_.
