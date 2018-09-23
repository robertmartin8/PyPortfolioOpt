.. _other-optimisers:

################
Other Optimisers
################

In addition to optimisers that rely on the covariance matrix in the style of
Markowitz, recent developments in portfolio optimisation have seen a number
of alternative optimisation schemes.

Value-at-Risk
=============

The value-at-risk is a measure of tail risk that estimates how much a portfolio
will lose in a day with a given probability. Alternatively, it is the maximum
loss with a confidence of beta. In fact, a more useful measure is the
**expected shortfall**, or **conditional value-at-risk** (CVaR), which is the
mean of all losses so severe that they only occur with a probability
:math:`1-\beta`.

.. math::
    CVaR_\beta = \frac{1}{1-\beta} \int_0^{1-\beta} VaR_\gamma(X) d\gamma

To approximate the CVaR for a portfolio, we will follow these steps:

1. Generate the portfolio returns, i.e the weighted sum of individual asset returns.
2. Fit a Gaussian KDE to these returns, then resample.
3. Compute the value-at-risk as the :math:`1-\beta` quantile of sampled returns.
4. Calculate the mean of all the sample returns that are below the value-at-risk.

Though CVaR optimisation can be transformed into a linear programming problem [1]_, I
have opted to keep things simple using the `NoisyOpt <https://noisyopt.readthedocs.io/en/latest/>`_
library, which is suited for optimising noisy functions.

.. warning::
    Caveat emptor: this functionality is still experimental. Although I have
    used the CVaR optimisation, I've noticed that it is very inconsistent
    (which to some extent is expected because of its stochastic nature).
    However, the optimiser doesn't always find a minimum, and it fails
    silently. Additionally, the weight bounds are not treated as hard bounds.


.. automodule:: pypfopt.value_at_risk

    .. autoclass:: CVAROpt
        :members:

        .. automethod:: __init__

    .. caution::
        Currently, we have not implemented any performance function. If you
        would like to calculate the actual CVaR of the resulting portfolio,
        please import the function from `objective_functions`.


Hierarchical Risk Parity (HRP)
==============================

Hierarchical Risk Parity is a novel portfolio optimisation method developed by
Marcos Lopez de Prado [2]_. Though a detailed explanation can be found in the
linked paper, here is a rough overview of how HRP works:


1. From a universe of assets, form a distance matrix based on the correlation
   of the assets.
2. Using this distance matrix, cluster the assets into a tree via hierarchical
   clustering
3. Within each branch of the tree, form the minimum variance portfolio (normally
   between just two assets).
4. Iterate over each level, optimally combining the mini-portfolios at each node.


The advantages of this are that it does not require inversion of the covariance
matrix as with traditional quadratic optimisers, and seems to produce diverse
portfolios that perform well out of sample.


.. automodule:: pypfopt.hierarchical_risk_parity

    .. autofunction:: hrp_portfolio

.. note::
    Because the HRP functionality doesn't inherit from ``BaseOptimizer``, you will
    have to implement pre-processing and post-processing methods on your own.

References
==========

.. [1] Rockafellar and Uryasev (2011) `Optimization of conditional value-at-risk <http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf>`_.
.. [2] López de Prado, M. (2016). `Building Diversified Portfolios that Outperform Out of Sample <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_. The Journal of Portfolio Management, 42(4), 59–69.
