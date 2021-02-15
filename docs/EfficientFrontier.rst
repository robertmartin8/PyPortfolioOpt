.. _efficient-frontier:

###############################
Efficient Frontier Optimisation
###############################

Mathematical optimisation is a very difficult problem in general, particularly when we are dealing
with complex objectives and constraints. However, **convex optimisation** problems are a well-understood
class of problems, which happen to be incredibly useful for finance. A convex problem has the following form:

.. math::

    \begin{equation*}
    \begin{aligned}
    & \underset{\mathbf{x}}{\text{minimise}} & & f(\mathbf{x}) \\
    & \text{subject to} & & g_i(\mathbf{x}) \leq 0, i = 1, \ldots, m\\
    &&& A\mathbf{x} = b,\\
    \end{aligned}
    \end{equation*}

where :math:`\mathbf{x} \in \mathbb{R}^n`, and :math:`f(\mathbf{x}), g_i(\mathbf{x})` are convex functions. [1]_

Fortunately, portfolio optimisation problems (with standard objectives and constraints) are convex. This
allows us to immediately apply the vast body of theory as well as the refined solving routines -- accordingly,
the main difficulty is inputting our specific problem into a solver.

PyPortfolioOpt aims to do the hard work for you, allowing for one-liners like ``ef.min_volatility()``
to generate a portfolio that minimises the volatility, while at the same time allowing for more
complex problems to be built up from modular units. This is all possible thanks to
`cvxpy <https://www.cvxpy.org/>`_, the *fantastic* python-embedded modelling
language for convex optimisation upon which PyPortfolioOpt's efficient frontier functionality lies.

As a brief aside, I should note that while "efficient frontier" optimisation is technically a very
specific method, I tend to use it as a blanket term (interchangeably with mean-variance
optimisation) to refer to anything similar, such as minimising variance.

.. tip::

    You can find complete examples in the relevant cookbook `recipe <https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb>`_.


Structure
=========

As shown in the definition of a convex problem, there are essentially two things we need to specify:
the optimisation objective, and the optimisation constraints. For example, the classic portfolio
optimisation problem is to **minimise risk** subject to a **return constraint** (i.e the portfolio
must return more than a certain amount). From an implementation perspective, however, there is
not much difference between an objective and a constraint. Consider a similar problem, which is to
**maximize return** subject to a **risk constraint** -- now, the role of risk and return have swapped.

To that end, PyPortfolioOpt defines an :py:mod:`objective_functions` module that contains objective functions
(which can also act as constraints, as we have just seen). The actual optimisation occurs in the :py:class:`efficient_frontier.EfficientFrontier` class.
This class provides straightforward methods for optimising different objectives (all documented below).

However, PyPortfolioOpt was designed so that you can easily add new constraints or objective terms to an existing problem.
For example, adding a regularisation objective (explained below) to a minimum volatility objective is as simple as::

    ef = EfficientFrontier(expected_returns, cov_matrix)  # setup
    ef.add_objective(objective_functions.L2_reg)  # add a secondary objective
    ef.min_volatility()  # find the portfolio that minimises volatility and L2_reg

.. tip::

    If you would like to plot the efficient frontier, take a look at the :ref:`plotting` module.

Basic Usage
===========

.. automodule:: pypfopt.efficient_frontier

    .. autoclass:: EfficientFrontier

        .. automethod:: __init__

            .. note::

                As of v0.5.0, you can pass a collection (list or tuple) of (min, max) pairs
                representing different bounds for different assets.

            .. tip::

                If you want to generate short-only portfolios, there is a quick hack. Multiply
                your expected returns by -1, then optimise a long-only portfolio.

        .. automethod:: min_volatility

        .. automethod:: max_sharpe

            .. caution::

                Because ``max_sharpe()`` makes a variable substitution, additional objectives may
                not work as intended.


        .. automethod:: max_quadratic_utility

            .. note::

                ``pypfopt.black_litterman`` provides a method for calculating the market-implied
                risk-aversion parameter, which gives a useful estimate in the absence of other
                information!

        .. automethod:: efficient_risk

            .. caution::

                If you pass an unreasonable target into :py:meth:`efficient_risk` or
                :py:meth:`efficient_return`, the optimiser will fail silently and return
                weird weights. *Caveat emptor* applies!

        .. automethod:: efficient_return

        .. automethod:: portfolio_performance

            .. tip::

                If you would like to use the ``portfolio_performance`` function independently of any
                optimiser (e.g for debugging purposes), you can use::

                    from pypfopt import base_optimizer

                    base_optimizer.portfolio_performance(
                        weights, expected_returns, cov_matrix, verbose=True, risk_free_rate=0.02
                    )

.. note::

    PyPortfolioOpt defers to cvxpy's default choice of solver. If you would like to explicitly
    choose the solver, simply pass the optional ``solver = "ECOS"`` kwarg to the constructor.
    You can choose from any of the `supported solvers <https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver>`_,
    and pass in solver params via ``solver_options`` (a ``dict``). 

Adding objectives and constraints
=================================

EfficientFrontier inherits from the BaseConvexOptimizer class. In particular, the functions to
add constraints and objectives are documented below:


.. class:: pypfopt.base_optimizer.BaseConvexOptimizer

    .. automethod:: add_constraint

    .. automethod:: add_sector_constraints

    .. automethod:: add_objective


Objective functions
===================

.. automodule:: pypfopt.objective_functions
    :members:



.. _L2-Regularisation:

More on L2 Regularisation
=========================

As has been discussed in the :ref:`user-guide`, efficient frontier optimisation often
results in many weights being negligible, i.e the efficient portfolio does not end up
including most of the assets. This is expected behaviour, but it may be undesirable
if you need a certain number of assets in your portfolio.

In order to coerce the efficient frontier optimiser to produce more non-negligible
weights, we add what can be thought of as a "small weights penalty" to all
of the objective functions, parameterised by :math:`\gamma` (``gamma``). Considering
the minimum variance objective for instance, we have:

.. math::
    \underset{w}{\text{minimise}} ~ \left\{w^T \Sigma w \right\} ~~~ \longrightarrow ~~~
    \underset{w}{\text{minimise}} ~ \left\{w^T \Sigma w + \gamma w^T w \right\}

Note that :math:`w^T w` is the same as the sum of squared weights (I didn't
write this explicitly to reduce confusion caused by :math:`\Sigma` denoting both the
covariance matrix and the summation operator). This term reduces the number of
negligible weights, because it has a minimum value when all weights are
equally distributed, and maximum value in the limiting case where the entire portfolio
is allocated to one asset. I refer to it as **L2 regularisation** because it has
exactly the same form as the L2 regularisation term in machine learning, though
a slightly different purpose (in ML it is used to keep weights small while here it is
used to make them larger).

.. note::

    In practice, :math:`\gamma` must be tuned to achieve the level
    of regularisation that you want. However, if the universe of assets is small
    (less than 20 assets), then ``gamma=1`` is a good starting point. For larger
    universes, or if you want more non-negligible weights in the final portfolio,
    increase ``gamma``.


Efficient Semivariance
======================

The mean-variance optimisation methods described above can be used whenever you have a vector
of expected returns and a covariance matrix.

However, you may want to construct the efficient frontier for an entirely different risk model.
For example, instead of penalising volatility (which is symmetric), you may want to penalise
only the downside deviation (with the idea that upside volatility actually benefits you!)

There are two approaches to the mean-semivariance optimisation problem. The first is to use a
heuristic (i.e "quick and dirty") solution: pretending that the semicovariance matrix
(implemented in :py:mod:`risk_models`) is a typical covariance matrix and doing standard
mean-variance optimisation. It can be shown that this *does not* yield a portfolio that
is efficient in mean-semivariance space (though it might be a good-enough approximation).

Fortunately, it is possible to write mean-semivariance optimisation as a convex problem
(albeit one with many variables), that can be solved to give an "exact" solution.
For example, to maximise return for a target semivariance
:math:`s^*` (long-only), we would solve the following problem:

.. math::

    \begin{equation*}
    \begin{aligned}
    & \underset{w}{\text{maximise}} & & w^T \mu \\
    & \text{subject to} & & n^T n \leq s^*  \\
    &&& B w - p + n = 0 \\
    &&& w^T \mathbf{1} = 1 \\
    &&& n \geq 0 \\
    &&& p \geq 0. \\
    \end{aligned}
    \end{equation*}

Here, **B** is the :math:`T \times N` (scaled) matrix of excess returns:
``B = (returns - benchmark) / sqrt(T)``. Additional linear equality constraints and
convex inequality constraints can be added. 

PyPortfolioOpt allows users to optimise along the efficient semivariance frontier
via the :py:class:`EfficientSemivariance` class. :py:class:`EfficientSemivariance` inherits from
:py:class:`EfficientFrontier`, so it has the same utility methods 
(e.g :py:func:`add_constraint`, :py:func:`portfolio_performance`), but finds portfolios on the mean-semivariance
frontier. Note  that some of the parent methods, like :py:func:`max_sharpe` and :py:func:`min_volatility`
are not applicable to mean-semivariance portfolios, so calling them returns an error.

:py:class:`EfficientSemivariance` has a slightly different API to :py:class:`EfficientFrontier`. Instead of passing
in a covariance matrix, you should past in a dataframe of historical/simulated returns (this can be constructed
from your price dataframe using the helper method :py:func:`expected_returns.returns_from_prices`). Here
is a full example, in which we seek the portfolio that minimises the semivariance for a target
annual return of 20%::

    from pypfopt import expected_returns, EfficientSemivariance

    df = ... # your dataframe of prices
    mu = expected_returns.mean_historical_returns(df)
    historical_returns = expected_returns.returns_from_prices(df)

    es = EfficientSemivariance(mu, historical_returns)
    es.efficient_return(0.20)

    # We can use the same helper methods as before
    weights = es.clean_weights() 
    print(weights)
    es.portfolio_performance(verbose=True)

The ``portfolio_performance`` method outputs the expected portfolio return, semivariance,
and the Sortino ratio (like the Sharpe ratio, but for downside deviation).

Interested readers should refer to Estrada (2007) [2]_ for more details. I'd like to thank 
`Philipp Schiele <https://github.com/phschiele>`_ for authoring the bulk
of the efficient semivariance functionality and documentation (all errors are my own). The
implementation is based on Markowitz et al (2019) [3]_.

.. caution::

    Finding portfolios on the mean-semivariance frontier is computationally harder
    than standard mean-variance optimisation: our implementation uses ``2T + N`` optimisation variables, 
    meaning that for 50 assets and 3 years of data, there are about 1500 variables.  
    While :py:class:`EfficientSemivariance` allows for additional constraints/objectives in principle,
    you are much more likely to run into solver errors. I suggest that you keep :py:class:`EfficientSemivariance`
    problems small and minimally constrained.

.. autoclass:: pypfopt.efficient_frontier.EfficientSemivariance
    :members:
    :exclude-members: max_sharpe, min_volatility

Efficient CVaR
==============

The **conditional value-at-risk** (a.k.a **expected shortfall**) is a popular measure of tail risk. The CVaR can be
thought of as the average of losses that occur on "very bad days", where "very bad" is quantified by the parameter
:math:`\beta`.

For example, if we calculate the CVaR to be 10% for :math:`\beta = 0.95`, we can be 95% confident that the worst-case
average daily loss will be 3%. Put differently, the CVaR is the average of all losses so severe that they only occur
:math:`(1-\beta)\%` of the time. 

While CVaR is quite an intuitive concept, a lot of new notation is required to formulate it mathematically (see
the `wiki page <https://en.wikipedia.org/wiki/Expected_shortfall>`_ for more details). We will adopt the following
notation: 

- *w* for the vector of portfolio weights
- *r* for a vector of asset returns (daily), with probability distribution :math:`p(r)`.
- :math:`L(w, r) = - w^T r` for the loss of the portfolio
- :math:`\alpha` for the portfolio value-at-risk (VaR) with confidence :math:`\beta`.

The CVaR can then be written as:

.. math::
    CVaR(w, \beta) = \frac{1}{1-\beta} \int_{L(w, r) \geq \alpha (w)} L(w, r) p(r)dr.

This is a nasty expression to optimise because we are essentially integrating over VaR values. The key insight
of Rockafellar and Uryasev (2001) [4]_ is that we can can equivalently optimise the following convex function:

.. math:: 
    F_\beta (w, \alpha) = \alpha + \frac{1}{1-\beta} \int [-w^T r - \alpha]^+ p(r) dr,

where :math:`[x]^+ = \max(x, 0)`. The authors prove that minimising :math:`F_\beta(w, \alpha)` over all
:math:`w, \alpha` minimises the CVaR. Suppose we have a sample of *T* daily returns (these
can either be historical or simulated). The integral in the expression becomes a sum, so the CVaR optimisation
problem reduces to a linear program:

.. math::

    \begin{equation*}
    \begin{aligned}
    & \underset{w, \alpha}{\text{minimise}} & & \alpha + \frac{1}{1-\beta} \frac 1 T \sum_{i=1}^T u_i \\
    & \text{subject to} & & u_i \geq 0  \\
    &&&  u_i \geq -w^T r_i - \alpha. \\
    \end{aligned}
    \end{equation*}

This formulation introduces a new variable for each datapoint (similar to Efficient Semivariance), so
you may run into performance issues for long returns dataframes. At the same time, you should aim to
provide a sample of data that is large enough to include tail events. 

.. autoclass:: pypfopt.efficient_frontier.EfficientCVaR
    :members:
    :exclude-members: max_sharpe, min_volatility, max_quadratic_utility

Efficient CDaR
==============

The **conditional drawdown at risk** is a more exotic measure of tail risk. It tries to alleviate the problem
of above measure in that it takes into account the timespan of serious decreases in value. The CDaR can be
thought of as the average of losses that occur on "very bad periods", where "very bad" is quantified by the parameter
:math:`\beta`. The drawdown is defined as the difference in non-compounded return to the previous peak.

Put differently, the CDaR is the average of all drawdowns so severe that they only occur
:math:`(1-\beta)\%` of the time. When :math:`\beta = 1` CDaR is simply the maximum drawdown.

While drawdown is quite an intuitive concept, a lot of new notation is required to formulate it mathematically (see
`the wiki page <https://en.wikipedia.org/wiki/Drawdown_(economics)>`_ for more details). We will adopt the following
notation:

- *w* for the vector of portfolio weights
- *r* for a vector of cumulative asset returns (daily), with probability distribution :math:`p(r(t))`.
- :math:`D(w, r, t) = \max_{\tau<t}(w^T r(\tau))-w^T r(t)` for the drawdown of the portfolio
- :math:`\alpha` for the portfolio drawdown (DaR) with confidence :math:`\beta`.

The CDaR can then be written as:

.. math::
    CDaR(w, \beta) = \frac{1}{1-\beta} \int_{D(w, r, t) \geq \alpha (w)} D(w, r, t) p(r(t))dr(t).

This is a nasty expression to optimise because we are essentially integrating over VaR values. The key insight
of Chekhlov, Rockafellar and Uryasev (2005) [5]_ is that we can can equivalently optimise the following convex function. Analogous
to CVaR, this can be transformed to a linear problem.

.. autoclass:: pypfopt.efficient_frontier.EfficientCDaR
    :members:
    :exclude-members: max_sharpe, min_volatility, max_quadratic_utility


.. _custom-optimisation:

Custom optimisation problems
============================

Previously we described an API for adding constraints and objectives to one of the core
optimisation problems in the :py:class:`EfficientFrontier` class. However, what if you aren't interested
in anything related to ``max_sharpe()``, ``min_volatility()``, ``efficient_risk()`` etc and want to
set up a completely new problem to optimise for some custom objective?

The :py:class:`EfficientFrontier` class inherits from the ``BaseConvexOptimizer``, which allows you to
define your own optimisation problem. You can either optimise some generic ``convex_objective``
(which *must* be built using ``cvxpy`` atomic functions -- see `here <https://www.cvxpy.org/tutorial/functions/index.html>`_)
or a ``nonconvex_objective``, which uses ``scipy.optimize`` as the backend and thus has a completely
different API. For more examples, check out this `cookbook recipe
<https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/3-Advanced-Mean-Variance-Optimisation.ipynb>`_.

    .. class:: pypfopt.base_optimizer.BaseConvexOptimizer

        .. automethod:: convex_objective

        .. automethod:: nonconvex_objective


References
==========

.. [1] Boyd, S.; Vandenberghe, L. (2004). `Convex Optimization <https://web.stanford.edu/~boyd/cvxbook/>`_.
.. [2] Estrada, J (2007). `Mean-Semivariance Optimization: A Heuristic Approach <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1028206>`_.
.. [3] Markowitz, H.; Starer, D.; Fram, H.; Gerber, S. (2019). `Avoiding the Downside <https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf>`_.
.. [4] Rockafellar, R.; Uryasev, D. (2001). `Optimization of conditional value-at-risk <https://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf>`_
.. [5] Chekhlov, A.; Rockafellar, R.; Uryasev, D. (2005). `Drawdown measure in portfolio optimization <http://www.math.columbia.edu/~chekhlov/IJTheoreticalAppliedFinance.8.1.2005.pdf>`_
