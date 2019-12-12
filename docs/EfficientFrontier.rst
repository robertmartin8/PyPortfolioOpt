.. _efficient-frontier:

###############################
Efficient Frontier Optimisation
###############################

The implementation of efficient frontier optimisation in PyPortfolioOpt is separated
into the :py:mod:`objective_functions` and :py:mod:`efficient_frontier` modules. It
was designed this way because in my mind there is a clear conceptual separation
between the optimisation objective and the actual optimisation method – if we
wanted to use something other than mean-variance optimisation via quadratic programming,
these objective functions would still be applicable.

It should be noted that while efficient frontier optimisation is technically a very
specific method, I tend to use it as a blanket term (interchangeably with mean-variance
optimisation) to refer to anything similar, such as minimising variance.

Optimisation
============

PyPortfolioOpt uses `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.
I realise that most python optimisation projects use `cvxopt <https://cvxopt.org/>`_
instead, but I do think that scipy.optimize is far cleaner and much more readable
(as per the Zen of Python, "Readability counts"). That being said, scipy.optimize
arguably has worse documentation, though ultimately I felt that it was intuitive
enough to justify the lack of explained examples. Because they are both based on
`LAPACK <http://www.netlib.org/lapack/>`_, I don't see why performance should
differ significantly, but if it transpires that cvxopt is faster by an order of
magnitude, I will definitely consider switching.

.. tip::

    If you would like to plot the efficient frontier, take a look at the :ref:`cla`.  


.. automodule:: pypfopt.efficient_frontier

    .. autoclass:: EfficientFrontier
        :members:
        :exclude-members: custom_objective

        .. automethod:: __init__

            .. note::

                As of v0.5.0, you can pass a collection (list or tuple) of (min, max) pairs
                representing different bounds for different assets.

        .. automethod:: max_sharpe

            .. note::

                If you want to generate short-only portfolios, there is a quick hack. Multiply
                your expected returns by -1, then maximise a long-only portfolio.

        .. automethod:: max_unconstrained_utility

            .. note::

                pypfopt.BlackLitterman provides a method for calculating the market-implied
                risk-aversion parameter, which gives a useful estimate in the absence of other
                information!

.. caution::

    If you pass an unreasonable target into :py:meth:`efficient_risk` or
    :py:meth:`efficient_return`, the optimiser will fail silently and return
    weird weights. *Caveat emptor* applies!

Objective functions
===================

.. automodule:: pypfopt.objective_functions
    :members:

One of the experimental features implemented in PyPortfolioOpt is the L2 regularisation
parameter ``gamma``, which is discussed below.

.. _L2-Regularisation:

L2 Regularisation
=================

As has been discussed in the :ref:`user-guide`, efficient frontier optimisation often
results in many weights being negligible, i.e the efficient portfolio does not end up
including most of the assets. This is expected behaviour, but it may be undesirable
if you need a certain number of assets in your portfolio.

In order to coerce the efficient frontier optimiser to produce more non-negligible
weights, I have added what can be thought of as a "small weights penalty" to all
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

.. _custom-objectives:

Custom objectives
=================

Though it is simple enough to modify ``objective_functions.py`` to implement
a custom objective (indeed, this is the recommended approach for long-term use),
I understand that most users would find it much more convenient to pass a
custom objective into the optimiser without having to edit the source files.

Thus, v0.2.0 introduces a simple API within the ``EfficientFrontier`` object for
optimising your own objective function.

The first step is to define the objective function, which must take an array
of weights as input (with optional additional arguments), and return a single
float corresponding to the cost. As an example, we will pretend that L2
regularisation is not built-in and re-implement it below::

    def my_objective_function(weights, cov_matrix, k):
        variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return variance + k * (weights ** 2).sum()

Next, we instantiate the ``EfficientFrontier`` object, and pass the objectives
function (and all required arguments) into ``custom_objective()``::

    ef = EfficientFrontier(mu, S)
    weights = ef.custom_objective(my_objective_function, ef.cov_matrix, 0.3)


.. caution::
    It is assumed that the objective function you define will be solvable
    by sequential quadratic programming. If this isn't the case, you may
    experience silent failure.
