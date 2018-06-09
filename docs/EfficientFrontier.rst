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

Optimisation
============

Optimisation uses `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.
I realise that most python optimisation projects use `cvxopt <https://cvxopt.org/>`_
instead, but I do think that scipy.optimize is far cleaner and much more readable
(as per the Zen ofPython, "Readability counts"). That being said, scipy.optimize
arguably has worse documentation, though in the end I felt that it was intuitive
enough to justify the lack of explained examples. Because they are both based on
`LAPACK <http://www.netlib.org/lapack/>`_, I don't see why performance should
differ significantly, but if it transpires that cvxopt is faster by an order of
magnitude, I will definitely consider switching.


.. automodule:: pypfopt.efficient_frontier

    .. autoclass:: EfficientFrontier
        :members:

        .. automethod:: __init__

            .. note::

                As a rule of thumb, any parameters that can apply to all optimisers
                are instance variables (passed when you are initialising the object).

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
the minimum volatility objective for instance, we have:

.. math::
    \underset{w}{\text{minimise}} ~ \left\{w^T \Sigma w \right\} ~~~ \longrightarrow ~~~
    \underset{w}{\text{minimise}} ~ \left\{w^T \Sigma w + \gamma w^T w \right\}

Note that :math:`w^T w` is the same as the sum of squared weights (I didn't
write this explicitly to reduce confusion caused by :math:`\Sigma` denoting both the
covariance matrix and the summation operator). This term serves the purpose of
reducing neglibile weights, because it has a minimum value when all weights are
equally distributed, and maximum value in the limiting case where the entire portfolio
is allocated to one asset. I refer to it as **L2 regularisation** because it has
exactly the same form as the L2 regularisation term in machine learning, though
a slightly different purpose (in ML it is used to keep weights small).

.. note::

    In practice, :math:`\gamma` must be tuned to achieve the level
    of regularisation that you want. However, if the universe of assets is small
    (less than 20 assets), then ``gamma=1`` is a good starting point, but try
    increasing it further if you want more assets in the final portfolio.
