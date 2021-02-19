.. _faq:

####
FAQs
####

Constraining the number of assets
---------------------------------

Unfortunately, cardinality constraints are not convex, making them difficult to implement.

However, we can treat it as a mixed-integer program and solve (provided you have access to a solver). 
or small problems with less than 1000 variables and constraints, you can use the community version of CPLEX:
``pip install cplex``. In the below example, we limit the portfolio to at most 10 assets::

    ef = EfficientFrontier(mu, S, solver=cp.CPLEX)
    booleans = cp.Variable(len(ef.tickers), boolean=True)
    ef.add_constraint(lambda x: x <= booleans)
    ef.add_constraint(lambda x: cp.sum(booleans) <= 10)
    ef.min_volatility()

This does not play well with ``max_sharpe``, and needs to be modified for different bounds.
See `this issue <https://github.com/robertmartin8/PyPortfolioOpt/issues/243>`_ for further discussion.

Tracking error
--------------

Tracking error can either be used as an objective (as described in :ref:`efficient-frontier`) or
as a constraint. This is an example of adding a tracking error constraint::

    from objective functions import ex_ante_tracking_error

    benchmark_weights = ...  # benchmark

    ef = EfficientFrontier(mu, S)
    ef.add_constraint(ex_ante_tracking_error, cov_matrix=ef.cov_matrix,
                      benchmark_weights=benchmark_weights)
    ef.min_volatility()


