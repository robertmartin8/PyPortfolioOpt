.. _roadmap:

#####################
Roadmap and Changelog
#####################


Roadmap
=======

These are some of the things that I am thinking of adding in the near future. If you
have any other feature requests, please raise them using GitHub
`issues <https://github.com/robertmartin8/PyPortfolioOpt/issues>`_

- Custom utility functions, including risk aversion
- Plotting the efficient frontier
- Different optimisation objectives
- Monte Carlo optimisation with custom distributions
- Black-Litterman portfolio selection
- Open-source backtests using either `Backtrader <https://www.backtrader.com/>`_ or
  `Zipline <https://github.com/quantopian/zipline>`_.
- Genetic optimisation methods (tentative)
- Further support for different risk/return models

0.2.1
-----

- Included python 3.7 in travis build
- Merged PR from `schneiderfelipe <https://github.com/schneiderfelipe>`_ to fix error message.

0.2.0
=====

- Hierarchical Risk Parity optimisation
- Semicovariance matrix
- Exponential covariance matrix
- CVaR optimisation
- Better support for custom objective functions
- Multiple bug fixes (including minimum volatility vs minimum variance)
- Refactored so all optimisers inherit from a ``BaseOptimizer``.


0.1.1
-----

Minor bug fixes and documentation


0.1.0
=====

Initial release:

- Efficient frontier (max sharpe, min variance, target risk/return)
- L2 regularisation
- Discrete allocation
- Mean historical returns, exponential mean returns
- Sample covariance, sklearn wrappers.
- Tests
- Docs
