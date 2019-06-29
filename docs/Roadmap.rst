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

0.3.3
-----

- Migrated the project internally to use the ``poetry`` dependency manager. Will still keep ``setup.py`` and 
  ``requirements.txt``, but ``poetry`` is now the recommended way to interact with ``PyPortfolioOpt``

0.3.1
-----

- Merged `PR <https://github.com/robertmartin8/PyPortfolioOpt/pull/23>`_ from `TommyBark <https://github.com/TommyBark>`_
  fixing a bug in the arguments of a call to ``portfolio_performance``.

0.3.0
=====

- Merged an amazing PR from `Dingyuan Wang <https://github.com/gumblex>`_ that rearchitects
  the project to make it more self-consistent and extensible.
- New algorithm: ML de Prado's CLA
- New algorithms for converting continuous allocation to discrete (using linear
  programming).
- Merged a `PR <https://github.com/robertmartin8/PyPortfolioOpt/pull/22>`_ implementing Single Factor and
  Constant Correlation shrinkage.

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
