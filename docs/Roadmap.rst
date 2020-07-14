.. _roadmap:

#####################
Roadmap and Changelog
#####################


Roadmap
=======

These are some of the things that I am thinking of adding in the near future. If you
have any other feature requests, please raise them using GitHub
`issues <https://github.com/robertmartin8/PyPortfolioOpt/issues>`_

- Optimising for higher moments (i.e skew and kurtosis)
- Factor modelling: doable but not sure if it fits within the API.
- Proper CVaR optimisation – remove NoisyOpt and use linear programming
- Monte Carlo optimisation with custom distributions
- Open-source backtests using either `Backtrader <https://www.backtrader.com/>`_ or
  `Zipline <https://github.com/quantopian/zipline>`_.
- Further support for different risk/return models

1.2.0
=====

- Added Idzorek's method for calculating the ``omega`` matrix given percentage confidences.
- Fixed max sharpe to allow for custom constraints
- Grouped sector constraints
- Improved error tracebacks
- Adding new cookbook for examples (in progress).
- Packaging: added bettter instructions for windows, added docker support.


1.2.1
-----

Fixed critical ordering bug in sector constraints

1.2.2
-----

Matplotlib now required dependency; support for pandas 1.0. 

1.2.3
-----

- Added support for changing solvers and verbose output
- Changed dict to OrderedDict to support python 3.5
- Improved packaging/dependencies: simplified requirements.txt, improved processes before pushing.

1.2.4
-----

- Fixed bug in Ledoit-Wolf shrinkage calculation.
- Fixed bug in plotting docs that caused them not to render. 

1.1.0
=====

- Multiple additions and improvements to ``risk_models``:
    
  - Introduced a new API, in which the function ``risk_models.risk_matrix(method="...")`` allows
    all the different risk models to be called. This should make testing easier.
  - All methods now accept returns data instead of prices, if you set the flag ``returns_data=True``.
- Automatically fix non-positive semidefinite covariance matrices!

- Additions and improvements to ``expected_returns``:

  - Introduced a new API, in which the function ``expected_returns.return_model(method="...")`` allows
    all the different return models to be called. This should make testing easier.
  - Added option to 'properly' compound returns.
  - Added the James-Stein shrinkage estimator
  - Added the CAPM return model.

- ``from pypfopt import plotting``: moved all plotting functionality into a new class and added
  new plots. All other plotting functions (scattered in different classes) have been retained,
  but are now deprecated.


1.0.0
=====

- Migrated backend from ``scipy`` to ``cvxpy`` and made significant breaking changes to the API

  - PyPortfolioOpt is now significantly more robust and numerically stable.
  - These changes will not affect basic users, who can still access features like ``max_sharpe()``.
  - However, additional objectives and constraints (including L2 regularisation) are now 
    explicitly added before optimising some 'primary' objective.

- Added basic plotting capabilities for the efficient frontier, hierarchical clusters, 
  and HRP dendrograms.
- Added a basic transaction cost objective.
- Made breaking changes to some modules and classes so that PyPortfolioOpt is easier to extend
  in future:
  
  - Replaced ``BaseScipyOptimizer`` with ``BaseConvexOptimizer``
  - ``hierarchical_risk_parity`` was replaced by ``hierarchical_portfolios`` to leave the door open for other hierarchical methods.
  - Sadly, removed CVaR optimisation for the time being until I can properly fix it.

1.0.1
-----

Fixed minor issues in CLA: weight bound bug, ``efficient_frontier`` needed weights to be called, ``set_weights`` not needed.

1.0.2
-----

Fixed small but important bug where passing ``expected_returns=None`` fails. According to the docs, users
should be able to only pass covariance if they want to only optimise min volatility.


0.5.0
=====

- Black-Litterman model and docs.
- Custom bounds per asset
- Improved ``BaseOptimizer``, adding a method that writes weights
  to text and fixing a bug in ``set_weights``.
- Unconstrained quadratic utility optimisation (analytic)
- Revamped docs, with information on types of attributes and
  more examples.

0.5.1
-----

Fixed an error with dot products by amending the pandas requirements.

0.5.2
-----

Made PuLP, sklearn, noisyopt optional dependencies to improve installation
experience.

0.5.3
-----

- Fixed an optimisation bug in ``EfficientFrontier.efficient_risk``. An error is now
  thrown if optimisation fails.
- Added a hidden API to change the scipy optimiser method. 

0.5.4
-----

- Improved the Black-Litterman linear algebra to avoid inverting the uncertainty matrix. 
  It is now possible to have 100% confidence in views.
- Clarified regarding the role of tau.
- Added a ``pipfile`` for ``pipenv`` users.
- Removed Value-at-risk from docs to discourage usage until it is properly fixed.

0.5.5
-----

Began migration to cvxpy by changing the discrete allocation backend from PuLP to cvxpy. 

0.4.0
=====

- Major improvements to ``discrete_allocation``. Added functionality to allocate shorts;
  modified the linear programming method suggested by `Dingyuan Wang <https://github.com/gumblex>`_;
  added postprocessing section to User Guide.
- Further refactoring and docs for ``HRPOpt``.
- Major documentation update, e.g to support custom optimisers

0.4.1
-----

- Added CLA back in after getting permission from Dr Marcos López de Prado
- Added more tests for different risk models.

0.4.2
-----

- Minor fix for ``clean_weights``
- Removed official support for python 3.4.
- Minor improvement to semicovariance, thanks to `Felipe Schneider <https://github.com/schneiderfelipe>`_.

0.4.3
-----

- Added ``prices_from_returns`` utility function and provided better docs for ``returns_from_prices``.
- Added ``cov_to_corr`` method to produce correlation matrices from covariance matrices.
- Fixed readme examples.



0.3.0
=====

- Merged an amazing PR from `Dingyuan Wang <https://github.com/gumblex>`_ that rearchitects
  the project to make it more self-consistent and extensible.
- New algorithm: ML de Prado's CLA
- New algorithms for converting continuous allocation to discrete (using linear
  programming).
- Merged a `PR <https://github.com/robertmartin8/PyPortfolioOpt/pull/22>`__ implementing Single Factor and
  Constant Correlation shrinkage.

0.3.1
-----

Merged `PR <https://github.com/robertmartin8/PyPortfolioOpt/pull/23>`__ from `TommyBark <https://github.com/TommyBark>`_ 
fixing a bug in the arguments of a call to ``portfolio_performance``.

0.3.3
-----

Migrated the project internally to use the ``poetry`` dependency manager. Will still keep ``setup.py`` and ``requirements.txt``, but ``poetry`` is now the recommended way to interact with PyPortfolioOpt.

0.3.4
-----

Refactored shrinkage models, including single factor and constant correlation.



0.2.0
=====

- Hierarchical Risk Parity optimisation
- Semicovariance matrix
- Exponential covariance matrix
- CVaR optimisation
- Better support for custom objective functions
- Multiple bug fixes (including minimum volatility vs minimum variance)
- Refactored so all optimisers inherit from a ``BaseOptimizer``.

0.2.1
-----

- Included python 3.7 in travis build
- Merged PR from `schneiderfelipe <https://github.com/schneiderfelipe>`_ to fix an error message.


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

0.1.1
-----

Minor bug fixes and documentation
