###########
Risk Models
###########

In addition to the expected asset returns, mean-variance optimisation requires a
**risk model**, some way of quantifying asset risk. The most common risk model is
probably the covariance matrix, an abstract statistical entity that describes the
volatility of the asset returns and how they vary with one another. This is
important because one of the principles of diversification is that risk can be
reduced by making many uncorrelated bets (and correlation is just normalised
covariance).

The problem, however, is that in practice we do not have access to the covariance
matrix (in the same way that we don't have access to expected returns) – the only
thing we can do is to make estimates based on past data. The most straightforward
approach is to just calculate the **sample covariance matrix** based on historical
returns, but relatively recent (post 2000) research indicates that there are much
more robust statistical estimators of the covariance matrix.

.. attention::

    Estimation of the covariance matrix is a very deep and actively-researched
    topic that involves statistics, neconometrics, and numerical/computational
    approaches. Please note that I am not an expert, but I have made an effort
    to familiarise myself with the seminal papers in the field.


.. automodule:: risk_models

    .. autofunction:: sample_cov

        Much like the mean historical return, this is the textbook default approach. The
        entries in the sample covariance matrix (which we denote as *S*) are the sample
        covariances between the *i* th and *j* th asset (the diagonals consist of
        variances). Although the sample covariance matrix is an unbiased estimator of the
        covariance matrix, i.e :math:`E(S) = \Sigma`, in practice it suffers from
        misspecification error and a lack of robustness. This is particularly problematic
        in mean-variance optimisation, because the optimiser may give extra credence to
        the erroneous values.

        .. note::

            This should *not* be your default choice! Please use a shrinkage estimator
            instead.


    .. autofunction:: min_cov_determinant

        The minimum covariance determinant (MCD) estimator is designed to be robust to
        outliers and 'contaminated' data [3]_. An efficient estimator is implemented in the
        :py:mod:`sklearn.covariance` module, which is based on the algorithm presented in
        Rousseeuw 1999 [4]_.


Shrinkage estimators
====================

A great starting point for those interested in understanding shrinkage estimators is
*Honey, I Shrunk the Sample Covariance Matrix* [1]_ by Ledoit and Wolf, which does a
good job at capturing the intuition behind shrinkage estimators. We will adopt the
notation used therein. I have written a summary of this article, which is available
on my `website <http://reasonabledeviations.science/notes/papers/ledoit_wolf_covariance/>`_.
A more rigorous reference can be found in Ledoit and Wolf (2001) [2]_.

The essential idea is that the unbiased but often misspecified sample covariance can be
combined with a structured estimator :math:`F`, using the below formula (where
:math:`\delta` is the shrinkage constant):

.. math::
    \hat{\Sigma} = \delta F + (1-\delta) S

It is called shrinkage, because it can be thought of as "shrinking" the sample
covariance matrix towards the other estimator, which is accordingly called the
**shrinkage target**. There are many possible choices for a shrinkage target,
but popular choices include:

- The diagonal matrix with sample variances on the diagonals and zeroes elsewhere,
  i.e assuming no covariance between assets.
- Sharpe's single factor (or single-index) model, which basically uses a stock's
  beta to the market as a risk model.
- The constant-correlation model, in which all pairwise correlations are set to
  the average correlation (sample variances are unchanged)

The optimal shrinkage constant :math:`\delta` depends on the choice of shrinkage
target, and the actual formula for the constant depends on the implementation.
PyPortfolioOpt offers two methods for calculating the shrinkage constant:

- Ledoit-Wolf shrinkage, using the formulae in their 2004 paper [5]_.
- Oracle approximating shrinkage (OAS), invented by Chen et al. (2010) [6]_, which
  has a lower mean-squared error than Ledoit-Wolf shrinkage when samples are
  Gaussian or near-Gaussian.

.. tip::

    For most use cases, I would just go with Ledoit Wolf shrinkage, as recommended by
    `Quantopian <https://www.quantopian.com/>`_ in their lecture series on quantitative
    finance.


.. autoclass:: CovarianceShrinkage
    :members:

    .. automethod:: __init__


References
==========

.. [1] Ledoit, O., & Wolf, M. (2003). `Honey, I Shrunk the Sample Covariance Matrix <http://www.ledoit.net/honey.pdf>`_ The Journal of Portfolio Management, 30(4), 110–119. https://doi.org/10.3905/jpm.2004.110
.. [2] Ledoit, O., & Wolf, M. (2001). `Improved estimation of the covariance matrix of stock returns with an application to portfolio selection <http://www.ledoit.net/ole2.pdf>`_, 10, 603–621.
.. [3] 	Rousseeuw, P., J (1984). `Least median of squares regression <http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/LeastMedianOfSquares.pdf>`_. The Journal of the American Statistical Association, 79, 871-880.
.. [4] Rousseeuw, P., J (1999). `A Fast Algorithm for the Minimum Covariance Determinant Estimator <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.45.5870&rep=rep1&type=pdf>`_. The Journal of the American Statistical Association, 41, 212-223.
.. [5] Ledoit, O., & Wolf, M. (2004) `A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices <http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf>`_, Journal of Multivariate Analysis, 88(2), 365-411
.. [6] Chen et al. (2010),  `Shrinkage Algorithms for MMSE Covariance Estimation <https://arxiv.org/pdf/0907.4698.pdf>`_, IEEE Transactions on Signals Processing, 58(10), 5016-5029.
