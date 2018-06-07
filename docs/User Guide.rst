.. _user-guide:

##########
User Guide
##########

This is designed to be a pratical guide, mostly aimed at users who are interested in a quick way of
optimally combining some assets (most likely equities). However, I point out areas that a more advanced
user might like to take note of, as they may be suitable springboards for more advanced optimisation
techniques. Details about the parameters are left for the respective documentation pages (please see
the sidebar). 

PyPortfolioOpt is designed with modulatrity in mind; the below flowchart sums up the current 
functionality and overall layout of PyPortfolioOpt.

.. image:: ../media/conceptual_flowchart_v1-grey.png

Processing historical prices
============================

Efficient frontier optimisation requires two things: the expected returns of the assets, and the 
covariance matrix (or more generally, a *risk model* quantifying asset risk). PyPortfolioOpt provides 
methods for estimating both (located in :py:mod:`expected_returns` and :py:mod:`risk_models`
respectively), but also supports users who would like to use their own models. 

However, I assume that most users will (at least initially) prefer to use the built-ins. 
In this case, all you need to supply is a dataset of historical prices for your assets.
This dataset should look something like the one below::

                    XOM        RRC        BBY         MA        PFE        JPM  
    date
    2010-01-04  54.068794  51.300568  32.524055  22.062426  13.940202  35.175220
    2010-01-05  54.279907  51.993038  33.349487  21.997149  13.741367  35.856571
    2010-01-06  54.749043  51.690697  33.090542  22.081820  13.697187  36.053574
    2010-01-07  54.577045  51.593170  33.616547  21.937523  13.645634  36.767757
    2010-01-08  54.358093  52.597733  32.297466  21.945297  13.756095  36.677460

The index should consist of dates or timestamps, and each column should represent the
timeseries of prices for an asset. A dataset of real-life stock prices has been included
in the `tests folder <https://github.com/robertmartin8/PyPortfolioOpt/tree/master/tests>`_
of the GitHub repo. 

.. note::

    Pricing data does not have to be daily, but the frequency should ideally
    be the same across all assets (workarounds exist but are not pretty).

After reading your historical prices into a pandas dataframe ``df``, you need to decide between
the avaialble methods for estimating expected returns and the covariance matrix.
Sensible defaults are :py:func:`expected_returns.mean_historical_return()` and
the Ledoit Wolf shrinkage estimate of the covariance matrix found in
:py:class:`risk_models.CovarianceShrinkage`. It is simply a matter of applying the relevant
functions to the price dataset:

.. code:: python

    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage

    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()

``mu`` will then be a pandas series of estimated expected returns for each asset, and ``S`` will
be the estimated covariance matrix (part of it is shown below)::

            GOOG      AAPL        FB      BABA      AMZN        GE       AMD  \
    GOOG  0.045529  0.022143  0.006389  0.003720  0.026085  0.015815  0.021761
    AAPL  0.022143  0.207037  0.004334  0.002954  0.058200  0.038102  0.084053
    FB    0.006389  0.004334  0.029233  0.003770  0.007619  0.003008  0.005804
    BABA  0.003720  0.002954  0.003770  0.013438  0.004176  0.002011  0.006332
    AMZN  0.026085  0.058200  0.007619  0.004176  0.276365  0.038169  0.075657
    GE    0.015815  0.038102  0.003008  0.002011  0.038169  0.083405  0.048580
    AMD   0.021761  0.084053  0.005804  0.006332  0.075657  0.048580  0.388916


Now that we have expected returns and a risk model, we are ready to move on to the actual
portfolio optimisation.


Efficient Frontier Optimisation
===============================

Efficient Frontier Optimisation is based on `Harry Markowitz's 1952 classic 
<https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1952.tb01525.x>`_, which turned 
portfolio management into a science. The key insight is that by combining assets with different 
expected returns and volatilities, one can decide on a matehmatically optimal allocation.

If :math:`w` is the weight vector of stocks with expected returns :math:`\mu`, then the
portfolio return is equal to each stock's weight multiplied by its return, i.e :math:`w^T \mu`.
The portfolio risk in terms of the covariance matrix :math:`\Sigma` is given by :math:`w^T \Sigma w`.
Portfolio optimisation can then be regarded as a convex optimisation problem, and a solution can be
found using quadratic programming. If we denote the target return as :math:`\mu^*`, the precise
statement of the long-only portfolio optimisation problem is as follows:

.. math::

    \begin{equation*}
    \begin{aligned}
    & \underset{w}{\text{minimise}} & & w^T \Sigma w \\
    & \text{subject to} & & w^T\mu \geq \mu^*\\
    &&& w^T\mathbf{1} = 1 \\
    &&& w_i \geq 0 \\
    \end{aligned}
    \end{equation*}

If we vary the target return, we will get a different set of weights (i.e a different portfolio) -
the set of all these optimal portfolios is referred to as the **efficient frontier**.

.. image:: ../media/efficient_frontier.png
   :align: center

Each dot on this diagram represents a different possible portfolio, with darker blue corresponding
to 'better' portfolios (in terms of the Sharpe Ratio). The dotted black line is the efficient frontier
itself. The triangular markers correspond to the best portfolios for different optimisation objectives.

The Sharpe ratio is the portfolio's return less the risk free rate, per unit risk (volatility).

.. math::
    SR = \frac{R_P - R_f}{\sigma}

It is particularly important because it measures the portfolio returns, adjusted for risk.
So in practice, rather than trying to minimise volatility for a given target return
(as per Markowitz 1952), it often makes more sense to just find the portfolio that maximises
the Sharpe ratio. 

This is implemented in...


