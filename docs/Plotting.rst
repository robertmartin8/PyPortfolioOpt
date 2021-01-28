.. _plotting:

########
Plotting
########

All of the optimisation functions in :py:class:`EfficientFrontier` produce a single optimal portfolio.
However, you may want to plot the entire efficient frontier. This efficient frontier can be thought
of in several different ways:

1. The set of all :py:func:`efficient_risk` portfolios for a range of target risks
2. The set of all :py:func:`efficient_return` portfolios for a range of target returns
3. The set of all :py:func:`max_quadratic_utility` portfolios for a range of risk aversions.

The :py:mod:`plotting` module provides support for all three of these approaches. To produce
a plot of the efficient frontier, you should instantiate your :py:class:`EfficientFrontier` object
and add constraints like you normally would, but *before* calling an optimisation function (e.g with
:py:func:`ef.max_sharpe`), you should pass this the instantiated object into :py:func:`plot.plot_efficient_frontier`:: 

    ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
    ef.add_constraint(lambda w: w[0] >= 0.2)
    ef.add_constraint(lambda w: w[2] == 0.15)
    ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

    # 100 portfolios with risks between 0.10 and 0.30
    risk_range = np.linspace(0.10, 0.40, 100)
    ax = plotting.plot_efficient_frontier(ef, ef_param="risk", ef_param_range=risk_range,
                               show_assets=True, showfig=True)

This produces the following plot -- you can set attributes using the returned ``ax`` object: 

    .. image:: ../media/ef_plot.png
        :width: 80%
        :align: center
        :alt: the Efficient Frontier


.. automodule:: pypfopt.plotting

    .. tip::

        To save the plot, pass ``filename="somefile.png"`` as a keyword argument to any of
        the plotting functions. This (along with some other kwargs) get passed through
        :py:func:`_plot_io` before being returned.

    .. autofunction:: _plot_io

    .. autofunction:: plot_covariance

    .. image:: ../media/corrplot.png
        :align: center
        :width: 80%
        :alt: plot of the covariance matrix

    .. autofunction:: plot_dendrogram

    .. image:: ../media/dendrogram.png
        :width: 80%
        :align: center
        :alt: return clusters

    .. autofunction:: plot_efficient_frontier

    .. image:: ../media/cla_plot.png
        :width: 80%
        :align: center
        :alt: the Efficient Frontier

    .. autofunction:: plot_weights

    .. image:: ../media/weight_plot.png
        :width: 80%
        :align: center
        :alt: bar chart to show weights
