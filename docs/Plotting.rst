.. _plotting:

########
Plotting
########

All of the optimization functions in :py:class:`EfficientFrontier` produce a single optimal portfolio.
However, you may want to plot the entire efficient frontier. This efficient frontier can be thought
of in several different ways:

1. The set of all :py:func:`efficient_risk` portfolios for a range of target risks
2. The set of all :py:func:`efficient_return` portfolios for a range of target returns
3. The set of all :py:func:`max_quadratic_utility` portfolios for a range of risk aversions.

The :py:mod:`plotting` module provides support for all three of these approaches. To produce
a plot of the efficient frontier, you should instantiate your :py:class:`EfficientFrontier` object
and add constraints like you normally would, but *before* calling an optimization function (e.g with
:py:func:`ef.max_sharpe`), you should pass this the instantiated object into :py:func:`plot.plot_efficient_frontier`:: 

    ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
    ef.add_constraint(lambda w: w[0] >= 0.2)
    ef.add_constraint(lambda w: w[2] == 0.15)
    ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    plt.show()

This produces the following plot:

    .. image:: ../media/ef_plot.png
        :width: 80%
        :align: center
        :alt: the Efficient Frontier

You can explicitly pass a range of parameters (risk, utility, or returns) to generate a frontier:: 
    
    # 100 portfolios with risks between 0.10 and 0.30
    risk_range = np.linspace(0.10, 0.40, 100)
    plotting.plot_efficient_frontier(ef, ef_param="risk", ef_param_range=risk_range,
                                    show_assets=True, showfig=True)


We can easily generate more complex plots. The following script plots both the efficient frontier and
randomly generated (suboptimal) portfolios, coloured by the Sharpe ratio::

    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the tangency portfolio
    ef.max_sharpe()
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt(np.diag(w @ S @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ef_scatter.png", dpi=200)
    plt.show()

This is the result:

    .. image:: ../media/ef_scatter.png
        :width: 80%
        :align: center
        :alt: the Efficient Frontier with random portfolios

Documentation reference
=======================

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
