.. _plotting:

########
Plotting
########


.. automodule:: pypfopt.plotting

    .. autofunction:: _plot_io

    .. tip::

        To save the plot, pass ``filename="somefile.png"`` as a keyword argument to any of
        the methods below.

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
